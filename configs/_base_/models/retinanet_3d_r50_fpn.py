# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        # resnet架构
        depth=18,
        # 输入通道数
        in_channels=1,
        # stem部分输出通道数，默认与base_channel一致
        stem_channels=None,
        # 基准输出channel
        base_channels=2,
        # resnet使用的阶段数，默认为4
        num_stages=4,
        # 每个stage第一个block的stride，用于减半长宽。第一个stage用maxpool减半长宽
        strides=(1, 2, 2, 2),
        # 每阶段膨胀卷积参数
        dilations=(1, 1, 1, 1),
        # 输出的中间特征图，默认4阶段全部输出
        out_indices=(0, 1, 2, 3),
        # 实现风格
        style='pytorch',
        # 是否在stem部分使用深度卷积
        deep_stem=False,
        # 下采样是否使用全局池化，否则使用卷积下采样
        avg_down=False,
        # avg pool 配置
        avg_cfg=dict(type='AvgPool3d'),
        # 需要冻结的层
        frozen_stages=-1,
        # 卷积层配置，可以更改为Conv1d, Conv2d, Conv3d。但是不要去改kernel等参数，因为这个是公用的，不同层不一样
        conv_cfg=dict(type='Conv3d'),
        # bn层配置，可以更改为BN1d, BN2d, BN3d
        norm_cfg=dict(type='BN3d', requires_grad=True),
        # max pool配置
        maxpool_cfg=dict(type='MaxPool3d'),
        # bn层是否使用eval模式，不更新参数
        norm_eval=False,
        # 加载预训练模型，或者初始化方法
        init_cfg=dict(type='Kaiming', layer='Conv3d'),
        # dict(type='Pretrained', checkpoint='../checkpoints/resnet/resnet50-19c8e357.pth')
        # dict(type='Kaiming', layer='Conv2d')
        # 是否在最后一个bn层初始化为0
        zero_init_residual=True,

        # 以下参数不用
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        plugins=None,
        with_cp=False,
        pretrained=None
    ),
    neck=dict(
        type='FPN',
        # 经过backbone输出的不同stage特征图channel
        in_channels=[2, 4, 8, 16],
        # 经过neck输出的channel
        out_channels=2,
        # neck需要输出多少个特征图
        num_outs=4,
        # 从backbone输出stage的起始编号
        start_level=0,
        # 从backbone输出stage的结束编号
        end_level=-1,
        # 增加卷积获得额外的输出，否则使用maxpool获得额外的输出。可选参数：on_input, on_lateral, on_output
        add_extra_convs='on_input',
        # 是否在extra_convs之前添加relu
        relu_before_extra_convs=True,
        # 是否在lateral卷积层之前使用BN
        no_norm_on_lateral=True,
        # 卷积层配置
        conv_cfg=dict(type='Conv3d'),
        # BN层配置
        norm_cfg=dict(type='BN3d', requires_grad=True),
        # 激活层配置
        act_cfg=dict(type='ReLU'),
        # 上采样配置，用在functional.interpolate中
        upsample_cfg=dict(mode='nearest'),
        # 加载预训练模型，或者初始化配置
        init_cfg=dict(type='Xavier', layer='Conv3d', distribution='uniform')
    ),
    bbox_head=dict(
        type='RetinaHead',
        # 数据集的类别数，不包括背景
        num_classes=1,
        # 输入特征图维度
        in_channels=2,
        # RetinaHead的检测头需要堆叠的卷积层个数
        stacked_convs=4,
        # 卷积层配置
        conv_cfg=dict(type='Conv3d'),
        # BN配置
        norm_cfg=dict(type='BN3d', requires_grad=True),
        # anchor生成
        anchor_generator=dict(
            type='AnchorGenerator3D',
            # 多阶段特征图相对于原始图像的stride
            strides=[4, 8, 16, 32],
            # anchor的depth和height和width与base_size的比例
            ratios=[(2, 1, 1)],
            # anchor的缩放比例，不可以与octave_base_scale和scales_per_octave同时设置
            scales=[1.0],
            # 多阶段特征图的anchor的基础大小，如果None，则使用strides（挺合理）
            base_sizes=(10, 20, 20, 30),
            # 是否先乘上缩放因子，再乘上宽高比
            scale_major=True,
            # The base scale of octave
            octave_base_scale=None,
            # Number of scales for each octave
            scales_per_octave=None,

            # 调整anchor中心点，一般不用
            centers=None,
            center_offset=0.
        ),
        # RetinaHead的初始化方法
        init_cfg=dict(
            type='Normal',
            layer='Conv3d',
            std=0.01,
            override=dict(
                type='Normal',
                name='retina_cls',
                std=0.01,
                bias_prob=0.01
            )
        ),

        # RetinaHead检测头使用的hidden layer的channel
        feat_channels=2,
        # 当使用IoULoss类型时，设置为true，将会使用绝对坐标计算loss
        reg_decoded_bbox=False,
        # 包围盒编解码
        bbox_coder=dict(
            type='DeltaXYZWHDBBoxCoder',
            # 标准化回归参数均值，对于3D包围盒要写成[.0, .0, .0, .0， .0， .0]
            target_means=[.0, .0, .0, .0, .0, .0],
            # 标准化回归参数方差
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            # 将预测的proposal还原为包围盒后，裁剪超出图像边界的部分
            clip_border=True,
            # YOLOF的中心偏移参数
            add_ctr_clamp=False,
            ctr_clamp=32
        ),
        # 分类loss
        loss_cls=dict(
            type='FocalLoss',
            # 将预测概率调整到-1~1区间，区别于softmax方法
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0,
            activated=False
        ),
        # 包围盒回归loss
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=1.0)
    ),
    # 训练配置，在train.py代码中使用，会被添加到bbox_head中
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            # 正样本bbox的IoU threshold，高于它为正样本
            pos_iou_thr=0.4,
            # 负样本bbox的IoU 的threshold，低于他为负样本
            neg_iou_thr=0.3,
            # 丢弃样本：在neg_iou_thr和pos_iou_thr之间的样本丢弃
            # 确定为正样本的最小IoU，以防存在gt bbox没有对应的bbox。与match_low_quality相关
            min_pos_iou=0,
            # 如果gt box有多个bbox有相同的最大IoU，将他们全部对应到gt box。前提条件是match_low_quality，我觉得也最好不用。
            gt_max_assign_all=False,
            # 不忽略任何bboxes
            ignore_iof_thr=-1,
            # 是否要计算bboxes 和 gt_bboxes_ignore 之间的iof，前提条件是ignore_iof_thr
            ignore_wrt_candidates=True,
            # 是否允许低质量匹配，与min_pos_iou相关。看了代码，我认为不要用这个更好。
            # 但是加上这个好像训练收敛了，我觉得原因在于我的anchor没有构造好，导致gt对应不到anchor
            match_low_quality=False,
            # 在gpu上实现
            gpu_assign_thr=-1,
            # 计算IoU的方法
            # 这里很重要，需要在这里调试。比如换一下anchor的大小以及比例，看看在不同的iou范围能有多少正样本和负样本
            iou_calculator=dict(type='BboxOverlaps3D')
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    # 测试配置，会被添加到bbox_head中
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.4,
        # 得写一个nms3D
        nms=dict(type='nms', iou_threshold=0.4),
        max_per_img=100
    ),
    # 弃用
    pretrained=None,
    # 整个网络模型的初始化方法
    init_cfg=None
)
