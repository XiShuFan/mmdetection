# 图片标准化参数，目前不使用
img_norm_cfg = dict(mean=[0.], std=[1.0], to_rgb=False)

# 训练过程数据预处理流程
train_pipeline = [
    dict(type='LoadNiiFromFile', to_float32=True),
    dict(type='LoadNiiAnnotations', with_bbox=True, with_label=True),
    dict(type='Pad3D', size_divisor=32),
    # 主要将数据转换成tensor，并放到DataContainer中
    dict(type='DefaultFormatBundle3D'),
    dict(type='Collect3D', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# 测试过程数据预处理流程
test_pipeline = [
    dict(type='LoadNiiFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300, 300),
        scale_factor=None,
        flip=False,
        flip_direction='horizontal',
        transforms=[
            dict(type='Resize3D'),
            dict(type='Pad3D', size_divisor=32),
            dict(type='DefaultFormatBundle3D'),
            dict(type='Collect3D', keys=['img'])
        ]
    )
]
data_root = 'D:/Dataset/'
# 数据集设置
data = dict(
    # 本配置中没有指明dataloader，在代码中会自动按照下面两个参数设置
    # 每张gpu处理多少张图片
    samples_per_gpu=1,
    # 加载图片用多少个worker
    workers_per_gpu=2,
    # 训练数据集
    train=dict(
        type='ToothDataset',
        # 标注文件
        ann_file=data_root+'ToothCOCO/annotations/instances_train2023.json',
        # 数据集预处理，到最终输出的流程
        pipeline=train_pipeline,
        # 指定的物体类别，如果None，则使用默认的所有类别
        classes=None,
        # ann_file存在的目录，不用管
        data_root=None,
        # 图像所在目录
        img_prefix=data_root+'ToothCOCO/train2023/',
        # 分割图所在目录
        seg_prefix=None,
        # 分割图片的后缀
        seg_suffix='.gz',
        # 不用管
        proposal_file=None,
        # 设置为true，则annotation不会被加载
        test_mode=False,
        # 不存在包围盒的图片被过滤掉
        filter_empty_gt=True,
        # 不用管
        file_client_args=dict(backend='disk')
    ),
    # 验证数据集
    val=dict(
        type='ToothDataset',
        ann_file=data_root+'ToothCOCO/annotations/instances_val2023.json',
        img_prefix=data_root+'ToothCOCO/val2023/',
        pipeline=test_pipeline
    ),
    # 测试数据集
    test=dict(
        type='ToothDataset',
        ann_file=data_root+'ToothCOCO/annotations/instances_val2023.json',
        img_prefix=data_root+'ToothCOCO/val2023/',
        pipeline=test_pipeline
    )
)
