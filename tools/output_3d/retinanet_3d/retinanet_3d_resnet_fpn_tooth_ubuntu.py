model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=1,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style='pytorch',
        deep_stem=False,
        avg_down=False,
        avg_cfg=dict(type='AvgPool3d'),
<<<<<<< HEAD
        frozen_stages=-1,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='BN3d', requires_grad=True),
        maxpool_cfg=dict(type='MaxPool3d'),
        norm_eval=False,
        init_cfg=dict(type='Kaiming', layer='Conv3d'),
=======
        frozen_stages=1,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='BN3d', requires_grad=True),
        maxpool_cfg=dict(type='MaxPool3d'),
        norm_eval=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/media/g704-server/新加卷/XiShuFan/pretrain/resnet_50.pth'
        ),
>>>>>>> aa75a67a1379e724299839e5e47dc718e207a6fb
        zero_init_residual=True,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        plugins=None,
        with_cp=False,
        pretrained=None),
    neck=dict(
        type='FPN',
<<<<<<< HEAD
        in_channels=[16, 32, 64, 128],
        out_channels=16,
        num_outs=4,
        start_level=0,
=======
        in_channels=[256, 512, 1024, 2048],
        out_channels=16,
        num_outs=3,
        start_level=1,
>>>>>>> aa75a67a1379e724299839e5e47dc718e207a6fb
        end_level=-1,
        add_extra_convs='on_input',
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='BN3d', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(mode='nearest'),
        init_cfg=dict(type='Xavier', layer='Conv3d', distribution='uniform')),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=16,
        stacked_convs=4,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='BN3d', requires_grad=True),
        anchor_generator=dict(
            type='AnchorGenerator3D',
<<<<<<< HEAD
            strides=[4, 8, 16, 32],
            ratios=[(1.5, 1, 1), (2.5, 1, 1), (3.5, 1, 1)],
            scales=[1.0],
            base_sizes=(10, 20, 20, 30),
=======
            strides=[8, 16, 32],
            ratios=[(1.5, 1.0, 1.0), (2.0, 1.0, 0.5), (2.0, 1.0, 1.0),
                    (1.5, 1.0, 0.5), (1.5, 0.5, 0.5)],
            scales=[1.0],
            base_sizes=(25, 25, 25),
>>>>>>> aa75a67a1379e724299839e5e47dc718e207a6fb
            scale_major=True,
            octave_base_scale=None,
            scales_per_octave=None,
            centers=None,
            center_offset=0.0),
        init_cfg=dict(
            type='Normal',
            layer='Conv3d',
            std=0.01,
            override=dict(
                type='Normal', name='retina_cls', std=0.01, bias_prob=0.01)),
        feat_channels=16,
        reg_decoded_bbox=False,
        bbox_coder=dict(
            type='DeltaXYZWHDBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            clip_border=True,
            add_ctr_clamp=False,
            ctr_clamp=32),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
<<<<<<< HEAD
            loss_weight=1.0,
            activated=False),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.3,
            neg_iou_thr=0.2,
=======
            loss_weight=1.5,
            activated=False),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=1.5)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
>>>>>>> aa75a67a1379e724299839e5e47dc718e207a6fb
            min_pos_iou=0,
            gt_max_assign_all=False,
            ignore_iof_thr=-1,
            ignore_wrt_candidates=True,
            match_low_quality=True,
            gpu_assign_thr=-1,
            iou_calculator=dict(type='BboxOverlaps3D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
<<<<<<< HEAD
        score_thr=0.6,
        nms=dict(type='nms3d', iou_threshold=0.6),
=======
        score_thr=0.4,
        nms=dict(type='nms3d', iou_threshold=0.3),
>>>>>>> aa75a67a1379e724299839e5e47dc718e207a6fb
        max_per_img=100),
    pretrained=None,
    init_cfg=None)
img_norm_cfg = dict(mean=[0.0], std=[1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadNiiFromFile', to_float32=True),
    dict(type='LoadNiiAnnotations', with_bbox=True, with_label=True),
    dict(type='Pad3D', size_divisor=32),
    dict(type='DefaultFormatBundle3D'),
    dict(type='Collect3D', keys=['img', 'gt_bboxes', 'gt_labels'])
]
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
        ])
]
data_root = '/media/g704-server/新加卷/XiShuFan/Dataset/'
data = dict(
<<<<<<< HEAD
    samples_per_gpu=2,
=======
    samples_per_gpu=1,
>>>>>>> aa75a67a1379e724299839e5e47dc718e207a6fb
    workers_per_gpu=2,
    train=dict(
        type='ToothDataset',
        ann_file=
        '/media/g704-server/新加卷/XiShuFan/Dataset/ToothCOCO/annotations/instances_train2023.json',
        pipeline=[
            dict(type='LoadNiiFromFile', to_float32=True),
            dict(type='LoadNiiAnnotations', with_bbox=True, with_label=True),
            dict(type='Pad3D', size_divisor=32),
            dict(type='DefaultFormatBundle3D'),
            dict(type='Collect3D', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=None,
        data_root=None,
        img_prefix=
        '/media/g704-server/新加卷/XiShuFan/Dataset/ToothCOCO/train2023/',
        seg_prefix=None,
        seg_suffix='.gz',
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
        file_client_args=dict(backend='disk')),
    val=dict(
        type='ToothDataset',
        ann_file=
        '/media/g704-server/新加卷/XiShuFan/Dataset/ToothCOCO/annotations/instances_val2023.json',
        img_prefix='/media/g704-server/新加卷/XiShuFan/Dataset/ToothCOCO/val2023/',
        pipeline=[
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
                ])
        ]),
    test=dict(
        type='ToothDataset',
        ann_file=
        '/media/g704-server/新加卷/XiShuFan/Dataset/ToothCOCO/annotations/instances_val2023.json',
        img_prefix='/media/g704-server/新加卷/XiShuFan/Dataset/ToothCOCO/val2023/',
        pipeline=[
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
                ])
        ]))
<<<<<<< HEAD
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
=======
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
>>>>>>> aa75a67a1379e724299839e5e47dc718e207a6fb
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=20,
<<<<<<< HEAD
    warmup_ratio=0.001,
    step=[40, 80])
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=5)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
=======
    warmup_ratio=0.0001,
    step=[70, 90])
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=10)
log_config = dict(interval=2, hooks=[dict(type='TextLoggerHook')])
>>>>>>> aa75a67a1379e724299839e5e47dc718e207a6fb
momentum_config = None
timer_config = dict(type='IterTimerHook')
custom_hooks = [dict(type='NumClassCheckHook')]
evaluation = dict(
    interval=1, metric='bbox', out_dir='/media/g704-server/xsf_tmp_dir')
<<<<<<< HEAD
runner = dict(type='EpochBasedRunner', max_epochs=12)
=======
runner = dict(type='EpochBasedRunner', max_epochs=100)
>>>>>>> aa75a67a1379e724299839e5e47dc718e207a6fb
workflow = [('train', 1)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
work_dir = './output_3d/retinanet_3d'
auto_resume = False
gpu_ids = [0]
