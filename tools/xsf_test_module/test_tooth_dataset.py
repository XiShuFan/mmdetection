from mmdet.datasets import build_dataset, build_dataloader
from mmdet.apis import init_random_seed

# 训练过程数据预处理流程
train_pipeline = [
    # 这些预处理流程，到处理数据集的时候再看
    dict(type='LoadNiiFromFile'),
    dict(type='LoadNiiAnnotations', with_bbox=True),
    dict(type='Pad3D', size_divisor=32),
    # 主要将数据转换成tensor，并放到DataContainer中
    dict(type='DefaultFormatBundle3D'),
    dict(type='Collect3D', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_cfg = dict(
    type='ToothDataset',
    # 标注文件
    ann_file='D:/Dataset/ToothCOCO/annotations/instances_train2023.json',
    # 数据集预处理，到最终输出的流程
    pipeline=train_pipeline,
    # 指定的物体类别，如果None，则使用默认的所有类别
    classes=None,
    # ann_file存在的目录，不用管
    data_root=None,
    # 图像所在目录
    img_prefix='D:/Dataset/ToothCOCO/train2023/',
    # 分割图所在目录
    seg_prefix=None,
    # 分割图片的后缀
    seg_suffix='.png',
    # 不用管
    proposal_file=None,
    # 设置为true，则annotation不会被加载
    test_mode=False,
    # 不存在包围盒的图片被过滤掉
    filter_empty_gt=True,
    # 不用管
    file_client_args=dict(backend='disk')
)

train_loader_cfg = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=1,
        dist=False,
        seed=init_random_seed(None, device='cpu'),
        runner_type='EpochBasedRunner',
        persistent_workers=False
)



test_pipeline = [
    dict(type='LoadNiiFromFile', to_float32=True),
    dict(type='LoadNiiAnnotations', with_bbox=True),
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
            dict(type='Collect3D', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]
    )
]

test_cfg = dict(
    type='ToothDataset',
    ann_file='D:/Dataset/ToothCOCO/annotations/instances_val2023.json',
    img_prefix='D:/Dataset/ToothCOCO/val2023/',
    pipeline=test_pipeline
)




test_loader_cfg = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=1,
        dist=False,
        seed=init_random_seed(None, device='cpu'),
        runner_type='EpochBasedRunner',
        persistent_workers=False
)

if __name__ == '__main__':
    dataset = build_dataset(test_cfg)
    print('\tdataset done')
    dataloader = build_dataloader(dataset, **test_loader_cfg)
    print('\tdataloader done')

    for i, data_batch in enumerate(dataloader):
        # 输出图像确认一下正确性
        import SimpleITK as sitk

        img1_tensor = data_batch['img']._data[0][0].squeeze(dim=0)
        img1_array = img1_tensor.cpu().numpy()
        img1_itk = sitk.GetImageFromArray(img1_array)
        sitk.WriteImage(img1_itk, f"D:/Dataset/ToothCOCO/temp/{data_batch['img_metas']._data[0][0]['ori_filename']}")

        img2_tensor = data_batch['img']._data[0][1].squeeze(dim=0)
        img2_array = img2_tensor.cpu().numpy()
        img2_itk = sitk.GetImageFromArray(img2_array)
        sitk.WriteImage(img2_itk, f"D:/Dataset/ToothCOCO/temp/{data_batch['img_metas']._data[0][1]['ori_filename']}")
        print(i)


