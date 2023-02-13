from mmdet.core import DeltaXYZWHDBBoxCoder
import torch

bbox_coder_cfg = dict(
    # 标准化回归参数均值，对于3D包围盒要写成[.0, .0, .0, .0， .0， .0]
    target_means=[.0, .0, .0, .0, .0, .0],
    # 标准化回归参数方差
    target_stds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    # 将预测的proposal还原为包围盒后，裁剪超出图像边界的部分
    clip_border=True,
    # YOLOF的中心偏移参数
    add_ctr_clamp=False,
    ctr_clamp=32
)

def test_encode(coder, bboxes, gt_bboxes):
    return coder.encode(bboxes, gt_bboxes)

def test_decode(coder, bboxes, pred_bboxes):
    return coder.decode(bboxes, pred_bboxes)

if __name__ == '__main__':
    coder = DeltaXYZWHDBBoxCoder(**bbox_coder_cfg)

    bboxes_dim2 = torch.tensor(
        [
            [1, 2, 2, 4, 5, 5],
            [1.3, 2.5, 2.5, 4.6, 5.2, 5.2]
        ]
    )

    gt_bboxes_dim2 = torch.tensor(
        [
            [2, 3, 3, 5, 6, 6],
            [2.3, 3.5, 3.5, 5.6, 6.2, 6.2]
        ]
    )

    pred_bboxes_dim2 = torch.tensor(
        [
            [0.2, 0.5, 0.5, 1, 2, 2],
            [0.6, 0.5, 0.5, 3, 1, 1]
        ]
    )

    bboxes_dim3 = torch.tensor(
        [
            [
                [1, 2, 2, 4, 5, 5],
                [1.3, 2.5, 2.5, 4.6, 5.2, 5.2]
            ],
            [
                [1, 2, 2, 4, 5, 5],
                [1.3, 2.5, 2.5, 4.6, 5.2, 5.2]
            ],
        ]
    )

    gt_bboxes_dim3 = torch.tensor(
        [
            [
                [2, 3, 3, 5, 6, 6],
                [2.3, 3.5, 3.5, 5.6, 6.2, 6.2]
            ],
            [
                [2, 3, 3, 5, 6, 6],
                [2.3, 3.5, 3.5, 5.6, 6.2, 6.2]
            ]
        ]
    )

    pred_bboxes_dim3 = torch.tensor(
        [
            [
                [0.2, 0.5, 0.5, 1, 2, 2],
                [0.6, 0.5, 0.5, 3, 1, 1]
            ],
            [
                [0.2, 0.5, 0.5, 1, 2, 2],
                [0.6, 0.5, 0.5, 3, 1, 1]
            ],
        ]
    )

    result = test_decode(coder, bboxes_dim3, pred_bboxes_dim3)

    print(result)

