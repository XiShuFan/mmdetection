from mmdet.models import FocalLoss
from mmdet.models import L1Loss
import torch

def test_focal_loss():
    cfg = dict(
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        reduction='mean',
        loss_weight=1.0,
        activated=False
    )
    loss = FocalLoss(**cfg)
    pred = torch.tensor(
        [
            [0.1, 0.1, 0.1, 0.6, 0.06, 0.04],
            [1, 0, 0, 0, 0, 0]
        ]
    )

    target = torch.tensor(
        [
            0,
            5
        ]
    )
    result = loss(pred, target)
    return result

def test_l1_loss():
    cfg = dict(reduction='mean', loss_weight=1.0)
    loss = L1Loss(**cfg)
    pred = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [2.3, 5.1, 23, 2.3, 3.6, 5.2]
        ]
    )

    target = torch.tensor(
        [
            [2.1, 2.3, 2.3, 7.1, 5.5, 6.6],
            [9, 5, 6, 7, 2.1, 2]
        ]
    )
    result = loss(pred, target)
    return result


if __name__ == '__main__':
    print(test_focal_loss())