from mmdet.core import BboxOverlaps3D
import torch


iou_calculator = BboxOverlaps3D()
bboxes1 = torch.tensor([[1, 2, 2, 3, 4, 4]])
bboxes2 = torch.tensor([[1, 2, 3, 3, 4, 4], [5, 5, 5, 6, 6, 6]])

# iou矩阵
iou_mat = iou_calculator(bboxes1, bboxes2)
print(iou_mat.shape)
print(iou_mat)
