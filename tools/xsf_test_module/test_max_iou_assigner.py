from mmdet.core import MaxIoUAssigner
import torch

cfg = dict(
    # 正样本bbox的IoU threshold
    pos_iou_thr=0.5,
    # 负样本bbox的IoU 的threshold
    neg_iou_thr=0.4,
    # 确定为正样本的最小IoU，以防存在gt bbox没有对应的bbox。与match_low_quality相关
    min_pos_iou=0,
    # 如果gt box有多个bbox有相同的最大IoU，将他们全部对应到gt box
    gt_max_assign_all=True,
    # 不忽略任何bboxes
    ignore_iof_thr=-1,
    # 是否要计算bboxes 和 gt_bboxes_ignore 之间的iof
    ignore_wrt_candidates=True,
    # 是否允许低质量匹配，与min_pos_iou相关
    match_low_quality=False,
    # 在gpu上实现
    gpu_assign_thr=-1,
    # 计算IoU的方法
    iou_calculator=dict(type='BboxOverlaps3D')
)

assigner = MaxIoUAssigner(**cfg)

# bbox和gt
gt = torch.tensor([[1, 2, 2, 3, 4, 4], [6, 6, 6, 7, 7, 7]])
labels = torch.tensor([102, 550])
# 最后一个为丢弃样本
bboxes = torch.tensor([[1, 2, 3, 3, 4, 4], [5, 5, 5, 6, 6, 6], [1, 2, 2, 3, 4, 4], [6, 6, 6, 7.1, 7.1, 7.6], [6, 6, 6, 7.1, 7.1, 7.9]])


assign_mat = assigner.assign(bboxes, gt, gt_labels=labels)

print(assign_mat)