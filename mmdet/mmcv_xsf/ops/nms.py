import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mmcv.utils import deprecated_api_warning

array_like_type = Union[Tensor, np.ndarray]

def nmsop3d(bboxes, scores, iou_threshold):
    """
    将预测的包围盒nms处理，得到不重叠的包围盒
    Args:
        bboxes: Tensor
        scores: Tensor
        iou_threshold:

    Returns:

    """
    device = bboxes.device
    bboxes = bboxes.cpu().numpy()
    scores = scores.cpu().numpy()
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    z1 = bboxes[:, 2]
    x2 = bboxes[:, 3]
    y2 = bboxes[:, 4]
    z2 = bboxes[:, 5]
    # 计算每个包围盒的体积
    areas = (z2 - z1) * (y2 - y1) * (x2 - x1)
    # 需要保留的包围盒索引
    keep = []
    # 按照预测分数降序排序，选取预测分数最大的包围盒
    index = scores.argsort()[::-1]

    while index.size > 0:
        # 每次都是当前预测值最大的
        i = index[0]
        keep.append(i)

        # 寻找左上角最大点
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        z11 = np.maximum(z1[i], z1[index[1:]])

        # 寻找右下角最小点
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        z22 = np.minimum(z2[i], z2[index[1:]])

        w = np.maximum(0, x22 - x11)
        h = np.maximum(0, y22 - y11)
        d = np.maximum(0, z22 - z11)

        overlaps = w * h * d

        union = areas[i] + areas[index[1:]] - overlaps
        # 避免除0
        eps = np.full((union.shape[0],), 1e-6)
        union = np.maximum(union, eps)

        ious = overlaps / union

        idx = np.where(ious <= iou_threshold)[0]
        index = index[idx+1]
    return torch.tensor(keep).to(device)

class NMSop3d(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, bboxes: Tensor, scores: Tensor, iou_threshold: float,
                offset: int, score_threshold: float, max_num: int) -> Tensor:
        is_filtering_by_score = score_threshold > 0
        if is_filtering_by_score:
            valid_mask = scores > score_threshold
            bboxes, scores = bboxes[valid_mask], scores[valid_mask]
            valid_inds = torch.nonzero(
                valid_mask, as_tuple=False).squeeze(dim=1)

        # 不知道这里加载的是什么
        if offset != 0:
            raise ValueError('but this should be 0')
        inds = nmsop3d(bboxes, scores, iou_threshold)

        if max_num > 0:
            inds = inds[:max_num]
        if is_filtering_by_score:
            inds = valid_inds[inds]
        return inds

    @staticmethod
    def symbolic(g, bboxes, scores, iou_threshold, offset, score_threshold,
                 max_num):
        from ..onnx import is_custom_op_loaded
        has_custom_op = is_custom_op_loaded()
        # TensorRT nms plugin is aligned with original nms in ONNXRuntime
        is_trt_backend = os.environ.get('ONNX_BACKEND') == 'MMCVTensorRT'
        if has_custom_op and (not is_trt_backend):
            return g.op(
                'mmcv::NonMaxSuppression',
                bboxes,
                scores,
                iou_threshold_f=float(iou_threshold),
                offset_i=int(offset))
        else:
            from torch.onnx.symbolic_opset9 import select, squeeze, unsqueeze

            from ..onnx.onnx_utils.symbolic_helper import _size_helper

            boxes = unsqueeze(g, bboxes, 0)
            scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)

            if max_num > 0:
                max_num = g.op(
                    'Constant',
                    value_t=torch.tensor(max_num, dtype=torch.long))
            else:
                dim = g.op('Constant', value_t=torch.tensor(0))
                max_num = _size_helper(g, bboxes, dim)
            max_output_per_class = max_num
            iou_threshold = g.op(
                'Constant',
                value_t=torch.tensor([iou_threshold], dtype=torch.float))
            score_threshold = g.op(
                'Constant',
                value_t=torch.tensor([score_threshold], dtype=torch.float))
            nms_out = g.op('NonMaxSuppression', boxes, scores,
                           max_output_per_class, iou_threshold,
                           score_threshold)
            return squeeze(
                g,
                select(
                    g, nms_out, 1,
                    g.op(
                        'Constant',
                        value_t=torch.tensor([2], dtype=torch.long))), 1)


@deprecated_api_warning({'iou_thr': 'iou_threshold'})
def nms3d(boxes: array_like_type,
        scores: array_like_type,
        iou_threshold: float,
        offset: int = 0,
        score_threshold: float = 0,
        max_num: int = -1) -> Tuple[array_like_type, array_like_type]:
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either torch tensor or numpy array. GPU NMS will be used
    if the input is gpu tensor, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
        scores (torch.Tensor or np.ndarray): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).
        score_threshold (float): score threshold for NMS.
        max_num (int): maximum number of boxes after NMS.

    Returns:
        tuple: kept dets (boxes and scores) and indice, which always have
        the same data type as the input.

    Example:
        >>> boxes = np.array([[49.1, 32.4, 51.0, 35.9],
        >>>                   [49.3, 32.9, 51.0, 35.3],
        >>>                   [49.2, 31.8, 51.0, 35.4],
        >>>                   [35.1, 11.5, 39.1, 15.7],
        >>>                   [35.6, 11.8, 39.3, 14.2],
        >>>                   [35.3, 11.5, 39.9, 14.5],
        >>>                   [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
        >>> scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],\
               dtype=np.float32)
        >>> iou_threshold = 0.6
        >>> dets, inds = nms(boxes, scores, iou_threshold)
        >>> assert len(inds) == len(dets) == 3
    """
    assert isinstance(boxes, (Tensor, np.ndarray))
    assert isinstance(scores, (Tensor, np.ndarray))
    is_numpy = False
    if isinstance(boxes, np.ndarray):
        is_numpy = True
        boxes = torch.from_numpy(boxes)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    assert boxes.size(1) == 6
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    inds = NMSop3d.apply(boxes, scores, iou_threshold, offset, score_threshold,
                       max_num)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    if is_numpy:
        dets = dets.cpu().numpy()
        inds = inds.cpu().numpy()
    return dets, inds



def batched_nms_3d(boxes: Tensor,
                    scores: Tensor,
                    idxs: Tensor,
                    nms_cfg: Optional[Dict],
                    class_agnostic: bool = False) -> Tuple[Tensor, Tensor]:
    r"""Performs non-maximum suppression in a batched fashion.

    Returns:
        tuple: kept dets and indice.

        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 7:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :3].max() + boxes[..., 3:6].max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :3] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 3:7]],
                                      dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms3d')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep