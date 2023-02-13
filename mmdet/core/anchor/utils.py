# Copyright (c) OpenMMLab. All rights reserved.
import torch


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def anchor_inside_flags(flat_anchors,
                        valid_flags,
                        img_shape,
                        allowed_border=0):
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int, optional): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a \
            valid range.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def anchor_inside_flags_3d(flat_anchors,
                        valid_flags,
                        img_shape,
                        allowed_border=0):
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int, optional): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a \
            valid range.
    """
    img_d, img_h, img_w = img_shape[:3]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] >= -allowed_border) & \
            (flat_anchors[:, 3] < img_w + allowed_border) & \
            (flat_anchors[:, 4] < img_h + allowed_border) & \
            (flat_anchors[:, 5] < img_d + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def calc_region(bbox, ratio, featmap_size=None):
    """Calculate a proportional bbox region.

    The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

    Args:
        bbox (Tensor): Bboxes to calculate regions, shape (n, 4).
        ratio (float): Ratio of the output region.
        featmap_size (tuple): Feature map size used for clipping the boundary.

    Returns:
        tuple: x1, y1, x2, y2
    """
    x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
    y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
    x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
    y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1])
        y1 = y1.clamp(min=0, max=featmap_size[0])
        x2 = x2.clamp(min=0, max=featmap_size[1])
        y2 = y2.clamp(min=0, max=featmap_size[0])
    return (x1, y1, x2, y2)


def calc_region_3d(bbox, ratio, featmap_size=None):
    """Calculate a proportional bbox region.

    The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

    Args:
        bbox (Tensor): Bboxes to calculate regions, shape (n, 4).
        ratio (float): Ratio of the output region.
        featmap_size (tuple): Feature map size used for clipping the boundary.

    Returns:
        tuple: x1, y1, x2, y2
    """
    x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[3]).long()
    y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[4]).long()
    z1 = torch.round((1 - ratio) * bbox[2] + ratio * bbox[5]).long()
    x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[3]).long()
    y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[4]).long()
    z2 = torch.round(ratio * bbox[2] + (1 - ratio) * bbox[5]).long()
    if featmap_size is not None:
        z1 = z1.clamp(min=0, max=featmap_size[2])
        x1 = x1.clamp(min=0, max=featmap_size[1])
        y1 = y1.clamp(min=0, max=featmap_size[0])
        z2 = z2.clamp(min=0, max=featmap_size[2])
        x2 = x2.clamp(min=0, max=featmap_size[1])
        y2 = y2.clamp(min=0, max=featmap_size[0])
    return (x1, y1, z1, x2, y2, z2)
