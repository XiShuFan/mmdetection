from mmdet.core import AnchorGenerator
import torch

cfg = dict(
    # 多阶段特征图相对于原始图像的stride
    strides=[16, 32],
    # anchor的depth和height和width的比例
    ratios=[1.0],
    # anchor的缩放比例，不可以与octave_base_scale和scales_per_octave同时设置
    scales=[1.0],
    # 多阶段特征图的anchor的基础大小，如果None，则使用strides（挺合理）
    base_sizes=[9, 18],
    # 是否先乘上缩放因子，再乘上宽高比
    scale_major=True,
    # The base scale of octave
    octave_base_scale=None,
    # Number of scales for each octave
    scales_per_octave=None,

    # 调整anchor中心点，一般不用
    centers=None,
    center_offset=0.
)

anchor_generator = AnchorGenerator(**cfg)
all_anchors = anchor_generator.grid_priors([(2, 2), (1, 1)], device='cpu')
print(all_anchors)

