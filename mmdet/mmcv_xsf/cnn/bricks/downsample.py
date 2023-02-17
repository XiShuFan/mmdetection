from mmcv.cnn.bricks.registry import DOWNSAMPLE_LAYERS
from torch import nn
from typing import Dict, Optional

DOWNSAMPLE_LAYERS.register_module('MaxPool1d', module=nn.MaxPool1d)
DOWNSAMPLE_LAYERS.register_module('MaxPool2d', module=nn.MaxPool2d)
DOWNSAMPLE_LAYERS.register_module('MaxPool3d', module=nn.MaxPool3d)

DOWNSAMPLE_LAYERS.register_module('AvgPool1d', module=nn.AvgPool1d)
DOWNSAMPLE_LAYERS.register_module('AvgPool2d', module=nn.AvgPool2d)
DOWNSAMPLE_LAYERS.register_module('AvgPool3d', module=nn.AvgPool3d)

def build_downsample_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build down sample layer.

    Args:
        cfg (None or dict): The down sample layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an down sample layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding down sample layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding down sample layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        # cfg_ = dict(type='MaxPool2d')
        raise ValueError('cfg must not be None')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in DOWNSAMPLE_LAYERS:
        raise KeyError(f'Unrecognized layer type {layer_type}')
    else:
        downsample_layer = DOWNSAMPLE_LAYERS.get(layer_type)

    layer = downsample_layer(*args, **kwargs, **cfg_)

    return layer