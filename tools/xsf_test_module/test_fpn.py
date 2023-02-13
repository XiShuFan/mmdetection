from mmdet.models import FPN
import torch

fpn_2d_cfg = dict(
    # 经过backbone输出的不同stage特征图channel
    in_channels=[256, 512, 1024, 2048],
    # 经过neck输出的channel
    out_channels=256,
    # neck需要输出多少个特征图
    num_outs=5,
    # 从backbone输出stage的起始编号
    start_level=1,
    # 从backbone输出stage的结束编号
    end_level=-1,
    # 增加卷积获得额外的输出，否则使用maxpool获得额外的输出。可选参数：on_input, on_lateral, on_output
    add_extra_convs='on_input',
    # 是否在extra_convs之前添加relu
    relu_before_extra_convs=False,
    # 是否在lateral卷积层之前使用BN
    no_norm_on_lateral=False,
    # 卷积层配置
    conv_cfg=dict(type='Conv2d'),
    # BN层配置
    norm_cfg=dict(type='BN2d', requires_grad=True),
    # 激活层配置
    act_cfg=dict(type='ReLU'),
    # 上采样配置，用在functional.interpolate中
    upsample_cfg=dict(mode='nearest'),
    # 加载预训练模型，或者初始化配置
    init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')
)


fpn_3d_cfg = dict(
    # 经过backbone输出的不同stage特征图channel
    in_channels=[2, 4, 8, 16],
    # 经过neck输出的channel
    out_channels=16,
    # neck需要输出多少个特征图
    num_outs=6,
    # 从backbone输出stage的起始编号
    start_level=0,
    # 从backbone输出stage的结束编号
    end_level=-1,
    # 增加卷积获得额外的输出，否则使用maxpool获得额外的输出。可选参数：on_input, on_lateral, on_output
    add_extra_convs='on_input',
    # 是否在extra_convs之前添加relu
    relu_before_extra_convs=True,
    # 是否在lateral卷积层之前使用BN
    no_norm_on_lateral=True,
    # 卷积层配置
    conv_cfg=dict(type='Conv3d'),
    # BN层配置
    norm_cfg=dict(type='BN3d', requires_grad=True),
    # 激活层配置
    act_cfg=dict(type='ReLU'),
    # 上采样配置，用在functional.interpolate中
    upsample_cfg=dict(mode='nearest'),
    # 加载预训练模型，或者初始化配置
    init_cfg=dict(type='Xavier', layer='Conv3d', distribution='uniform')
)


def test_2d(device):
    fpn = FPN(**fpn_2d_cfg)
    fpn.init_weights()
    fpn.to(device)
    fpn.eval()

    inputs_2d = [
        torch.rand([2, 256, 256, 256]).to(device),
        torch.rand([2, 512, 128, 128]).to(device),
        torch.rand([2, 1024, 64, 64]).to(device),
        torch.rand([2, 2048, 32, 32]).to(device)
    ]

    outputs = fpn(inputs_2d)
    return outputs


def test_3d(device):
    fpn = FPN(**fpn_3d_cfg)
    fpn.init_weights()
    fpn.to(device)
    fpn.eval()

    inputs_3d = [
        torch.rand([2, 2, 128, 128, 128]).to(device),
        torch.rand([2, 4, 64, 64, 64]).to(device),
        torch.rand([2, 8, 32, 32, 32]).to(device),
        torch.rand([2, 16, 16, 16, 16]).to(device)
    ]

    outputs = fpn(inputs_3d)
    return outputs


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    outputs = test_3d(device)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')
