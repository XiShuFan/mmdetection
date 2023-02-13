from mmdet.models import ResNet
import torch

# 配置dict
resnet_2d_cfg = dict(
    # resnet50架构
    depth=50,
    # 输入通道数
    in_channels=3,
    # stem部分输出通道数，默认与base_channel一致
    stem_channels=None,
    # 基准输出channel
    base_channels=64,
    # resnet使用的阶段数，默认为4
    num_stages=4,
    # 每个stage第一个block的stride，用于减半长宽。第一个stage用maxpool减半长宽
    strides=(1, 2, 2, 2),
    # 每阶段膨胀卷积参数
    dilations=(1, 1, 1, 1),
    # 输出的中间特征图，默认4阶段全部输出
    out_indices=(0, 1, 2, 3),
    # 实现风格
    style='pytorch',
    # 是否在stem部分使用深度卷积
    deep_stem=False,
    # 下采样是否使用全局池化，否则使用卷积下采样
    avg_down=False,
    # avg pool 配置
    avg_cfg=dict(tyoe='AvgPool2d'),
    # 需要冻结的层
    frozen_stages=1,
    # 卷积层配置，可以更改为Conv1d, Conv2d, Conv3d。但是不要去改kernel等参数，因为这个是公用的，不同层不一样
    conv_cfg=dict(type='Conv2d'),
    # bn层配置，可以更改为BN1d, BN2d, BN3d
    norm_cfg=dict(type='BN2d', requires_grad=True),
    # max pool配置
    maxpool_cfg=dict(type='MaxPool2d'),
    # bn层是否使用eval模式，不更新参数
    norm_eval=True,
    # 加载预训练模型，或者初始化方法
    init_cfg=dict(type='Kaiming', layer='Conv2d'),
    # dict(type='Pretrained', checkpoint='../checkpoints/resnet/resnet50-19c8e357.pth')
    # dict(type='Kaiming', layer='Conv2d')

    # 以下参数不用
    dcn=None,
    stage_with_dcn=(False, False, False, False),
    plugins=None,
    with_cp=False,
    # 是否在最后一个bn层初始化为0
    zero_init_residual=True,
    pretrained=None
)

resnet_3d_cfg = dict(
    # resnet18架构
    depth=18,
    # 输入通道数
    in_channels=1,
    # stem部分输出通道数，默认与base_channel一致
    stem_channels=None,
    # 基准输出channel
    base_channels=2,
    # resnet使用的阶段数，默认为4
    num_stages=4,
    # 每个stage第一个block的stride，用于减半长宽。第一个stage用maxpool减半长宽
    strides=(1, 2, 2, 2),
    # 每阶段膨胀卷积参数
    dilations=(1, 1, 1, 1),
    # 输出的中间特征图，默认4阶段全部输出
    out_indices=(0, 1, 2, 3),
    # 实现风格
    style='pytorch',
    # 是否在stem部分使用深度卷积
    deep_stem=True,
    # 下采样是否使用全局池化，否则使用卷积下采样
    avg_down=True,
    # avg pool 配置
    avg_cfg=dict(type='AvgPool3d'),
    # 需要冻结的层
    frozen_stages=1,
    # 卷积层配置，可以更改为Conv1d, Conv2d, Conv3d。但是不要去改kernel等参数，因为这个是公用的，不同层不一样
    conv_cfg=dict(type='Conv3d'),
    # bn层配置，可以更改为BN1d, BN2d, BN3d
    norm_cfg=dict(type='BN3d', requires_grad=True),
    # max pool配置
    maxpool_cfg=dict(type='MaxPool3d'),
    # bn层是否使用eval模式，不更新参数
    norm_eval=False,
    # 加载预训练模型，或者初始化方法
    init_cfg=dict(type='Kaiming', layer='Conv3d'),
    # dict(type='Pretrained', checkpoint='../checkpoints/resnet/resnet50-19c8e357.pth')
    # dict(type='Kaiming', layer='Conv2d')

    # 以下参数不用
    dcn=None,
    stage_with_dcn=(False, False, False, False),
    plugins=None,
    with_cp=False,
    # 是否在最后一个bn层初始化为0
    zero_init_residual=True,
    pretrained=None
)


def test_2d(device):
    net = ResNet(**resnet_2d_cfg)
    net.init_weights()
    net.to(device)
    net.eval()

    inputs_2d = torch.rand(2, 3, 1024, 1024)
    inputs_2d = inputs_2d.to(device)
    outs = net.forward(inputs_2d)

    return outs


def test_3d(device):
    net = ResNet(**resnet_3d_cfg)
    net.init_weights()
    net.to(device)
    net.eval()
    inputs_3d = torch.rand(2, 1, 512, 512, 512)
    inputs_3d = inputs_3d.to(device)

    outs = net.forward(inputs_3d)
    return outs


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    outs = test_3d(device)
    for out in outs:
        print(out.shape)


