import torch.nn as nn
from models.convnext import InvertedBottleneck, _resnext


def ConvNeXt(num_classes):
    return _resnext(InvertedBottleneck, [2, 2, 4, 2], num_classes, inplanes=32, dim=[32, 64, 128, 256])


def ConvNeXt_kernel7(num_classes):
    return _resnext(InvertedBottleneck, [2, 2, 4, 2], num_classes, inplanes=32, dim=[32, 64, 128, 256], kernel_size=7)


def ConvNeXt_nostem(num_classes):
    net = _resnext(InvertedBottleneck, [2, 2, 4, 2], num_classes, inplanes=32, dim=[32, 64, 128, 256])
    net.stem = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
    )
    return net


def ConvNeXtkernel7_nostem(num_classes):
    net = _resnext(InvertedBottleneck, [2, 2, 4, 2], num_classes, inplanes=32, dim=[32, 64, 128, 256], kernel_size=7)
    net.stem = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
    )
    return net


def ConvNeXtkernel7_increasedim(num_classes):
    return _resnext(InvertedBottleneck, [2, 2, 4, 2], num_classes, inplanes=42, dim=[42, 84, 168, 336], kernel_size=7)


def ConvNeXtkernel7_increasedim_nostem(num_classes):
    net = _resnext(InvertedBottleneck, [2, 2, 4, 2], num_classes, inplanes=36, dim=[36, 72, 144, 288], kernel_size=7)
    net.stem = nn.Sequential(
        nn.Conv2d(3, 36, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(36),
        nn.ReLU(inplace=True),
    )
    return net

