from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from models.resnext import Bottleneck, InvertedBottleneck, _resnext, ResNet


def ResNext(num_classes) -> ResNet:
    return _resnext(Bottleneck, [2, 2, 4, 2], num_classes, inplanes=32, dim=[32, 64, 128, 256])


def ResNext_nostem(num_classes) -> ResNet:
    net = _resnext(Bottleneck, [2, 2, 4, 2], num_classes)
    net.stem = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
    )
    return net
