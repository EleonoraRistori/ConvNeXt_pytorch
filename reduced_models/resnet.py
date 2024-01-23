from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from models.resnet import _resnet, ResNet, Bottleneck


def resnet_reduced(num_classes) -> ResNet:
    return _resnet(Bottleneck, [2, 2, 4, 2], num_classes, groups=1, width_per_group=24)


def resnet_nostem(num_classes) -> ResNet:
    net = _resnet(Bottleneck, [2, 2, 4, 2], num_classes, groups=1, width_per_group=24)
    net.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


def resnet_small(num_classes) -> ResNet:
    return _resnet(Bottleneck, [1, 1, 2, 1], num_classes, groups=1, width_per_group=24)



