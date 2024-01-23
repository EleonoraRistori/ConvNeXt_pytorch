from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


from models.resnext import _resnext


class InvertedBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            # groups: int = 1,
            # base_width: int = 96,
            kernel_size: int = 3,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 96.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = nn.Identity()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, groups=planes, padding='same', dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Linear(planes, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.conv3 = nn.Linear(planes * self.expansion, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)
        out = out.permute(0, 3, 1, 2)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[InvertedBottleneck],
            layers: List[int],
            num_classes: int = 100,
            zero_init_residual: bool = False,
            inplanes: int = 64,
            dim: List[int] = (64, 128, 256, 512),
            kernel_size: int = 3,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.kernel_size = kernel_size
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


        self.layer1 = self._make_layer(block, dim[0], layers[0])
        self.layer2 = self._make_layer(block, dim[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, dim[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, dim[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[InvertedBottleneck],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=2, stride=2)
                )
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.kernel_size, previous_dilation, norm_layer
            )
        )

        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.stem(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnext(
        block: Type[InvertedBottleneck],
        layers: List[int],
        num_classes: int,
        inplanes: int = 64,
        dim: List[int] = (64, 128, 256, 512),
        kernel_size: int = 3,
        # groups: int = 1,
        # width_per_group: int = 64,
        **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, num_classes, False, inplanes, dim, kernel_size, **kwargs)

    return model


def ConvNeXt(num_classes):
    return _resnext(InvertedBottleneck, [3, 3, 9, 3], num_classes, inplanes=96, dim=[96, 192, 398, 768])


def ConvNeXt_kernel7(num_classes):
    return _resnext(InvertedBottleneck, [3, 3, 9, 3], num_classes, inplanes=64, dim=[64, 128, 256, 512], kernel_size=7)


