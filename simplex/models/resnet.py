from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import torch
import torch.nn as nn
from simplex_models import Linear as SimpLinear
from simplex_models import Conv2d as SimpConv
from simplex_models import BatchNorm2d as SimpBN
import math


def simp_conv3x3(in_planes: int, out_planes: int, stride: int = 1,
                 groups: int = 1, dilation: int = 1,
                 fix_points=None):
    """3x3 convolution with padding"""
    return SimpConv(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        fix_points=fix_points
    )


def simp_conv1x1(in_planes: int, out_planes: int, stride: int = 1,
                 fix_points=None):
    """1x1 convolution with padding"""
    return SimpConv(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        fix_points=fix_points
    )


class SimplexBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            fix_points=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = SimpBN
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = simp_conv3x3(inplanes, planes, stride,
                                  fix_points=fix_points)
        self.bn1 = norm_layer(planes, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = simp_conv3x3(planes, planes, fix_points=fix_points)
        self.bn2 = norm_layer(planes, fix_points=fix_points)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, coeffs_t) -> Tensor:
        identity = x

        out = self.conv1(x, coeffs_t)
        out = self.bn1(out, coeffs_t)
        out = self.relu(out)

        out = self.conv2(out, coeffs_t)
        out = self.bn2(out, coeffs_t)

        if self.downsample is not None:
            identity = self.downsample(x, coeffs_t)

        out += identity
        out = self.relu(out)

        return out


class Resnet18Simplex(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            fix_points=[False],

    ) -> None:
        super().__init__()
        block = SimplexBasicBlock
        layers = [2, 2, 2, 2]

        if norm_layer is None:
            norm_layer = SimpBN
        self._norm_layer = norm_layer

        self.inplanes = 64
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

        self.conv1 = SimpConv(3, self.inplanes, kernel_size=7, stride=2,
                              padding=3, bias=False, fix_points=fix_points)
        self.bn1 = norm_layer(self.inplanes, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0],
                                       fix_points=fix_points)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       fix_points=fix_points)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       fix_points=fix_points)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       fix_points=fix_points)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = SimpLinear(512 * block.expansion, num_classes,
                             fix_points=fix_points)

        for m in self.modules():
            if isinstance(m, SimpConv):
                for idx, flag in enumerate(fix_points):
                    nn.init.kaiming_normal_(getattr(m, f"weight_{idx}"),
                                            mode="fan_out",
                                            nonlinearity="relu")
            elif isinstance(m, (SimpBN, nn.GroupNorm)):
                for idx, flag in enumerate(fix_points):
                    nn.init.constant_(getattr(m, f"weight_{idx}"), 1)
                    nn.init.constant_(getattr(m, f"bias_{idx}"), 0)

    def _make_layer(
            self,
            block,
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            fix_points=None
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                simp_conv1x1(self.inplanes, planes * block.expansion, stride,
                             fix_points=fix_points),
                norm_layer(planes * block.expansion, fix_points=fix_points),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer,
                fix_points=fix_points
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer, fix_points=fix_points
                )
            )

        return layers

    def _forward_impl(self, x: Tensor, coeffs_t) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x, coeffs_t)
        x = self.bn1(x, coeffs_t)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layer1:
            x = layer(x, coeffs_t)
        for layer in self.layer2:
            x = layer(x, coeffs_t)
        for layer in self.layer3:
            x = layer(x, coeffs_t)
        for layer in self.layer4:
            x = layer(x, coeffs_t)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x, coeffs_t)

        return x

    def forward(self, x: Tensor, coeffs_t) -> Tensor:
        return self._forward_impl(x, coeffs_t)
