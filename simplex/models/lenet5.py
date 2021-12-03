#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries

# Internal libraries
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import torch
import torch.nn as nn
from simplex_models import Linear as SimpLinear
from simplex_models import Conv2d as SimpConv


class Lenet5Simplex(nn.Module):
    def __init__(self, num_classes=10, fix_points=[True]):
        super().__init__()
        self.conv1 = SimpConv(in_channels=1, out_channels=6,
                              stride=1, kernel_size=5, fix_points=fix_points)
        self.conv2 = SimpConv(in_channels=6, out_channels=16,
                              stride=1, kernel_size=5, fix_points=fix_points)
        self.conv3 = SimpConv(in_channels=16, out_channels=120, kernel_size=5,
                            fix_points=fix_points)
        self.fc2 = SimpLinear(120, 84, fix_points=fix_points)
        self.fc3 = SimpLinear(84, num_classes, fix_points=fix_points)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, coeffs_t=[1]):
        x = self.pool(self.relu(self.conv1(x, coeffs_t)))
        x = self.pool(self.relu(self.conv2(x, coeffs_t)))
        x = self.relu(self.conv3(x, coeffs_t))
        x = x.view(x.shape[0], -1)
        x = self.fc2(x, coeffs_t)
        x = self.fc3(x, coeffs_t)
        return x