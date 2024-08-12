# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm, GRN

from validation import epochVal_metrics_test
from torchvision import transforms
import os
from dataloaders import dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
import pandas as pd

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self,mode,drop_rate=0.2, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0.2, head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        #self.head = nn.Linear(dims[-1], num_classes)
        self.actv = nn.ReLU()
        self.head = nn.Linear(dims[-1], num_classes)
        self.head1 = 0
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)


        self.mode = mode
    def replace_head(self, num_classes,dims):
        # Replace the final classification layer(s) with new layers
        self.head = nn.Linear(dims[-1], num_classes)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        # x = self.forward_features(x)
        # if self.drop_rate > 0:
        #     x = self.drop_layer(x)
        # x = self.head(x)
        # return x
        x = self.forward_features(x)
        z = self.head1(x)
        y = self.actv(z)
        y = self.head(y)
        return y,z


def convnextv2_atto(pretrained=True, **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    if pretrained:
        model = load_pretrained_weights(model, 'convnextv2_atto_22k_224_ema.pt')
    return model


def convnextv2_femto(pretrained=True, **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    if pretrained:
        model = load_pretrained_weights(model, 'convnextv2_femto_22k_224_ema.pt')
        model.replace_head(7)

    return model


def convnext_pico(classnumber,pretrained=True, **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    if pretrained:
        model = load_pretrained_weights(model, 'convnextv2_pico_22k_224_ema.pt')
        model.replace_head(classnumber)
    return model


def convnextv2_nano(classnumber,pretrained=True, **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640],**kwargs)
    if pretrained:
        model = load_pretrained_weights(model,  'convnextv2_nano_22k_224_ema.pt')
        model.replace_head(num_classes=classnumber,dims=[80,160,320,640])  # Assuming you want to change to 7 classes

    return model


def convnextv2_tiny(classnumber,pretrained=True, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        model = load_pretrained_weights(model, './convnextv2_tiny_22k_224_ema.pt')
        model.replace_head(num_classes=classnumber,dims=[96, 192, 384, 768])  # Assuming you want to change to 7 classes
    dims = [96, 192, 384, 768]
    model.head = nn.Linear(192, classnumber)
    model.head1 = nn.Linear(768, 192)
    return model


def convnextv2_base(pretrained=True, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        model = load_pretrained_weights(model, 'convnextv2_base_22k_224_ema.pt')
    return model


def convnextv2_large(pretrained=True, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        model = load_pretrained_weights(model, 'convnextv2_large_22k_224_ema.pt')
    return model


def convnextv2_huge(pretrained=True, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    if pretrained:
        model = load_pretrained_weights(model, 'convnextv2_huge_22k_224_ema.pt')
    return model


def load_pretrained_weights(model, pretrained_path, key='model_state_dict'):
    # Load the pretrained model
    pretrained_model = torch.load(pretrained_path)

    # Access the correct state dictionary key
    model.load_state_dict(pretrained_model["model"])


    return model

