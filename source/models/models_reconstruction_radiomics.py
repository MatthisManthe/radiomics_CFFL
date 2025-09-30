from monai.networks.nets import AutoEncoder
import warnings
from typing import Optional, Sequence, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch import sigmoid
from torch.nn import functional as F
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export


class ResBlockUp(nn.Module):
    def __init__(self, filters_in, filters_out, act=True):
        super(ResBlockUp, self).__init__()
        self.act = act
        self.conv1_block = nn.Sequential(
            nn.Conv3d(filters_in, filters_in, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm3d(filters_in),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2_block = nn.Sequential(
            nn.Conv3d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.BatchNorm3d(filters_out))

        self.conv3_block = nn.Sequential(
            nn.Conv3d(filters_in, filters_out, 3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm3d(filters_out),
            nn.LeakyReLU(0.2, inplace=True))

        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        if self.act:
            conv2 = self.lrelu(conv2)
        conv3 = self.conv3_block(x)
        if self.act:
            conv3 = self.lrelu(conv3)

        return conv2 + conv3
    
    
class Decoder(nn.Module):
    
    def __init__(self, n_features=93, n_channels=1, gf_dim=16):
        super(Decoder, self).__init__()
        self.gf_dim = gf_dim
        self.fc = nn.Sequential(
            nn.Linear(n_features, gf_dim*32*2*2*2),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.res1 = ResBlockUp(gf_dim * 32, gf_dim * 16)
        self.res2 = ResBlockUp(gf_dim * 16, gf_dim * 8)
        self.res3 = ResBlockUp(gf_dim * 8, gf_dim * 4)
        self.res4 = ResBlockUp(gf_dim * 4, gf_dim * 2)
        self.res5 = ResBlockUp(gf_dim * 2, gf_dim * 1)
        self.res6 = ResBlockUp(gf_dim * 1, gf_dim * 1)
        self.conv_1_block = nn.Sequential(
            nn.Conv3d(gf_dim, gf_dim, 3, stride=1, padding=1),
            nn.BatchNorm3d(gf_dim),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Conv3d(gf_dim, n_channels, 3, stride=1, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.gf_dim*32, 2, 2, 2)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.conv_1_block(x)
        x = self.conv2(x)
        return x