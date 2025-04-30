# UNet implementation with a few modifications that help it perform better/learn faster!
# Modifications are:
# - use of dilation in the convlayers
# - introduction of a split factor which simply scales the numer of parameters of the network down

import torch
from torch import nn, optim, logical_xor, logical_not
import torch.nn.functional as F
from torchvision.transforms import CenterCrop
from torchvision import transforms
from PIL import Image
from torch.nn import Upsample
import matplotlib.pyplot as plt
import os
import sys
import glob
import torchvision
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as f

parent = os.path.dirname((os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent)

import sys

""" Full assembly of the parts to form the complete network """

from networks.baseline_unet_parts import *


# n-channels should be the "normal n-channels" i.e. 3 if the image is rgb
class ModNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        split_factor,
        dilation_in,
        bilinear=False,
        dtype=torch.float32,
    ):
        super(ModNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.split_factor = split_factor
        self.dilation_input = dilation_in

        self.inc = DoubleConv(
            n_channels, (64 // self.split_factor), dilation=self.dilation_input
        )
        self.down1 = Down(
            64 // self.split_factor,
            128 // self.split_factor,
            dtype=dtype,
            dilation=self.dilation_input,
        )
        self.down2 = Down(
            128 // self.split_factor,
            256 // self.split_factor,
            dtype=dtype,
            dilation=self.dilation_input,
        )
        self.down3 = Down(
            256 // self.split_factor,
            512 // self.split_factor,
            dtype=dtype,
            dilation=self.dilation_input,
        )
        factor = 2 if bilinear else 1
        self.down4 = Down(
            512 // self.split_factor,
            1024 // self.split_factor,
            dtype=dtype,
            dilation=self.dilation_input,
        )
        self.up1 = Up(
            1024 // self.split_factor,
            512 // self.split_factor,
            bilinear=bilinear,
            dtype=dtype,
            dilation=self.dilation_input,
        )
        self.up2 = Up(
            512 // self.split_factor,
            256 // self.split_factor,
            bilinear=bilinear,
            dtype=dtype,
            dilation=self.dilation_input,
        )
        self.up3 = Up(
            256 // self.split_factor,
            128 // self.split_factor,
            bilinear=bilinear,
            dtype=dtype,
            dilation=self.dilation_input,
        )
        self.up4 = Up(
            128 // self.split_factor,
            64 // self.split_factor,
            bilinear=bilinear,
            dtype=dtype,
            dilation=self.dilation_input,
        )
        self.outc = OutConv(64 // self.split_factor, n_classes, dtype=dtype)

    def forward(self, x):
        """
        # unfold might do the same as my manual implementation
        x = x.to("cpu")
        print(x.shape)
        x = x.unfold(2, self.split_factor, self.split_factor).unfold(3, self.split_factor, self.split_factor)
        print(x.shape)
        x = x.permute(0, 4, 5, 1, 2, 3).flatten(0, 2)
        print(x.shape)
        x = x.to("mps")
        """

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
