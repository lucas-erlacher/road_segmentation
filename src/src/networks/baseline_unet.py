# Vanilla UNet without any bells and whistles

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

parent = os.path.dirname((os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent)

import sys


""" Full assembly of the parts to form the complete network """

from networks.baseline_unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
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


# pixel-wise distance
class UNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        # remove unneccessary 0-th dim (if present)
        if len(y_hat.shape) == 3:
            y_hat = y_hat.squeeze(0)
        if len(y.shape) == 3:
            y = y.squeeze(0)
        # compute pixel-wise distance
        diff = torch.abs(y_hat - y)
        dist = torch.sum(diff)
        return dist


# for debugging purposes
if __name__ == "__main__":
    # import data
    path = "cil-road-segmentation-2022/training/"
    in_prefix = "images/*"
    out_prefix = "groundtruth/*"
    in_ims = []
    out_ims = []
    convert_tensor = transforms.ToTensor()
    for file_path in glob.glob(os.path.join(path, in_prefix)):
        in_ims.append(convert_tensor(Image.open(file_path).convert("RGB")))
    for file_path in glob.glob(os.path.join(path, out_prefix)):
        out_ims.append(convert_tensor(Image.open(file_path).convert("RGB")))
    in_ims = torch.stack(in_ims)
    out_ims = torch.stack(out_ims)
    out_ims = out_ims.norm(dim=1) > 1
    out_ims = out_ims.to(torch.long)
    dataset = TensorDataset(in_ims, out_ims)
    trainloader = DataLoader(dataset, batch_size=2)
    # training elements
    net = UNet(n_channels=3, n_classes=2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    # training loop
    num_epochs = 10
    for i in range(num_epochs):
        epoch_loss = 0
        for x, y in trainloader:
            optimizer.zero_grad()
            y_hat = net(x)
            loss = loss_fn(y_hat, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(epoch_loss)
