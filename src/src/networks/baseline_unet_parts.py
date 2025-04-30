# building blocks for a vanilla U-Net. Only thing I added is dilation.

import torch
import torch.nn as nn
import torch.nn.functional as F

# I added dilation and it really helped make road masks thicker (and helped the model not get confused by the road labels from OSM
# which are often much thinner than the real road)
# THOUGH: perhaps it made the detection of small/thin roads worse, so we might want to test dilation in more detail ...
dilation_fact = 2


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        dilation,
        mid_channels=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=(2 if dilation else 1),
                bias=False,
                dtype=dtype,
                dilation=(dilation_fact if dilation else 1),
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=(2 if dilation else 1),
                bias=False,
                dtype=dtype,
                dilation=(dilation_fact if dilation else 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x.to(dtype=self.dtype))


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dilation, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dilation=dilation, dtype=dtype),
        )

    def forward(self, x):
        return self.maxpool_conv(x.to(dtype=self.dtype))


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels, out_channels, dilation, bilinear=True, dtype=torch.float32
    ):
        super().__init__()
        self.dtype = dtype

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                in_channels // 2,
                dilation=dilation,
                dtype=dtype,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2, dtype=dtype
            )
            self.conv = DoubleConv(
                in_channels, out_channels, dilation=dilation, dtype=dtype
            )

    def forward(self, x1, x2):
        x1 = self.up(x1.to(dtype=self.dtype))
        x2 = x2.to(dtype=self.dtype)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, dtype=torch.float32):
        super(OutConv, self).__init__()
        self.dtype = dtype
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, dtype=dtype)

    def forward(self, x):
        return self.conv(x.to(dtype=self.dtype))
