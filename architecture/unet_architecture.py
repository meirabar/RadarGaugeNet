import torch
from torch.utils.data import Dataset,Subset
from torch.utils.data import DataLoader
import random 
from torch import nn
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_channels=16, clamp_value=5, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.init_channels = init_channels  # Start with this number of channels
        self.bilinear = bilinear
        self.clamp_value = clamp_value
        
        self.inc = DoubleConv(n_channels, self.init_channels)
        self.down1 = Down(self.init_channels, self.init_channels * 2)
        self.down2 = Down(self.init_channels * 2, self.init_channels * 4)
        self.down3 = Down(self.init_channels * 4, self.init_channels * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(self.init_channels * 8, self.init_channels * 16 // factor)
        
        self.up1 = Up(self.init_channels * 16, self.init_channels * 8 // factor, bilinear)
        self.up2 = Up(self.init_channels * 8, self.init_channels * 4 // factor, bilinear)
        self.up3 = Up(self.init_channels * 4, self.init_channels * 2 // factor, bilinear)
        self.up4 = Up(self.init_channels * 2, self.init_channels, bilinear)
        
        self.last = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.outc = OutConv(self.init_channels, n_classes)
        self.act_last = nn.ReLU()  

    def forward(self, x):
        x0 = x
        x0 = torch.clamp(x,min = 0, max = self.clamp_value)
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.outc(x)
        x = torch.cat([x, x0], dim=1)
        x = self.last(x)
        x = self.act_last(x)
        # x = gaussian_blur(x)
        
        return x    
 