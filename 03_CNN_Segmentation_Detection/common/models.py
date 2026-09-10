"""03 分割家族模型：UNetMini / FCNMini（共享编码器容量，唯一差跳跃连接）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import count_flops


def _double_conv(in_ch: int, out_ch: int):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class UNetMini(nn.Module):
    """U-Net mini：enc 32→64→128→256(瓶颈)，三级上采样+跳跃拼接，out 3×128×128。"""

    def __init__(self, in_ch: int = 3, n_classes: int = 3):
        super().__init__()
        self.e1 = _double_conv(in_ch, 32)
        self.e2 = _double_conv(32, 64)
        self.e3 = _double_conv(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = _double_conv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = _double_conv(256, 128)   # 128(up)+128(skip)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = _double_conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = _double_conv(64, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.d3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class FCNMini(nn.Module):
    """FCN 基线：与 UNetMini 共享编码器+瓶颈，三级转置卷积上采样，不拼接 skips，多一层 refine。"""

    def __init__(self, in_ch: int = 3, n_classes: int = 3):
        super().__init__()
        self.e1 = _double_conv(in_ch, 32)
        self.e2 = _double_conv(32, 64)
        self.e3 = _double_conv(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = _double_conv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c3 = _double_conv(128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c2 = _double_conv(64, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.refine = _double_conv(32, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        x = self.c3(self.up3(b))
        x = self.c2(self.up2(x))
        x = self.refine(self.up1(x))
        return self.head(x)


def seg_flops(model) -> int:
    return count_flops(model, (1, 3, 128, 128))
