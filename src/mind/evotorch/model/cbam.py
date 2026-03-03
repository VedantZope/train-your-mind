from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from mind.evotorch.model.shared import FeatureNet


@dataclass(frozen=True)
class CBAMConfig:
    in_channels: int = 1
    hidden_channels: int = 8
    out_channels: int = 8
    kernel_size: int = 3
    reduction: int = 2
    spatial_kernel: int = 7
    dilation: int = 1


class ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.avg_pool(x))
        mx = self.mlp(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAMBlock(nn.Module):
    def __init__(self, cfg: CBAMConfig):
        super().__init__()
        pad = (cfg.kernel_size // 2) * cfg.dilation
        self.conv1 = nn.Conv3d(
            cfg.in_channels,
            cfg.hidden_channels,
            cfg.kernel_size,
            padding=pad,
            dilation=cfg.dilation,
        )
        self.conv2 = nn.Conv3d(
            cfg.hidden_channels,
            cfg.out_channels,
            cfg.kernel_size,
            padding=pad,
            dilation=cfg.dilation,
        )
        self.act = nn.GELU()
        self.ca = ChannelAttention3D(cfg.out_channels, cfg.reduction)
        self.sa = SpatialAttention3D(cfg.spatial_kernel)
        self.use_residual = cfg.in_channels == cfg.out_channels
        self.proj = None
        if cfg.out_channels != 1:
            self.proj = nn.Conv3d(cfg.out_channels, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = x * self.ca(x)
        x = x * self.sa(x)
        if self.proj is not None:
            x = self.proj(x)
        if self.use_residual:
            x = x + identity
        return x


def build(dilation: int = 1) -> Tuple[torch.nn.Module, List[CBAMConfig]]:
    cfgs = [CBAMConfig(dilation=dilation)]
    blocks = [CBAMBlock(cfg) for cfg in cfgs]
    return FeatureNet(blocks), cfgs
