from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from mind.evotorch.model.shared import FeatureNet


@dataclass(frozen=True)
class ResGNConfig:
    in_channels: int = 1
    hidden_channels: int = 32
    out_channels: int = 8
    kernel_size: int = 3
    num_groups: int = 4
    dilation: int = 1


class ResGNBlock(nn.Module):
    def __init__(self, cfg: ResGNConfig):
        super().__init__()
        pad = (cfg.kernel_size // 2) * cfg.dilation
        self.conv1 = nn.Conv3d(
            cfg.in_channels,
            cfg.hidden_channels,
            cfg.kernel_size,
            padding=pad,
            dilation=cfg.dilation,
        )
        g1 = cfg.num_groups if cfg.hidden_channels % cfg.num_groups == 0 else 1
        self.gn1 = nn.GroupNorm(g1, cfg.hidden_channels)
        self.conv2 = nn.Conv3d(
            cfg.hidden_channels,
            cfg.out_channels,
            cfg.kernel_size,
            padding=pad,
            dilation=cfg.dilation,
        )
        g2 = cfg.num_groups if cfg.out_channels % cfg.num_groups == 0 else 1
        self.gn2 = nn.GroupNorm(g2, cfg.out_channels)
        self.act = nn.GELU()
        if cfg.in_channels != cfg.out_channels:
            self.proj = nn.Conv3d(cfg.in_channels, cfg.out_channels, kernel_size=1)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        if self.proj is not None:
            identity = self.proj(identity)
        return x + identity


def build(dilation: int = 1, num_blocks: int = 1) -> Tuple[torch.nn.Module, List[ResGNConfig]]:
    cfgs = [ResGNConfig(dilation=dilation) for _ in range(num_blocks)]
    blocks = [ResGNBlock(cfg) for cfg in cfgs]
    return FeatureNet(blocks), cfgs
