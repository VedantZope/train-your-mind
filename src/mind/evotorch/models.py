from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn


@dataclass(frozen=True)
class ConvBlockConfig:
    in_channels: int = 1
    hidden_channels: int = 8
    out_channels: int = 1
    kernel_size: int = 3
    activation: str = "relu"


class TwoLayerConv(nn.Module):
    def __init__(self, cfg: ConvBlockConfig):
        super().__init__()
        pad = cfg.kernel_size // 2
        self.conv1 = nn.Conv3d(cfg.in_channels, cfg.hidden_channels, cfg.kernel_size, padding=pad)
        self.conv2 = nn.Conv3d(cfg.hidden_channels, cfg.out_channels, cfg.kernel_size, padding=pad)
        if cfg.activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif cfg.activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {cfg.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x


class FeatureNet(nn.Module):
    def __init__(self, blocks: List[TwoLayerConv]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x

    def freeze_blocks(self, num_blocks: int) -> None:
        for i, blk in enumerate(self.blocks):
            requires_grad = i >= num_blocks
            for p in blk.parameters():
                p.requires_grad = requires_grad


def build_feature_net(block_cfgs: List[ConvBlockConfig]) -> FeatureNet:
    blocks = [TwoLayerConv(cfg) for cfg in block_cfgs]
    return FeatureNet(blocks)
