from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from mind.evotorch.model.shared import FeatureNet


@dataclass(frozen=True)
class BasicBlockConfig:
    in_channels: int = 1
    hidden_channels: int = 8
    out_channels: int = 1
    kernel_size: int = 3
    activation: str = "relu"


class TwoLayerConv(nn.Module):
    def __init__(self, cfg: BasicBlockConfig):
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


def build() -> Tuple[torch.nn.Module, List[BasicBlockConfig]]:
    cfgs = [BasicBlockConfig()]
    blocks = [TwoLayerConv(cfg) for cfg in cfgs]
    return FeatureNet(blocks), cfgs
