from __future__ import annotations

from typing import List

import torch
from torch import nn


class FeatureNet(nn.Module):
    def __init__(self, blocks: List[nn.Module]):
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
