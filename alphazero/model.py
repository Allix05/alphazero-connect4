"""Dual-head (policy + value) residual CNN, the standard AlphaZero network
architecture scaled down for a 6x7 board.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import ACTION_SIZE, COLS, ROWS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNet(nn.Module):
    """Input: (N, 2, ROWS, COLS) -> policy logits (N, COLS), value (N, 1)."""

    def __init__(self, channels: int = 64, num_res_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * ROWS * COLS, ACTION_SIZE),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * ROWS * COLS, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.res_blocks(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value.squeeze(-1)

    @torch.no_grad()
    def predict(self, board_tensor: torch.Tensor, device: torch.device):
        """board_tensor: (2, ROWS, COLS) numpy-free tensor, unbatched."""
        self.eval()
        x = board_tensor.unsqueeze(0).to(device)
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs, float(value.item())
