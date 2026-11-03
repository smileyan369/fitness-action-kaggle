"""Temporal models for pose-sequence action recognition.

Two backbones:
  - TemporalModel: Conv1D + 2x BiLSTM (default)
  - TCNModel:      dilated causal Conv1D stack (fast, good for edge)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    def __init__(
        self,
        in_dim: int = 71,
        n_action: int = 5,
        n_phase: int = 0,
        hidden: int = 96,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_phase = n_phase
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm1 = nn.LSTM(128, hidden, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden * 2, 64, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Dropout(dropout + 0.1)
        )
        self.head_action = nn.Linear(128, n_action)
        self.head_phase = nn.Linear(128, n_phase) if n_phase > 0 else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, T, F) -> dict with logits."""
        x = self.stem(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        z = self.fc(self.norm(x[:, -1]))
        out = {"action": self.head_action(z)}
        if self.head_phase is not None:
            out["phase"] = self.head_phase(z)
        return out

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.bn(self.conv(x))))


class TCNModel(nn.Module):
    """Causal-ish dilated TCN; padding keeps length, no future leakage in eval."""

    def __init__(
        self,
        in_dim: int = 71,
        n_action: int = 5,
        n_phase: int = 0,
        hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_phase = n_phase
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[TCNBlock(hidden, d, dropout) for d in (1, 2, 4, 8)]
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.GELU(), nn.Dropout(dropout)
        )
        self.head_action = nn.Linear(128, n_action)
        self.head_phase = nn.Linear(128, n_phase) if n_phase > 0 else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x.transpose(1, 2))
        x = self.blocks(x)
        z = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)
        z = self.fc(z)
        out = {"action": self.head_action(z)}
        if self.head_phase is not None:
            out["phase"] = self.head_phase(z)
        return out

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(arch: str, **kwargs) -> nn.Module:
    arch = arch.lower()
    if arch in ("bilstm", "lstm", "temporal"):
        return TemporalModel(**kwargs)
    if arch == "tcn":
        return TCNModel(**kwargs)
    raise ValueError(f"unknown arch: {arch}")
