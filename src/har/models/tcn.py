"""Dilated 1D-CNN (Temporal Convolutional Network) for HAR sequence classification.

Architecture:
- Stack of Conv1d blocks with exponentially increasing dilations (1, 2, 4, 8, ...)
- Each block: Conv1d → BatchNorm → ReLU → Dropout, with residual connection
- Causal padding to preserve time dimension
- Global average pooling over time → [B, hidden_dim]
- Then static concat + classifier (handled by base class)

Requirements: R5.1, R5.2, R5.3, R5.4, R5.5
"""

from __future__ import annotations

import torch
import torch.nn as nn

from har.models.base import HARModel, register_arch


class TCNBlock(nn.Module):
    """Single TCN block with causal convolution, BatchNorm, ReLU, dropout, and residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        padding: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.padding = padding  # for causal trimming

        # Residual connection (1x1 conv if channel mismatch)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] → [B, out_channels, T]"""
        residual = self.residual(x)

        out = self.conv(x)
        # Causal trim: remove the extra padding from the right
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out + residual


@register_arch("tcn")
class TCNModel(HARModel):
    """Dilated 1D-CNN (TCN) for sequence classification.

    Architecture:
    - Stack of Conv1d blocks with exponentially increasing dilations (1, 2, 4, 8, ...)
    - Each block: Conv1d → BatchNorm → ReLU → Dropout, with residual connection
    - Causal padding to preserve time dimension
    - Global average pooling over time → [B, hidden_dim]
    - Then static concat + classifier (handled by base class)

    Hyperparameters (from config):
    - hidden_dim: number of channels in conv layers
    - depth: number of conv blocks
    - kernel_size: odd integer [1, 31]
    - dropout: dropout rate [0.0, 0.9]
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        static_dim: int = 0,
        kernel_size: int = 5,
        **kwargs: object,
    ) -> None:
        super().__init__(input_channels, hidden_dim, depth, dropout, static_dim, **kwargs)

        layers = []
        in_ch = input_channels
        for i in range(depth):
            dilation = 2**i
            out_ch = hidden_dim
            # Causal padding: (kernel_size - 1) * dilation
            padding = (kernel_size - 1) * dilation
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, padding, dropout))
            in_ch = out_ch

        self.tcn_layers = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pool

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode [B, 300, C] → [B, hidden_dim] via dilated convolutions + global avg pool."""
        # Conv1d expects [B, C, T]
        x = seq.transpose(1, 2)  # [B, C, 300]

        for layer in self.tcn_layers:
            x = layer(x)

        # Global average pool: [B, hidden_dim, T] → [B, hidden_dim, 1] → [B, hidden_dim]
        x = self.pool(x).squeeze(-1)
        return x
