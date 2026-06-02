"""Transformer encoder for HAR sequence classification.

Architecture:
- Input projection: Linear(input_channels, hidden_dim)
- Sinusoidal positional encoding (fixed, not learned)
- nn.TransformerEncoder(num_layers=depth, d_model=hidden_dim, nhead=nhead)
- Mean-pool over time dimension → [B, hidden_dim]
- Then static concat + classifier (handled by base class)

Requirements: R5.1, R5.2, R5.3, R5.4, R5.5
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from har.models.base import HARModel, register_arch


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Generates a [1, max_len, d_model] buffer of sin/cos position embeddings
    that is added to the input sequence. Handles both even and odd d_model.
    """

    def __init__(self, d_model: int, max_len: int = 300) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model: cos columns may be one fewer than sin columns
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: [B, T, d_model] input tensor

        Returns:
            [B, T, d_model] tensor with positional encoding added
        """
        return x + self.pe[:, : x.size(1), :]


@register_arch("transformer")
class TransformerModel(HARModel):
    """Transformer encoder for sequence classification.

    Architecture:
    - Input projection: Linear(input_channels, hidden_dim)
    - Sinusoidal positional encoding (fixed, not learned)
    - nn.TransformerEncoder(num_layers=depth, d_model=hidden_dim, nhead=nhead)
    - Mean-pool over time dimension → [B, hidden_dim]
    - Then static concat + classifier (base class)

    Hyperparameters (from config):
    - hidden_dim: model dimension (d_model) for the transformer
    - depth: number of transformer encoder layers
    - dropout: dropout rate [0.0, 0.9]
    - nhead: number of attention heads (hidden_dim must be divisible by nhead)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        static_dim: int = 0,
        nhead: int = 4,
        **kwargs: object,
    ) -> None:
        super().__init__(input_channels, hidden_dim, depth, dropout, static_dim, **kwargs)

        self.input_proj = nn.Linear(input_channels, hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=300)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode [B, 300, C] → [B, hidden_dim] via Transformer encoder + mean pool.

        Args:
            seq: [B, 300, C] input sequence tensor

        Returns:
            [B, hidden_dim] pooled representation
        """
        # Project input channels to hidden_dim
        x = self.input_proj(seq)  # [B, 300, hidden_dim]

        # Add positional encoding
        x = self.pos_encoding(x)  # [B, 300, hidden_dim]

        # Transformer encoder
        x = self.transformer(x)  # [B, 300, hidden_dim]

        # Mean pool over time
        x = x.mean(dim=1)  # [B, hidden_dim]
        return x
