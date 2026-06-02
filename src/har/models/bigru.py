"""Bidirectional GRU architecture for HAR sequence classification.

Requirements: R5.1, R5.2, R5.3, R5.4, R5.5
"""

import torch
import torch.nn as nn

from har.models.base import HARModel, register_arch


@register_arch("bigru")
class BiGRUModel(HARModel):
    """Bidirectional GRU for sequence classification.

    Architecture:
    - nn.GRU(bidirectional=True, batch_first=True)
    - depth = num_layers
    - hidden = per-direction hidden size (so output is 2*hidden per step)
    - Take final hidden states from both directions, concatenate → [B, 2*hidden]
    - Project to hidden_dim → then static concat + classifier (base class)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        static_dim: int = 0,
        **kwargs: object,
    ) -> None:
        super().__init__(input_channels, hidden_dim, depth, dropout, static_dim, **kwargs)

        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_dim // 2,  # per-direction, so total = hidden_dim
            num_layers=depth,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if depth > 1 else 0.0,
        )
        # Project from 2*(hidden_dim//2) = hidden_dim to hidden_dim
        # (identity if sizes match, linear projection otherwise)
        gru_output_dim = 2 * (hidden_dim // 2)
        if gru_output_dim != hidden_dim:
            self.proj = nn.Linear(gru_output_dim, hidden_dim)
        else:
            self.proj = nn.Identity()

    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode [B, 300, C] → [B, hidden_dim] via BiGRU.

        Takes the final hidden states from both forward and backward
        directions (last layer) and concatenates them.
        """
        # GRU output: [B, 300, 2*hidden_per_dir]
        # hidden: [num_layers*2, B, hidden_per_dir]
        _output, hidden = self.gru(seq)

        # Take final hidden states from both directions (last layer)
        # hidden shape: [num_layers*2, B, hidden_per_dir]
        # Last layer forward: hidden[-2], last layer backward: hidden[-1]
        forward_hidden = hidden[-2]  # [B, hidden_per_dir]
        backward_hidden = hidden[-1]  # [B, hidden_per_dir]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)  # [B, 2*hidden_per_dir]

        return self.proj(combined)  # [B, hidden_dim]
