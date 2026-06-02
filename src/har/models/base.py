"""Abstract base class for HAR sequence models and architecture registry.

Provides:
- HARModel: abstract nn.Module with shape validation, static-feature concat,
  and classification head.
- register_arch: decorator to register concrete architectures.
- build_model: factory that instantiates a model from a config dict.

Requirements: R5.1, R5.2, R5.5, R5.6, R5.7
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from har.config import ConfigError


class HARModel(nn.Module, ABC):
    """Abstract base class for HAR sequence models.

    All models accept:
    - seq: [batch_size, 300, C] — sequence input
    - static: [batch_size, F] or None — optional static feature vector

    All models produce:
    - logits: [batch_size, 6] — class logits (finite values only)
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
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout_rate = dropout
        self.static_dim = static_dim
        self.num_classes = 6

        # Static feature projection (if static features are provided)
        if static_dim > 0:
            self.static_proj = nn.Linear(static_dim, hidden_dim)
        else:
            self.static_proj = None

        # Classification head (takes hidden_dim or hidden_dim*2 if static features)
        classifier_input = hidden_dim * 2 if static_dim > 0 else hidden_dim
        self.classifier = nn.Linear(classifier_input, self.num_classes)

    def forward(
        self, seq: torch.Tensor, static: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass with shape validation.

        Args:
            seq: [B, 300, C] sequence tensor
            static: [B, F] static feature tensor or None

        Returns:
            [B, 6] logit tensor with finite values

        Raises:
            ValueError: On shape mismatch (R5.7)
        """
        # Shape validation
        if seq.ndim != 3 or seq.shape[1] != 300 or seq.shape[2] != self.input_channels:
            raise ValueError(
                f"Expected seq shape [B, 300, {self.input_channels}], "
                f"got {list(seq.shape)}"
            )
        if static is not None:
            if static.ndim != 2 or static.shape[1] != self.static_dim:
                raise ValueError(
                    f"Expected static shape [B, {self.static_dim}], "
                    f"got {list(static.shape)}"
                )
            if static.shape[0] != seq.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: seq has {seq.shape[0]}, "
                    f"static has {static.shape[0]}"
                )

        # Get sequence representation from subclass
        seq_repr = self.encode_sequence(seq)  # [B, hidden_dim]

        # Combine with static features if present
        if static is not None and self.static_proj is not None:
            static_repr = self.static_proj(static)  # [B, hidden_dim]
            combined = torch.cat([seq_repr, static_repr], dim=1)  # [B, hidden_dim*2]
        else:
            combined = seq_repr  # [B, hidden_dim]

        logits = self.classifier(combined)  # [B, 6]
        return logits

    @abstractmethod
    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode the sequence into a fixed-size representation.

        Args:
            seq: [B, 300, C] input tensor

        Returns:
            [B, hidden_dim] representation
        """
        ...


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

_ARCH_REGISTRY: dict[str, type] = {}


def register_arch(name: str):
    """Decorator to register a model architecture."""

    def decorator(cls: type) -> type:
        _ARCH_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(
    config: dict, input_channels: int, static_dim: int = 0
) -> HARModel:
    """Build a model from configuration.

    Args:
        config: Model config dict with 'arch', 'hidden_dim', 'depth',
                'dropout', etc.
        input_channels: Number of input channels (C dimension)
        static_dim: Dimension of static feature vector (0 if none)

    Returns:
        Instantiated HARModel subclass

    Raises:
        ConfigError: If architecture identifier is unsupported (R5.6)
    """
    arch = config.get("arch", "tcn")
    if arch not in _ARCH_REGISTRY:
        raise ConfigError(
            field="model.arch",
            value=arch,
            message=(
                f"Unsupported architecture: '{arch}'. "
                f"Supported: {list(_ARCH_REGISTRY.keys())}"
            ),
        )

    model_cls = _ARCH_REGISTRY[arch]
    return model_cls(
        input_channels=input_channels,
        hidden_dim=config.get("hidden_dim", 128),
        depth=config.get("depth", 6),
        dropout=config.get("dropout", 0.2),
        static_dim=static_dim,
        kernel_size=config.get("kernel_size", 5),
        nhead=config.get("nhead", 4),
    )
