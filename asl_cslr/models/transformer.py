"""
Lightweight Transformer encoder for skeleton CSLR (Family C, §7.3).

Replaces BiLSTM with multi-head self-attention for potentially
better long-range temporal modeling.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer input.

    Args:
        d_model: Model dimensionality.
        max_len: Maximum sequence length to pre-compute.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model) with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerSequenceEncoder(nn.Module):
    """Transformer encoder for temporal skeleton sequences.

    Applies positional encoding + standard Transformer encoder layers.

    Args:
        input_dim: Input feature dimension (typically conv_dim).
        hidden_dim: Transformer model dimension.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward inner dimension (default: 4 * hidden_dim).
        dropout: Dropout probability.
        max_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int | None = None,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()

        if ff_dim is None:
            ff_dim = 4 * hidden_dim

        self.output_dim = hidden_dim

        # Project input to hidden_dim if needed
        self.input_projection = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.pos_encoding = PositionalEncoding(hidden_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, input_dim) input tensor.
            lengths: (B,) original sequence lengths. Used to create
                     padding mask for attention.

        Returns:
            (B, T, hidden_dim) output tensor.
        """
        x = self.input_projection(x)
        x = self.pos_encoding(x)

        # Create padding mask if lengths provided
        src_key_padding_mask = None
        if lengths is not None:
            B, T = x.shape[0], x.shape[1]
            src_key_padding_mask = torch.arange(T, device=x.device).unsqueeze(
                0
            ) >= lengths.unsqueeze(1)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        return x
