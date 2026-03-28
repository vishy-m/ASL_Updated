"""
Bidirectional LSTM encoder for temporal sequence modeling (§7.1).
"""

import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for skeleton sequences.

    Takes temporal conv output (B, T, conv_dim) and produces contextual
    representations (B, T, 2*hidden_size).

    Args:
        input_dim: Input feature dimension (typically conv_dim).
        hidden_size: LSTM hidden size per direction.
        num_layers: Number of LSTM layers.
        dropout: Dropout between LSTM layers.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_dim = 2 * hidden_size

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, input_dim) input tensor.
            lengths: (B,) original sequence lengths for packing. Optional.

        Returns:
            (B, T, 2*hidden_size) output tensor.
        """
        if lengths is not None:
            # Pack padded sequences for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True
            )
        else:
            output, _ = self.lstm(x)

        return output
