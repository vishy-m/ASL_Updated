"""
ISLR model: Isolated Sign Language Recognition (§8.1).

Combines TemporalConvEncoder + BiLSTMEncoder (or Transformer) +
ClassificationHead for single-sign classification.
"""

import torch
import torch.nn as nn

from .temporal_conv import TemporalConvEncoder, MultiScaleTemporalConv
from .bilstm import BiLSTMEncoder
from .transformer import TransformerSequenceEncoder
from .heads import ClassificationHead


class ISLRModel(nn.Module):
    """Complete ISLR model for isolated sign recognition.

    Architecture: Temporal Conv → Sequence Encoder → Classification Head

    Args:
        input_dim: Per-frame feature dimension (104 or 208 with motion).
        num_classes: Number of gloss classes.
        conv_dim: Temporal conv output channels.
        conv_layers: Number of temporal conv blocks.
        conv_kernel_size: Conv kernel size.
        conv_dropout: Dropout in conv blocks.
        encoder_type: 'bilstm' or 'transformer'.
        lstm_hidden_size: BiLSTM hidden size per direction.
        lstm_layers: Number of BiLSTM layers.
        lstm_dropout: BiLSTM dropout.
        transformer_hidden: Transformer hidden dim.
        transformer_layers: Number of Transformer layers.
        transformer_heads: Number of attention heads.
        fc_dropout: Dropout before classification.
        pool: Pooling strategy for classification head.
        multi_scale: Use multi-scale temporal conv (Family B).
        multi_scale_kernels: Kernel sizes for multi-scale branches.
    """

    def __init__(
        self,
        input_dim: int = 104,
        num_classes: int = 2000,
        conv_dim: int = 384,
        conv_layers: int = 3,
        conv_kernel_size: int = 5,
        conv_dropout: float = 0.1,
        encoder_type: str = "bilstm",
        lstm_hidden_size: int = 384,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        transformer_hidden: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        fc_dropout: float = 0.2,
        pool: str = "mean",
        multi_scale: bool = False,
        multi_scale_kernels: list[int] | None = None,
    ):
        super().__init__()

        # Temporal conv encoder
        if multi_scale:
            self.conv_encoder = MultiScaleTemporalConv(
                input_dim=input_dim,
                conv_dim=conv_dim,
                kernel_sizes=multi_scale_kernels,
                dropout=conv_dropout,
                num_layers=conv_layers,
            )
        else:
            self.conv_encoder = TemporalConvEncoder(
                input_dim=input_dim,
                conv_dim=conv_dim,
                num_layers=conv_layers,
                kernel_size=conv_kernel_size,
                dropout=conv_dropout,
            )

        # Sequence encoder
        self.encoder_type = encoder_type
        if encoder_type == "bilstm":
            self.seq_encoder = BiLSTMEncoder(
                input_dim=conv_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                dropout=lstm_dropout,
            )
            head_input_dim = self.seq_encoder.output_dim
        elif encoder_type == "transformer":
            self.seq_encoder = TransformerSequenceEncoder(
                input_dim=conv_dim,
                hidden_dim=transformer_hidden,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                dropout=lstm_dropout,
            )
            head_input_dim = self.seq_encoder.output_dim
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Classification head
        self.head = ClassificationHead(
            input_dim=head_input_dim,
            num_classes=num_classes,
            dropout=fc_dropout,
            pool=pool,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, input_dim) skeleton sequences.
            lengths: (B,) original sequence lengths.

        Returns:
            (B, num_classes) classification logits.
        """
        x = self.conv_encoder(x)          # (B, T, conv_dim)
        x = self.seq_encoder(x, lengths)  # (B, T, encoder_dim)
        logits = self.head(x, lengths)    # (B, num_classes)
        return logits

    def get_backbone_state_dict(self) -> dict:
        """Extract conv + seq encoder weights for CSLR initialization."""
        state = {}
        for name, tensor in self.state_dict().items():
            if name.startswith("conv_encoder.") or name.startswith("seq_encoder."):
                state[name] = tensor.detach().clone()
        return state
