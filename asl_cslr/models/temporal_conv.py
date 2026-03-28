"""
Temporal convolutional encoders for skeleton sequences.

Implements both standard single-scale (Family A, §7.1) and multi-scale
(Family B, §7.2) temporal convolutions.
"""

import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    """Single temporal convolution block: Conv1d + ReLU + BatchNorm + Dropout.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Conv1d kernel size (odd recommended for same padding).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2  # Same padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C_in, T) input tensor.

        Returns:
            (B, C_out, T) output tensor.
        """
        return self.dropout(self.relu(self.bn(self.conv(x))))


class TemporalConvEncoder(nn.Module):
    """Stack of temporal conv blocks (Family A, §7.1).

    Takes (B, T, D_in) skeleton input and produces (B, T, conv_dim).

    Args:
        input_dim: Number of input features per frame (e.g., 104 or 208).
        conv_dim: Number of channels in conv layers.
        num_layers: Number of conv blocks.
        kernel_size: Kernel size for all blocks.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 104,
        conv_dim: int = 384,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            out_ch = conv_dim
            layers.append(TemporalConvBlock(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch

        self.layers = nn.ModuleList(layers)
        self.output_dim = conv_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, D_in) input tensor.

        Returns:
            (B, T, conv_dim) output tensor.
        """
        # Conv1d expects (B, C, T) format
        x = x.transpose(1, 2)  # (B, D_in, T)

        for layer in self.layers:
            x = layer(x)

        return x.transpose(1, 2)  # (B, T, conv_dim)


class MultiScaleTemporalConv(nn.Module):
    """Multi-scale temporal convolution (Family B, §7.2).

    Applies parallel conv branches with different kernel sizes and
    concatenates their outputs before projecting back to conv_dim.

    Args:
        input_dim: Number of input features per frame.
        conv_dim: Output channels per branch and final projection dim.
        kernel_sizes: List of kernel sizes for parallel branches.
        dropout: Dropout probability.
        num_layers: Number of stacked multi-scale blocks.
    """

    def __init__(
        self,
        input_dim: int = 104,
        conv_dim: int = 384,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.1,
        num_layers: int = 3,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 9]

        self.num_branches = len(kernel_sizes)
        self.output_dim = conv_dim

        # Build stacked MS blocks
        self.blocks = nn.ModuleList()
        in_ch = input_dim

        for _ in range(num_layers):
            branches = nn.ModuleList([
                TemporalConvBlock(in_ch, conv_dim, ks, dropout)
                for ks in kernel_sizes
            ])
            # Project concatenated branches back to conv_dim
            projection = nn.Sequential(
                nn.Conv1d(conv_dim * self.num_branches, conv_dim, kernel_size=1),
                nn.BatchNorm1d(conv_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.blocks.append(nn.ModuleDict({
                "branches": branches,
                "projection": projection,
            }))
            in_ch = conv_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, D_in) input tensor.

        Returns:
            (B, T, conv_dim) output tensor.
        """
        x = x.transpose(1, 2)  # (B, D_in, T)

        for block in self.blocks:
            branch_outputs = [branch(x) for branch in block["branches"]]
            x = torch.cat(branch_outputs, dim=1)  # (B, conv_dim * num_branches, T)
            x = block["projection"](x)             # (B, conv_dim, T)

        return x.transpose(1, 2)  # (B, T, conv_dim)
