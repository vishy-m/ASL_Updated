"""
Classification and CTC output heads.

ClassificationHead: for ISLR (cross-entropy, single label per sequence).
CTCHead: for CSLR (CTC loss, gloss sequence per skeleton sequence).
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Classification head for isolated sign recognition (ISLR).

    Pools over the temporal dimension and projects to class logits.

    Args:
        input_dim: Input feature dimension from the encoder.
        num_classes: Number of gloss classes (vocabulary size).
        dropout: Dropout before the final linear layer.
        pool: Pooling strategy — 'mean', 'max', or 'last'.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        pool: str = "mean",
    ):
        super().__init__()
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, D) encoder output.
            lengths: (B,) original sequence lengths for masked pooling.

        Returns:
            (B, num_classes) logits.
        """
        if self.pool == "mean":
            if lengths is not None:
                # Masked mean pooling
                mask = torch.arange(x.size(1), device=x.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)  # (B, T)
                mask = mask.unsqueeze(2).float()  # (B, T, 1)
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pool == "max":
            pooled = x.max(dim=1).values
        elif self.pool == "last":
            if lengths is not None:
                pooled = x[
                    torch.arange(x.size(0), device=x.device), lengths - 1
                ]
            else:
                pooled = x[:, -1]
        else:
            raise ValueError(f"Unknown pooling: {self.pool}")

        return self.fc(self.dropout(pooled))


class CTCHead(nn.Module):
    """CTC output head for continuous sign language recognition (CSLR).

    Projects encoder output to per-frame gloss probabilities
    (including CTC blank token at index 0).

    Args:
        input_dim: Input feature dimension from the encoder.
        num_classes: Number of gloss classes including CTC blank.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, D) encoder output.

        Returns:
            (B, T, num_classes) log-probabilities for CTC.
        """
        return torch.nn.functional.log_softmax(self.fc(x), dim=-1)
