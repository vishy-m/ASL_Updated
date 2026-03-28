"""
Learning rate schedulers with warmup support.
"""

import math

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 50,
    warmup_epochs: int = 5,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build a learning rate scheduler with optional linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: 'cosine' or 'linear'.
        epochs: Total training epochs.
        warmup_epochs: Number of warmup epochs.

    Returns:
        An LR scheduler instance.
    """
    if scheduler_type == "cosine":
        if warmup_epochs <= 0:
            return CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

        if warmup_epochs >= epochs:
            start_factor = 1.0 / max(warmup_epochs + 1, 2)
            return LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=max(warmup_epochs, 1),
            )

        start_factor = 1.0 / max(warmup_epochs + 1, 2)
        warmup = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - warmup_epochs, 1),
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    elif scheduler_type == "linear":

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs + 1, 1)
            return max(0.0, 1.0 - (epoch - warmup_epochs) / max(
                epochs - warmup_epochs, 1
            ))

        return LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
