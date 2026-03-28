"""
Evaluation metrics for ISLR and CSLR (§10).

- Top-k accuracy for ISLR
- Macro-averaged accuracy across glosses
- Word Error Rate (WER) for CSLR gloss sequences
"""

import torch
import editdistance


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[float]:
    """Compute top-k accuracy.

    Args:
        logits: (B, C) prediction logits.
        labels: (B,) ground truth labels.
        topk: Tuple of k values.

    Returns:
        List of accuracy values (fractions, not percentages).
    """
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(labels.unsqueeze(1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:, :k].any(dim=1).float().mean().item()
        results.append(correct_k)

    return results


def macro_averaged_accuracy(
    all_preds: list[int],
    all_labels: list[int],
    num_classes: int,
) -> float:
    """Compute macro-averaged per-class accuracy.

    Ensures rare glosses contribute equally to the metric.

    Args:
        all_preds: List of predicted label IDs.
        all_labels: List of ground truth label IDs.
        num_classes: Total number of classes.

    Returns:
        Macro-averaged accuracy (0-1).
    """
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes

    for pred, label in zip(all_preds, all_labels):
        per_class_total[label] += 1
        if pred == label:
            per_class_correct[label] += 1

    accuracies = []
    for c in range(num_classes):
        if per_class_total[c] > 0:
            accuracies.append(per_class_correct[c] / per_class_total[c])

    return sum(accuracies) / len(accuracies) if accuracies else 0.0


def compute_wer(
    references: list[list[int]],
    hypotheses: list[list[int]],
) -> float:
    """Compute Word Error Rate (WER) over gloss sequences.

    WER = sum(edit_distance(ref, hyp)) / sum(len(ref))

    Args:
        references: List of reference gloss ID sequences.
        hypotheses: List of hypothesis gloss ID sequences.

    Returns:
        WER as a fraction (0 = perfect, >1 = more errors than reference words).
    """
    total_edits = 0
    total_ref_length = 0

    for ref, hyp in zip(references, hypotheses):
        total_edits += editdistance.eval(ref, hyp)
        total_ref_length += len(ref)

    if total_ref_length == 0:
        raise ValueError("Cannot compute WER with zero reference tokens")

    return total_edits / total_ref_length


def compute_cer(
    references: list[list[int]],
    hypotheses: list[list[int]],
) -> float:
    """Compute Character Error Rate (CER) over gloss sequences.

    Treats each gloss ID as a "character" for character-level edit distance.
    Equivalent to WER when each gloss is a single token.

    Args:
        references: List of reference gloss ID sequences.
        hypotheses: List of hypothesis gloss ID sequences.

    Returns:
        CER as a fraction.
    """
    return compute_wer(references, hypotheses)
