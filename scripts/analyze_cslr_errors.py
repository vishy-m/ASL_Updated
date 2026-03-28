#!/usr/bin/env python3
"""Detailed CSLR evaluation with token-level error analysis."""

from __future__ import annotations

import argparse
import logging
from collections import Counter

import torch
from torch.utils.data import DataLoader

from asl_cslr.data.dataset import CSLRDataset, cslr_collate_fn
from asl_cslr.data.vocab import GlossVocab
from asl_cslr.models.cslr_model import (
    CSLRModel,
    DualStreamCSLRModel,
    suppress_ctc_special_tokens,
)
from asl_cslr.training.metrics import compute_cer, compute_wer
from asl_cslr.utils.device import get_autocast_context, get_device
from asl_cslr.utils.logging import setup_logging
from asl_cslr.utils.model_config import (
    resolve_frame_feature_dim,
    resolve_motion_dim,
    resolve_single_stream_input_dim,
)

logger = logging.getLogger(__name__)


def _build_model(config: dict, num_classes: int, device: torch.device):
    common_kwargs = dict(
        num_classes=num_classes,
        conv_dim=config["model"]["conv_dim"],
        conv_layers=config["model"]["conv_layers"],
        conv_kernel_size=config["model"]["conv_kernel_size"],
        conv_dropout=config["model"]["conv_dropout"],
        lstm_hidden_size=config["model"]["lstm_hidden_size"],
        lstm_layers=config["model"]["lstm_layers"],
        lstm_dropout=config["model"]["lstm_dropout"],
        multi_scale=config["model"].get("multi_scale", False),
        multi_scale_kernels=config["model"].get("multi_scale_kernels"),
    )
    if config["model"].get("dual_stream", False):
        model = DualStreamCSLRModel(
            pose_dim=resolve_frame_feature_dim(config["model"]),
            motion_dim=resolve_motion_dim(config["model"]),
            fusion=config["model"].get("fusion", "concat"),
            **common_kwargs,
        )
    else:
        model = CSLRModel(
            input_dim=resolve_single_stream_input_dim(config["model"]),
            encoder_type=config["model"].get("encoder_type", "bilstm"),
            transformer_hidden=config["model"].get("transformer_hidden", 256),
            transformer_layers=config["model"].get("transformer_layers", 4),
            transformer_heads=config["model"].get("transformer_heads", 4),
            **common_kwargs,
        )
    return model.to(device)


def _forward(model, batch: dict, device: torch.device, use_amp: bool):
    input_lengths = batch["input_lengths"].to(device)
    if "features_pose" in batch:
        features_pose = batch["features_pose"].to(device)
        features_motion = batch["features_motion"].to(device)
        with get_autocast_context(device, use_amp):
            log_probs = model(features_pose, features_motion, input_lengths)
    else:
        features = batch["features"].to(device)
        with get_autocast_context(device, use_amp):
            log_probs = model(features, input_lengths)
    return log_probs, input_lengths


def _align(ref: list[int], hyp: list[int]) -> list[tuple[str, int | None, int | None]]:
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = (i - 1, 0, "delete")
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = (0, j - 1, "insert")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best_cost = dp[i - 1][j - 1]
                best_prev = (i - 1, j - 1, "match")
            else:
                best_cost = dp[i - 1][j - 1] + 1
                best_prev = (i - 1, j - 1, "substitute")

            delete_cost = dp[i - 1][j] + 1
            if delete_cost < best_cost:
                best_cost = delete_cost
                best_prev = (i - 1, j, "delete")

            insert_cost = dp[i][j - 1] + 1
            if insert_cost < best_cost:
                best_cost = insert_cost
                best_prev = (i, j - 1, "insert")

            dp[i][j] = best_cost
            back[i][j] = best_prev

    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            break
        prev_i, prev_j, op = prev
        ref_id = ref[i - 1] if op in {"match", "substitute", "delete"} and i > 0 else None
        hyp_id = hyp[j - 1] if op in {"match", "substitute", "insert"} and j > 0 else None
        ops.append((op, ref_id, hyp_id))
        i, j = prev_i, prev_j
    ops.reverse()
    return ops


def main():
    parser = argparse.ArgumentParser(description="Analyze CSLR token errors.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level))
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    if config["model"]["type"] != "cslr":
        raise ValueError("This analyzer only supports CSLR checkpoints")

    vocab = GlossVocab.load(config["data"]["vocab_path"])
    model = _build_model(config, len(vocab), device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    manifest = config["data"][f"{args.split}_manifest"]
    expected_frame_feature_dim = resolve_frame_feature_dim(config["model"])
    dataset = CSLRDataset(
        manifest,
        vocab,
        t_max=config["training"]["t_max"],
        use_motion=config["model"].get("use_motion", False),
        dual_stream=config["model"].get("dual_stream", False),
        frame_stride=config["data"].get("frame_stride", 1),
        expected_frame_feature_dim=expected_frame_feature_dim,
        required_schema_version=(2 if expected_frame_feature_dim == 208 else None),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=cslr_collate_fn)

    all_refs: list[list[int]] = []
    all_hyps: list[list[int]] = []
    use_amp = config["training"].get("mixed_precision", True)
    ignore_ids = set(vocab.special_indices(include_blank=False))

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            label_lengths = batch["label_lengths"]
            log_probs, input_lengths = _forward(model, batch, device, use_amp)
            log_probs = suppress_ctc_special_tokens(
                log_probs,
                vocab.special_indices(include_blank=False),
            )
            decoded = model.decode_with_lengths(
                log_probs,
                lengths=input_lengths,
                ignore_ids=ignore_ids,
            )

            offset = 0
            for i in range(len(label_lengths)):
                ll = label_lengths[i].item()
                ref_ids = labels[offset: offset + ll].tolist()
                offset += ll
                all_refs.append(ref_ids)
                all_hyps.append(decoded[i])

    wer = compute_wer(all_refs, all_hyps)
    cer = compute_cer(all_refs, all_hyps)

    ref_counts: Counter[int] = Counter()
    hyp_counts: Counter[int] = Counter()
    correct_counts: Counter[int] = Counter()
    deletion_counts: Counter[int] = Counter()
    insertion_counts: Counter[int] = Counter()
    substitution_counts: Counter[tuple[int, int]] = Counter()
    blank_sequences = 0

    for ref, hyp in zip(all_refs, all_hyps):
        if not hyp:
            blank_sequences += 1
        ref_counts.update(ref)
        hyp_counts.update(hyp)
        for op, ref_id, hyp_id in _align(ref, hyp):
            if op == "match" and ref_id is not None:
                correct_counts[ref_id] += 1
            elif op == "delete" and ref_id is not None:
                deletion_counts[ref_id] += 1
            elif op == "insert" and hyp_id is not None:
                insertion_counts[hyp_id] += 1
            elif op == "substitute" and ref_id is not None and hyp_id is not None:
                substitution_counts[(ref_id, hyp_id)] += 1

    token_rows = []
    for token_id, total in ref_counts.items():
        token_rows.append(
            (
                correct_counts[token_id] / total,
                total,
                vocab.decode(token_id),
                correct_counts[token_id],
                deletion_counts[token_id],
            )
        )
    token_rows.sort(key=lambda row: (row[0], row[1], row[2]))

    logger.info("Split: %s", args.split)
    logger.info("Samples: %d", len(all_refs))
    logger.info("WER: %.4f", wer)
    logger.info("CER: %.4f", cer)
    logger.info("Blank hypotheses: %d", blank_sequences)

    logger.info("Weakest gloss recall:")
    for recall, total, gloss, correct, deletions in token_rows[: args.top_k]:
        logger.info(
            "  %s | recall=%.3f | ref=%d | correct=%d | deletions=%d",
            gloss,
            recall,
            total,
            correct,
            deletions,
        )

    logger.info("Top substitutions:")
    for (ref_id, hyp_id), count in substitution_counts.most_common(args.top_k):
        logger.info(
            "  %s -> %s | %d",
            vocab.decode(ref_id),
            vocab.decode(hyp_id),
            count,
        )

    logger.info("Top insertions:")
    for hyp_id, count in insertion_counts.most_common(args.top_k):
        logger.info("  %s | %d", vocab.decode(hyp_id), count)

    logger.info("Top deletions:")
    for ref_id, count in deletion_counts.most_common(args.top_k):
        logger.info("  %s | %d", vocab.decode(ref_id), count)

    logger.info("Prediction volume:")
    for hyp_id, count in hyp_counts.most_common(args.top_k):
        logger.info("  %s | %d", vocab.decode(hyp_id), count)


if __name__ == "__main__":
    main()
