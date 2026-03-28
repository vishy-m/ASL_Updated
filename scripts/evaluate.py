#!/usr/bin/env python3
"""
CLI: Evaluate a trained model on a test split.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/cslr/best.pt --split test
    python scripts/evaluate.py --checkpoint checkpoints/islr/best.pt --mode islr
"""

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from asl_cslr.utils.logging import setup_logging
from asl_cslr.utils.device import get_device, get_autocast_context
from asl_cslr.data.vocab import GlossVocab
from asl_cslr.data.dataset import (
    ISLRDataset, CSLRDataset,
    islr_collate_fn, cslr_collate_fn,
)
from asl_cslr.models.islr_model import ISLRModel
from asl_cslr.models.cslr_model import (
    CSLRModel,
    DualStreamCSLRModel,
    suppress_ctc_special_tokens,
)
from asl_cslr.training.metrics import (
    compute_accuracy,
    macro_averaged_accuracy,
    compute_cer,
    compute_wer,
)
from asl_cslr.utils.model_config import (
    resolve_frame_feature_dim,
    resolve_motion_dim,
    resolve_single_stream_input_dim,
)

logger = logging.getLogger(__name__)


def _mask_special_logits(logits: torch.Tensor, vocab: GlossVocab) -> torch.Tensor:
    masked = logits.clone()
    special_indices = vocab.special_indices(include_blank=True)
    if special_indices:
        masked[:, special_indices] = torch.finfo(masked.dtype).min
    return masked


def _build_cslr_eval_model(config: dict, num_classes: int, device: torch.device):
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
        input_dim = resolve_single_stream_input_dim(config["model"])
        model = CSLRModel(
            input_dim=input_dim,
            encoder_type=config["model"].get("encoder_type", "bilstm"),
            transformer_hidden=config["model"].get("transformer_hidden", 256),
            transformer_layers=config["model"].get("transformer_layers", 4),
            transformer_heads=config["model"].get("transformer_heads", 4),
            **common_kwargs,
        )
    return model.to(device)


def _mask_ctc_special_log_probs(log_probs: torch.Tensor, vocab: GlossVocab) -> torch.Tensor:
    """Suppress non-blank special tokens before CTC decoding."""
    return suppress_ctc_special_tokens(
        log_probs,
        vocab.special_indices(include_blank=False),
    )


def _forward_cslr_eval(model, batch, device, use_amp):
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument("--mode", choices=["islr", "cslr"], help="Model mode (auto-detected from checkpoint config if omitted).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    device = get_device()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    mode = args.mode or config["model"]["type"]

    vocab = GlossVocab.load(config["data"]["vocab_path"])
    num_classes = len(vocab)

    logger.info(f"Evaluating {mode.upper()} model on {args.split} split")

    if mode == "islr":
        _evaluate_islr(ckpt, config, vocab, num_classes, device, args)
    elif mode == "cslr":
        _evaluate_cslr(ckpt, config, vocab, num_classes, device, args)


def _evaluate_islr(ckpt, config, vocab, num_classes, device, args):
    """Run ISLR evaluation."""
    input_dim = resolve_single_stream_input_dim(config["model"])
    model = ISLRModel(
        input_dim=input_dim,
        num_classes=num_classes,
        conv_dim=config["model"]["conv_dim"],
        conv_layers=config["model"]["conv_layers"],
        conv_kernel_size=config["model"]["conv_kernel_size"],
        conv_dropout=config["model"]["conv_dropout"],
        multi_scale=config["model"].get("multi_scale", False),
        multi_scale_kernels=config["model"].get("multi_scale_kernels"),
        encoder_type=config["model"].get("encoder_type", "bilstm"),
        lstm_hidden_size=config["model"]["lstm_hidden_size"],
        lstm_layers=config["model"]["lstm_layers"],
        lstm_dropout=config["model"]["lstm_dropout"],
        fc_dropout=config["model"].get("fc_dropout", 0.2),
        pool=config["model"].get("pool", "mean"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    manifest = config["data"][f"{args.split}_manifest"]
    expected_frame_feature_dim = resolve_frame_feature_dim(config["model"])
    required_schema_version = 2 if expected_frame_feature_dim == 208 else None
    dataset = ISLRDataset(
        manifest,
        vocab,
        use_motion=config["model"].get("use_motion", False),
        expected_frame_feature_dim=expected_frame_feature_dim,
        required_schema_version=required_schema_version,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Evaluation split is empty: {manifest}")
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=islr_collate_fn)

    all_logits = []
    all_labels = []
    use_amp = config["training"].get("mixed_precision", True)

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            with get_autocast_context(device, use_amp):
                logits = model(features, lengths)
            logits = _mask_special_logits(logits, vocab)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    if not all_logits or not all_labels:
        raise RuntimeError(f"ISLR evaluation produced no batches for split: {manifest}")
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    top1, top5 = compute_accuracy(all_logits, all_labels, topk=(1, 5))
    all_preds = all_logits.argmax(dim=1).tolist()
    macro = macro_averaged_accuracy(all_preds, all_labels.tolist(), len(vocab))

    logger.info(f"Results on {args.split}:")
    logger.info(f"  Top-1 Accuracy: {top1:.4f}")
    logger.info(f"  Top-5 Accuracy: {top5:.4f}")
    logger.info(f"  Macro Accuracy: {macro:.4f}")
    logger.info(f"  Samples: {len(all_labels)}")


def _evaluate_cslr(ckpt, config, vocab, num_classes, device, args):
    """Run CSLR evaluation with greedy CTC decode."""
    model = _build_cslr_eval_model(config, num_classes, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    manifest = config["data"][f"{args.split}_manifest"]
    expected_frame_feature_dim = resolve_frame_feature_dim(config["model"])
    dataset = CSLRDataset(
        manifest, vocab,
        t_max=config["training"]["t_max"],
        use_motion=config["model"].get("use_motion", False),
        dual_stream=config["model"].get("dual_stream", False),
        frame_stride=config["data"].get("frame_stride", 1),
        expected_frame_feature_dim=expected_frame_feature_dim,
        required_schema_version=(2 if expected_frame_feature_dim == 208 else None),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=cslr_collate_fn)
    if len(dataset) == 0:
        raise RuntimeError(f"Evaluation split is empty: {manifest}")

    all_refs = []
    all_hyps = []
    use_amp = config["training"].get("mixed_precision", True)
    ignore_ids = set(vocab.special_indices(include_blank=False))

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            label_lengths = batch["label_lengths"]

            log_probs, input_lengths = _forward_cslr_eval(
                model, batch, device, use_amp
            )
            log_probs = _mask_ctc_special_log_probs(log_probs, vocab)
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

    if not all_refs:
        raise RuntimeError("Evaluation produced no CSLR reference sequences")

    wer = compute_wer(all_refs, all_hyps)
    cer = compute_cer(all_refs, all_hyps)

    logger.info(f"Results on {args.split}:")
    logger.info(f"  WER: {wer:.4f}")
    logger.info(f"  CER: {cer:.4f}")
    logger.info(f"  Sentences: {len(all_refs)}")


if __name__ == "__main__":
    main()
