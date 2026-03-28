"""Shared checkpoint loaders for online inference entrypoints."""

from __future__ import annotations

from pathlib import Path

import torch

from asl_cslr.data.vocab import GlossVocab
from asl_cslr.models.cslr_model import CSLRModel
from asl_cslr.models.islr_model import ISLRModel
from asl_cslr.utils.model_config import (
    resolve_frame_feature_dim,
    resolve_motion_dim,
    resolve_single_stream_input_dim,
)


def _merge_model_config(ckpt_model: dict, overrides: dict | None = None) -> dict:
    model_cfg = dict(ckpt_model or {})
    if overrides:
        model_cfg.update({k: v for k, v in overrides.items() if v is not None})
    return model_cfg


def load_online_islr_model(
    checkpoint_path: str | Path,
    vocab_path: str | Path,
    device: torch.device,
    model_overrides: dict | None = None,
):
    """Load an ISLR model and vocab for online inference."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mcfg = _merge_model_config(ckpt.get("config", {}).get("model", {}), model_overrides)
    vocab = GlossVocab.load(vocab_path)
    model = ISLRModel(
        input_dim=resolve_single_stream_input_dim(mcfg),
        num_classes=len(vocab),
        conv_dim=mcfg["conv_dim"],
        conv_layers=mcfg["conv_layers"],
        conv_kernel_size=mcfg["conv_kernel_size"],
        conv_dropout=mcfg["conv_dropout"],
        encoder_type=mcfg.get("encoder_type", "bilstm"),
        lstm_hidden_size=mcfg["lstm_hidden_size"],
        lstm_layers=mcfg["lstm_layers"],
        lstm_dropout=mcfg["lstm_dropout"],
        transformer_hidden=mcfg.get("transformer_hidden", 256),
        transformer_layers=mcfg.get("transformer_layers", 4),
        transformer_heads=mcfg.get("transformer_heads", 4),
        fc_dropout=mcfg.get("fc_dropout", 0.2),
        pool=mcfg.get("pool", "mean"),
        multi_scale=mcfg.get("multi_scale", False),
        multi_scale_kernels=mcfg.get("multi_scale_kernels"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.online_use_motion = bool(mcfg.get("use_motion", False))
    model.online_frame_feature_dim = resolve_frame_feature_dim(mcfg)
    model.online_motion_dim = resolve_motion_dim(mcfg)
    model.eval()
    return model, vocab, ckpt, mcfg


def load_online_cslr_model(
    checkpoint_path: str | Path,
    vocab_path: str | Path,
    device: torch.device,
    model_overrides: dict | None = None,
):
    """Load a CSLR model and vocab for online inference."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mcfg = _merge_model_config(ckpt.get("config", {}).get("model", {}), model_overrides)
    if mcfg.get("dual_stream", False):
        raise ValueError("Online runtime does not yet support dual-stream CSLR checkpoints")

    vocab = GlossVocab.load(vocab_path)
    model = CSLRModel(
        input_dim=resolve_single_stream_input_dim(mcfg),
        num_classes=len(vocab),
        conv_dim=mcfg["conv_dim"],
        conv_layers=mcfg["conv_layers"],
        conv_kernel_size=mcfg["conv_kernel_size"],
        conv_dropout=mcfg["conv_dropout"],
        encoder_type=mcfg.get("encoder_type", "bilstm"),
        transformer_hidden=mcfg.get("transformer_hidden", 256),
        transformer_layers=mcfg.get("transformer_layers", 4),
        transformer_heads=mcfg.get("transformer_heads", 4),
        lstm_hidden_size=mcfg["lstm_hidden_size"],
        lstm_layers=mcfg["lstm_layers"],
        lstm_dropout=mcfg["lstm_dropout"],
        multi_scale=mcfg.get("multi_scale", False),
        multi_scale_kernels=mcfg.get("multi_scale_kernels"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.online_use_motion = bool(mcfg.get("use_motion", False))
    model.online_frame_feature_dim = resolve_frame_feature_dim(mcfg)
    model.online_motion_dim = resolve_motion_dim(mcfg)
    model.eval()
    return model, vocab, ckpt, mcfg
