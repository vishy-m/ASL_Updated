"""Helpers for resolving model feature dimensions from config dictionaries."""

from __future__ import annotations


def resolve_frame_feature_dim(model_cfg: dict) -> int:
    """Return the per-frame feature width before optional motion concatenation."""
    if model_cfg.get("frame_feature_dim") is not None:
        return int(model_cfg["frame_feature_dim"])
    return int(model_cfg.get("input_dim", 104))


def resolve_motion_dim(model_cfg: dict) -> int:
    """Return the motion feature width for the configured schema."""
    if model_cfg.get("motion_dim") is not None:
        return int(model_cfg["motion_dim"])
    return resolve_frame_feature_dim(model_cfg)


def resolve_single_stream_input_dim(model_cfg: dict) -> int:
    """Return the actual single-stream model input width.

    Backward compatibility:
    - legacy configs stored `input_dim` as the base frame width and doubled it
      implicitly when `use_motion=True`
    - new configs store the actual single-stream width in `input_dim` and the
      base frame width in `frame_feature_dim`
    """
    configured = int(model_cfg.get("input_dim", 104))
    if not model_cfg.get("use_motion", False):
        return configured
    if model_cfg.get("frame_feature_dim") is not None:
        return configured
    return configured * 2
