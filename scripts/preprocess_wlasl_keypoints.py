#!/usr/bin/env python3
"""Preprocess Kaggle WLASL holistic keypoints into canonical skeleton features."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from asl_cslr.data.preprocessing import preprocess_wlasl_holistic_keypoints


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess flattened WLASL holistic keypoints."
    )
    parser.add_argument(
        "--keypoints-root",
        default="data/raw/kaggle/wlasl_keypoints/output_V_WLASL",
        help="Directory containing per-video WLASL holistic .npy files.",
    )
    parser.add_argument(
        "--metadata",
        default="data/raw/kaggle/wlasl_processed/WLASL_v0.3.json",
        help="WLASL metadata JSON used to recover gloss labels and splits.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/keypoints/wlasl_kaggle_keypoints",
        help="Output directory for canonical .npz features.",
    )
    parser.add_argument(
        "--manifest-path",
        default="data/processed/manifests/wlasl_kaggle_keypoints.jsonl",
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=1,
        help="Keep every Nth frame from the source keypoints.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[1]
    preprocess_wlasl_holistic_keypoints(
        keypoints_root=(repo_root / args.keypoints_root).resolve(),
        metadata_path=(repo_root / args.metadata).resolve(),
        output_dir=(repo_root / args.output_dir).resolve(),
        manifest_path=(repo_root / args.manifest_path).resolve(),
        downsample_factor=args.downsample_factor,
        compute_velocity=True,
    )


if __name__ == "__main__":
    main()
