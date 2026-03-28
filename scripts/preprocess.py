#!/usr/bin/env python3
"""
CLI: Run preprocessing for a specific dataset.

Usage:
    python scripts/preprocess.py --dataset wlasl --config configs/preprocessing.json
    python scripts/preprocess.py --dataset how2sign --data-dir data/raw/how2sign
"""

import argparse
import logging
import shutil
from pathlib import Path

from asl_cslr.utils.logging import setup_logging
from asl_cslr.utils.io import load_json_config
from asl_cslr.data.preprocessing import (
    preprocess_asl_citizen_keypoints,
    preprocess_how2sign,
    preprocess_asllvd,
    preprocess_ncslgr,
    preprocess_wlasl,
)
from asl_cslr.data.label_maps import (
    build_wlasl_label_map,
    build_asllvd_label_map,
    build_bu_label_map,
    load_label_map,
)

logger = logging.getLogger(__name__)


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    """Resolve config-relative paths against the repository root."""
    path = Path(path_like)
    return path if path.is_absolute() else (base_dir / path)


def _first_existing_path(candidates: list[Path], *, label: str) -> Path:
    """Return the first existing path from a candidate list."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {label}. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _reset_outputs(manifest_path: Path, output_dir: Path, *, clean_output: bool):
    """Reset derived outputs before a clean preprocessing run."""
    if manifest_path.exists():
        logger.info("Removing stale manifest: %s", manifest_path)
        manifest_path.unlink()
    if clean_output and output_dir.exists():
        logger.info("Removing stale output directory: %s", output_dir)
        shutil.rmtree(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ASL datasets into skeleton .npz files."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["asl_citizen", "how2sign", "asllvd", "ncslgr", "wlasl"],
        help="Dataset to preprocess.",
    )
    parser.add_argument(
        "--config",
        default="configs/preprocessing.json",
        help="Preprocessing config file.",
    )
    parser.add_argument(
        "--data-dir",
        help="Override raw data directory.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory for .npz files.",
    )
    parser.add_argument(
        "--annotations",
        help="Path to annotations file (dataset-specific).",
    )
    parser.add_argument(
        "--label-map",
        help="Path to pre-built label map JSON.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the derived output directory before preprocessing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(args.config, repo_root)
    config = load_json_config(config_path)
    downsample = config["temporal"]["downsample_factor"]
    compute_vel = config["motion_features"]["compute_velocity"]
    mediapipe_config = config.get("mediapipe", {})
    paths = config.get("paths", {})
    raw_paths = paths.get("raw", {})
    processed_paths = paths.get("processed", {})

    label_map = None
    if args.label_map:
        label_map = load_label_map(_resolve_path(args.label_map, repo_root))

    manifest_root = _resolve_path(
        processed_paths.get("manifests", "data/processed/manifests"),
        repo_root,
    )
    keypoints_root = _resolve_path(
        processed_paths.get("keypoints", "data/processed/keypoints"),
        repo_root,
    )
    base_output = _resolve_path(
        args.output_dir or keypoints_root / args.dataset,
        repo_root,
    )
    manifest = manifest_root / f"{args.dataset}.jsonl"

    if args.dataset == "how2sign":
        data_dir = _resolve_path(
            args.data_dir or raw_paths.get("how2sign", "data/raw/how2sign"),
            repo_root,
        )
        base_output = _resolve_path(
            args.output_dir or keypoints_root / "how2sign",
            repo_root,
        )
        manifest = manifest_root / "how2sign.jsonl"

        # How2Sign has 3 splits with different directory structures
        splits = config.get("how2sign", {}).get(
            "splits",
            [
                {
                    "name": "test",
                    "keypoints_subdir": "archive/Test_data/Test_data",
                    "annotations": "annotations/test_annotations.csv",
                },
                {
                    "name": "val",
                    "keypoints_subdir": "archive/Val_data/Validation_data",
                    "annotations": "annotations/val_annotations.csv",
                },
                {
                    "name": "train",
                    "keypoints_subdir": "archive/Train_data/Train_data",
                    "annotations": "annotations/train_annotations.csv",
                },
            ],
        )

        _reset_outputs(manifest, base_output, clean_output=args.clean_output)
        base_output.mkdir(parents=True, exist_ok=True)

        for split_cfg in splits:
            split_name = split_cfg["name"]
            kp_dir = data_dir / split_cfg["keypoints_subdir"]
            ann_path = data_dir / split_cfg["annotations"]
            if args.annotations:
                ann_path = _resolve_path(args.annotations, repo_root)
            logger.info(f"Processing How2Sign split: {split_name}")
            preprocess_how2sign(
                keypoints_dir=kp_dir,
                annotations_path=ann_path,
                output_dir=base_output / split_name,
                manifest_path=manifest,
                split=split_name,
                downsample_factor=downsample,
                compute_velocity=compute_vel,
                mediapipe_config=mediapipe_config,
            )

    elif args.dataset == "asl_citizen":
        data_dir = _resolve_path(
            args.data_dir
            or raw_paths.get("asl_citizen", "data/raw/kaggle/asl_citizen_keypoints"),
            repo_root,
        )
        base_output = _resolve_path(
            args.output_dir or keypoints_root / "asl_citizen",
            repo_root,
        )
        manifest = manifest_root / "asl_citizen.jsonl"
        _reset_outputs(manifest, base_output, clean_output=args.clean_output)
        preprocess_asl_citizen_keypoints(
            keypoints_root=data_dir,
            output_dir=base_output,
            manifest_path=manifest,
            downsample_factor=downsample,
            compute_velocity=compute_vel,
            mediapipe_config=mediapipe_config,
        )

    elif args.dataset == "asllvd":
        data_dir = _resolve_path(
            args.data_dir or raw_paths.get("asllvd", "data/raw/asllvd"),
            repo_root,
        )
        annotations = _resolve_path(
            args.annotations or data_dir / "tokens.json",
            repo_root,
        )
        _reset_outputs(manifest, base_output, clean_output=args.clean_output)
        preprocess_asllvd(
            video_dir=data_dir,
            token_table_path=annotations,
            output_dir=base_output,
            manifest_path=manifest,
            label_map=label_map,
            downsample_factor=downsample,
            compute_velocity=compute_vel,
            mediapipe_config=mediapipe_config,
        )

    elif args.dataset == "ncslgr":
        data_dir = _resolve_path(
            args.data_dir or raw_paths.get("ncslgr", "data/raw/asllrp_ncslgr"),
            repo_root,
        )
        video_dir = _first_existing_path(
            [
                data_dir / "videos",
                data_dir / "video",
            ],
            label="NCSLGR videos",
        )
        annotation_dir = _first_existing_path(
            [
                data_dir / "annotations",
                data_dir / "signstream",
            ],
            label="NCSLGR annotations",
        )
        _reset_outputs(manifest, base_output, clean_output=args.clean_output)
        preprocess_ncslgr(
            video_dir=video_dir,
            annotation_dir=annotation_dir,
            output_dir=base_output,
            manifest_path=manifest,
            label_map=label_map,
            downsample_factor=downsample,
            compute_velocity=compute_vel,
            mediapipe_config=mediapipe_config,
        )

    elif args.dataset == "wlasl":
        data_dir = _resolve_path(
            args.data_dir or raw_paths.get("wlasl", "data/raw/wlasl"),
            repo_root,
        )
        annotations = _first_existing_path(
            [
                _resolve_path(args.annotations, repo_root)
                if args.annotations else data_dir / "wlasl_video_mapping.json",
                _resolve_path(args.annotations, repo_root)
                if args.annotations else data_dir / "WLASL_v0.3.json",
                data_dir / "start_kit" / "WLASL_v0.3.json",
            ],
            label="WLASL metadata",
        )
        clips_dir = _first_existing_path(
            [
                data_dir / "videos",
                data_dir / "raw_videos",
                data_dir / "start_kit" / "raw_videos",
                data_dir / "start_kit" / "videos",
            ],
            label="WLASL video directory",
        )
        _reset_outputs(manifest, base_output, clean_output=args.clean_output)
        preprocess_wlasl(
            clips_dir=clips_dir,
            metadata_path=annotations,
            output_dir=base_output,
            manifest_path=manifest,
            label_map=label_map,
            downsample_factor=downsample,
            compute_velocity=compute_vel,
            mediapipe_config=mediapipe_config,
        )

    logger.info(f"Preprocessing complete for {args.dataset}")


if __name__ == "__main__":
    main()
