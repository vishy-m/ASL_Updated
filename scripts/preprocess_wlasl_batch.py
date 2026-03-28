#!/usr/bin/env python3
"""Batch WLASL preprocessing with clip trimming and single MediaPipe init.

Defaults are resolved relative to the repository root so the script can run
from any working directory without path drift.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else base_dir / path


def process_video_clip(
    video_path,
    *,
    frame_start=0,
    frame_end=-1,
    downsample_factor=2,
    mediapipe_config=None,
    landmarker=None,
    timestamp_offset_ms=0,
):
    """Extract a skeleton sequence from a video clip using the shared pipeline."""
    import cv2

    from asl_cslr.data.preprocessing import _video_to_skeletons
    from asl_cslr.data.skeleton import FEATURE_DIM

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), timestamp_offset_ms

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start = max(frame_start, 0)
    end = frame_end if frame_end > 0 else total_frames
    end = min(end, total_frames)
    max_frames = 300
    if (end - start) > max_frames:
        end = start + max_frames

    sequence, next_timestamp_ms = _video_to_skeletons(
        video_path,
        downsample_factor=downsample_factor,
        start_time=start / fps,
        end_time=end / fps,
        mediapipe_config=mediapipe_config,
        landmarker=landmarker,
        timestamp_offset_ms=timestamp_offset_ms,
    )
    return sequence, next_timestamp_ms


def _write_manifest_entry(manifest_path: Path, entry: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _write_manifest_entries(manifest_path: Path, entries: list[dict]):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess WLASL clips in batches.")
    parser.add_argument(
        "--config",
        default="configs/preprocessing.json",
        help="Preprocessing config file used for shared MediaPipe and path defaults.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to the parent directory of this script.",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Path to WLASL clip mapping JSON.",
    )
    parser.add_argument(
        "--clips-dir",
        default=None,
        help="Directory containing cropped WLASL clips.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for skeleton .npz files.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Manifest path for the processed samples.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory containing MediaPipe task models.",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=None,
        help="Keep every Nth frame.",
    )
    parser.add_argument(
        "--glosses",
        nargs="*",
        default=None,
        help="Optional canonical glosses to keep (for focused demo rebuilds).",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the derived WLASL keypoints directory before rebuilding.",
    )
    parser.add_argument(
        "--dataset-name",
        default="wlasl",
        help="Dataset label written into the output manifest.",
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

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from asl_cslr.data.label_maps import clean_wlasl_gloss
    from asl_cslr.data.preprocessing import (
        _save_skeleton,
        _probe_video_frame_size,
        _get_cached_video_landmarker,
        _close_landmarker_cache,
    )
    from asl_cslr.utils.io import load_json_config

    config = load_json_config(_resolve_path(args.config, repo_root))
    raw_paths = config.get("paths", {}).get("raw", {})
    processed_paths = config.get("paths", {}).get("processed", {})
    mediapipe_config = dict(config.get("mediapipe", {}))
    wlasl_root = _resolve_path(raw_paths.get("wlasl", "data/raw/wlasl"), repo_root)
    keypoints_root = _resolve_path(
        processed_paths.get("keypoints", "data/processed/keypoints"),
        repo_root,
    )
    manifests_root = _resolve_path(
        processed_paths.get("manifests", "data/processed/manifests"),
        repo_root,
    )

    mapping_path = _resolve_path(
        args.mapping or (wlasl_root / "wlasl_video_mapping.json"),
        repo_root,
    )
    clips_dir = _resolve_path(
        args.clips_dir or (wlasl_root / "videos"),
        repo_root,
    )
    output_dir = _resolve_path(
        args.output_dir or (keypoints_root / "wlasl"),
        repo_root,
    )
    manifest_path = _resolve_path(
        args.manifest_path or (manifests_root / "wlasl.jsonl"),
        repo_root,
    )

    task_model_path = mediapipe_config.get("task_model_path")
    if task_model_path:
        mediapipe_config["task_model_path"] = str(_resolve_path(task_model_path, repo_root))
        models_dir = Path(mediapipe_config["task_model_path"]).parent
    else:
        models_dir = _resolve_path(args.models_dir or "models/mediapipe", repo_root)
        mediapipe_config["task_model_path"] = str(models_dir / "holistic_landmarker.task")

    downsample_factor = (
        args.downsample_factor
        if args.downsample_factor is not None
        else int(config.get("temporal", {}).get("downsample_factor", 2))
    )

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    selected_glosses = None
    if args.glosses:
        selected_glosses = {clean_wlasl_gloss(gloss) for gloss in args.glosses}
        mapping = [
            item for item in mapping
            if clean_wlasl_gloss(item.get("gloss", "")) in selected_glosses
        ]
        logger.info(
            "Filtered WLASL mapping to %s glosses: %s",
            len(selected_glosses),
            sorted(selected_glosses),
        )

    # Sort deterministically so reruns produce the same output order.
    mapping = sorted(
        mapping,
        key=lambda item: (
            item.get("split", ""),
            item.get("gloss", ""),
            item.get("video_id", ""),
            item.get("frame_start", 0),
            item.get("frame_end", -1),
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest_path.unlink()
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "MediaPipe Holistic ready. Processing %s videos with clip trimming (downsample=%s)...",
        len(mapping),
        downsample_factor,
    )
    landmarker_cache = {}

    processed = 0
    cached = 0
    failed = 0
    start_time = time.time()
    next_timestamp_by_size: dict[tuple[int, int], int] = {}
    manifest_entries: list[dict] = []
    dataset_name = str(args.dataset_name).strip() or "wlasl"

    for i, item in enumerate(mapping):
        vid = item["video_id"]
        vpath = _resolve_path(item["video_path"], repo_root) if item.get("video_path") else clips_dir / f"{vid}.mp4"
        gloss = item.get("gloss", "")
        split = item.get("split", "train")
        frame_start = item.get("frame_start", 0)
        frame_end = item.get("frame_end", -1)

        npz_path = output_dir / f"{vid}.npz"

        if npz_path.exists():
            try:
                with np.load(str(npz_path)) as npz:
                    T = npz["X"].shape[0]
                canonical = clean_wlasl_gloss(gloss)
                entry = {
                    "id": vid,
                    "features_path": str(npz_path),
                    "glosses": [canonical],
                    "num_frames": T,
                    "split": split,
                    "dataset": dataset_name,
                }
                manifest_entries.append(entry)
                cached += 1
                continue
            except Exception:
                pass

        if not vpath.exists():
            failed += 1
            continue

        try:
            frame_size = _probe_video_frame_size(vpath)
            landmarker = _get_cached_video_landmarker(
                frame_size,
                landmarker_cache,
                mediapipe_config=mediapipe_config,
            )
            sequence, next_timestamp_ms = process_video_clip(
                vpath,
                frame_start=frame_start,
                frame_end=frame_end,
                downsample_factor=downsample_factor,
                mediapipe_config=mediapipe_config,
                landmarker=landmarker,
                timestamp_offset_ms=next_timestamp_by_size.get(frame_size, 0),
            )
            next_timestamp_by_size[frame_size] = next_timestamp_ms
            T = sequence.shape[0]
            if T == 0:
                failed += 1
                continue

            _save_skeleton(npz_path, sequence, compute_velocity=True)
            canonical = clean_wlasl_gloss(gloss)

            entry = {
                "id": vid,
                "features_path": str(npz_path),
                "glosses": [canonical],
                "num_frames": T,
                "split": split,
                "dataset": dataset_name,
            }
            manifest_entries.append(entry)
            processed += 1

        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning("Failed %s: %s", vid, e)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            total_done = processed + cached
            rate = total_done / elapsed if elapsed > 0 else 0
            remaining = len(mapping) - i - 1
            eta = remaining / rate / 60 if rate > 0 else 0
            logger.info(
                "[%s/%s] processed=%s, cached=%s, failed=%s, %.1f vid/s, ETA=%.0fmin",
                i + 1,
                len(mapping),
                processed,
                cached,
                failed,
                rate,
                eta,
            )

    _close_landmarker_cache(landmarker_cache)
    _write_manifest_entries(manifest_path, manifest_entries)

    elapsed = time.time() - start_time
    logger.info(
        "WLASL complete in %.1fmin: processed=%s, cached=%s, failed=%s",
        elapsed / 60,
        processed,
        cached,
        failed,
    )

    split_counts = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                split_counts[e["split"]] = split_counts.get(e["split"], 0) + 1
        logger.info("Split counts: %s", split_counts)


if __name__ == "__main__":
    main()
