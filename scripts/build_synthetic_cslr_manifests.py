#!/usr/bin/env python3
"""Build exact-label synthetic CSLR manifests from goal-vocabulary WLASL clips."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from asl_cslr.data.dataset import load_manifest
from asl_cslr.data.demo import build_vocab_file, synthesize_cslr_split, write_manifest
from asl_cslr.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (base_dir / path)


def _load_goal_glosses(goal_metadata_path: Path, explicit_glosses: list[str] | None) -> list[str]:
    if explicit_glosses:
        return [gloss.upper() for gloss in explicit_glosses if gloss]

    metadata = json.loads(goal_metadata_path.read_text(encoding="utf-8"))
    goal_glosses = metadata.get("goal_glosses", [])
    if not goal_glosses:
        raise ValueError(f"No goal_glosses found in {goal_metadata_path}")
    return [str(gloss).upper() for gloss in goal_glosses]


def _wait_for_artifacts(manifests: list[dict], timeout_sec: float = 30.0):
    """Wait for generated files to become visible on disk before returning."""
    deadline = time.time() + timeout_sec
    missing = []
    while time.time() < deadline:
        missing = [
            entry["features_path"]
            for entry in manifests
            if not Path(entry["features_path"]).exists()
        ]
        if not missing:
            return
        time.sleep(0.5)
    raise RuntimeError(
        "Timed out waiting for generated synthetic artifacts to appear: "
        f"{missing[:5]}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build synthetic exact-label CSLR manifests from goal WLASL clips.",
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/processed/manifests",
        help="Directory containing the rebuilt goal ISLR manifests.",
    )
    parser.add_argument(
        "--synthetic-dir",
        default="data/processed/synthetic_cslr",
        help="Directory for generated synthetic CSLR .npz files.",
    )
    parser.add_argument(
        "--goal-metadata",
        default="data/processed/manifests/goal_manifest_metadata.json",
        help="Path to the goal metadata JSON produced by build_training_manifests.py.",
    )
    parser.add_argument(
        "--glosses",
        nargs="*",
        default=None,
        help="Optional explicit goal glosses. Defaults to goal_manifest_metadata.json.",
    )
    parser.add_argument("--train-sequences", type=int, default=4096)
    parser.add_argument("--val-sequences", type=int, default=512)
    parser.add_argument("--test-sequences", type=int, default=512)
    parser.add_argument("--min-sequence-len", type=int, default=2)
    parser.add_argument("--max-sequence-len", type=int, default=5)
    parser.add_argument("--transition-frames-min", type=int, default=3)
    parser.add_argument("--transition-frames-max", type=int, default=7)
    parser.add_argument("--speed-jitter", type=float, default=0.12)
    parser.add_argument("--repeat-gloss-probability", type=float, default=0.12)
    parser.add_argument(
        "--manifest-prefix",
        default="cslr_synthetic",
        help="Prefix for the generated manifest and vocab filenames.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    repo_root = Path(__file__).resolve().parents[1]
    manifest_dir = _resolve_path(args.manifest_dir, repo_root)
    synthetic_dir = _resolve_path(args.synthetic_dir, repo_root)
    goal_metadata_path = _resolve_path(args.goal_metadata, repo_root)
    glosses = _load_goal_glosses(goal_metadata_path, args.glosses)
    gloss_set = set(glosses)

    split_specs = [
        ("train", args.train_sequences, 101),
        ("val", args.val_sequences, 211),
        ("test", args.test_sequences, 307),
    ]
    split_counts: dict[str, dict[str, int]] = {}
    generated_coverage: dict[str, dict[str, int]] = {}
    for split, num_sequences, seed in split_specs:
        source_path = manifest_dir / f"islr_goal_{split}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source manifest: {source_path}")
        source_entries = [
            entry
            for entry in load_manifest(source_path)
            if entry.get("glosses") and entry["glosses"][0] in gloss_set
        ]
        source_glosses = {entry["glosses"][0] for entry in source_entries}
        missing = [gloss for gloss in glosses if gloss not in source_glosses]
        if missing:
            raise RuntimeError(
                f"Source split {split} is missing goal glosses required for synthetic CSLR: {missing}"
            )
        split_counts[split] = {
            gloss: sum(entry["glosses"][0] == gloss for entry in source_entries)
            for gloss in glosses
        }
        split_output_dir = synthetic_dir / split
        if split_output_dir.exists():
            shutil.rmtree(split_output_dir)
        manifests = synthesize_cslr_split(
            source_entries=source_entries,
            output_dir=split_output_dir,
            split=split,
            num_sequences=num_sequences,
            seed=seed,
            min_sequence_len=args.min_sequence_len,
            max_sequence_len=args.max_sequence_len,
            transition_frames_min=args.transition_frames_min,
            transition_frames_max=args.transition_frames_max,
            speed_jitter=args.speed_jitter,
            repeat_gloss_probability=args.repeat_gloss_probability,
            required_glosses=glosses,
        )
        generated_coverage[split] = {
            gloss: sum(gloss in entry["glosses"] for entry in manifests)
            for gloss in glosses
        }
        out_path = manifest_dir / f"{args.manifest_prefix}_{split}.jsonl"
        write_manifest(manifests, out_path)
        _wait_for_artifacts(manifests)
        logger.info("Wrote %s with %d sequences", out_path, len(manifests))

    vocab_path = manifest_dir / f"{args.manifest_prefix}_vocab.json"
    build_vocab_file(glosses, vocab_path)

    metadata = {
        "goal_glosses": glosses,
        "goal_metadata_path": str(goal_metadata_path),
        "source_manifest_dir": str(manifest_dir),
        "synthetic_dir": str(synthetic_dir),
        "manifest_prefix": args.manifest_prefix,
        "source_split_counts": split_counts,
        "generated_sequences": {
            "train": args.train_sequences,
            "val": args.val_sequences,
            "test": args.test_sequences,
        },
        "generated_gloss_coverage": generated_coverage,
        "min_sequence_len": args.min_sequence_len,
        "max_sequence_len": args.max_sequence_len,
        "transition_frames_min": args.transition_frames_min,
        "transition_frames_max": args.transition_frames_max,
        "speed_jitter": args.speed_jitter,
        "repeat_gloss_probability": args.repeat_gloss_probability,
    }
    metadata_path = manifest_dir / f"{args.manifest_prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", metadata_path)
    logger.info("Wrote %s", vocab_path)


if __name__ == "__main__":
    main()
