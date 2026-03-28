#!/usr/bin/env python3
"""Build a small demo dataset from true WLASL gloss labels.

This builder creates:
1. Stratified ISLR manifests from isolated WLASL clips.
2. Synthetic CSLR manifests formed by concatenating those isolated clips.

The goal is a cleaner end-to-end demo path than the pseudo-labeled How2Sign
pilot, while keeping the label set intentionally small.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from asl_cslr.data.demo import (
    build_vocab_file,
    load_wlasl_entries,
    preserve_source_splits,
    select_gloss_entries,
    stratified_islr_splits,
    synthesize_cslr_split,
    write_manifest,
)

logger = logging.getLogger(__name__)


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
        "Timed out waiting for generated demo artifacts to appear: "
        f"{missing[:5]}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build a clean WLASL demo dataset.")
    parser.add_argument(
        "--manifest-dir",
        default="data/processed/manifests",
        help="Directory containing processed ISLR manifests, or a single WLASL JSONL manifest.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/demo",
        help="Output directory for demo manifests and synthetic CSLR clips.",
    )
    parser.add_argument(
        "--glosses",
        nargs="+",
        default=["COLOR", "DRINK", "HOT", "LIKE", "WRONG"],
        help="Glosses to include in the demo set.",
    )
    parser.add_argument(
        "--cslr-train-seqs",
        type=int,
        default=1024,
        help="Number of synthetic CSLR train sequences.",
    )
    parser.add_argument(
        "--cslr-val-seqs",
        type=int,
        default=192,
        help="Number of synthetic CSLR val sequences.",
    )
    parser.add_argument(
        "--cslr-test-seqs",
        type=int,
        default=192,
        help="Number of synthetic CSLR test sequences.",
    )
    parser.add_argument(
        "--min-sequence-len",
        type=int,
        default=3,
        help="Minimum number of glosses per synthetic CSLR sequence.",
    )
    parser.add_argument(
        "--max-sequence-len",
        type=int,
        default=5,
        help="Maximum number of glosses per synthetic CSLR sequence.",
    )
    parser.add_argument(
        "--transition-frames-min",
        type=int,
        default=4,
        help="Minimum interpolated transition frames between gloss clips.",
    )
    parser.add_argument(
        "--transition-frames-max",
        type=int,
        default=8,
        help="Maximum interpolated transition frames between gloss clips.",
    )
    parser.add_argument(
        "--speed-jitter",
        type=float,
        default=0.15,
        help="Temporal speed jitter applied to each sampled gloss clip.",
    )
    parser.add_argument(
        "--repeat-gloss-probability",
        type=float,
        default=0.08,
        help="Probability of repeating the previous gloss in a synthetic sequence.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["preserve_source", "stratified"],
        default="preserve_source",
        help="How to derive demo ISLR splits from the source manifest entries.",
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
    manifest_dir = repo_root / args.manifest_dir
    output_dir = repo_root / args.output_dir
    manifest_out = output_dir / "manifests"
    synthetic_out = output_dir / "synthetic_cslr"
    glosses = [gloss.upper() for gloss in args.glosses]

    wlasl_entries = load_wlasl_entries(manifest_dir)
    grouped = select_gloss_entries(wlasl_entries, glosses)
    missing = [gloss for gloss, entries in grouped.items() if not entries]
    if missing:
        raise RuntimeError(f"No processed WLASL entries found for glosses: {missing}")

    if args.split_strategy == "preserve_source":
        islr_splits = preserve_source_splits(grouped)
    else:
        islr_splits = stratified_islr_splits(grouped)
    for split, entries in islr_splits.items():
        out_path = manifest_out / f"islr_demo_{split}.jsonl"
        write_manifest(entries, out_path)
        logger.info(
            "Wrote %s with %d entries (%s)",
            out_path,
            len(entries),
            json.dumps({gloss: sum(e['glosses'][0] == gloss for e in entries) for gloss in glosses}),
        )

    vocab_path = manifest_out / "demo_vocab.json"
    build_vocab_file(glosses, vocab_path)
    logger.info("Wrote demo vocab to %s", vocab_path)

    synthetic_specs = [
        ("train", args.cslr_train_seqs, 11),
        ("val", args.cslr_val_seqs, 17),
        ("test", args.cslr_test_seqs, 23),
    ]
    for split, num_sequences, seed in synthetic_specs:
        split_output_dir = synthetic_out / split
        if split_output_dir.exists():
            shutil.rmtree(split_output_dir)
        entries = synthesize_cslr_split(
            source_entries=islr_splits[split],
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
        out_path = manifest_out / f"cslr_demo_{split}.jsonl"
        write_manifest(entries, out_path)
        _wait_for_artifacts(entries)
        logger.info("Wrote %s with %d synthetic sequences", out_path, len(entries))

    metadata = {
        "glosses": glosses,
        "source_manifest_dir": str(manifest_dir),
        "output_dir": str(output_dir),
        "min_sequence_len": args.min_sequence_len,
        "max_sequence_len": args.max_sequence_len,
        "transition_frames_min": args.transition_frames_min,
        "transition_frames_max": args.transition_frames_max,
        "speed_jitter": args.speed_jitter,
        "repeat_gloss_probability": args.repeat_gloss_probability,
        "split_strategy": args.split_strategy,
    }
    metadata_path = manifest_out / "demo_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
