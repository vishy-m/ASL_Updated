#!/usr/bin/env python3
"""Build a larger synthetic CSLR dataset from multiple isolated-sign sources.

This is the reproducible bridge from extra Kaggle isolated-sign corpora into
our CSLR pipeline:
1. Load one or more processed isolated-sign manifests.
2. Select a shared, well-supported gloss vocabulary.
3. Merge the filtered clips into deterministic ISLR source manifests.
4. Synthesize multi-word CSLR sequences from those clips.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from asl_cslr.data.dataset import load_manifest
from asl_cslr.data.demo import (
    build_vocab_file,
    preserve_source_splits,
    select_gloss_entries,
    stratified_islr_splits,
    synthesize_cslr_split,
    wait_for_feature_paths,
    write_manifest,
)
from asl_cslr.data.label_maps import clean_wlasl_gloss

logger = logging.getLogger(__name__)


def _canonical_source_group(dataset: str) -> str:
    dataset_name = (dataset or "").strip().lower()
    if dataset_name.startswith("wlasl"):
        return "wlasl"
    return dataset_name or "unknown"


def _source_clip_key(entry: dict, dataset: str) -> tuple[str, str]:
    raw_id = str(
        entry.get("source_id")
        or entry.get("raw_id")
        or entry.get("clip_id")
        or entry.get("id", "")
    )
    return _canonical_source_group(dataset), raw_id


def _load_isolated_entries(manifest_paths: list[Path]) -> list[dict]:
    entries: list[dict] = []
    seen_source_clips: set[tuple[str, str]] = set()
    for manifest_path in manifest_paths:
        source_entries = load_manifest(manifest_path)
        for entry in source_entries:
            glosses = entry.get("glosses") or []
            if not glosses:
                continue
            dataset = str(entry.get("dataset") or manifest_path.stem)
            source_key = _source_clip_key(entry, dataset)
            if source_key in seen_source_clips:
                continue
            seen_source_clips.add(source_key)
            normalized = dict(entry)
            normalized["dataset"] = dataset
            normalized["id"] = f"{dataset}:{entry.get('id', '')}"
            normalized["glosses"] = [clean_wlasl_gloss(glosses[0])]
            entries.append(normalized)
    return sorted(
        entries,
        key=lambda entry: (
            entry.get("dataset", ""),
            entry.get("split", ""),
            entry.get("glosses", [""])[0],
            entry.get("id", ""),
        ),
    )


def _rank_shared_glosses(
    entries: list[dict],
    *,
    max_glosses: int,
    min_total_count: int,
    min_count_per_source: int,
    min_train_count: int,
    min_val_count: int,
    min_test_count: int,
    min_source_datasets: int,
    preferred_glosses: list[str],
) -> list[str]:
    totals: Counter[str] = Counter()
    split_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    dataset_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for entry in entries:
        gloss = entry["glosses"][0]
        split = str(entry.get("split", "train")).lower()
        dataset = str(entry.get("dataset", "unknown"))
        totals[gloss] += 1
        split_counts[gloss][split] += 1
        dataset_counts[gloss][dataset] += 1

    preferred_order = {
        clean_wlasl_gloss(gloss): idx for idx, gloss in enumerate(preferred_glosses)
    }
    candidates: list[tuple] = []
    for gloss, total_count in totals.items():
        if total_count < min_total_count:
            continue
        if split_counts[gloss]["train"] < min_train_count:
            continue
        if split_counts[gloss]["val"] < min_val_count:
            continue
        if split_counts[gloss]["test"] < min_test_count:
            continue
        if len(dataset_counts[gloss]) < min_source_datasets:
            continue

        per_source = dataset_counts[gloss]
        if min(per_source.values()) < min_count_per_source:
            continue
        min_source_count = min(per_source.values())
        preferred_idx = preferred_order.get(gloss, 10_000)
        candidates.append(
            (
                preferred_idx,
                -total_count,
                -min_source_count,
                -split_counts[gloss]["train"],
                gloss,
            )
        )

    candidates.sort()
    return [gloss for *_rest, gloss in candidates[:max_glosses]]


def _collect_gloss_metadata(entries: list[dict], glosses: list[str]) -> dict[str, dict]:
    selected = set(glosses)
    split_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    dataset_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for entry in entries:
        gloss = entry["glosses"][0]
        if gloss not in selected:
            continue
        split_counts[gloss][str(entry.get("split", "train")).lower()] += 1
        dataset_counts[gloss][str(entry.get("dataset", "unknown"))] += 1

    return {
        gloss: {
            "split_counts": dict(split_counts[gloss]),
            "dataset_counts": dict(dataset_counts[gloss]),
        }
        for gloss in glosses
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a shared isolated-source CSLR dataset from multiple manifests."
    )
    parser.add_argument(
        "--source-manifest",
        action="append",
        required=True,
        help="Processed isolated-sign manifest path. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/kaggle_shared",
        help="Output directory for merged manifests and synthetic CSLR clips.",
    )
    parser.add_argument(
        "--max-glosses",
        type=int,
        default=20,
        help="Maximum number of shared glosses to keep.",
    )
    parser.add_argument(
        "--preferred-glosses",
        nargs="*",
        default=[],
        help="Optional preferred gloss ordering to keep familiar live signs first.",
    )
    parser.add_argument(
        "--preferred-glosses-file",
        default=None,
        help="Optional newline-delimited preferred gloss file.",
    )
    parser.add_argument("--min-total-count", type=int, default=20)
    parser.add_argument("--min-count-per-source", type=int, default=3)
    parser.add_argument("--min-train-count", type=int, default=8)
    parser.add_argument("--min-val-count", type=int, default=2)
    parser.add_argument("--min-test-count", type=int, default=2)
    parser.add_argument("--min-source-datasets", type=int, default=2)
    parser.add_argument(
        "--split-strategy",
        choices=["preserve_source", "stratified"],
        default="preserve_source",
    )
    parser.add_argument("--cslr-train-seqs", type=int, default=4096)
    parser.add_argument("--cslr-val-seqs", type=int, default=768)
    parser.add_argument("--cslr-test-seqs", type=int, default=768)
    parser.add_argument("--min-sequence-len", type=int, default=2)
    parser.add_argument("--max-sequence-len", type=int, default=4)
    parser.add_argument("--transition-frames-min", type=int, default=4)
    parser.add_argument("--transition-frames-max", type=int, default=8)
    parser.add_argument("--speed-jitter", type=float, default=0.18)
    parser.add_argument("--repeat-gloss-probability", type=float, default=0.10)
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
    source_manifests = [
        (Path(path) if Path(path).is_absolute() else (repo_root / path)).resolve()
        for path in args.source_manifest
    ]
    output_dir = (repo_root / args.output_dir).resolve()
    manifest_dir = output_dir / "manifests"
    synthetic_dir = output_dir / "synthetic_cslr"

    preferred_glosses = list(args.preferred_glosses)
    if args.preferred_glosses_file:
        preferred_path = Path(args.preferred_glosses_file)
        if not preferred_path.is_absolute():
            preferred_path = (repo_root / preferred_path).resolve()
        preferred_glosses.extend(
            gloss.strip()
            for gloss in preferred_path.read_text(encoding="utf-8").splitlines()
            if gloss.strip()
        )

    entries = _load_isolated_entries(source_manifests)
    if not entries:
        raise RuntimeError("No isolated-sign entries were loaded from the provided manifests.")

    glosses = _rank_shared_glosses(
        entries,
        max_glosses=args.max_glosses,
        min_total_count=args.min_total_count,
        min_count_per_source=args.min_count_per_source,
        min_train_count=args.min_train_count,
        min_val_count=args.min_val_count,
        min_test_count=args.min_test_count,
        min_source_datasets=args.min_source_datasets,
        preferred_glosses=preferred_glosses,
    )
    if not glosses:
        raise RuntimeError("No glosses satisfied the shared-source selection criteria.")

    grouped = select_gloss_entries(entries, glosses)
    if args.split_strategy == "preserve_source":
        islr_splits = preserve_source_splits(grouped)
    else:
        islr_splits = stratified_islr_splits(grouped)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    combined_entries = []
    for split in ("train", "val", "test"):
        split_entries = islr_splits[split]
        combined_entries.extend(split_entries)
        write_manifest(split_entries, manifest_dir / f"islr_demo_{split}.jsonl")
        logger.info("Wrote %s split with %d entries", split, len(split_entries))

    write_manifest(combined_entries, manifest_dir / "isolated_source.jsonl")
    build_vocab_file(glosses, manifest_dir / "demo_vocab.json")

    specs = [
        ("train", args.cslr_train_seqs, 11),
        ("val", args.cslr_val_seqs, 17),
        ("test", args.cslr_test_seqs, 23),
    ]
    for split, num_sequences, seed in specs:
        split_output_dir = synthetic_dir / split
        if split_output_dir.exists():
            shutil.rmtree(split_output_dir)
        generated = synthesize_cslr_split(
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
        write_manifest(generated, manifest_dir / f"cslr_demo_{split}.jsonl")
        wait_for_feature_paths(generated)
        logger.info("Wrote CSLR %s split with %d sequences", split, len(generated))

    metadata = {
        "source_manifests": [str(path) for path in source_manifests],
        "selected_glosses": glosses,
        "selection": {
            "max_glosses": args.max_glosses,
            "min_total_count": args.min_total_count,
            "min_count_per_source": args.min_count_per_source,
            "min_train_count": args.min_train_count,
            "min_val_count": args.min_val_count,
            "min_test_count": args.min_test_count,
            "min_source_datasets": args.min_source_datasets,
            "preferred_glosses": [clean_wlasl_gloss(gloss) for gloss in preferred_glosses],
            "split_strategy": args.split_strategy,
        },
        "gloss_metadata": _collect_gloss_metadata(entries, glosses),
        "synthetic": {
            "train_sequences": args.cslr_train_seqs,
            "val_sequences": args.cslr_val_seqs,
            "test_sequences": args.cslr_test_seqs,
            "min_sequence_len": args.min_sequence_len,
            "max_sequence_len": args.max_sequence_len,
            "transition_frames_min": args.transition_frames_min,
            "transition_frames_max": args.transition_frames_max,
            "speed_jitter": args.speed_jitter,
            "repeat_gloss_probability": args.repeat_gloss_probability,
        },
    }
    (manifest_dir / "demo_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Selected glosses: %s", ", ".join(glosses))


if __name__ == "__main__":
    main()
