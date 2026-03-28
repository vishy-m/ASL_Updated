#!/usr/bin/env python3
"""Build small deterministic pilot manifests for end-to-end pipeline checks.

The pilot selection uses a shared vocabulary between WLASL and How2Sign:
frequent WLASL glosses are intersected with deterministic tokens extracted
from How2Sign sentence text. This keeps the pilot compact while still being
repeatable and cross-dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from asl_cslr.data.dataset import load_manifest
from asl_cslr.data.pilot import entry_glosses, select_balanced_subset
from asl_cslr.data.label_maps import (
    clean_wlasl_gloss,
    extract_how2sign_pilot_labels,
)
from asl_cslr.data.vocab import GlossVocab
from asl_cslr.utils.io import load_json_config

logger = logging.getLogger(__name__)


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else base_dir / path


def _load_entries_from_candidates(candidates: list[Path]) -> tuple[list[dict], Path]:
    """Load the first available manifest, or concatenate multiple split manifests."""
    primary = candidates[0]
    if primary.exists():
        return load_manifest(primary), primary

    existing = [candidate for candidate in candidates[1:] if candidate.exists()]
    if existing:
        entries: list[dict] = []
        for candidate in existing:
            entries.extend(load_manifest(candidate))
        return entries, existing[0].parent

    raise FileNotFoundError(
        "None of the candidate manifests exist: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _sort_entries(entries: list[dict]) -> list[dict]:
    return sorted(
        entries,
        key=lambda entry: (
            entry.get("split", ""),
            entry.get("id", ""),
            entry.get("features_path", ""),
        ),
    )


def _select_shared_glosses(
    wlasl_entries: list[dict],
    how2sign_entries: list[dict],
    shared_vocab_size: int,
    min_shared_frequency: int,
    seed_glosses: set[str] | None = None,
) -> list[str]:
    """Pick a small shared vocabulary from both corpora."""
    wlasl_counts: Counter[str] = Counter()
    for entry in wlasl_entries:
        glosses = entry.get("glosses", [])
        if glosses:
            wlasl_counts[clean_wlasl_gloss(glosses[0])] += 1

    how2sign_counts: Counter[str] = Counter()
    for entry in how2sign_entries:
        labels = entry.get("glosses") or extract_how2sign_pilot_labels(
            entry.get("sentence", "")
        )
        how2sign_counts.update(labels)

    shared = set(wlasl_counts) & set(how2sign_counts)
    if seed_glosses:
        shared &= seed_glosses
    ranked = []
    for gloss in shared:
        combined = wlasl_counts[gloss] + how2sign_counts[gloss]
        if combined >= min_shared_frequency:
            ranked.append((gloss, combined, wlasl_counts[gloss], how2sign_counts[gloss]))

    ranked.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
    return [gloss for gloss, _, _, _ in ranked[:shared_vocab_size]]


def _filter_wlasl_entries(entries: list[dict], allowed: set[str]) -> list[dict]:
    filtered = []
    for entry in _sort_entries(entries):
        canonical = entry.get("glosses", [])
        if not canonical:
            continue
        gloss = clean_wlasl_gloss(canonical[0])
        if gloss in allowed:
            filtered.append(entry)
    return filtered


def _filter_how2sign_entries(entries: list[dict], allowed: set[str]) -> list[dict]:
    filtered = []
    for entry in _sort_entries(entries):
        labels = entry.get("glosses") or extract_how2sign_pilot_labels(
            entry.get("sentence", ""),
            allowed,
        )
        labels = [label for label in labels if label in allowed]
        if not labels:
            continue

        input_length = entry.get("num_frames")
        if input_length is not None:
            try:
                if len(labels) > int(input_length):
                    continue
            except (TypeError, ValueError):
                pass

        new_entry = dict(entry)
        new_entry["glosses"] = labels
        filtered.append(new_entry)

    return filtered


def _write_jsonl(entries: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build deterministic pilot manifests for ISLR and CSLR."
    )
    parser.add_argument(
        "--config",
        default="configs/preprocessing.json",
        help="Preprocessing config file.",
    )
    parser.add_argument(
        "--wlasl-manifest",
        default=None,
        help="Processed WLASL manifest path. Defaults to config paths.",
    )
    parser.add_argument(
        "--how2sign-manifest",
        default=None,
        help="Processed How2Sign manifest path. Defaults to config paths.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Pilot output directory. Defaults to config paths.",
    )
    parser.add_argument(
        "--shared-vocab-size",
        type=int,
        default=None,
        help="Number of shared labels to keep.",
    )
    parser.add_argument(
        "--min-shared-frequency",
        type=int,
        default=None,
        help="Minimum combined frequency needed for a label to enter the pilot.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Maximum number of samples to keep per split for each dataset.",
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
    config = load_json_config(_resolve_path(args.config, repo_root))
    paths = config.get("paths", {})
    processed_paths = paths.get("processed", {})
    pilot_config = config.get("pilot", {})

    pilot_root = _resolve_path(
        args.output_dir or processed_paths.get("pilot", "data/processed/pilot"),
        repo_root,
    )
    manifest_dir = pilot_root / "manifests"

    wlasl_manifest = _resolve_path(
        args.wlasl_manifest
        or (Path(processed_paths.get("manifests", "data/processed/manifests")) / "wlasl.jsonl"),
        repo_root,
    )
    how2sign_manifest = _resolve_path(
        args.how2sign_manifest
        or (Path(processed_paths.get("manifests", "data/processed/manifests")) / "how2sign.jsonl"),
        repo_root,
    )

    shared_vocab_size = (
        args.shared_vocab_size
        if args.shared_vocab_size is not None
        else int(pilot_config.get("shared_vocab_size", 8))
    )
    min_shared_frequency = (
        args.min_shared_frequency
        if args.min_shared_frequency is not None
        else int(pilot_config.get("min_shared_frequency", 2))
    )
    max_samples_per_split = (
        args.max_samples_per_split
        if args.max_samples_per_split is not None
        else int(pilot_config.get("max_samples_per_split", 64))
    )
    seed_glosses = None
    seed_vocab_path = pilot_config.get("seed_vocab_path")
    if seed_vocab_path:
        resolved_seed_vocab = _resolve_path(seed_vocab_path, repo_root)
        if resolved_seed_vocab.exists():
            seed_vocab = GlossVocab.load(resolved_seed_vocab)
            seed_glosses = {
                gloss for gloss in seed_vocab.itos if not gloss.startswith("<")
            }
            logger.info("Using seed pilot vocab from %s", resolved_seed_vocab)

    wlasl_entries, resolved_wlasl = _load_entries_from_candidates(
        [
            wlasl_manifest,
            manifest_dir.parent.parent / "manifests" / "islr_train.jsonl",
            manifest_dir.parent.parent / "manifests" / "islr_val.jsonl",
            manifest_dir.parent.parent / "manifests" / "islr_test.jsonl",
        ]
    )
    how2sign_entries, resolved_how2sign = _load_entries_from_candidates(
        [
            how2sign_manifest,
            manifest_dir.parent.parent / "how2sign" / "manifest.jsonl",
            manifest_dir.parent.parent / "manifests" / "cslr_train.jsonl",
            manifest_dir.parent.parent / "manifests" / "cslr_val.jsonl",
            manifest_dir.parent.parent / "manifests" / "cslr_test.jsonl",
        ]
    )

    logger.info("Loading manifests:")
    logger.info("  WLASL: %s", resolved_wlasl)
    logger.info("  How2Sign: %s", resolved_how2sign)

    selected_glosses = _select_shared_glosses(
        wlasl_entries,
        how2sign_entries,
        shared_vocab_size=shared_vocab_size,
        min_shared_frequency=min_shared_frequency,
        seed_glosses=seed_glosses,
    )
    if not selected_glosses:
        raise RuntimeError(
            "No shared pilot glosses were found. "
            "Lower the minimum frequency or inspect the cleaned labels."
        )

    allowed = set(selected_glosses)
    logger.info("Selected pilot vocab (%s labels): %s", len(selected_glosses), selected_glosses)

    vocab = GlossVocab()
    for gloss in selected_glosses:
        vocab.add_gloss(gloss)
    vocab_path = manifest_dir / "pilot_vocab.json"
    vocab.save(vocab_path)

    wlasl_pilot = _filter_wlasl_entries(wlasl_entries, allowed)
    how2sign_pilot = _filter_how2sign_entries(
        how2sign_entries,
        allowed,
    )

    for dataset_name, entries in (
        ("islr", wlasl_pilot),
        ("cslr", how2sign_pilot),
    ):
        split_groups: dict[str, list[dict]] = {}
        for entry in entries:
            split_groups.setdefault(entry.get("split", "train"), []).append(entry)

        for split, split_entries in split_groups.items():
            split_entries = select_balanced_subset(
                split_entries, max_samples=max_samples_per_split
            )
            out_path = manifest_dir / f"{dataset_name}_pilot_{split}.jsonl"
            _write_jsonl(split_entries, out_path)
            label_counts: Counter[str] = Counter()
            for entry in split_entries:
                label_counts.update(entry_glosses(entry))
            logger.info("Wrote %s (%s entries)", out_path, len(split_entries))
            logger.info("  Label counts: %s", dict(sorted(label_counts.items())))

    logger.info("Pilot vocab written to %s", vocab_path)
    logger.info("Pilot manifests are available under %s", manifest_dir)


if __name__ == "__main__":
    main()
