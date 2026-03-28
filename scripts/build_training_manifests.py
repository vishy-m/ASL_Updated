#!/usr/bin/env python3
"""Build reproducible ISLR/CSLR training manifests from source manifests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from asl_cslr.data.manifests import rebuild_training_manifests
from asl_cslr.utils.io import load_json_config
from asl_cslr.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (base_dir / path)


def _first_existing_path(candidates: list[Path], *, label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {label}. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build reproducible ISLR/CSLR manifests after preprocessing.",
    )
    parser.add_argument(
        "--config",
        default="configs/preprocessing.json",
        help="Preprocessing config file with manifest_build defaults.",
    )
    parser.add_argument(
        "--wlasl-manifest",
        default=None,
        help="Canonical WLASL source manifest. Defaults to data/processed/manifests/wlasl.jsonl.",
    )
    parser.add_argument(
        "--how2sign-manifest",
        default=None,
        help="Canonical How2Sign source manifest. Defaults to data/processed/manifests/how2sign.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for task manifests and vocab files. Defaults to config paths.processed.manifests.",
    )
    parser.add_argument("--goal-shared-vocab-size", type=int, default=None)
    parser.add_argument(
        "--goal-glosses",
        nargs="*",
        default=None,
        help="Optional preferred goal gloss list. Viability filters still apply before filling from the ranked overlap.",
    )
    parser.add_argument("--goal-min-wlasl-frequency", type=int, default=None)
    parser.add_argument("--goal-min-how2sign-frequency", type=int, default=None)
    parser.add_argument("--goal-min-glosses-per-sequence", type=int, default=None)
    parser.add_argument(
        "--goal-split-strategy",
        choices=["preserve_source", "stratified_per_gloss"],
        default=None,
    )
    parser.add_argument("--goal-min-val-per-gloss", type=int, default=None)
    parser.add_argument("--goal-min-test-per-gloss", type=int, default=None)
    parser.add_argument("--cslr-full-min-glosses-per-sequence", type=int, default=None)
    parser.add_argument("--cslr-full-vocab-max-size", type=int, default=None)
    parser.add_argument("--cslr-full-vocab-min-frequency", type=int, default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    repo_root = Path(__file__).resolve().parents[1]
    config = load_json_config(_resolve_path(args.config, repo_root))
    paths = config.get("paths", {})
    processed_paths = paths.get("processed", {})
    manifest_cfg = config.get("manifest_build", {})
    goal_cfg = manifest_cfg.get("goal", {})
    cslr_full_cfg = manifest_cfg.get("cslr_full", {})

    output_dir = _resolve_path(
        args.output_dir or processed_paths.get("manifests", "data/processed/manifests"),
        repo_root,
    )
    wlasl_manifest = _first_existing_path(
        [
            _resolve_path(args.wlasl_manifest, repo_root)
            if args.wlasl_manifest else output_dir / "wlasl.jsonl",
            repo_root / "data/processed/wlasl/manifest.jsonl",
        ],
        label="WLASL source manifest",
    )
    how2sign_manifest = _first_existing_path(
        [
            _resolve_path(args.how2sign_manifest, repo_root)
            if args.how2sign_manifest else output_dir / "how2sign.jsonl",
            repo_root / "data/processed/how2sign/manifest.jsonl",
        ],
        label="How2Sign source manifest",
    )

    metadata = rebuild_training_manifests(
        wlasl_manifest_path=wlasl_manifest,
        how2sign_manifest_path=how2sign_manifest,
        output_dir=output_dir,
        goal_shared_vocab_size=(
            args.goal_shared_vocab_size
            if args.goal_shared_vocab_size is not None
            else int(goal_cfg.get("shared_vocab_size", 30))
        ),
        goal_glosses=args.goal_glosses or goal_cfg.get("seed_glosses"),
        goal_min_wlasl_frequency=(
            args.goal_min_wlasl_frequency
            if args.goal_min_wlasl_frequency is not None
            else int(goal_cfg.get("min_wlasl_frequency", 4))
        ),
        goal_min_how2sign_frequency=(
            args.goal_min_how2sign_frequency
            if args.goal_min_how2sign_frequency is not None
            else int(goal_cfg.get("min_how2sign_frequency", 10))
        ),
        goal_min_glosses_per_sequence=(
            args.goal_min_glosses_per_sequence
            if args.goal_min_glosses_per_sequence is not None
            else int(goal_cfg.get("min_glosses_per_sequence", 2))
        ),
        goal_split_strategy=(
            args.goal_split_strategy
            if args.goal_split_strategy is not None
            else str(goal_cfg.get("split_strategy", "preserve_source"))
        ),
        goal_min_val_per_gloss=(
            args.goal_min_val_per_gloss
            if args.goal_min_val_per_gloss is not None
            else int(goal_cfg.get("min_val_per_gloss", 1))
        ),
        goal_min_test_per_gloss=(
            args.goal_min_test_per_gloss
            if args.goal_min_test_per_gloss is not None
            else int(goal_cfg.get("min_test_per_gloss", 1))
        ),
        cslr_full_min_glosses_per_sequence=(
            args.cslr_full_min_glosses_per_sequence
            if args.cslr_full_min_glosses_per_sequence is not None
            else int(cslr_full_cfg.get("min_glosses_per_sequence", 1))
        ),
        cslr_full_vocab_max_size=(
            args.cslr_full_vocab_max_size
            if args.cslr_full_vocab_max_size is not None
            else int(cslr_full_cfg.get("vocab_max_size", 3000))
        ),
        cslr_full_vocab_min_frequency=(
            args.cslr_full_vocab_min_frequency
            if args.cslr_full_vocab_min_frequency is not None
            else int(cslr_full_cfg.get("vocab_min_frequency", 1))
        ),
        extra_goal_stopwords=set(goal_cfg.get("stopword_glosses", [])),
    )

    logger.info("Goal glosses: %s", ", ".join(metadata["goal_glosses"]))


if __name__ == "__main__":
    main()
