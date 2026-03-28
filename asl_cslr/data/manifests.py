"""
Reproducible manifest builders for ISLR and CSLR training.

Turns canonical per-dataset source manifests into task-specific train/val/test
splits and vocab files. This keeps the end-to-end pipeline reproducible after
preprocessing instead of relying on stale checked-in artifacts.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path

from .dataset import load_manifest
from .label_maps import clean_wlasl_gloss, tokenize_how2sign_sentence
from .vocab import build_vocab

logger = logging.getLogger(__name__)

VALID_SPLITS = ("train", "val", "test")
DEFAULT_GOAL_STOPWORDS = {
    "A",
    "AN",
    "AND",
    "ARE",
    "AS",
    "AT",
    "BE",
    "FOR",
    "HE",
    "I",
    "IN",
    "IS",
    "IT",
    "ME",
    "MY",
    "OF",
    "ON",
    "OR",
    "OUR",
    "SHE",
    "THAT",
    "THE",
    "THEY",
    "THIS",
    "TO",
    "US",
    "WE",
    "WITH",
    "YOU",
    "YOUR",
}


def _canonical_split(split: str | None) -> str:
    split_name = (split or "train").strip().lower()
    return split_name if split_name in VALID_SPLITS else "train"


def _sort_entries(entries: list[dict]) -> list[dict]:
    return sorted(
        entries,
        key=lambda entry: (
            _canonical_split(entry.get("split")),
            entry.get("id", ""),
            entry.get("features_path", ""),
        ),
    )


def _primary_gloss(entry: dict) -> str | None:
    glosses = entry.get("glosses", [])
    if not glosses:
        return None
    return clean_wlasl_gloss(glosses[0])


def _write_jsonl(entries: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _write_json(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _ctc_min_required_frames(glosses: list[str]) -> int:
    if not glosses:
        return 0
    adjacent_repeats = sum(
        1
        for prev, cur in zip(glosses[:-1], glosses[1:])
        if prev == cur
    )
    return len(glosses) + adjacent_repeats


def _tokenize_how2sign_entry(entry: dict) -> list[str]:
    return tokenize_how2sign_sentence(entry.get("sentence", ""))


def _goal_gloss_score(wlasl_count: int, how2sign_count: int) -> float:
    """Rank goal glosses by isolated support and continuous-usage coverage."""
    return float(wlasl_count) * math.log1p(float(how2sign_count))


def select_shared_goal_glosses(
    wlasl_entries: list[dict],
    how2sign_entries: list[dict],
    *,
    shared_vocab_size: int = 30,
    min_wlasl_frequency: int = 4,
    min_how2sign_frequency: int = 10,
    stopwords: set[str] | None = None,
    min_train_per_gloss: int = 0,
    min_val_per_gloss: int = 0,
    min_test_per_gloss: int = 0,
) -> list[str]:
    """Pick the CSLR goal vocabulary from the WLASL∩How2Sign overlap.

    Candidates must satisfy the isolated-sign and continuous-data floors.
    Ranking then uses a WLASL-backed score with log-scaled How2Sign coverage
    so the selected live vocabulary remains viable for synthetic CSLR while
    still favoring words that occur often in continuous signing.
    """
    stopword_set = set(DEFAULT_GOAL_STOPWORDS)
    if stopwords:
        stopword_set.update(stopwords)

    wlasl_counts: Counter[str] = Counter()
    for entry in wlasl_entries:
        glosses = entry.get("glosses", [])
        if glosses:
            wlasl_counts[clean_wlasl_gloss(glosses[0])] += 1

    how2sign_counts: Counter[str] = Counter()
    for entry in how2sign_entries:
        how2sign_counts.update(_tokenize_how2sign_entry(entry))

    min_required_wlasl = max(
        int(min_wlasl_frequency),
        max(int(min_train_per_gloss), 0)
        + max(int(min_val_per_gloss), 0)
        + max(int(min_test_per_gloss), 0),
    )

    ranked = []
    for gloss in set(wlasl_counts) & set(how2sign_counts):
        if gloss in stopword_set:
            continue
        if wlasl_counts[gloss] < min_required_wlasl:
            continue
        if how2sign_counts[gloss] < min_how2sign_frequency:
            continue
        ranked.append((gloss, wlasl_counts[gloss], how2sign_counts[gloss]))

    ranked.sort(
        key=lambda item: (
            -_goal_gloss_score(item[1], item[2]),
            -item[1],
            -item[2],
            item[0],
        )
    )
    return [gloss for gloss, _, _ in ranked[:shared_vocab_size]]


def build_islr_split_entries(
    wlasl_entries: list[dict],
    *,
    allowed_glosses: set[str] | None = None,
    selection_name: str | None = None,
) -> dict[str, list[dict]]:
    """Build deterministic ISLR train/val/test splits from WLASL source entries."""
    grouped = {split: [] for split in VALID_SPLITS}
    for entry in _sort_entries(wlasl_entries):
        glosses = entry.get("glosses", [])
        if not glosses:
            continue
        gloss = clean_wlasl_gloss(glosses[0])
        if allowed_glosses and gloss not in allowed_glosses:
            continue

        split = _canonical_split(entry.get("split"))
        new_entry = dict(entry)
        new_entry["glosses"] = [gloss]
        if selection_name:
            new_entry["selection"] = selection_name
        grouped[split].append(new_entry)

    return grouped


def stratify_islr_entries_by_gloss(
    entries: list[dict],
    *,
    min_val_per_gloss: int = 1,
    min_test_per_gloss: int = 1,
) -> dict[str, list[dict]]:
    """Reassign a focused ISLR subset so every gloss appears in val/test.

    The source WLASL split is still used as a preference order, but for small
    goal vocabularies we deterministically rebalance clips to avoid evaluation
    holes where a target gloss never appears in validation or test.
    """
    grouped: dict[str, list[dict]] = {}
    for entry in _sort_entries(entries):
        gloss = _primary_gloss(entry)
        if gloss is None:
            continue
        grouped.setdefault(gloss, []).append(entry)

    split_map = {split: [] for split in VALID_SPLITS}
    for gloss in sorted(grouped):
        ordered = grouped[gloss]
        used: set[int] = set()

        def pick(preferred_splits: tuple[str, ...]) -> int | None:
            if len(used) >= max(len(ordered) - 1, 0):
                return None
            for desired in preferred_splits:
                for idx, entry in enumerate(ordered):
                    if idx in used:
                        continue
                    if _canonical_split(entry.get("split")) == desired:
                        used.add(idx)
                        return idx
            for idx, _entry in enumerate(ordered):
                if idx not in used:
                    used.add(idx)
                    return idx
            return None

        for _ in range(max(min_test_per_gloss, 0)):
            idx = pick(("test", "val", "train"))
            if idx is None:
                break
            entry = dict(ordered[idx])
            entry["split"] = "test"
            split_map["test"].append(entry)

        for _ in range(max(min_val_per_gloss, 0)):
            idx = pick(("val", "test", "train"))
            if idx is None:
                break
            entry = dict(ordered[idx])
            entry["split"] = "val"
            split_map["val"].append(entry)

        for idx, entry in enumerate(ordered):
            if idx in used:
                continue
            updated = dict(entry)
            updated["split"] = "train"
            split_map["train"].append(updated)

    for split in split_map:
        split_map[split] = _sort_entries(split_map[split])

    return split_map


def build_cslr_split_entries(
    how2sign_entries: list[dict],
    *,
    allowed_glosses: set[str] | None = None,
    min_glosses_per_sequence: int = 1,
    selection_name: str | None = None,
    label_source: str = "sentence_tokens",
) -> dict[str, list[dict]]:
    """Build deterministic CSLR train/val/test splits from How2Sign source data."""
    grouped = {split: [] for split in VALID_SPLITS}

    for entry in _sort_entries(how2sign_entries):
        glosses = _tokenize_how2sign_entry(entry)
        if allowed_glosses is not None:
            glosses = [gloss for gloss in glosses if gloss in allowed_glosses]
        if len(glosses) < min_glosses_per_sequence:
            continue

        num_frames = entry.get("num_frames")
        if num_frames is not None:
            try:
                if _ctc_min_required_frames(glosses) > int(num_frames):
                    continue
            except (TypeError, ValueError):
                pass

        split = _canonical_split(entry.get("split"))
        new_entry = dict(entry)
        new_entry["glosses"] = glosses
        new_entry["num_glosses"] = len(glosses)
        new_entry["label_source"] = label_source
        if selection_name:
            new_entry["selection"] = selection_name
        grouped[split].append(new_entry)

    return grouped


def _collect_all_glosses(grouped_entries: dict[str, list[dict]]) -> list[str]:
    glosses: list[str] = []
    for split in VALID_SPLITS:
        for entry in grouped_entries.get(split, []):
            glosses.extend(entry.get("glosses", []))
    return glosses


def _write_split_bundle(
    grouped_entries: dict[str, list[dict]],
    *,
    output_dir: Path,
    prefix: str,
    vocab_path: Path,
    vocab_max_size: int = 3000,
    vocab_min_frequency: int = 1,
):
    for split in VALID_SPLITS:
        _write_jsonl(
            grouped_entries.get(split, []),
            output_dir / f"{prefix}_{split}.jsonl",
        )

    vocab = build_vocab(
        _collect_all_glosses(grouped_entries),
        max_size=vocab_max_size,
        min_frequency=vocab_min_frequency,
    )
    vocab.save(vocab_path)
    return vocab


def rebuild_training_manifests(
    *,
    wlasl_manifest_path: str | Path,
    how2sign_manifest_path: str | Path,
    output_dir: str | Path,
    goal_shared_vocab_size: int = 30,
    goal_min_wlasl_frequency: int = 4,
    goal_min_how2sign_frequency: int = 10,
    goal_min_glosses_per_sequence: int = 2,
    goal_split_strategy: str = "preserve_source",
    goal_min_val_per_gloss: int = 1,
    goal_min_test_per_gloss: int = 1,
    cslr_full_min_glosses_per_sequence: int = 1,
    cslr_full_vocab_max_size: int = 3000,
    cslr_full_vocab_min_frequency: int = 1,
    extra_goal_stopwords: set[str] | None = None,
    goal_glosses: list[str] | None = None,
) -> dict:
    """Build reproducible task manifests and vocab files from source manifests."""
    output_dir = Path(output_dir)
    wlasl_entries = load_manifest(wlasl_manifest_path)
    how2sign_entries = load_manifest(how2sign_manifest_path)

    if not wlasl_entries:
        raise ValueError(f"WLASL source manifest is empty: {wlasl_manifest_path}")
    if not how2sign_entries:
        raise ValueError(f"How2Sign source manifest is empty: {how2sign_manifest_path}")

    explicit_goal_glosses = bool(goal_glosses)
    requested_goal_glosses = (
        [clean_wlasl_gloss(gloss) for gloss in goal_glosses if gloss]
        if goal_glosses
        else []
    )
    ranked_goal_glosses = select_shared_goal_glosses(
        wlasl_entries,
        how2sign_entries,
        shared_vocab_size=max(goal_shared_vocab_size, len(requested_goal_glosses), 256),
        min_wlasl_frequency=goal_min_wlasl_frequency,
        min_how2sign_frequency=goal_min_how2sign_frequency,
        stopwords=extra_goal_stopwords,
        min_train_per_gloss=1,
        min_val_per_gloss=goal_min_val_per_gloss,
        min_test_per_gloss=goal_min_test_per_gloss,
    )
    if requested_goal_glosses:
        ranked_goal_set = set(ranked_goal_glosses)
        goal_glosses = []
        for gloss in requested_goal_glosses:
            if gloss in ranked_goal_set and gloss not in goal_glosses:
                goal_glosses.append(gloss)
                if len(goal_glosses) >= goal_shared_vocab_size:
                    break
        for gloss in ranked_goal_glosses:
            if len(goal_glosses) >= goal_shared_vocab_size:
                break
            if gloss in goal_glosses:
                continue
            goal_glosses.append(gloss)
    else:
        goal_glosses = ranked_goal_glosses[:goal_shared_vocab_size]
    goal_gloss_set = set(goal_glosses)

    if not goal_glosses:
        raise ValueError("Goal glossary selection produced zero shared glosses")

    full_islr = build_islr_split_entries(wlasl_entries)
    goal_islr = build_islr_split_entries(
        wlasl_entries,
        allowed_glosses=goal_gloss_set,
        selection_name="goal_shared_vocab",
    )
    if goal_split_strategy == "stratified_per_gloss":
        goal_islr = stratify_islr_entries_by_gloss(
            [
                entry
                for split in VALID_SPLITS
                for entry in goal_islr.get(split, [])
            ],
            min_val_per_gloss=goal_min_val_per_gloss,
            min_test_per_gloss=goal_min_test_per_gloss,
        )
    full_cslr = build_cslr_split_entries(
        how2sign_entries,
        min_glosses_per_sequence=cslr_full_min_glosses_per_sequence,
        label_source="sentence_tokens",
    )
    goal_cslr = build_cslr_split_entries(
        how2sign_entries,
        allowed_glosses=goal_gloss_set,
        min_glosses_per_sequence=goal_min_glosses_per_sequence,
        selection_name="goal_shared_vocab",
        label_source="sentence_tokens_filtered",
    )

    if not any(goal_islr[split] for split in VALID_SPLITS):
        raise ValueError("Goal glossary produced zero ISLR samples")
    if not any(goal_cslr[split] for split in VALID_SPLITS):
        raise ValueError("Goal glossary produced zero CSLR samples")

    _write_split_bundle(
        full_islr,
        output_dir=output_dir,
        prefix="islr",
        vocab_path=output_dir / "islr_vocab.json",
    )
    goal_vocab = _write_split_bundle(
        goal_islr,
        output_dir=output_dir,
        prefix="islr_goal",
        vocab_path=output_dir / "goal_vocab.json",
    )
    _write_split_bundle(
        full_cslr,
        output_dir=output_dir,
        prefix="cslr_full",
        vocab_path=output_dir / "cslr_full_vocab.json",
        vocab_max_size=cslr_full_vocab_max_size,
        vocab_min_frequency=cslr_full_vocab_min_frequency,
    )

    for split in VALID_SPLITS:
        _write_jsonl(goal_cslr.get(split, []), output_dir / f"cslr_{split}.jsonl")
    goal_vocab.save(output_dir / "cslr_vocab.json")

    metadata = {
        "source_manifests": {
            "wlasl": str(wlasl_manifest_path),
            "how2sign": str(how2sign_manifest_path),
        },
        "goal_glosses": goal_glosses,
        "goal_selection": {
            "mode": (
                "seeded_shared_frequency_overlap"
                if explicit_goal_glosses
                else "shared_frequency_overlap"
            ),
            "shared_vocab_size": goal_shared_vocab_size,
            "min_wlasl_frequency": goal_min_wlasl_frequency,
            "min_how2sign_frequency": goal_min_how2sign_frequency,
            "min_glosses_per_sequence": goal_min_glosses_per_sequence,
            "split_strategy": goal_split_strategy,
            "min_val_per_gloss": goal_min_val_per_gloss,
            "min_test_per_gloss": goal_min_test_per_gloss,
            "stopwords": sorted(set(DEFAULT_GOAL_STOPWORDS) | set(extra_goal_stopwords or set())),
        },
        "requested_goal_glosses": requested_goal_glosses,
        "splits": {
            "islr": {split: len(full_islr.get(split, [])) for split in VALID_SPLITS},
            "islr_goal": {split: len(goal_islr.get(split, [])) for split in VALID_SPLITS},
            "cslr_full": {split: len(full_cslr.get(split, [])) for split in VALID_SPLITS},
            "cslr_goal": {split: len(goal_cslr.get(split, [])) for split in VALID_SPLITS},
        },
    }
    _write_json(metadata, output_dir / "goal_manifest_metadata.json")

    logger.info(
        "Rebuilt manifests: ISLR=%s goal-ISLR=%s CSLR(full)=%s CSLR(goal)=%s",
        metadata["splits"]["islr"],
        metadata["splits"]["islr_goal"],
        metadata["splits"]["cslr_full"],
        metadata["splits"]["cslr_goal"],
    )

    return metadata
