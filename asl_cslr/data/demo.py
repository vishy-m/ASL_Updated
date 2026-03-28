"""Helpers for building compact demo datasets from processed manifests."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset import load_manifest
from .skeleton import (
    NUM_COORDS,
    NUM_JOINTS,
    COORD_FEATURE_DIM,
    FEATURE_DIM,
    build_feature_frame,
    compute_motion_features,
    extract_coordinate_features,
    extract_observation_mask,
)
from .vocab import GlossVocab


def _entry_sort_key(entry: dict) -> tuple[str, str, str]:
    return (
        entry.get("split", ""),
        entry.get("id", ""),
        entry.get("features_path", ""),
    )


def load_wlasl_entries(manifest_source: str | Path) -> list[dict]:
    """Load processed WLASL entries from split manifests or one JSONL file."""
    manifest_source = Path(manifest_source)

    if manifest_source.is_file():
        return sorted(load_manifest(manifest_source), key=_entry_sort_key)

    split_paths = [manifest_source / f"islr_{split}.jsonl" for split in ("train", "val", "test")]
    if all(path.exists() for path in split_paths):
        entries: list[dict] = []
        for path in split_paths:
            entries.extend(load_manifest(path))
        return sorted(entries, key=_entry_sort_key)

    single_manifest = manifest_source / "wlasl.jsonl"
    if single_manifest.exists():
        return sorted(load_manifest(single_manifest), key=_entry_sort_key)

    raise FileNotFoundError(
        "Could not find WLASL manifests under "
        f"{manifest_source}. Expected split manifests or wlasl.jsonl."
    )


def select_gloss_entries(entries: list[dict], glosses: list[str]) -> dict[str, list[dict]]:
    """Group manifest entries by requested gloss."""
    grouped: dict[str, list[dict]] = {gloss: [] for gloss in glosses}
    allowed = set(glosses)
    for entry in entries:
        entry_glosses = entry.get("glosses") or []
        if not entry_glosses:
            continue
        gloss = entry_glosses[0]
        if gloss in allowed:
            grouped[gloss].append(entry)
    return grouped


def stratified_islr_splits(
    grouped_entries: dict[str, list[dict]],
    min_val_per_gloss: int = 1,
    min_test_per_gloss: int = 1,
) -> dict[str, list[dict]]:
    """Build deterministic per-gloss train/val/test splits for small demos."""
    split_map = {"train": [], "val": [], "test": []}

    for gloss, entries in grouped_entries.items():
        ordered = sorted(entries, key=_entry_sort_key)
        used: set[int] = set()

        def pick(preferred_splits: tuple[str, ...]) -> int | None:
            for desired in preferred_splits:
                for idx, entry in enumerate(ordered):
                    if idx in used:
                        continue
                    if entry.get("split") == desired:
                        used.add(idx)
                        return idx
            for idx, _entry in enumerate(ordered):
                if idx not in used:
                    used.add(idx)
                    return idx
            return None

        for _ in range(min_test_per_gloss):
            idx = pick(("test", "val", "train"))
            if idx is not None:
                entry = dict(ordered[idx])
                entry["split"] = "test"
                split_map["test"].append(entry)

        for _ in range(min_val_per_gloss):
            idx = pick(("val", "test", "train"))
            if idx is not None:
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
        split_map[split] = sorted(split_map[split], key=_entry_sort_key)

    return split_map


def preserve_source_splits(grouped_entries: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Preserve the original manifest split assignments for a demo subset."""
    split_map = {"train": [], "val": [], "test": []}

    for entries in grouped_entries.values():
        for entry in sorted(entries, key=_entry_sort_key):
            split = entry.get("split", "train")
            if split not in split_map:
                split = "train"
            updated = dict(entry)
            updated["split"] = split
            split_map[split].append(updated)

    for split in split_map:
        split_map[split] = sorted(split_map[split], key=_entry_sort_key)

    return split_map


def build_vocab_file(glosses: list[str], output_path: str | Path) -> Path:
    """Write a demo vocab JSON file."""
    output_path = Path(output_path)
    vocab = GlossVocab()
    for gloss in glosses:
        vocab.add_gloss(gloss)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(output_path)
    return output_path


def write_manifest(entries: list[dict], output_path: str | Path):
    """Write a JSONL manifest file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def wait_for_feature_paths(
    entries: list[dict],
    *,
    timeout_sec: float = 60.0,
    poll_interval_sec: float = 1.0,
) -> None:
    """Block until every manifest feature path is visible on disk.

    This guards against delayed materialization on slower external volumes,
    where large `.npz` batches can become visible a short time after the
    generator function returns.
    """
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    missing = [
        str(Path(entry["features_path"]))
        for entry in entries
        if not Path(entry["features_path"]).exists()
    ]
    while missing and time.monotonic() < deadline:
        time.sleep(max(poll_interval_sec, 0.01))
        missing = [
            str(Path(entry["features_path"]))
            for entry in entries
            if not Path(entry["features_path"]).exists()
        ]

    if missing:
        sample = missing[:5]
        raise RuntimeError(
            "Synthetic features did not materialize before timeout. "
            f"Missing {len(missing)} files, sample={sample}"
        )


def _resample_temporal_sequence(
    clip: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """Resample a skeleton clip along the time axis."""
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if clip.shape[0] == target_length:
        return clip.astype(np.float32, copy=False)
    if clip.shape[0] == 1:
        return np.repeat(clip.astype(np.float32), target_length, axis=0)

    coords = extract_coordinate_features(clip).reshape(
        clip.shape[0],
        NUM_JOINTS,
        NUM_COORDS,
    )
    observed_mask = extract_observation_mask(clip)
    if observed_mask is None:
        raise ValueError("Synthetic CSLR generation requires packed xyz+mask features")

    source_positions = np.arange(clip.shape[0], dtype=np.float32)
    target_positions = np.linspace(
        0.0,
        float(clip.shape[0] - 1),
        num=target_length,
        dtype=np.float32,
    )
    flat_coords = coords.reshape(clip.shape[0], -1)
    resampled_coords = np.empty((target_length, flat_coords.shape[1]), dtype=np.float32)
    for dim in range(flat_coords.shape[1]):
        resampled_coords[:, dim] = np.interp(
            target_positions,
            source_positions,
            flat_coords[:, dim],
        )

    nearest_indices = np.clip(
        np.rint(target_positions).astype(np.int32),
        0,
        clip.shape[0] - 1,
    )
    resampled_mask = observed_mask[nearest_indices]
    packed = np.concatenate(
        [
            resampled_coords.reshape(target_length, NUM_JOINTS, NUM_COORDS),
            resampled_mask[..., None],
        ],
        axis=2,
    )
    return packed.reshape(target_length, FEATURE_DIM).astype(np.float32, copy=False)


def _sample_clip_variant(
    clip: np.ndarray,
    rng: np.random.Generator,
    speed_jitter: float,
) -> np.ndarray:
    """Apply a small temporal speed perturbation to a source clip."""
    if clip.shape[0] <= 4 or speed_jitter <= 0.0:
        return clip.astype(np.float32, copy=False)

    scale = float(rng.uniform(1.0 - speed_jitter, 1.0 + speed_jitter))
    target_length = max(4, int(round(clip.shape[0] * scale)))
    return _resample_temporal_sequence(clip, target_length)


def _build_transition_frames(
    left_clip: np.ndarray,
    right_clip: np.ndarray,
    frames: int,
) -> np.ndarray | None:
    """Interpolate a short transition between consecutive gloss clips."""
    if frames <= 0:
        return None

    left_coords = extract_coordinate_features(left_clip).reshape(
        left_clip.shape[0],
        NUM_JOINTS,
        NUM_COORDS,
    )
    right_coords = extract_coordinate_features(right_clip).reshape(
        right_clip.shape[0],
        NUM_JOINTS,
        NUM_COORDS,
    )
    left_mask = extract_observation_mask(left_clip)
    right_mask = extract_observation_mask(right_clip)
    if left_mask is None or right_mask is None:
        raise ValueError("Synthetic CSLR transitions require packed xyz+mask features")

    start = left_coords[-1]
    end = right_coords[0]
    transition_mask = (
        (left_mask[-1] > 0.5) & (right_mask[0] > 0.5)
    ).astype(np.float32)
    weights = np.linspace(0.0, 1.0, num=frames + 2, dtype=np.float32)[1:-1]
    transition_coords = np.stack(
        [(1.0 - weight) * start + weight * end for weight in weights],
        axis=0,
    )
    return np.stack(
        [
            build_feature_frame(coords, transition_mask)
            for coords in transition_coords.astype(np.float32, copy=False)
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _validate_required_glosses(
    available_glosses: list[str],
    required_glosses: list[str] | None,
) -> list[str]:
    """Normalize and validate a requested required-gloss coverage list."""
    if not required_glosses:
        return []

    normalized = []
    seen = set()
    for gloss in required_glosses:
        gloss = str(gloss).upper()
        if gloss not in seen:
            normalized.append(gloss)
            seen.add(gloss)

    missing = [gloss for gloss in normalized if gloss not in set(available_glosses)]
    if missing:
        raise ValueError(
            "Cannot guarantee synthetic coverage for missing source glosses: "
            f"{missing}"
        )
    return normalized


def synthesize_cslr_split(
    source_entries: list[dict],
    output_dir: str | Path,
    split: str,
    num_sequences: int,
    seed: int,
    min_sequence_len: int = 2,
    max_sequence_len: int = 4,
    transition_frames_min: int = 0,
    transition_frames_max: int = 2,
    speed_jitter: float = 0.12,
    repeat_gloss_probability: float = 0.2,
    required_glosses: list[str] | None = None,
) -> list[dict]:
    """Build synthetic multi-word CSLR sequences from isolated WLASL clips."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ordered = sorted(source_entries, key=_entry_sort_key)
    if not ordered:
        return []

    by_gloss: dict[str, list[dict]] = defaultdict(list)
    for entry in ordered:
        by_gloss[entry["glosses"][0]].append(entry)
    glosses = sorted(by_gloss)
    required_glosses = _validate_required_glosses(glosses, required_glosses)
    sequence_lengths = [
        int(rng.integers(min_sequence_len, max_sequence_len + 1))
        for _ in range(num_sequences)
    ]
    if len(required_glosses) > sum(sequence_lengths):
        raise ValueError(
            "Not enough sampled synthetic sequence capacity to guarantee required "
            f"gloss coverage for {len(required_glosses)} glosses across "
            f"{num_sequences} sequences"
        )

    forced_glosses_by_sequence: list[list[str]] = [[] for _ in range(num_sequences)]
    if required_glosses:
        shuffled_required = [
            required_glosses[idx]
            for idx in rng.permutation(len(required_glosses))
        ]
        sequence_order = list(rng.permutation(num_sequences))
        cursor = 0
        for gloss in shuffled_required:
            placed = False
            for _ in range(num_sequences):
                seq_idx = sequence_order[cursor % num_sequences]
                cursor += 1
                if len(forced_glosses_by_sequence[seq_idx]) < sequence_lengths[seq_idx]:
                    forced_glosses_by_sequence[seq_idx].append(gloss)
                    placed = True
                    break
            if not placed:
                raise RuntimeError(
                    f"Could not place required gloss {gloss} into split {split}"
                )

    manifests: list[dict] = []
    for idx, seq_len in enumerate(sequence_lengths):
        chosen_glosses = list(forced_glosses_by_sequence[idx])
        prev_gloss = chosen_glosses[-1] if chosen_glosses else None

        while len(chosen_glosses) < seq_len:
            if prev_gloss is not None and rng.random() < repeat_gloss_probability:
                gloss = prev_gloss
            else:
                options = [gloss for gloss in glosses if gloss != prev_gloss] or glosses
                gloss = str(rng.choice(options))
            chosen_glosses.append(gloss)
            prev_gloss = gloss

        parts: list[np.ndarray] = []
        labels: list[str] = []
        for step, gloss in enumerate(chosen_glosses):
            options = by_gloss[gloss]
            source = options[int(rng.integers(0, len(options)))]
            with np.load(source["features_path"]) as data:
                clip = data["X"].astype(np.float32)
            if clip.ndim != 2 or clip.shape[1] != FEATURE_DIM:
                raise ValueError(
                    f"Synthetic CSLR expects packed {FEATURE_DIM}-dim features, "
                    f"got shape={clip.shape} for {source['id']}"
                )
            clip = _sample_clip_variant(clip, rng, speed_jitter)
            if parts:
                transition_frames = int(
                    rng.integers(
                        transition_frames_min,
                        transition_frames_max + 1,
                    )
                )
                transition = _build_transition_frames(parts[-1], clip, transition_frames)
                if transition is not None:
                    parts.append(transition)
            parts.append(clip)
            labels.append(gloss)

        sequence = np.concatenate(parts, axis=0)
        motion = compute_motion_features(sequence)

        sample_id = f"{split}_{idx:04d}"
        npz_path = output_dir / f"{sample_id}.npz"
        np.savez_compressed(
            npz_path,
            X=sequence.astype(np.float32),
            X_vel=motion["velocity"].astype(np.float32),
            schema_version=np.array(2, dtype=np.int32),
            num_joints=np.array(NUM_JOINTS, dtype=np.int32),
            num_coords=np.array(NUM_COORDS, dtype=np.int32),
            coord_feature_dim=np.array(COORD_FEATURE_DIM, dtype=np.int32),
            frame_feature_dim=np.array(FEATURE_DIM, dtype=np.int32),
        )
        manifests.append(
            {
                "id": sample_id,
                "features_path": str(npz_path),
                "glosses": labels,
                "num_frames": int(sequence.shape[0]),
                "split": split,
                "dataset": "synthetic_wlasl_cslr",
            }
        )

    if required_glosses:
        covered = {gloss for entry in manifests for gloss in entry["glosses"]}
        missing = [gloss for gloss in required_glosses if gloss not in covered]
        if missing:
            raise RuntimeError(
                f"Synthetic split {split} failed required gloss coverage: {missing}"
            )

    return manifests
