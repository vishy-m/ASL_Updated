"""
PyTorch datasets for ISLR and CSLR tasks.

Loads preprocessed .npz skeleton files and .jsonl manifests to create
training-ready datasets with proper padding and collation (§6).
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .vocab import GlossVocab
from .augmentation import SkeletonAugmentor
from .skeleton import (
    FEATURE_DIM,
    COORD_FEATURE_DIM,
    LEGACY_XY_COORD_DIM,
    compute_motion_features,
    extract_coordinate_features,
)

logger = logging.getLogger(__name__)
SUPPORTED_FEATURE_WIDTHS = {FEATURE_DIM, COORD_FEATURE_DIM, LEGACY_XY_COORD_DIM}
CURRENT_SCHEMA_VERSION = 2
PATH_VISIBILITY_TIMEOUT_SEC = 60.0
PATH_VISIBILITY_POLL_SEC = 0.5


def _ctc_min_required_frames(label_ids: list[int]) -> int:
    """Minimum input frames needed for a CTC target sequence."""
    if not label_ids:
        return 0

    adjacent_repeats = sum(
        1
        for prev, cur in zip(label_ids[:-1], label_ids[1:])
        if prev == cur
    )
    return len(label_ids) + adjacent_repeats


def _wait_for_visible_paths(
    paths: list[str | Path],
    *,
    timeout_sec: float | None = None,
    poll_interval_sec: float | None = None,
) -> list[str]:
    """Wait briefly for feature files to appear on disk."""
    normalized = [str(Path(path)) for path in paths]
    timeout = PATH_VISIBILITY_TIMEOUT_SEC if timeout_sec is None else timeout_sec
    poll_interval = (
        PATH_VISIBILITY_POLL_SEC if poll_interval_sec is None else poll_interval_sec
    )
    deadline = time.monotonic() + max(timeout, 0.0)
    missing = [path for path in normalized if not Path(path).exists()]
    while missing and time.monotonic() < deadline:
        time.sleep(max(poll_interval, 0.01))
        missing = [path for path in normalized if not Path(path).exists()]
    return missing


def _load_npz_with_retry(
    features_path: str | Path,
    *,
    timeout_sec: float | None = None,
    poll_interval_sec: float | None = None,
):
    """Open an `.npz` file, retrying briefly on transient visibility races."""
    normalized = str(Path(features_path))
    timeout = PATH_VISIBILITY_TIMEOUT_SEC if timeout_sec is None else timeout_sec
    poll_interval = (
        PATH_VISIBILITY_POLL_SEC if poll_interval_sec is None else poll_interval_sec
    )
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        try:
            return np.load(normalized)
        except (FileNotFoundError, EOFError, OSError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(max(poll_interval, 0.01))


def _maybe_augment_sequence(
    sequence: np.ndarray,
    augmentor: SkeletonAugmentor | None,
    *,
    min_length: int | None = None,
    attempts: int = 3,
) -> tuple[np.ndarray, bool]:
    """Apply augmentation while respecting optional minimum-length constraints."""
    if augmentor is None:
        return sequence, False

    for _ in range(max(1, attempts)):
        augmented = augmentor(sequence)
        if min_length is None or augmented.shape[0] >= min_length:
            return augmented.astype(np.float32, copy=False), True

    logger.debug(
        "Falling back to unaugmented sequence because all augmented candidates "
        "were shorter than the required minimum length (%s)",
        min_length,
    )
    return sequence.astype(np.float32, copy=True), False


def _validate_feature_schema(
    data,
    sample_id: str,
    *,
    expected_frame_feature_dim: int | None = None,
    required_schema_version: int | None = None,
) -> int:
    """Validate the stored feature schema for one `.npz` sample."""
    if "X" not in data:
        raise ValueError(f"Missing X array in features for sample {sample_id}")

    X = data["X"]
    if X.ndim != 2:
        raise ValueError(
            f"Expected X to be 2D for sample {sample_id}, got shape={X.shape}"
        )
    if X.shape[1] not in SUPPORTED_FEATURE_WIDTHS:
        raise ValueError(
            f"Unsupported X feature width {X.shape[1]} for sample {sample_id}"
        )
    if expected_frame_feature_dim is not None and X.shape[1] != expected_frame_feature_dim:
        raise ValueError(
            f"Sample {sample_id} has X feature width {X.shape[1]}, "
            f"expected {expected_frame_feature_dim}"
        )

    if required_schema_version is not None:
        if "schema_version" not in data:
            raise ValueError(
                f"Sample {sample_id} is missing schema_version metadata; "
                f"expected schema >= {required_schema_version}"
            )
        stored_schema_version = int(np.asarray(data["schema_version"]).item())
        if stored_schema_version < required_schema_version:
            raise ValueError(
                f"Sample {sample_id} uses schema_version={stored_schema_version}, "
                f"expected schema >= {required_schema_version}"
            )

    if "frame_feature_dim" in data:
        stored_frame_dim = int(np.asarray(data["frame_feature_dim"]).item())
        if stored_frame_dim != X.shape[1]:
            raise ValueError(
                f"Schema metadata mismatch for {sample_id}: "
                f"frame_feature_dim={stored_frame_dim} but X.shape[1]={X.shape[1]}"
            )

    expected_coord_dim = extract_coordinate_features(X[:1]).shape[1]
    if "coord_feature_dim" in data:
        stored_coord_dim = int(np.asarray(data["coord_feature_dim"]).item())
        if stored_coord_dim != expected_coord_dim:
            raise ValueError(
                f"Schema metadata mismatch for {sample_id}: "
                f"coord_feature_dim={stored_coord_dim} but expected {expected_coord_dim}"
            )

    return expected_coord_dim


def _load_motion_features(data, sequence: np.ndarray, sample_id: str) -> np.ndarray:
    """Load motion features aligned to the current pose sequence."""
    expected_width = extract_coordinate_features(sequence[:1]).shape[1]
    if "X_vel" in data:
        velocity = data["X_vel"].astype(np.float32, copy=False)
        if velocity.ndim != 2 or velocity.shape[1] != expected_width:
            logger.warning(
                "Recomputing motion features for %s due to incompatible stored "
                "X_vel shape %s (expected width=%s)",
                sample_id,
                tuple(velocity.shape),
                expected_width,
            )
            velocity = compute_motion_features(sequence)["velocity"].astype(np.float32)
        if velocity.shape[0] >= sequence.shape[0]:
            velocity = velocity[: sequence.shape[0]]
        else:
            velocity = compute_motion_features(sequence)["velocity"].astype(np.float32)
    else:
        velocity = compute_motion_features(sequence)["velocity"].astype(np.float32)

    return velocity.astype(np.float32, copy=False)


def _build_motion_features(
    data,
    sequence: np.ndarray,
    sample_id: str,
    *,
    use_motion: bool,
    force_recompute: bool,
) -> np.ndarray:
    """Append motion features, recomputing them when augmentation changed time."""
    if not use_motion:
        return sequence.astype(np.float32, copy=False)

    if force_recompute:
        velocity = compute_motion_features(sequence)["velocity"].astype(np.float32)
    else:
        velocity = _load_motion_features(data, sequence, sample_id)

    return np.concatenate([sequence, velocity], axis=1).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str | Path) -> list[dict]:
    """Load a .jsonl manifest file.

    Each line is a JSON object with keys:
        id, features_path, glosses, num_frames, split, dataset
    """
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _validate_manifest_feature_paths(
    entries: list[dict],
    manifest_path: str | Path,
    *,
    preview_limit: int = 10,
):
    """Fail fast when a manifest references missing feature files."""
    feature_paths = [
        entry.get("features_path")
        for entry in entries
        if entry.get("features_path")
    ]
    missing_paths = set(
        _wait_for_visible_paths(
            feature_paths,
            timeout_sec=PATH_VISIBILITY_TIMEOUT_SEC,
            poll_interval_sec=PATH_VISIBILITY_POLL_SEC,
        )
    )
    missing = []
    for entry in entries:
        features_path = entry.get("features_path")
        normalized = str(features_path) if features_path else "<missing>"
        if not features_path or normalized in missing_paths:
            missing.append((entry.get("id", "<unk>"), normalized))
            if len(missing) >= preview_limit:
                break

    if missing:
        preview = ", ".join(f"{sample_id}:{path}" for sample_id, path in missing)
        raise FileNotFoundError(
            f"Manifest {manifest_path} references missing feature files. "
            f"Examples: {preview}"
        )


def _validate_manifest_glosses(
    entries: list[dict],
    vocab: GlossVocab,
    manifest_path: str | Path,
    *,
    preview_limit: int = 10,
):
    """Fail fast when a manifest contains empty or out-of-vocab gloss labels."""
    invalid: list[str] = []
    for entry in entries:
        glosses = entry.get("glosses") or []
        if not glosses:
            invalid.append(f"{entry.get('id', '<unk>')}:<missing-glosses>")
        else:
            unknown = [gloss for gloss in glosses if gloss not in vocab]
            if unknown:
                invalid.append(f"{entry.get('id', '<unk>')}:{unknown}")
        if len(invalid) >= preview_limit:
            break

    if invalid:
        raise ValueError(
            f"Manifest {manifest_path} contains empty or out-of-vocab gloss labels. "
            f"Examples: {', '.join(invalid)}"
        )


def _preflight_feature_schemas(
    entries: list[dict],
    manifest_path: str | Path,
    *,
    expected_frame_feature_dim: int | None = None,
    required_schema_version: int | None = None,
    preview_limit: int = 10,
):
    """Validate feature-schema contracts eagerly before training starts."""
    if expected_frame_feature_dim is None and required_schema_version is None:
        return

    failures: list[str] = []
    for entry in entries:
        try:
            with _load_npz_with_retry(entry["features_path"]) as data:
                _validate_feature_schema(
                    data,
                    entry.get("id", "<unk>"),
                    expected_frame_feature_dim=expected_frame_feature_dim,
                    required_schema_version=required_schema_version,
                )
        except Exception as exc:
            failures.append(f"{entry.get('id', '<unk>')}: {exc}")
            if len(failures) >= preview_limit:
                break

    if failures:
        raise ValueError(
            f"Manifest {manifest_path} failed feature-schema preflight. "
            f"Examples: {'; '.join(failures)}"
        )


# ---------------------------------------------------------------------------
# ISLR Dataset (§6.1) — Isolated Sign Language Recognition
# ---------------------------------------------------------------------------

class ISLRDataset(Dataset):
    """Dataset for isolated sign recognition (single sign per sample).

    Each sample is a skeleton sequence paired with a single gloss label ID.
    Used for training the ISLR backbone (Stage 1).

    Args:
        manifest_path: Path to .jsonl manifest.
        vocab: GlossVocab instance for encoding glosses.
        t_max: Optional maximum sequence length (truncate if exceeded).
        use_motion: Whether to concatenate velocity features.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        vocab: GlossVocab,
        t_max: int | None = None,
        use_motion: bool = False,
        augmentor: SkeletonAugmentor | None = None,
        expected_frame_feature_dim: int | None = None,
        required_schema_version: int | None = None,
    ):
        self.entries = load_manifest(manifest_path)
        _validate_manifest_feature_paths(self.entries, manifest_path)
        self.vocab = vocab
        self.t_max = t_max
        self.use_motion = use_motion
        self.augmentor = augmentor
        self.expected_frame_feature_dim = expected_frame_feature_dim
        self.required_schema_version = required_schema_version
        _validate_manifest_glosses(self.entries, vocab, manifest_path)
        _preflight_feature_schemas(
            self.entries,
            manifest_path,
            expected_frame_feature_dim=self.expected_frame_feature_dim,
            required_schema_version=self.required_schema_version,
        )

        logger.info(
            f"ISLRDataset loaded: {len(self.entries)} samples from {manifest_path}"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]

        with _load_npz_with_retry(entry["features_path"]) as data:
            _validate_feature_schema(
                data,
                entry["id"],
                expected_frame_feature_dim=self.expected_frame_feature_dim,
                required_schema_version=self.required_schema_version,
            )
            X = data["X"].astype(np.float32, copy=False)  # (T, 104)
            X, did_augment = _maybe_augment_sequence(X, self.augmentor)

            # Truncate if needed before adding motion features.
            if self.t_max and X.shape[0] > self.t_max:
                X = X[: self.t_max]
                did_augment = did_augment or self.augmentor is not None

            X = _build_motion_features(
                data,
                X,
                entry["id"],
                use_motion=self.use_motion,
                force_recompute=did_augment,
            )

        # Encode label (ISLR = single gloss)
        gloss = entry["glosses"][0] if entry["glosses"] else "<unk>"
        label_id = self.vocab.encode(gloss)

        return {
            "id": entry["id"],
            "features": torch.from_numpy(X).float(),        # (T, D)
            "label": torch.tensor(label_id, dtype=torch.long),
            "length": X.shape[0],
        }


# ---------------------------------------------------------------------------
# CSLR Dataset (§6.2) — Continuous Sign Language Recognition
# ---------------------------------------------------------------------------

class CSLRDataset(Dataset):
    """Dataset for continuous sign recognition (gloss sequence per sample).

    Each sample is a skeleton sequence paired with a sequence of gloss IDs
    for CTC training. Used for CSLR models (Stage 2+).

    Args:
        manifest_path: Path to .jsonl manifest.
        vocab: GlossVocab instance for encoding glosses.
        t_max: Maximum sequence length in frames.
        use_motion: Whether to concatenate velocity features.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        vocab: GlossVocab,
        t_max: int = 256,
        use_motion: bool = False,
        augmentor: SkeletonAugmentor | None = None,
        dual_stream: bool = False,
        frame_stride: int = 1,
        expected_frame_feature_dim: int | None = None,
        required_schema_version: int | None = None,
    ):
        self.entries = load_manifest(manifest_path)
        original_count = len(self.entries)
        if original_count == 0:
            raise ValueError(f"CSLR manifest is empty: {manifest_path}")
        _validate_manifest_feature_paths(self.entries, manifest_path)
        self.vocab = vocab
        self.t_max = t_max
        self.use_motion = use_motion
        self.augmentor = augmentor
        self.dual_stream = dual_stream
        self.frame_stride = max(1, int(frame_stride))
        self.expected_frame_feature_dim = expected_frame_feature_dim
        self.required_schema_version = required_schema_version
        _validate_manifest_glosses(self.entries, vocab, manifest_path)
        _preflight_feature_schemas(
            self.entries,
            manifest_path,
            expected_frame_feature_dim=self.expected_frame_feature_dim,
            required_schema_version=self.required_schema_version,
        )
        if self.dual_stream and not self.use_motion:
            raise ValueError("CSLR dual_stream requires use_motion=True")
        self.entries = self._filter_valid_entries(self.entries)
        if not self.entries:
            raise ValueError(
                "All CSLR samples were filtered out before training/evaluation: "
                f"{manifest_path}"
            )

        logger.info(
            f"CSLRDataset loaded: {len(self.entries)} samples from {manifest_path}"
        )

    def _effective_input_length(self, entry: dict) -> int | None:
        """Best-effort estimate of the number of frames available to CTC."""
        num_frames = entry.get("num_frames")
        if num_frames is not None:
            try:
                effective = int(num_frames)
                if self.frame_stride > 1:
                    effective = (effective + self.frame_stride - 1) // self.frame_stride
                return min(effective, self.t_max)
            except (TypeError, ValueError):
                pass

        features_path = entry.get("features_path")
        if not features_path:
            return None

        try:
            with _load_npz_with_retry(features_path) as data:
                effective = int(data["X"].shape[0])
                if self.frame_stride > 1:
                    effective = (effective + self.frame_stride - 1) // self.frame_stride
                return min(effective, self.t_max)
        except Exception:
            logger.warning(f"Could not inspect features for {entry.get('id', '<unk>')}")
            return None

    def _filter_valid_entries(self, entries: list[dict]) -> list[dict]:
        """Drop CSLR samples that cannot satisfy CTC length constraints."""
        valid_entries = []
        filtered = 0

        for entry in entries:
            label_ids = self.vocab.encode_sequence(entry.get("glosses", []))
            label_length = len(label_ids)
            min_required_frames = _ctc_min_required_frames(label_ids)
            input_length = self._effective_input_length(entry)

            if input_length is None:
                valid_entries.append(entry)
                continue

            if min_required_frames > input_length:
                filtered += 1
                logger.warning(
                    "Skipping CSLR sample %s: ctc_required_frames=%s > input_length=%s",
                    entry.get("id", "<unk>"),
                    min_required_frames,
                    input_length,
                )
                continue

            valid_entries.append(entry)

        if filtered:
            logger.info(f"Filtered {filtered} invalid CSLR samples before training")

        return valid_entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        label_ids = self.vocab.encode_sequence(entry["glosses"])
        min_required_frames = _ctc_min_required_frames(label_ids)

        with _load_npz_with_retry(entry["features_path"]) as data:
            _validate_feature_schema(
                data,
                entry["id"],
                expected_frame_feature_dim=self.expected_frame_feature_dim,
                required_schema_version=self.required_schema_version,
            )
            pose = data["X"].astype(np.float32, copy=False)  # (T, 104)
            pose, did_augment = _maybe_augment_sequence(
                pose,
                self.augmentor,
                min_length=min_required_frames,
            )

            if self.frame_stride > 1:
                pose = pose[:: self.frame_stride]

            # Truncate before adding motion features so X and X_vel stay aligned.
            if pose.shape[0] > self.t_max:
                pose = pose[: self.t_max]
                did_augment = did_augment or self.augmentor is not None

            motion = None
            if self.use_motion:
                if did_augment or self.frame_stride > 1:
                    motion = compute_motion_features(pose)["velocity"].astype(np.float32)
                else:
                    motion = _load_motion_features(data, pose, entry["id"])

        if min_required_frames > pose.shape[0]:
            raise ValueError(
                f"CSLR sample {entry['id']} is invalid after preprocessing: "
                f"ctc_required_frames={min_required_frames} > input_length={pose.shape[0]}"
            )

        sample = {
            "id": entry["id"],
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "input_length": pose.shape[0],
            "label_length": len(label_ids),
        }
        if self.dual_stream:
            sample["features_pose"] = torch.from_numpy(pose).float()
            sample["features_motion"] = torch.from_numpy(motion).float()
        else:
            features = (
                np.concatenate([pose, motion], axis=1).astype(np.float32, copy=False)
                if self.use_motion
                else pose.astype(np.float32, copy=False)
            )
            sample["features"] = torch.from_numpy(features).float()
        return sample


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def islr_collate_fn(batch: list[dict]) -> dict:
    """Collate for ISLR batches — pad sequences to max length in batch.

    Returns:
        ids: List of sample IDs.
        features: (B, T_max, D) padded features.
        labels: (B,) label IDs.
        lengths: (B,) original sequence lengths.
    """
    ids = [item["id"] for item in batch]
    features = pad_sequence(
        [item["features"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    labels = torch.stack([item["label"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

    return {
        "ids": ids,
        "features": features,
        "labels": labels,
        "lengths": lengths,
    }


def cslr_collate_fn(batch: list[dict]) -> dict:
    """Collate for CSLR batches — pad both input and label sequences.

    Returns:
        ids: List of sample IDs.
        features: (B, T_max, D) padded input features.
        labels: (total_label_length,) concatenated label sequences (for CTC).
        input_lengths: (B,) original input sequence lengths.
        label_lengths: (B,) original label sequence lengths.
    """
    ids = [item["id"] for item in batch]

    # CTC expects concatenated labels
    labels = torch.cat([item["labels"] for item in batch])

    input_lengths = torch.tensor(
        [item["input_length"] for item in batch], dtype=torch.long
    )
    label_lengths = torch.tensor(
        [item["label_length"] for item in batch], dtype=torch.long
    )

    collated = {
        "ids": ids,
        "labels": labels,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
    }
    if "features_pose" in batch[0]:
        collated["features_pose"] = pad_sequence(
            [item["features_pose"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
        collated["features_motion"] = pad_sequence(
            [item["features_motion"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
    else:
        collated["features"] = pad_sequence(
            [item["features"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
    return collated
