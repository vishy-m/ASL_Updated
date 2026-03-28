"""
Per-dataset preprocessing pipelines.

Converts raw dataset formats (videos, keypoints, annotations) into the
canonical 52-joint skeleton .npz files and .jsonl manifest entries.
Implements §5.1–5.4 of the project plan.
"""

import json
import logging
import pickle
import xml.etree.ElementTree as ET
from hashlib import sha1
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from .mediapipe_tasks import create_holistic_landmarker, create_mp_image
from .skeleton import (
    NUM_JOINTS,
    NUM_COORDS,
    COORD_FEATURE_DIM,
    FEATURE_DIM,
    MEDIAPIPE_POSE_INDICES,
    IDX_LEFT_SHOULDER,
    IDX_RIGHT_SHOULDER,
    IDX_MID_SHOULDERS,
    normalize_frame,
    build_feature_frame,
    compute_motion_features,
    fill_missing_joints,
    extract_skeleton_from_holistic_result,
)
from .label_maps import (
    clean_asl_citizen_gloss,
    clean_wlasl_gloss,
    clean_asllvd_gloss,
    clean_bu_gloss,
    clean_how2sign_gloss,
    extract_how2sign_pilot_labels,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _save_skeleton(
    output_path: Path,
    X: np.ndarray,
    compute_velocity: bool = True,
    compute_acceleration: bool = False,
    compress: bool = True,
):
    """Save skeleton sequence and optional motion features to .npz.

    Args:
        output_path: Path for the .npz file.
        X: Normalized skeleton sequence of shape (T, D).
        compute_velocity: Whether to compute and store velocity features.
        compute_acceleration: Whether to compute and store acceleration.
        compress: Whether to use compressed npz format.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "X": X.astype(np.float32),
        "schema_version": np.array(2, dtype=np.int32),
        "num_joints": np.array(NUM_JOINTS, dtype=np.int32),
        "num_coords": np.array(NUM_COORDS, dtype=np.int32),
        "coord_feature_dim": np.array(COORD_FEATURE_DIM, dtype=np.int32),
        "frame_feature_dim": np.array(FEATURE_DIM, dtype=np.int32),
    }

    if compute_velocity or compute_acceleration:
        motion = compute_motion_features(X, compute_acceleration)
        if compute_velocity:
            save_dict["X_vel"] = motion["velocity"].astype(np.float32)
        if compute_acceleration and "acceleration" in motion:
            save_dict["X_acc"] = motion["acceleration"].astype(np.float32)

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(str(output_path), **save_dict)


def _append_manifest(manifest_path: Path, entry: dict):
    """Append one JSONL entry to a manifest file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _write_manifest_entries(
    manifest_path: Path,
    entries: list[dict],
    *,
    append: bool = False,
):
    """Write a batch of JSONL entries in one pass."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(manifest_path, mode, encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")


def _stable_how2sign_sample_id(row: dict, split: str, index: int) -> str:
    """Create a deterministic How2Sign sample identifier.

    The raw annotation tables can contain duplicate sentence IDs or repeated
    sentence names. Hashing the source fields keeps the output name stable
    across reruns while avoiding collisions when the same sentence ID appears
    more than once.
    """
    sentence_id = row.get("SENTENCE_ID", "").strip() or f"row{index:06d}"
    sentence_name = row.get("SENTENCE_NAME", "").strip()
    video_id = row.get("VIDEO_ID", "").strip()
    start = row.get("START_REALIGNED", "").strip()
    end = row.get("END_REALIGNED", "").strip()
    sentence = row.get("SENTENCE", "").strip()
    key = "|".join([split, sentence_id, sentence_name, video_id, start, end, sentence])
    digest = sha1(key.encode("utf-8")).hexdigest()[:10]
    safe_sentence_id = clean_how2sign_gloss(sentence_id) or f"ROW{index:06d}"
    return f"{split}_{safe_sentence_id}_{digest}"


def _sorted_how2sign_rows(rows: list[dict]) -> list[dict]:
    """Sort How2Sign rows deterministically before preprocessing."""
    def _key(row: dict):
        return (
            row.get("VIDEO_ID", ""),
            row.get("SENTENCE_ID", ""),
            row.get("SENTENCE_NAME", ""),
            row.get("START_REALIGNED", ""),
            row.get("END_REALIGNED", ""),
            row.get("SENTENCE", ""),
        )

    return sorted(rows, key=_key)


def _single_score_passes_thresholds(
    score: float,
    *,
    visibility_threshold: float | None = None,
    presence_threshold: float | None = None,
) -> bool:
    """Evaluate a single quality score against visibility/presence thresholds.

    Some archived formats expose only one score (or a binary observed proxy)
    instead of separate visibility and presence values. In those cases we use
    the strongest configured threshold as the acceptance cutoff.
    """
    thresholds = [
        float(value)
        for value in (visibility_threshold, presence_threshold)
        if value is not None
    ]
    if not thresholds:
        return True
    return float(score) >= max(thresholds)


def _default_mediapipe_config() -> dict:
    """Return default MediaPipe Holistic thresholds for preprocessing."""
    return {
        "min_face_detection_confidence": 0.5,
        "min_face_suppression_threshold": 0.5,
        "min_face_landmarks_confidence": 0.5,
        "min_pose_detection_confidence": 0.5,
        "min_pose_suppression_threshold": 0.5,
        "min_pose_landmarks_confidence": 0.5,
        "min_hand_landmarks_confidence": 0.5,
        "pose_visibility_threshold": 0.5,
        "pose_presence_threshold": 0.5,
        "hand_visibility_threshold": None,
        "hand_presence_threshold": None,
    }


def _normalize_mediapipe_config(config: dict | None) -> dict:
    """Normalize MediaPipe config values with backward-compatible defaults."""
    merged = _default_mediapipe_config()
    if not config:
        return merged

    merged.update({
        "min_face_detection_confidence": config.get(
            "min_face_detection_confidence",
            config.get("min_detection_confidence", 0.5),
        ),
        "min_face_suppression_threshold": config.get(
            "min_face_suppression_threshold", 0.5
        ),
        "min_face_landmarks_confidence": config.get(
            "min_face_landmarks_confidence", 0.5
        ),
        "min_pose_detection_confidence": config.get(
            "min_pose_detection_confidence",
            config.get("min_detection_confidence", 0.5),
        ),
        "min_pose_suppression_threshold": config.get(
            "min_pose_suppression_threshold", 0.5
        ),
        "min_pose_landmarks_confidence": config.get(
            "min_pose_landmarks_confidence",
            config.get("min_tracking_confidence", 0.5),
        ),
        "min_hand_landmarks_confidence": config.get(
            "min_hand_landmarks_confidence",
            config.get("min_tracking_confidence", 0.5),
        ),
        "pose_visibility_threshold": config.get("pose_visibility_threshold", 0.5),
        "pose_presence_threshold": config.get("pose_presence_threshold", 0.5),
        "hand_visibility_threshold": config.get("hand_visibility_threshold"),
        "hand_presence_threshold": config.get("hand_presence_threshold"),
    })
    if config.get("task_model_path") is not None:
        merged["task_model_path"] = config["task_model_path"]
    return merged


def _init_holistic_landmarker(
    running_mode: mp.tasks.vision.RunningMode,
    mediapipe_config: dict | None = None,
):
    """Initialize a Holistic Landmarker for the requested running mode."""
    cfg = _normalize_mediapipe_config(mediapipe_config)
    return create_holistic_landmarker(
        running_mode=running_mode,
        model_path=cfg.get("task_model_path"),
        min_face_detection_confidence=cfg["min_face_detection_confidence"],
        min_face_suppression_threshold=cfg["min_face_suppression_threshold"],
        min_face_landmarks_confidence=cfg["min_face_landmarks_confidence"],
        min_pose_detection_confidence=cfg["min_pose_detection_confidence"],
        min_pose_suppression_threshold=cfg["min_pose_suppression_threshold"],
        min_pose_landmarks_confidence=cfg["min_pose_landmarks_confidence"],
        min_hand_landmarks_confidence=cfg["min_hand_landmarks_confidence"],
    )


def _probe_video_frame_size(video_path: str | Path) -> tuple[int, int]:
    """Return the frame size for a video clip without decoding the full stream."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return width, height


def _get_cached_video_landmarker(
    frame_size: tuple[int, int],
    cache: dict[tuple[int, int], object],
    mediapipe_config: dict | None = None,
):
    """Reuse VIDEO-mode landmarkers only across clips with matching dimensions."""
    landmarker = cache.get(frame_size)
    if landmarker is None:
        landmarker = _init_holistic_landmarker(
            mp.tasks.vision.RunningMode.VIDEO,
            mediapipe_config=mediapipe_config,
        )
        cache[frame_size] = landmarker
    return landmarker


def _close_landmarker_cache(cache: dict[tuple[int, int], object]):
    """Best-effort shutdown for cached landmarkers."""
    for landmarker in cache.values():
        try:
            landmarker.close()
        except Exception as exc:  # pragma: no cover - cleanup safety net
            logger.warning("Could not close cached landmarker cleanly: %s", exc)


def _video_to_skeletons(
    video_path: str | Path,
    downsample_factor: int = 2,
    start_time: float | None = None,
    end_time: float | None = None,
    mediapipe_config: dict | None = None,
    landmarker=None,
    running_mode: mp.tasks.vision.RunningMode | None = None,
    timestamp_offset_ms: int = 0,
) -> tuple[np.ndarray, int]:
    """Extract skeleton sequence from a video file using MediaPipe.

    Args:
        video_path: Path to video file.
        downsample_factor: Keep every n-th frame.
        start_time: Optional start time in seconds.
        end_time: Optional end time in seconds.

    Returns:
        Tuple of:
            - normalized skeleton sequence of shape (T, 104)
            - next monotonically increasing timestamp offset in milliseconds
    """
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be positive")

    if running_mode is None:
        running_mode = mp.tasks.vision.RunningMode.VIDEO

    mp_cfg = _normalize_mediapipe_config(mediapipe_config)
    owns_landmarker = landmarker is None
    if landmarker is None:
        landmarker = _init_holistic_landmarker(
            running_mode,
            mediapipe_config=mp_cfg,
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    start_frame = int(start_time * fps) if start_time else 0
    end_frame = int(end_time * fps) if end_time else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_raw = []
    prev_joints = None
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        clip_frame_idx = frame_idx - start_frame
        if clip_frame_idx % downsample_factor == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = create_mp_image(frame_rgb)
            if running_mode == mp.tasks.vision.RunningMode.IMAGE:
                result = landmarker.detect(mp_image)
            else:
                timestamp_ms = timestamp_offset_ms + int(
                    round((clip_frame_idx / max(fps, 1e-6)) * 1000.0)
                )
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
            joints, observed_mask = extract_skeleton_from_holistic_result(
                result,
                prev_joints=prev_joints,
                fill=True,
                pose_visibility_threshold=mp_cfg.get("pose_visibility_threshold"),
                pose_presence_threshold=mp_cfg.get("pose_presence_threshold"),
                hand_visibility_threshold=mp_cfg.get("hand_visibility_threshold"),
                hand_presence_threshold=mp_cfg.get("hand_presence_threshold"),
                return_observed_mask=True,
            )
            prev_joints = joints
            normalized = normalize_frame(joints, observed_mask=observed_mask)
            frames_raw.append(build_feature_frame(normalized, observed_mask))

        frame_idx += 1

    cap.release()
    if owns_landmarker:
        landmarker.close()

    if running_mode == mp.tasks.vision.RunningMode.IMAGE:
        next_timestamp_ms = timestamp_offset_ms
    else:
        next_timestamp_ms = timestamp_offset_ms + int(
            round((max(frame_idx - start_frame, 0) / max(fps, 1e-6)) * 1000.0)
        ) + 1

    if not frames_raw:
        logger.warning(f"No frames extracted from {video_path}")
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), next_timestamp_ms

    return np.array(frames_raw, dtype=np.float32), next_timestamp_ms



# ---------------------------------------------------------------------------
# How2Sign preprocessing (§5.1)
# ---------------------------------------------------------------------------

def preprocess_how2sign(
    keypoints_dir: str | Path,
    annotations_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    split: str = "train",
    downsample_factor: int = 2,
    compute_velocity: bool = True,
    pilot_glosses: set[str] | None = None,
    mediapipe_config: dict | None = None,
):
    """Preprocess How2Sign dataset from provided 2D keypoints.

    How2Sign provides pre-extracted MediaPipe Holistic keypoints as .npy
    files (shape: T×1662) and TSV annotation files with English translations.
    This function remaps keypoints to the canonical 52-joint layout,
    normalizes, and saves .npz files.

    Args:
        keypoints_dir: Directory containing How2Sign .npy keypoint files.
            Files are named like ``{SENTENCE_NAME}.npy``.
        annotations_path: Path to TSV annotation file with columns:
            VIDEO_ID, VIDEO_NAME, SENTENCE_ID, SENTENCE_NAME,
            START_REALIGNED, END_REALIGNED, SENTENCE.
        output_dir: Directory for output .npz files.
        manifest_path: Path for output .jsonl manifest.
        split: Data split name (train/val/test) for manifest entries.
        downsample_factor: Temporal downsampling factor.
        compute_velocity: Whether to compute velocity features.
        pilot_glosses: Optional set of allowed pseudo-gloss labels. When
            provided, How2Sign sentence text is deterministically tokenized
            and filtered to these labels for pilot training.
        mediapipe_config: Threshold config used when remapping MediaPipe-format
            How2Sign keypoints.
    """
    import csv

    keypoints_dir = Path(keypoints_dir)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)

    logger.info(f"Preprocessing How2Sign ({split}) from {keypoints_dir}")
    mp_cfg = _normalize_mediapipe_config(mediapipe_config)

    # Load TSV annotations
    rows = []
    with open(annotations_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    rows = _sorted_how2sign_rows(rows)
    logger.info(f"Loaded {len(rows)} annotation rows for split={split}")

    processed = 0
    skipped = 0
    seen_sample_ids: set[str] = set()
    manifest_entries: list[dict] = []

    for row_idx, row in enumerate(tqdm(rows, desc=f"How2Sign-{split}")):
        sentence_name = row["SENTENCE_NAME"]
        sentence_id = row["SENTENCE_ID"]
        sentence_text = row.get("SENTENCE", "")
        sample_id = _stable_how2sign_sample_id(row, split, row_idx)

        if sample_id in seen_sample_ids:
            skipped += 1
            continue
        seen_sample_ids.add(sample_id)

        # Look up corresponding .npy keypoint file
        kp_path = keypoints_dir / f"{sentence_name}.npy"
        if not kp_path.exists():
            skipped += 1
            if skipped <= 5:
                logger.warning(f"Keypoints not found: {kp_path}")
            continue

        raw_kps = np.load(str(kp_path))

        # Temporal downsampling
        if downsample_factor > 1:
            raw_kps = raw_kps[::downsample_factor]

        T = raw_kps.shape[0]
        if T == 0:
            skipped += 1
            continue

        # Remap to canonical skeleton layout, preserve observed/imputed mask,
        # then normalize coordinates for model features.
        sequence = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        prev_joints = None
        for t in range(T):
            joints, observed_mask = _remap_how2sign_keypoints(
                raw_kps[t],
                return_observed_mask=True,
                pose_visibility_threshold=mp_cfg.get("pose_visibility_threshold"),
                pose_presence_threshold=mp_cfg.get("pose_presence_threshold"),
                hand_visibility_threshold=mp_cfg.get("hand_visibility_threshold"),
                hand_presence_threshold=mp_cfg.get("hand_presence_threshold"),
            )
            joints = fill_missing_joints(joints, prev_joints)
            normalized = normalize_frame(joints, observed_mask=observed_mask)
            sequence[t] = build_feature_frame(normalized, observed_mask)
            prev_joints = joints

        # Build manifest entry. When pilot glosses are provided, derive a
        # deterministic pseudo-gloss sequence from the available sentence text.
        glosses = []
        if pilot_glosses:
            glosses = extract_how2sign_pilot_labels(sentence_text, pilot_glosses)
            if not glosses:
                skipped += 1
                continue
            if len(glosses) > T:
                skipped += 1
                continue

        # Save .npz only after the sample is known to be valid for the manifest.
        npz_path = output_dir / f"{sample_id}.npz"
        _save_skeleton(npz_path, sequence, compute_velocity)

        entry = {
            "id": sample_id,
            "features_path": str(npz_path),
            "source_sentence_id": sentence_id,
            "source_sentence_name": sentence_name,
            "sentence": sentence_text,
            "glosses": glosses,
            "num_frames": T,
            "split": split,
            "dataset": "how2sign",
        }
        manifest_entries.append(entry)
        processed += 1

    _write_manifest_entries(
        manifest_path,
        manifest_entries,
        append=manifest_path.exists(),
    )

    logger.info(
        f"How2Sign ({split}) preprocessing complete: "
        f"{processed} processed, {skipped} skipped → {manifest_path}"
    )


# ---------------------------------------------------------------------------
# ASL Citizen keypoint preprocessing
# ---------------------------------------------------------------------------

def _remap_asl_citizen_keypoints(
    raw_frame: np.ndarray,
    *,
    return_observed_mask: bool = False,
    pose_visibility_threshold: float | None = 0.5,
    pose_presence_threshold: float | None = None,
    hand_visibility_threshold: float | None = None,
    hand_presence_threshold: float | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Remap one ASL Citizen keypoint frame into the canonical 52-joint layout.

    The Kaggle ``ASL-Citizen-Keypoints`` mirror stores per-frame tensors with
    shape ``(75, 4)``. We interpret them as:
      - pose: 33 MediaPipe-style landmarks
      - left hand: 21 landmarks
      - right hand: 21 landmarks

    The last channel is treated as a single observation-quality proxy
    (visibility/presence). This matches the dataset's compact landmark format
    closely enough for our canonical xyz+mask pipeline.
    """
    frame = np.asarray(raw_frame, dtype=np.float32)
    if frame.shape != (75, 4):
        raise ValueError(
            "Expected ASL Citizen keypoint frame with shape (75, 4), "
            f"got {frame.shape}"
        )

    joints = np.full((NUM_JOINTS, NUM_COORDS), np.nan, dtype=np.float32)
    observed_mask = np.zeros(NUM_JOINTS, dtype=np.float32)

    pose = frame[:33]
    left_hand = frame[33:54]
    right_hand = frame[54:75]

    for canon_idx, mp_idx in enumerate(MEDIAPIPE_POSE_INDICES):
        x, y, z, score = pose[mp_idx]
        if (
            np.isfinite([x, y, z, score]).all()
            and not (x == 0.0 and y == 0.0 and z == 0.0)
            and _single_score_passes_thresholds(
                score,
                visibility_threshold=pose_visibility_threshold,
                presence_threshold=pose_presence_threshold,
            )
        ):
            joints[canon_idx] = [x, y, z]
            observed_mask[canon_idx] = 1.0

    if (
        observed_mask[IDX_LEFT_SHOULDER] > 0.0
        and observed_mask[IDX_RIGHT_SHOULDER] > 0.0
    ):
        joints[IDX_MID_SHOULDERS] = (
            joints[IDX_LEFT_SHOULDER] + joints[IDX_RIGHT_SHOULDER]
        ) / 2.0
        observed_mask[IDX_MID_SHOULDERS] = 1.0

    for hand_offset, hand_frame in ((10, left_hand), (31, right_hand)):
        for idx in range(21):
            x, y, z, score = hand_frame[idx]
            if (
                np.isfinite([x, y, z, score]).all()
                and not (x == 0.0 and y == 0.0 and z == 0.0)
                and _single_score_passes_thresholds(
                    score,
                    visibility_threshold=hand_visibility_threshold,
                    presence_threshold=hand_presence_threshold,
                )
            ):
                joints[hand_offset + idx] = [x, y, z]
                observed_mask[hand_offset + idx] = 1.0

    if return_observed_mask:
        return joints, observed_mask
    return joints


def _iter_asl_citizen_keypoint_files(keypoints_root: Path):
    """Yield `(split, gloss, path)` tuples from an ASL Citizen keypoint tree."""
    for pkl_path in sorted(keypoints_root.rglob("*.pkl")):
        rel_parts = pkl_path.relative_to(keypoints_root).parts
        if len(rel_parts) < 3:
            continue

        if rel_parts[0].startswith("keypoints-"):
            if len(rel_parts) < 4:
                continue
            split = rel_parts[1]
            gloss = rel_parts[2]
        else:
            split = rel_parts[0]
            gloss = rel_parts[1]

        yield split.lower(), gloss, pkl_path


def preprocess_asl_citizen_keypoints(
    keypoints_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    downsample_factor: int = 2,
    compute_velocity: bool = True,
    mediapipe_config: dict | None = None,
):
    """Preprocess Kaggle ASL Citizen keypoints into canonical skeleton `.npz`.

    The Kaggle mirror already contains landmark tensors, so this path converts
    them directly into the shared xyz+mask feature schema instead of rerunning
    MediaPipe.
    """
    keypoints_root = Path(keypoints_root)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)

    logger.info("Preprocessing ASL Citizen keypoints from %s", keypoints_root)
    mp_cfg = _normalize_mediapipe_config(mediapipe_config)
    manifest_entries: list[dict] = []
    processed = 0
    skipped = 0

    for split, gloss_orig, pkl_path in tqdm(
        list(_iter_asl_citizen_keypoint_files(keypoints_root)),
        desc="ASL-Citizen",
    ):
        try:
            with open(pkl_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                logger.warning("Could not load %s: %s", pkl_path, exc)
            continue

        keypoints = np.asarray(payload.get("keypoints"), dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (75, 4):
            skipped += 1
            if skipped <= 5:
                logger.warning(
                    "Unexpected ASL Citizen tensor shape for %s: %s",
                    pkl_path,
                    keypoints.shape,
                )
            continue

        if downsample_factor > 1:
            keypoints = keypoints[::downsample_factor]

        T = keypoints.shape[0]
        if T == 0:
            skipped += 1
            continue

        sequence = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        prev_joints = None
        for t in range(T):
            joints, observed_mask = _remap_asl_citizen_keypoints(
                keypoints[t],
                return_observed_mask=True,
                pose_visibility_threshold=mp_cfg.get("pose_visibility_threshold"),
                pose_presence_threshold=mp_cfg.get("pose_presence_threshold"),
                hand_visibility_threshold=mp_cfg.get("hand_visibility_threshold"),
                hand_presence_threshold=mp_cfg.get("hand_presence_threshold"),
            )
            joints = fill_missing_joints(joints, prev_joints)
            normalized = normalize_frame(joints, observed_mask=observed_mask)
            sequence[t] = build_feature_frame(normalized, observed_mask)
            prev_joints = joints

        canonical = clean_asl_citizen_gloss(
            str(payload.get("class") or gloss_orig or pkl_path.parent.name)
        )
        sample_id = f"asl_citizen_{split}_{pkl_path.stem}"
        npz_path = output_dir / split / canonical / f"{pkl_path.stem}.npz"
        _save_skeleton(npz_path, sequence, compute_velocity)

        manifest_entries.append(
            {
                "id": sample_id,
                "features_path": str(npz_path),
                "glosses": [canonical],
                "num_frames": T,
                "split": split,
                "dataset": "asl_citizen",
                "source_path": str(pkl_path),
            }
        )
        processed += 1

    _write_manifest_entries(manifest_path, manifest_entries, append=False)
    logger.info(
        "ASL Citizen preprocessing complete: %s processed, %s skipped → %s",
        processed,
        skipped,
        manifest_path,
    )


def preprocess_wlasl_holistic_keypoints(
    keypoints_root: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    downsample_factor: int = 1,
    compute_velocity: bool = True,
    mediapipe_config: dict | None = None,
):
    """Preprocess flattened WLASL holistic keypoints into canonical skeleton `.npz`.

    The Kaggle WLASL holistic mirror stores one `.npy` per WLASL `video_id`,
    with frames shaped `(T, 1662)`. That matches the MediaPipe-Holistic-like
    layout handled by `_remap_how2sign_keypoints`, so we can convert the full
    corpus without rerunning landmark extraction.
    """
    keypoints_root = Path(keypoints_root)
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    video_index: dict[str, dict] = {}
    for item in metadata:
        canonical = clean_wlasl_gloss(item.get("gloss", ""))
        if not canonical:
            continue
        for inst in item.get("instances", []):
            video_id = str(inst.get("video_id", "")).strip()
            if not video_id or video_id in video_index:
                continue
            video_index[video_id] = {
                "gloss": canonical,
                "split": str(inst.get("split", "train")).strip().lower() or "train",
            }

    logger.info("Preprocessing WLASL holistic keypoints from %s", keypoints_root)
    mp_cfg = _normalize_mediapipe_config(mediapipe_config)
    manifest_entries: list[dict] = []
    processed = 0
    skipped = 0

    for npy_path in tqdm(sorted(keypoints_root.glob("*.npy")), desc="WLASL-Keypoints"):
        video_id = npy_path.stem
        meta = video_index.get(video_id)
        if meta is None:
            skipped += 1
            continue

        try:
            raw_sequence = np.asarray(np.load(npy_path), dtype=np.float32)
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                logger.warning("Could not load %s: %s", npy_path, exc)
            continue

        if raw_sequence.ndim != 2 or raw_sequence.shape[1] != 1662:
            skipped += 1
            if skipped <= 5:
                logger.warning(
                    "Unexpected WLASL keypoint tensor shape for %s: %s",
                    npy_path,
                    raw_sequence.shape,
                )
            continue

        if downsample_factor > 1:
            raw_sequence = raw_sequence[::downsample_factor]

        T = raw_sequence.shape[0]
        if T == 0:
            skipped += 1
            continue

        sequence = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        prev_joints = None
        for t in range(T):
            joints, observed_mask = _remap_how2sign_keypoints(
                raw_sequence[t],
                return_observed_mask=True,
                pose_visibility_threshold=mp_cfg.get("pose_visibility_threshold"),
                pose_presence_threshold=mp_cfg.get("pose_presence_threshold"),
                hand_visibility_threshold=mp_cfg.get("hand_visibility_threshold"),
                hand_presence_threshold=mp_cfg.get("hand_presence_threshold"),
            )
            joints = fill_missing_joints(joints, prev_joints)
            normalized = normalize_frame(joints, observed_mask=observed_mask)
            sequence[t] = build_feature_frame(normalized, observed_mask)
            prev_joints = joints

        canonical = meta["gloss"]
        split = meta["split"]
        npz_path = output_dir / split / canonical / f"{video_id}.npz"
        _save_skeleton(npz_path, sequence, compute_velocity)
        manifest_entries.append(
            {
                "id": video_id,
                "features_path": str(npz_path),
                "glosses": [canonical],
                "num_frames": T,
                "split": split,
                "dataset": "wlasl_kaggle_keypoints",
                "source_path": str(npy_path),
            }
        )
        processed += 1

    _write_manifest_entries(manifest_path, manifest_entries, append=False)
    logger.info(
        "WLASL holistic keypoint preprocessing complete: %s processed, %s skipped → %s",
        processed,
        skipped,
        manifest_path,
    )


# ---------------------------------------------------------------------------
# OpenPose BODY_25 → Canonical 52-joint mapping
# ---------------------------------------------------------------------------
#
# OpenPose BODY_25 indices:
#   0  Nose            9  Left Hip        18 Left Big Toe
#   1  Neck           10  Right Hip       19 Left Small Toe
#   2  Right Shoulder 11  Left Knee       20 Left Heel
#   3  Right Elbow    12  Right Knee      21 Right Big Toe
#   4  Right Wrist    13  Left Ankle      22 Right Small Toe
#   5  Left Shoulder  14  Right Ankle     23 Right Heel
#   6  Left Elbow     15  Right Eye       24 Background
#   7  Left Wrist     16  Left Eye
#   8  Mid Hip        17  Right Ear
#
# Mapping to our canonical 10 pose joints:
#   Canon 0 (NOSE)            ← BODY_25[0]
#   Canon 1 (LEFT_SHOULDER)   ← BODY_25[5]
#   Canon 2 (RIGHT_SHOULDER)  ← BODY_25[2]
#   Canon 3 (LEFT_ELBOW)      ← BODY_25[6]
#   Canon 4 (RIGHT_ELBOW)     ← BODY_25[3]
#   Canon 5 (LEFT_WRIST)      ← BODY_25[7]
#   Canon 6 (RIGHT_WRIST)     ← BODY_25[4]
#   Canon 7 (LEFT_HIP)        ← BODY_25[9]
#   Canon 8 (RIGHT_HIP)       ← BODY_25[10]
#   Canon 9 (MID_SHOULDERS)   ← computed 0.5*(BODY_25[5]+BODY_25[2])
#
# Hand joints (21 each) pass through directly:
#   Canon 10-30 ← OpenPose left hand 0-20
#   Canon 31-51 ← OpenPose right hand 0-20

OPENPOSE_BODY25_TO_CANON = {
    0: 0,   # Nose
    5: 1,   # Left Shoulder
    2: 2,   # Right Shoulder
    6: 3,   # Left Elbow
    3: 4,   # Right Elbow
    7: 5,   # Left Wrist
    4: 6,   # Right Wrist
    9: 7,   # Left Hip
    10: 8,  # Right Hip
    # 9 (MID_SHOULDERS) is computed
}


def _remap_how2sign_keypoints(
    raw_frame: np.ndarray,
    *,
    return_observed_mask: bool = False,
    pose_visibility_threshold: float | None = 0.5,
    pose_presence_threshold: float | None = None,
    hand_visibility_threshold: float | None = None,
    hand_presence_threshold: float | None = None,
    confidence_threshold: float = 0.05,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Remap a single How2Sign frame to the canonical 52-joint layout.

    How2Sign provides OpenPose keypoints as either:
      (a) Flat array: [x0,y0,c0, x1,y1,c1, ...] with body25 (75 values) +
          left_hand (63 values) + right_hand (63 values) = 201 total, OR
      (b) Shaped array: (num_keypoints, 3) with (x, y, confidence).

    This function handles both formats and maps to our canonical (52, 3)
    coordinate layout while preserving an explicit observed/imputed mask.

    Args:
        raw_frame: Raw OpenPose keypoints for one frame. Can be:
            - shape (201,): flattened [body25_75, lhand_63, rhand_63]
            - shape (67, 3): stacked [body25(25,3), lhand(21,3), rhand(21,3)]
            - shape (N, 3): body+hand keypoints as (x,y,conf) rows
            - shape (N, 2): body+hand keypoints as (x,y) rows (no conf)
        pose_visibility_threshold: Visibility threshold for MediaPipe-format
            1662-dim How2Sign frames.
        pose_presence_threshold: Presence threshold proxy for MediaPipe-format
            1662-dim How2Sign frames, which only expose pose visibility.
        hand_visibility_threshold: Visibility threshold proxy for MediaPipe-format
            1662-dim How2Sign frames. Archived hand arrays do not expose
            landmark confidence, so non-zero coordinates act as a binary proxy.
        hand_presence_threshold: Presence threshold proxy for MediaPipe-format
            1662-dim How2Sign frames. Archived hand arrays do not expose
            landmark confidence, so non-zero coordinates act as a binary proxy.
        confidence_threshold: Confidence threshold for OpenPose-format frames.

    Returns:
        Array of shape (52, 3) in canonical joint order. NaN for missing joints.
    """
    joints = np.full((NUM_JOINTS, NUM_COORDS), np.nan, dtype=np.float32)
    observed_mask = np.zeros(NUM_JOINTS, dtype=np.float32)
    raw = raw_frame.copy()

    # --- Detect and reshape format ---
    if raw.ndim == 1:
        if raw.size == 1662:
            # How2Sign .npy format (MediaPipe Holistic)
            # 0:132 -> Pose (33 * 4) [x, y, z, vis]
            # 132:1536 -> Face (468 * 3) [x, y, z]
            # 1536:1599 -> Left Hand (21 * 3) [x, y, z]
            # 1599:1662 -> Right Hand (21 * 3) [x, y, z]
            pose = raw[:132].reshape(33, 4)
            lhand_raw = raw[1536:1599].reshape(21, 3)
            rhand_raw = raw[1599:1662].reshape(21, 3)

            # Map MediaPipe Pose -> canonical pose joints
            for canon_idx, mp_idx in enumerate(MEDIAPIPE_POSE_INDICES):
                x, y, z, vis = pose[mp_idx]
                if (
                    not (x == 0.0 and y == 0.0)
                    and _single_score_passes_thresholds(
                        vis,
                        visibility_threshold=pose_visibility_threshold,
                        presence_threshold=pose_presence_threshold,
                    )
                ):
                    joints[canon_idx] = [x, y, z]
                    observed_mask[canon_idx] = 1.0

            # Compute synthetic MID_SHOULDERS
            if (
                not np.isnan(joints[IDX_LEFT_SHOULDER]).any()
                and not np.isnan(joints[IDX_RIGHT_SHOULDER]).any()
            ):
                joints[IDX_MID_SHOULDERS] = (
                    joints[IDX_LEFT_SHOULDER] + joints[IDX_RIGHT_SHOULDER]
                ) / 2.0
                observed_mask[IDX_MID_SHOULDERS] = 1.0

            # Map Left Hand
            for i in range(21):
                x, y, z = lhand_raw[i]
                proxy_score = 1.0 if not (x == 0.0 and y == 0.0) else 0.0
                if proxy_score > 0.0 and _single_score_passes_thresholds(
                    proxy_score,
                    visibility_threshold=hand_visibility_threshold,
                    presence_threshold=hand_presence_threshold,
                ):
                    joints[10 + i] = [x, y, z]
                    observed_mask[10 + i] = 1.0

            # Map Right Hand
            for i in range(21):
                x, y, z = rhand_raw[i]
                proxy_score = 1.0 if not (x == 0.0 and y == 0.0) else 0.0
                if proxy_score > 0.0 and _single_score_passes_thresholds(
                    proxy_score,
                    visibility_threshold=hand_visibility_threshold,
                    presence_threshold=hand_presence_threshold,
                ):
                    joints[31 + i] = [x, y, z]
                    observed_mask[31 + i] = 1.0

            if return_observed_mask:
                return joints, observed_mask
            return joints

        # Flattened (x,y,c) triplets: body25=75, lhand=63, rhand=63
        if raw.size >= 201:
            body = raw[:75].reshape(25, 3)
            lhand = raw[75:138].reshape(21, 3)
            rhand = raw[138:201].reshape(21, 3)
        elif raw.size >= 134:
            # (x,y) pairs only: body25=50, lhand=42, rhand=42
            body = raw[:50].reshape(25, 2)
            body = np.hstack([body, np.ones((25, 1))])  # fake confidence
            lhand = raw[50:92].reshape(21, 2)
            lhand = np.hstack([lhand, np.ones((21, 1))])
            rhand = raw[92:134].reshape(21, 2)
            rhand = np.hstack([rhand, np.ones((21, 1))])
        else:
            logger.warning(f"Unexpected flat keypoint size: {raw.size}")
            if return_observed_mask:
                return joints, observed_mask
            return joints
    elif raw.ndim == 2:
        ncols = raw.shape[1]
        if ncols == 3:
            # (N, 3) with (x, y, confidence)
            if raw.shape[0] >= 67:
                body = raw[:25]
                lhand = raw[25:46]
                rhand = raw[46:67]
            elif raw.shape[0] >= 25:
                body = raw[:25]
                lhand = None
                rhand = None
            else:
                logger.warning(f"Unexpected keypoint rows: {raw.shape[0]}")
                if return_observed_mask:
                    return joints, observed_mask
                return joints
        elif ncols == 2:
            # (N, 2) — no confidence scores
            if raw.shape[0] >= 67:
                body = np.hstack([raw[:25], np.ones((25, 1))])
                lhand = np.hstack([raw[25:46], np.ones((21, 1))])
                rhand = np.hstack([raw[46:67], np.ones((21, 1))])
            elif raw.shape[0] >= 25:
                body = np.hstack([raw[:25], np.ones((25, 1))])
                lhand = None
                rhand = None
            else:
                logger.warning(f"Unexpected keypoint shape: {raw.shape}")
                if return_observed_mask:
                    return joints, observed_mask
                return joints
        else:
            logger.warning(f"Unexpected keypoint columns: {ncols}")
            if return_observed_mask:
                return joints, observed_mask
            return joints
    else:
        logger.warning(f"Unexpected keypoint ndim: {raw.ndim}")
        if return_observed_mask:
            return joints, observed_mask
        return joints

    # --- Map body BODY_25 → canonical pose joints ---
    confidence_threshold = float(confidence_threshold)

    for op_idx, canon_idx in OPENPOSE_BODY25_TO_CANON.items():
        x, y, conf = body[op_idx]
        if conf > confidence_threshold and not (x == 0.0 and y == 0.0):
            joints[canon_idx] = [x, y, 0.0]
            observed_mask[canon_idx] = 1.0

    # Compute synthetic MID_SHOULDERS
    if (
        not np.isnan(joints[IDX_LEFT_SHOULDER]).any()
        and not np.isnan(joints[IDX_RIGHT_SHOULDER]).any()
    ):
        joints[IDX_MID_SHOULDERS] = (
            joints[IDX_LEFT_SHOULDER] + joints[IDX_RIGHT_SHOULDER]
        ) / 2.0
        observed_mask[IDX_MID_SHOULDERS] = 1.0

    # --- Map hand joints (21 each, direct pass-through) ---
    if lhand is not None:
        for i in range(21):
            x, y, conf = lhand[i]
            if conf > confidence_threshold and not (x == 0.0 and y == 0.0):
                joints[10 + i] = [x, y, 0.0]
                observed_mask[10 + i] = 1.0

    if rhand is not None:
        for i in range(21):
            x, y, conf = rhand[i]
            if conf > confidence_threshold and not (x == 0.0 and y == 0.0):
                joints[31 + i] = [x, y, 0.0]
                observed_mask[31 + i] = 1.0

    if return_observed_mask:
        return joints, observed_mask
    return joints


# ---------------------------------------------------------------------------
# ASLLVD preprocessing (§5.2)
# ---------------------------------------------------------------------------

def preprocess_asllvd(
    video_dir: str | Path,
    token_table_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    label_map: dict[str, str] | None = None,
    downsample_factor: int = 2,
    compute_velocity: bool = True,
    view: str = "frontal",
    mediapipe_config: dict | None = None,
):
    """Preprocess ASLLVD dataset from videos + token table.

    For each token: read video segment, run MediaPipe, extract skeleton.

    Args:
        video_dir: Directory containing ASLLVD videos.
        token_table_path: Path to CSV/JSON token annotation table.
        output_dir: Directory for output .npz files.
        manifest_path: Path for output .jsonl manifest.
        label_map: Optional pre-built label map for gloss normalization.
        downsample_factor: Temporal downsampling factor.
        compute_velocity: Whether to compute velocity features.
        view: Which camera view to use (default: "frontal").
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)

    logger.info(f"Preprocessing ASLLVD from {video_dir}")

    # Load token table
    with open(token_table_path, "r") as f:
        tokens = json.load(f)

    for token in tqdm(tokens, desc="ASLLVD"):
        token_id = token["token_id"]
        video_file = token.get("video_file", "")
        video_path = video_dir / video_file

        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            continue

        # Extract skeleton from video segment
        sequence, _ = _video_to_skeletons(
            video_path,
            downsample_factor=downsample_factor,
            start_time=token.get("start_time"),
            end_time=token.get("end_time"),
            mediapipe_config=mediapipe_config,
        )

        T = sequence.shape[0]
        if T == 0:
            continue

        # Save skeleton
        npz_path = output_dir / f"{token_id}.npz"
        _save_skeleton(npz_path, sequence, compute_velocity)

        # Normalize gloss
        orig_gloss = token.get("gloss", "")
        if label_map:
            key = f"{token.get('lexical_entry_id', '')}_{token.get('variant_id', '')}"
            canonical = label_map.get(key, clean_asllvd_gloss(orig_gloss))
        else:
            canonical = clean_asllvd_gloss(orig_gloss)

        entry = {
            "id": token_id,
            "features_path": str(npz_path),
            "glosses": [canonical],
            "num_frames": T,
            "split": token.get("split", "train"),
            "dataset": "asllvd",
        }
        _append_manifest(manifest_path, entry)

    logger.info(f"ASLLVD preprocessing complete → {manifest_path}")


# ---------------------------------------------------------------------------
# NCSLGR / BU preprocessing (§5.3)
# ---------------------------------------------------------------------------

def preprocess_ncslgr(
    video_dir: str | Path,
    annotation_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    label_map: dict[str, str] | None = None,
    downsample_factor: int = 2,
    compute_velocity: bool = True,
    t_max: int | None = None,
    mediapipe_config: dict | None = None,
):
    """Preprocess NCSLGR / BU continuous corpus.

    Parses ELAN/SignStream annotations, extracts video segments,
    runs MediaPipe for skeleton extraction.

    Args:
        video_dir: Directory containing utterance videos.
        annotation_dir: Directory containing ELAN .eaf or SignStream files.
        output_dir: Directory for output .npz files.
        manifest_path: Path for output .jsonl manifest.
        label_map: Optional pre-built BU label map.
        downsample_factor: Temporal downsampling factor.
        compute_velocity: Whether to compute velocity features.
        t_max: Optional max frames per utterance (truncate if exceeded).
    """
    video_dir = Path(video_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)

    logger.info(f"Preprocessing NCSLGR from {video_dir}")

    # Find annotation files
    eaf_files = list(annotation_dir.glob("*.eaf"))
    if not eaf_files:
        logger.warning(f"No .eaf files found in {annotation_dir}")
        return

    for eaf_path in tqdm(eaf_files, desc="NCSLGR"):
        utterances = _parse_elan_annotations(eaf_path)

        for utt in utterances:
            utt_id = utt["id"]
            video_path = video_dir / utt.get("video_file", eaf_path.stem + ".mp4")

            if not video_path.exists():
                logger.warning(f"Video not found: {video_path}")
                continue

            # Extract skeleton sequence
            sequence, _ = _video_to_skeletons(
                video_path,
                downsample_factor=downsample_factor,
                start_time=utt.get("start_time"),
                end_time=utt.get("end_time"),
                mediapipe_config=mediapipe_config,
            )

            T = sequence.shape[0]
            if T == 0:
                continue

            # Truncate if needed
            if t_max and T > t_max:
                sequence = sequence[:t_max]
                T = t_max

            # Save
            npz_path = output_dir / f"{utt_id}.npz"
            _save_skeleton(npz_path, sequence, compute_velocity)

            # Normalize gloss sequence
            glosses = []
            for g in utt.get("glosses", []):
                if label_map and g in label_map:
                    glosses.append(label_map[g])
                else:
                    glosses.append(clean_bu_gloss(g))

            entry = {
                "id": utt_id,
                "features_path": str(npz_path),
                "glosses": glosses,
                "num_frames": T,
                "split": utt.get("split", "train"),
                "dataset": "ncslgr",
            }
            _append_manifest(manifest_path, entry)

    logger.info(f"NCSLGR preprocessing complete → {manifest_path}")


def _parse_elan_annotations(
    eaf_path: Path,
    gloss_tier_names: list[str] | None = None,
    utterance_tier_name: str | None = None,
) -> list[dict]:
    """Parse an ELAN .eaf file to extract utterances and gloss sequences.

    Uses xml.etree.ElementTree to parse the .eaf XML directly. Falls back
    to pympi-ling if installed and the direct approach encounters issues.

    The parser works in two modes:
      1. If an utterance tier is specified, groups glosses within utterance
         boundaries to produce multi-gloss utterance entries.
      2. Otherwise, treats each gloss annotation as an individual entry
         (useful when there's no explicit utterance segmentation).

    Args:
        eaf_path: Path to the .eaf annotation file.
        gloss_tier_names: Tier names to search for gloss annotations.
            Default: tries common names like 'Gloss', 'gloss', 'Sign',
            'RH-IDgloss', 'LH-IDgloss', 'Main Gloss', 'Dominant Gloss'.
        utterance_tier_name: Optional tier name for utterance boundaries.
            If provided, glosses are grouped within utterance segments.

    Returns:
        List of utterance dicts with keys:
            - id: unique identifier (file_stem + index)
            - start_time: start time in seconds
            - end_time: end time in seconds
            - glosses: list of gloss strings in temporal order
            - video_file: inferred video filename
    """
    if gloss_tier_names is None:
        gloss_tier_names = [
            "Gloss", "gloss", "GLOSS",
            "Sign", "sign", "SIGN",
            "RH-IDgloss", "LH-IDgloss",
            "Main Gloss", "Dominant Gloss",
            "RH-IDgloss (R)", "LH-IDgloss (L)",
            "Glosses", "glosses",
        ]

    try:
        return _parse_eaf_xml(eaf_path, gloss_tier_names, utterance_tier_name)
    except Exception as e:
        logger.warning(f"XML parsing failed for {eaf_path.name}: {e}")
        logger.info("Attempting fallback with pympi-ling...")
        try:
            return _parse_eaf_pympi(eaf_path, gloss_tier_names, utterance_tier_name)
        except ImportError:
            logger.error(
                "pympi-ling not installed. Install with: pip install pympi-ling"
            )
            return []
        except Exception as e2:
            logger.error(f"pympi fallback also failed: {e2}")
            return []


def _parse_eaf_xml(
    eaf_path: Path,
    gloss_tier_names: list[str],
    utterance_tier_name: str | None,
) -> list[dict]:
    """Parse .eaf using xml.etree directly."""
    tree = ET.parse(eaf_path)
    root = tree.getroot()
    file_stem = eaf_path.stem

    # --- 1. Build time slot map: TIME_SLOT_ID → time in seconds ---
    time_slots = {}
    for ts in root.findall(".//TIME_SLOT"):
        ts_id = ts.get("TIME_SLOT_ID")
        ts_val = ts.get("TIME_VALUE")
        if ts_id and ts_val:
            time_slots[ts_id] = int(ts_val) / 1000.0  # ms → seconds

    # --- 2. Find the gloss tier ---
    gloss_tier = None
    all_tier_ids = []

    for tier_elem in root.findall(".//TIER"):
        tier_id = tier_elem.get("TIER_ID", "")
        all_tier_ids.append(tier_id)
        if tier_id in gloss_tier_names:
            gloss_tier = tier_elem
            break

    if gloss_tier is None:
        # Try case-insensitive partial match
        for tier_elem in root.findall(".//TIER"):
            tier_id = tier_elem.get("TIER_ID", "")
            for target in gloss_tier_names:
                if target.lower() in tier_id.lower():
                    gloss_tier = tier_elem
                    break
            if gloss_tier is not None:
                break

    if gloss_tier is None:
        logger.warning(
            f"No gloss tier found in {eaf_path.name}. "
            f"Available tiers: {all_tier_ids}"
        )
        return []

    logger.debug(f"Using gloss tier: {gloss_tier.get('TIER_ID')}")

    # --- 3. Extract gloss annotations with timing ---
    gloss_annots = []
    for annot in gloss_tier.findall(".//ALIGNABLE_ANNOTATION"):
        ts_ref1 = annot.get("TIME_SLOT_REF1")
        ts_ref2 = annot.get("TIME_SLOT_REF2")
        value_elem = annot.find("ANNOTATION_VALUE")

        if ts_ref1 in time_slots and ts_ref2 in time_slots and value_elem is not None:
            text = (value_elem.text or "").strip()
            if text:  # skip empty annotations
                gloss_annots.append({
                    "text": text,
                    "start": time_slots[ts_ref1],
                    "end": time_slots[ts_ref2],
                })

    # Sort by start time
    gloss_annots.sort(key=lambda g: g["start"])

    if not gloss_annots:
        logger.warning(f"No gloss annotations found in {eaf_path.name}")
        return []

    # --- 4. Infer video filename from header ---
    video_file = file_stem + ".mp4"
    media_desc = root.find(".//MEDIA_DESCRIPTOR")
    if media_desc is not None:
        media_url = media_desc.get("MEDIA_URL", "") or media_desc.get("RELATIVE_MEDIA_URL", "")
        if media_url:
            video_file = Path(media_url).name

    # --- 5. Group into utterances ---
    utterances = []

    if utterance_tier_name:
        # Group glosses within utterance boundaries
        utt_tier = None
        for tier_elem in root.findall(".//TIER"):
            if tier_elem.get("TIER_ID") == utterance_tier_name:
                utt_tier = tier_elem
                break

        if utt_tier is not None:
            for idx, annot in enumerate(utt_tier.findall(".//ALIGNABLE_ANNOTATION")):
                ts_ref1 = annot.get("TIME_SLOT_REF1")
                ts_ref2 = annot.get("TIME_SLOT_REF2")
                if ts_ref1 not in time_slots or ts_ref2 not in time_slots:
                    continue

                utt_start = time_slots[ts_ref1]
                utt_end = time_slots[ts_ref2]

                # Collect glosses within this utterance's time window
                utt_glosses = [
                    g["text"]
                    for g in gloss_annots
                    if g["start"] >= utt_start - 0.01 and g["end"] <= utt_end + 0.01
                ]

                if utt_glosses:
                    utterances.append({
                        "id": f"{file_stem}_utt{idx:04d}",
                        "start_time": utt_start,
                        "end_time": utt_end,
                        "glosses": utt_glosses,
                        "video_file": video_file,
                    })
        else:
            logger.warning(
                f"Utterance tier '{utterance_tier_name}' not found, "
                f"falling back to automatic segmentation"
            )

    if not utterances:
        # No utterance tier — auto-segment using gaps between glosses
        # Group consecutive glosses with < 0.5s gap into utterances
        GAP_THRESHOLD = 0.5  # seconds
        current_group = [gloss_annots[0]]

        for i in range(1, len(gloss_annots)):
            gap = gloss_annots[i]["start"] - gloss_annots[i - 1]["end"]
            if gap > GAP_THRESHOLD:
                # Emit current group as one utterance
                utterances.append({
                    "id": f"{file_stem}_utt{len(utterances):04d}",
                    "start_time": current_group[0]["start"],
                    "end_time": current_group[-1]["end"],
                    "glosses": [g["text"] for g in current_group],
                    "video_file": video_file,
                })
                current_group = [gloss_annots[i]]
            else:
                current_group.append(gloss_annots[i])

        # Emit final group
        if current_group:
            utterances.append({
                "id": f"{file_stem}_utt{len(utterances):04d}",
                "start_time": current_group[0]["start"],
                "end_time": current_group[-1]["end"],
                "glosses": [g["text"] for g in current_group],
                "video_file": video_file,
            })

    logger.info(
        f"Parsed {eaf_path.name}: {len(gloss_annots)} glosses → "
        f"{len(utterances)} utterances"
    )
    return utterances


def _parse_eaf_pympi(
    eaf_path: Path,
    gloss_tier_names: list[str],
    utterance_tier_name: str | None,
) -> list[dict]:
    """Fallback: parse .eaf using pympi-ling library."""
    import pympi

    eaf = pympi.Elan.Eaf(str(eaf_path))
    file_stem = eaf_path.stem

    # Find gloss tier
    available_tiers = eaf.get_tier_names()
    gloss_tier_id = None

    for name in gloss_tier_names:
        if name in available_tiers:
            gloss_tier_id = name
            break

    if gloss_tier_id is None:
        # Case-insensitive partial match
        for tier in available_tiers:
            for target in gloss_tier_names:
                if target.lower() in tier.lower():
                    gloss_tier_id = tier
                    break
            if gloss_tier_id:
                break

    if gloss_tier_id is None:
        logger.warning(
            f"No gloss tier found in {eaf_path.name}. "
            f"Available: {list(available_tiers)}"
        )
        return []

    # Get annotations: list of (start_ms, end_ms, value)
    annotations = eaf.get_annotation_data_for_tier(gloss_tier_id)
    annotations.sort(key=lambda a: a[0])

    if not annotations:
        return []

    # Auto-segment by gaps
    GAP_THRESHOLD_MS = 500
    utterances = []
    current_group = [annotations[0]]

    for i in range(1, len(annotations)):
        gap = annotations[i][0] - annotations[i - 1][1]
        if gap > GAP_THRESHOLD_MS:
            start_s = current_group[0][0] / 1000.0
            end_s = current_group[-1][1] / 1000.0
            glosses = [a[2].strip() for a in current_group if a[2].strip()]
            if glosses:
                utterances.append({
                    "id": f"{file_stem}_utt{len(utterances):04d}",
                    "start_time": start_s,
                    "end_time": end_s,
                    "glosses": glosses,
                    "video_file": f"{file_stem}.mp4",
                })
            current_group = [annotations[i]]
        else:
            current_group.append(annotations[i])

    if current_group:
        start_s = current_group[0][0] / 1000.0
        end_s = current_group[-1][1] / 1000.0
        glosses = [a[2].strip() for a in current_group if a[2].strip()]
        if glosses:
            utterances.append({
                "id": f"{file_stem}_utt{len(utterances):04d}",
                "start_time": start_s,
                "end_time": end_s,
                "glosses": glosses,
                "video_file": f"{file_stem}.mp4",
            })

    logger.info(
        f"pympi parsed {eaf_path.name}: {len(annotations)} glosses → "
        f"{len(utterances)} utterances"
    )
    return utterances


# ---------------------------------------------------------------------------
# WLASL preprocessing (§5.4)
# ---------------------------------------------------------------------------

def preprocess_wlasl(
    clips_dir: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    label_map: dict[str, str] | None = None,
    downsample_factor: int = 2,
    compute_velocity: bool = True,
    mediapipe_config: dict | None = None,
):
    """Preprocess WLASL dataset from cropped sign clips.

    For each clip: read frames, run MediaPipe, extract + normalize skeleton.

    Args:
        clips_dir: Directory containing WLASL video clips.
        metadata_path: Path to WLASL JSON metadata file.
        output_dir: Directory for output .npz files.
        manifest_path: Path for output .jsonl manifest.
        label_map: Optional pre-built WLASL label map.
        downsample_factor: Temporal downsampling factor.
        compute_velocity: Whether to compute velocity features.
    """
    clips_dir = Path(clips_dir)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)

    logger.info(f"Preprocessing WLASL from {clips_dir}")

    # Load WLASL metadata or the curated clip mapping if provided.
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    uses_mapping = bool(metadata) and "instances" not in metadata[0]

    if uses_mapping:
        iterable = sorted(
            metadata,
            key=lambda item: (
                item.get("split", ""),
                item.get("gloss", ""),
                item.get("video_id", ""),
            ),
        )
    else:
        iterable = metadata

    for entry_data in tqdm(iterable, desc="WLASL"):
        if uses_mapping:
            gloss_orig = entry_data.get("gloss", "")
            video_id = entry_data.get("video_id", "")
            mapping_video_path = entry_data.get("video_path")
            video_path = Path(mapping_video_path) if mapping_video_path else (clips_dir / f"{video_id}.mp4")
            split = entry_data.get("split", "train")
            signer_id = entry_data.get("signer_id")
            start_frame = max(int(entry_data.get("frame_start", 0) or 0), 0)
            end_frame = int(entry_data.get("frame_end", -1) or -1)
            instances = [
                {
                    "video_id": video_id,
                    "video_path": video_path,
                    "split": split,
                    "signer_id": signer_id,
                    "fps": entry_data.get("fps"),
                    "frame_start": start_frame,
                    "frame_end": end_frame,
                }
            ]
        else:
            gloss_orig = entry_data.get("gloss", "")
            instances = entry_data.get("instances", [])

        for instance in instances:
            video_id = instance.get("video_id", "")
            instance_video_path = instance.get("video_path")
            video_path = Path(instance_video_path) if instance_video_path else (clips_dir / f"{video_id}.mp4")

            if not video_path.exists():
                logger.warning(f"Clip not found: {video_path}")
                continue

            fps = instance.get("fps")
            start_frame = max(int(instance.get("frame_start", 0) or 0), 0)
            end_frame = int(instance.get("frame_end", -1) or -1)
            start_time = None
            end_time = None
            if fps and fps > 0:
                start_time = start_frame / float(fps)
                if end_frame > 0:
                    end_time = end_frame / float(fps)

            # Extract only the sign span when clip boundaries are available.
            sequence, _ = _video_to_skeletons(
                video_path,
                downsample_factor=downsample_factor,
                start_time=start_time,
                end_time=end_time,
                mediapipe_config=mediapipe_config,
            )

            T = sequence.shape[0]
            if T == 0:
                continue

            # Save
            npz_path = output_dir / f"{video_id}.npz"
            _save_skeleton(npz_path, sequence, compute_velocity)

            # Normalize gloss
            if label_map and gloss_orig in label_map:
                canonical = label_map[gloss_orig]
            else:
                canonical = clean_wlasl_gloss(gloss_orig)

            split = instance.get("split", "train")

            manifest_entry = {
                "id": video_id,
                "features_path": str(npz_path),
                "glosses": [canonical],
                "num_frames": T,
                "split": split,
                "dataset": "wlasl",
                "signer_id": instance.get("signer_id"),
            }
            _append_manifest(manifest_path, manifest_entry)

    logger.info(f"WLASL preprocessing complete → {manifest_path}")
