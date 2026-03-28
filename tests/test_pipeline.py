"""
End-to-end smoke test using synthetic data.

Tests the full pipeline: synthetic skeleton generation → .npz save →
manifest creation → vocab build → dataset loading → model forward pass →
loss computation → backward pass.

Run with:
    python -m pytest tests/test_pipeline.py -v
"""

import json
import csv
import importlib.util
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from asl_cslr.data.skeleton import (
    NUM_JOINTS,
    NUM_COORDS,
    FEATURE_DIM,
    COORD_FEATURE_DIM,
    LEGACY_XY_COORD_DIM,
    normalize_frame,
    build_feature_frame,
    compute_motion_features,
    fill_missing_joints,
    extract_skeleton_from_holistic_result,
)
from asl_cslr.data.augmentation import SkeletonAugmentor
from asl_cslr.data.vocab import GlossVocab, build_vocab
from asl_cslr.data.dataset import (
    ISLRDataset,
    CSLRDataset,
    islr_collate_fn,
    cslr_collate_fn,
)
from asl_cslr.data import dataset as dataset_module
from asl_cslr.models.temporal_conv import TemporalConvEncoder, MultiScaleTemporalConv
from asl_cslr.models.bilstm import BiLSTMEncoder
from asl_cslr.models.transformer import TransformerSequenceEncoder
from asl_cslr.models.heads import ClassificationHead, CTCHead
from asl_cslr.models.islr_model import ISLRModel
from asl_cslr.models.cslr_model import (
    CSLRModel,
    DualStreamCSLRModel,
    suppress_ctc_special_tokens,
)
from asl_cslr.training.metrics import compute_accuracy, compute_wer, macro_averaged_accuracy
from asl_cslr.training.scheduler import build_scheduler
from asl_cslr.training.train_cslr import _build_balanced_sampler as build_cslr_balanced_sampler
from asl_cslr.training.train_cslr import _configure_ctc_blank_row
from asl_cslr.training.train_cslr import _freeze_ctc_blank_gradients
from asl_cslr.training.train_cslr import _initialize_ctc_head_biases
from asl_cslr.training.train_cslr import _resolve_loader_workers as resolve_cslr_loader_workers
from asl_cslr.training.train_islr import _resolve_loader_workers as resolve_islr_loader_workers
from asl_cslr.data.preprocessing import _remap_how2sign_keypoints
from asl_cslr.data.preprocessing import preprocess_how2sign
from asl_cslr.data.preprocessing import (
    _remap_asl_citizen_keypoints,
    preprocess_asl_citizen_keypoints,
    preprocess_wlasl_holistic_keypoints,
)
from asl_cslr.data.pilot import select_balanced_subset
from asl_cslr.data.demo import _build_transition_frames, synthesize_cslr_split
from asl_cslr.data.label_maps import (
    clean_asl_citizen_gloss,
    clean_wlasl_gloss,
    extract_how2sign_pilot_labels,
)
from asl_cslr.data.manifests import (
    build_cslr_split_entries,
    build_islr_split_entries,
    rebuild_training_manifests,
    select_shared_goal_glosses,
    stratify_islr_entries_by_gloss,
)

_shared_builder_spec = importlib.util.spec_from_file_location(
    "build_shared_isolated_cslr_dataset",
    Path(__file__).resolve().parents[1] / "scripts" / "build_shared_isolated_cslr_dataset.py",
)
_shared_builder_module = importlib.util.module_from_spec(_shared_builder_spec)
assert _shared_builder_spec is not None and _shared_builder_spec.loader is not None
_shared_builder_spec.loader.exec_module(_shared_builder_module)
_load_isolated_entries = _shared_builder_module._load_isolated_entries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_glosses():
    """A small synthetic glossary."""
    return ["HELLO", "THANK-YOU", "PLEASE", "YES", "NO", "SORRY", "HELP", "GOODBYE"]


@pytest.fixture
def vocab(synth_glosses):
    """Build a vocab from synthetic glosses."""
    return build_vocab(synth_glosses)


@pytest.fixture
def synth_data_dir(tmp_path, vocab, synth_glosses):
    """Create synthetic .npz files and .jsonl manifests in a temp dir."""
    npz_dir = tmp_path / "keypoints"
    npz_dir.mkdir()

    manifest_path = tmp_path / "manifest.jsonl"
    entries = []

    for i in range(20):
        # Random sequence length between 15 and 60 frames
        T = np.random.randint(15, 61)

        # Generate plausible skeleton: joints near center with some variation
        X = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        for t in range(T):
            joints = np.random.randn(NUM_JOINTS, NUM_COORDS).astype(np.float32) * 0.1
            # Place shoulders roughly symmetric
            joints[1] = [-0.15, 0.0, 0.0]   # left shoulder
            joints[2] = [0.15, 0.0, 0.0]    # right shoulder
            joints[9] = [0.0, 0.0, 0.0]     # mid shoulders
            X[t] = build_feature_frame(
                normalize_frame(joints),
                np.ones(NUM_JOINTS, dtype=np.float32),
            )

        # Save .npz
        npz_path = npz_dir / f"sample_{i:04d}.npz"
        motion = compute_motion_features(X)
        np.savez_compressed(
            str(npz_path),
            X=X,
            X_vel=motion["velocity"],
        )

        # Assign 1-3 glosses per sample
        num_glosses = np.random.randint(1, 4)
        glosses = [synth_glosses[j % len(synth_glosses)] for j in range(i, i + num_glosses)]

        entry = {
            "id": f"sample_{i:04d}",
            "features_path": str(npz_path),
            "glosses": glosses,
            "num_frames": T,
            "split": "train" if i < 16 else "val",
            "dataset": "synthetic",
        }
        entries.append(entry)

    # Write manifest
    with open(manifest_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    return {
        "manifest_path": str(manifest_path),
        "npz_dir": str(npz_dir),
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Skeleton tests
# ---------------------------------------------------------------------------

class TestSkeleton:
    """Tests for skeleton representation and normalization."""

    def test_normalize_frame_shape(self):
        """Normalized output should keep the canonical joint/coord shape."""
        joints = np.random.randn(NUM_JOINTS, NUM_COORDS).astype(np.float32) * 0.1
        joints[1] = [-0.15, 0.0, 0.0]
        joints[2] = [0.15, 0.0, 0.0]
        joints[9] = [0.0, 0.0, 0.0]
        norm = normalize_frame(joints)
        assert norm.shape == (NUM_JOINTS, NUM_COORDS)

    def test_fill_missing_joints(self):
        """Missing joints (NaN) should be filled with previous frame's values."""
        prev = np.random.randn(NUM_JOINTS, NUM_COORDS).astype(np.float32)
        current = np.full((NUM_JOINTS, NUM_COORDS), np.nan)
        current[0] = [0.5, 0.5, 0.1]  # Nose is present

        filled = fill_missing_joints(current, prev)
        assert not np.isnan(filled).any(), "No NaN should remain after filling"
        np.testing.assert_array_equal(filled[0], [0.5, 0.5, 0.1])
        np.testing.assert_array_equal(filled[1], prev[1])

    def test_first_frame_missing_joints_fill_to_reference(self):
        """First-frame missing joints should normalize to the body reference point."""
        current = np.full((NUM_JOINTS, NUM_COORDS), np.nan, dtype=np.float32)
        current[1] = [-0.2, 0.0, 0.0]  # Left shoulder
        current[2] = [0.2, 0.0, 0.0]   # Right shoulder

        filled = fill_missing_joints(current, prev_joints_xy=None)
        normalized = normalize_frame(filled)

        np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(normalized[9], [0.0, 0.0, 0.0], atol=1e-6)

    def test_first_frame_missing_shoulders_uses_fallback_scale(self):
        """Missing shoulders on the first frame should not explode normalization."""
        current = np.full((NUM_JOINTS, NUM_COORDS), np.nan, dtype=np.float32)
        current[0] = [0.5, 0.4, 0.0]
        current[10] = [0.6, 0.4, -0.1]

        filled = fill_missing_joints(current, prev_joints_xy=None)
        normalized = normalize_frame(filled)

        assert np.isfinite(normalized).all()
        assert np.max(np.abs(normalized)) < 10.0

    def test_normalize_frame_prefers_observed_joints_over_imputed_shoulders(self):
        """Imputed stale shoulders should not define normalization when a mask is present."""
        joints = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
        observed = np.zeros(NUM_JOINTS, dtype=np.float32)

        # Stale forward-filled shoulders from an old frame.
        joints[1] = [-10.0, 0.0, 0.0]
        joints[2] = [10.0, 0.0, 0.0]

        # Current observed joints clustered around the actual signer location.
        joints[0] = [0.50, 0.50, 0.00]
        joints[10] = [0.60, 0.50, 0.00]
        observed[0] = 1.0
        observed[10] = 1.0

        normalized = normalize_frame(joints, observed_mask=observed)

        np.testing.assert_allclose(normalized[0], [-1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(normalized[10], [1.0, 0.0, 0.0], atol=1e-5)

    def test_motion_features_shape(self):
        """Motion features should have same T as input."""
        T = 30
        X = np.random.randn(T, FEATURE_DIM).astype(np.float32)
        motion = compute_motion_features(X, compute_acceleration=True)
        assert motion["velocity"].shape == (T, COORD_FEATURE_DIM)
        assert motion["acceleration"].shape == (T, COORD_FEATURE_DIM)

    def test_motion_features_ignore_imputed_gap_transitions(self):
        """Velocity should stay zero across observed→missing and missing→observed gaps."""
        joints = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
        observed = np.ones(NUM_JOINTS, dtype=np.float32)
        frame0 = build_feature_frame(joints, observed)

        joints1 = joints.copy()
        joints1[10] = [0.25, 0.0, 0.0]
        observed1 = observed.copy()
        frame1 = build_feature_frame(joints1, observed1)

        joints2 = joints1.copy()
        observed2 = observed.copy()
        observed2[10] = 0.0
        frame2 = build_feature_frame(joints2, observed2)

        joints3 = joints.copy()
        joints3[10] = [0.75, 0.0, 0.0]
        observed3 = observed.copy()
        frame3 = build_feature_frame(joints3, observed3)

        motion = compute_motion_features(
            np.stack([frame0, frame1, frame2, frame3], axis=0)
        )["velocity"].reshape(4, NUM_JOINTS, NUM_COORDS)

        np.testing.assert_allclose(motion[2, 10], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(motion[3, 10], [0.0, 0.0, 0.0], atol=1e-6)

    def test_acceleration_ignores_imputed_gap_transitions(self):
        """Acceleration should stay zero around missing-joint gaps."""
        joints = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
        observed = np.ones(NUM_JOINTS, dtype=np.float32)
        frame0 = build_feature_frame(joints, observed)

        joints1 = joints.copy()
        joints1[10] = [0.5, 0.0, 0.0]
        frame1 = build_feature_frame(joints1, observed)

        hidden_mask = observed.copy()
        hidden_mask[10] = 0.0
        frame2 = build_feature_frame(joints1, hidden_mask)

        joints3 = joints.copy()
        joints3[10] = [1.0, 0.0, 0.0]
        frame3 = build_feature_frame(joints3, observed)

        acceleration = compute_motion_features(
            np.stack([frame0, frame1, frame2, frame3], axis=0),
            compute_acceleration=True,
        )["acceleration"].reshape(4, NUM_JOINTS, NUM_COORDS)

        np.testing.assert_allclose(acceleration[2, 10], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(acceleration[3, 10], [0.0, 0.0, 0.0], atol=1e-6)

    def test_extract_skeleton_from_holistic_result(self):
        """Holistic Tasks results should map into the canonical 52-joint layout."""
        class _Lm:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        pose = [_Lm(i / 100.0, i / 200.0, -i / 300.0) for i in range(25)]
        left = [_Lm(0.2 + i / 100.0, 0.3 + i / 100.0, -0.05) for i in range(21)]
        right = [_Lm(0.6 + i / 100.0, 0.7 + i / 100.0, 0.05) for i in range(21)]
        result = type(
            "FakeHolisticResult",
            (),
            {
                "pose_landmarks": pose,
                "left_hand_landmarks": left,
                "right_hand_landmarks": right,
            },
        )()

        joints = extract_skeleton_from_holistic_result(result)

        assert joints.shape == (NUM_JOINTS, NUM_COORDS)
        np.testing.assert_allclose(joints[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(joints[10], [0.2, 0.3, -0.05])
        np.testing.assert_allclose(joints[31], [0.6, 0.7, 0.05])

    def test_low_visibility_pose_landmarks_are_treated_as_missing(self):
        """Visibility-gated pose joints should remain missing before filling."""
        class _Lm:
            def __init__(self, x, y, z, visibility=1.0, presence=1.0):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility
                self.presence = presence

        pose = [_Lm(0.1, 0.2, 0.0) for _ in range(33)]
        pose[11] = _Lm(10.0, 20.0, 0.0, visibility=0.0)
        pose[12] = _Lm(-10.0, -20.0, 0.0, visibility=0.0)
        result = type(
            "FakeHolisticResult",
            (),
            {
                "pose_landmarks": pose,
                "left_hand_landmarks": None,
                "right_hand_landmarks": None,
            },
        )()

        joints, observed_mask = extract_skeleton_from_holistic_result(
            result,
            fill=False,
            return_observed_mask=True,
        )

        assert observed_mask[1] == 0.0
        assert observed_mask[2] == 0.0
        assert np.isnan(joints[1]).all()
        assert np.isnan(joints[2]).all()

    def test_low_presence_hand_landmarks_are_treated_as_missing(self):
        """Presence-gated hand joints should remain missing before filling."""
        class _Lm:
            def __init__(self, x, y, z, visibility=1.0, presence=1.0):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility
                self.presence = presence

        pose = [_Lm(0.1, 0.2, 0.0) for _ in range(33)]
        left = [_Lm(0.2, 0.3, -0.1) for _ in range(21)]
        left[0] = _Lm(10.0, 10.0, -0.2, presence=0.0)
        result = type(
            "FakeHolisticResult",
            (),
            {
                "pose_landmarks": pose,
                "left_hand_landmarks": left,
                "right_hand_landmarks": None,
            },
        )()

        joints, observed_mask = extract_skeleton_from_holistic_result(
            result,
            fill=False,
            hand_presence_threshold=0.5,
            return_observed_mask=True,
        )

        assert observed_mask[10] == 0.0
        assert np.isnan(joints[10]).all()


# ---------------------------------------------------------------------------
# Label cleaning tests
# ---------------------------------------------------------------------------

class TestLabelCleaning:
    """Tests for gloss normalization and How2Sign pilot tokenization."""

    def test_wlasl_synonym_cleanup(self):
        """Contraction variants should map to canonical WLASL labels."""
        assert clean_wlasl_gloss("can't") == "CANNOT"
        assert clean_wlasl_gloss("won't") == "WILL_NOT"
        assert clean_wlasl_gloss("what's") == "WHAT_IS"

    def test_how2sign_pilot_token_extraction(self):
        """How2Sign sentence text should tokenize deterministically."""
        sentence = "Can't you say hello, please?"
        allowed = {"CANNOT", "YOU", "SAY", "HELLO", "PLEASE"}
        labels = extract_how2sign_pilot_labels(sentence, allowed)
        assert labels == ["CANNOT", "YOU", "SAY", "HELLO", "PLEASE"]

    def test_asl_citizen_gloss_cleanup_matches_wlasl(self):
        """ASL Citizen normalization should align with the shared isolated vocab."""
        assert clean_asl_citizen_gloss("fine1") == "FINE"
        assert clean_asl_citizen_gloss("not mind") == "NOT_MIND"


# ---------------------------------------------------------------------------
# How2Sign preprocessing tests
# ---------------------------------------------------------------------------

class TestHow2SignPreprocessing:
    """Tests for deterministic How2Sign preprocessing."""

    def test_duplicate_sentence_ids_get_unique_outputs(self, tmp_path):
        """Duplicate sentence IDs should not collide in output naming."""
        keypoints_dir = tmp_path / "keypoints"
        keypoints_dir.mkdir()
        output_dir = tmp_path / "processed"
        manifest_path = tmp_path / "manifest.jsonl"
        annotations_path = tmp_path / "annotations.tsv"

        for stem in ("clip_a", "clip_b"):
            raw = np.random.randn(4, 1662).astype(np.float32)
            np.save(keypoints_dir / f"{stem}.npy", raw)

        rows = [
            {
                "VIDEO_ID": "video_a",
                "VIDEO_NAME": "video_a.mp4",
                "SENTENCE_ID": "dup_001",
                "SENTENCE_NAME": "clip_a",
                "START_REALIGNED": "0.0",
                "END_REALIGNED": "1.0",
                "SENTENCE": "hello please",
            },
            {
                "VIDEO_ID": "video_b",
                "VIDEO_NAME": "video_b.mp4",
                "SENTENCE_ID": "dup_001",
                "SENTENCE_NAME": "clip_b",
                "START_REALIGNED": "0.0",
                "END_REALIGNED": "1.0",
                "SENTENCE": "hello please",
            },
        ]

        with open(annotations_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "VIDEO_ID",
                    "VIDEO_NAME",
                    "SENTENCE_ID",
                    "SENTENCE_NAME",
                    "START_REALIGNED",
                    "END_REALIGNED",
                    "SENTENCE",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)

        preprocess_how2sign(
            keypoints_dir=keypoints_dir,
            annotations_path=annotations_path,
            output_dir=output_dir,
            manifest_path=manifest_path,
            split="train",
            downsample_factor=2,
            compute_velocity=False,
            pilot_glosses={"HELLO", "PLEASE"},
        )

        with open(manifest_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        assert len(entries) == 2
        assert len({entry["id"] for entry in entries}) == 2
        assert all(entry["glosses"] == ["HELLO", "PLEASE"] for entry in entries)
        for entry in entries:
            assert Path(entry["features_path"]).exists()
            assert entry["source_sentence_id"] == "dup_001"


class TestAslCitizenPreprocessing:
    """Tests for Kaggle ASL Citizen keypoint preprocessing."""

    def test_remap_asl_citizen_keypoints(self):
        """ASL Citizen 75x4 frames should map into the canonical 52-joint layout."""
        frame = np.zeros((75, 4), dtype=np.float32)
        frame[0] = [0.1, 0.2, -0.1, 1.0]      # nose
        frame[11] = [0.3, 0.4, -0.2, 1.0]     # left shoulder
        frame[12] = [0.5, 0.4, -0.2, 1.0]     # right shoulder
        frame[33] = [0.2, 0.8, -0.05, 1.0]    # left hand wrist
        frame[54] = [0.8, 0.8, 0.05, 1.0]     # right hand wrist

        joints, observed = _remap_asl_citizen_keypoints(
            frame,
            return_observed_mask=True,
        )

        assert joints.shape == (NUM_JOINTS, NUM_COORDS)
        np.testing.assert_allclose(joints[0], [0.1, 0.2, -0.1])
        np.testing.assert_allclose(joints[1], [0.3, 0.4, -0.2])
        np.testing.assert_allclose(joints[2], [0.5, 0.4, -0.2])
        np.testing.assert_allclose(joints[9], [0.4, 0.4, -0.2])
        np.testing.assert_allclose(joints[10], [0.2, 0.8, -0.05])
        np.testing.assert_allclose(joints[31], [0.8, 0.8, 0.05])
        assert observed[10] == 1.0
        assert observed[31] == 1.0

    def test_preprocess_asl_citizen_keypoints(self, tmp_path):
        """ASL Citizen keypoint trees should preprocess into canonical manifests."""
        root = tmp_path / "asl_citizen"
        sample_dir = root / "keypoints-100" / "train" / "FINE1"
        sample_dir.mkdir(parents=True)
        sample_path = sample_dir / "123-FINE1.pkl"
        sequence = np.zeros((6, 75, 4), dtype=np.float32)
        sequence[:, 0] = [0.1, 0.2, -0.1, 1.0]
        sequence[:, 11] = [0.3, 0.4, -0.2, 1.0]
        sequence[:, 12] = [0.5, 0.4, -0.2, 1.0]
        sequence[:, 33] = [0.2, 0.8, -0.05, 1.0]
        sequence[:, 54] = [0.8, 0.8, 0.05, 1.0]

        import pickle

        with open(sample_path, "wb") as handle:
            pickle.dump({"keypoints": sequence, "class": "FINE1"}, handle)

        output_dir = tmp_path / "processed"
        manifest_path = tmp_path / "manifest.jsonl"
        preprocess_asl_citizen_keypoints(
            keypoints_root=root,
            output_dir=output_dir,
            manifest_path=manifest_path,
            downsample_factor=2,
            compute_velocity=False,
        )

        entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["glosses"] == ["FINE"]
        assert entry["split"] == "train"
        assert entry["dataset"] == "asl_citizen"
        assert Path(entry["features_path"]).exists()

    def test_shared_builder_deduplicates_wlasl_variants_by_clip_id(self, tmp_path):
        """Equivalent WLASL clip IDs from multiple manifests should collapse to one source sample."""
        manifest_a = tmp_path / "wlasl.jsonl"
        manifest_b = tmp_path / "wlasl_kaggle.jsonl"
        manifest_c = tmp_path / "asl_citizen.jsonl"

        records = {
            manifest_a: [{
                "id": "12345",
                "features_path": str(tmp_path / "a.npz"),
                "glosses": ["DOG"],
                "num_frames": 12,
                "split": "train",
                "dataset": "wlasl",
            }],
            manifest_b: [{
                "id": "12345",
                "features_path": str(tmp_path / "b.npz"),
                "glosses": ["DOG"],
                "num_frames": 12,
                "split": "train",
                "dataset": "wlasl_kaggle",
            }],
            manifest_c: [{
                "id": "12345",
                "features_path": str(tmp_path / "c.npz"),
                "glosses": ["DOG"],
                "num_frames": 12,
                "split": "train",
                "dataset": "asl_citizen",
            }],
        }

        for path, entries in records.items():
            with open(path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

        loaded = _load_isolated_entries([manifest_a, manifest_b, manifest_c])

        assert len(loaded) == 2
        assert {entry["dataset"] for entry in loaded} == {"wlasl", "asl_citizen"}

    def test_preprocess_wlasl_holistic_keypoints(self, tmp_path):
        """Flattened WLASL holistic `.npy` files should preprocess into canonical manifests."""
        keypoints_root = tmp_path / "wlasl_keypoints"
        keypoints_root.mkdir()
        metadata_path = tmp_path / "WLASL_v0.3.json"
        sample_path = keypoints_root / "12345.npy"

        frame = np.zeros(1662, dtype=np.float32)
        frame[0:4] = [0.1, 0.2, -0.1, 1.0]          # pose nose
        frame[44:48] = [0.3, 0.4, -0.2, 1.0]        # pose left shoulder
        frame[48:52] = [0.5, 0.4, -0.2, 1.0]        # pose right shoulder
        frame[1536:1539] = [0.2, 0.8, -0.05]        # left wrist
        frame[1599:1602] = [0.8, 0.8, 0.05]         # right wrist
        sequence = np.stack([frame, frame], axis=0)
        np.save(sample_path, sequence)

        metadata = [
            {
                "gloss": "book",
                "instances": [
                    {
                        "video_id": "12345",
                        "split": "train",
                    }
                ],
            }
        ]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        output_dir = tmp_path / "processed"
        manifest_path = tmp_path / "manifest.jsonl"
        preprocess_wlasl_holistic_keypoints(
            keypoints_root=keypoints_root,
            metadata_path=metadata_path,
            output_dir=output_dir,
            manifest_path=manifest_path,
            downsample_factor=1,
            compute_velocity=False,
        )

        entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["glosses"] == ["BOOK"]
        assert entry["split"] == "train"
        assert entry["dataset"] == "wlasl_kaggle_keypoints"
        assert Path(entry["features_path"]).exists()


# ---------------------------------------------------------------------------
# Dataset guard tests
# ---------------------------------------------------------------------------

class TestDatasetGuards:
    """Tests for invalid CSLR sample filtering."""

    def test_cslr_dataset_filters_label_longer_than_input(self, tmp_path):
        """Samples with too many labels should be filtered before training."""
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        manifest_path = tmp_path / "manifest.jsonl"

        valid_path = npz_dir / "valid.npz"
        invalid_path = npz_dir / "invalid.npz"
        X_valid = np.random.randn(4, FEATURE_DIM).astype(np.float32)
        X_invalid = np.random.randn(2, FEATURE_DIM).astype(np.float32)
        np.savez_compressed(valid_path, X=X_valid)
        np.savez_compressed(invalid_path, X=X_invalid)

        entries = [
            {
                "id": "valid",
                "features_path": str(valid_path),
                "glosses": ["HELLO", "PLEASE"],
                "num_frames": 4,
                "split": "train",
                "dataset": "synthetic",
            },
            {
                "id": "invalid",
                "features_path": str(invalid_path),
                "glosses": ["HELLO", "PLEASE", "YES"],
                "num_frames": 2,
                "split": "train",
                "dataset": "synthetic",
            },
        ]

        with open(manifest_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        vocab = build_vocab(["HELLO", "PLEASE", "YES"])
        dataset = CSLRDataset(manifest_path, vocab, t_max=4)

        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["id"] == "valid"

    def test_cslr_dataset_filters_repeated_labels_without_ctc_room(self, tmp_path):
        """Adjacent repeated glosses need an extra frame for CTC blank separation."""
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        manifest_path = tmp_path / "manifest.jsonl"

        valid_path = npz_dir / "valid_repeat.npz"
        invalid_path = npz_dir / "invalid_repeat.npz"
        np.savez_compressed(valid_path, X=np.random.randn(3, FEATURE_DIM).astype(np.float32))
        np.savez_compressed(invalid_path, X=np.random.randn(2, FEATURE_DIM).astype(np.float32))

        entries = [
            {
                "id": "valid_repeat",
                "features_path": str(valid_path),
                "glosses": ["HELLO", "HELLO"],
                "num_frames": 3,
                "split": "train",
                "dataset": "synthetic",
            },
            {
                "id": "invalid_repeat",
                "features_path": str(invalid_path),
                "glosses": ["HELLO", "HELLO"],
                "num_frames": 2,
                "split": "train",
                "dataset": "synthetic",
            },
        ]

        with open(manifest_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        vocab = build_vocab(["HELLO"])
        dataset = CSLRDataset(manifest_path, vocab, t_max=4)

        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["id"] == "valid_repeat"

    def test_cslr_dataset_fails_fast_on_missing_feature_path(self, tmp_path):
        manifest_path = tmp_path / "manifest.jsonl"
        entries = [
            {
                "id": "missing",
                "features_path": str(tmp_path / "missing.npz"),
                "glosses": ["HELLO"],
                "num_frames": 4,
                "split": "train",
                "dataset": "synthetic",
            }
        ]

        with open(manifest_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        vocab = build_vocab(["HELLO"])
        with pytest.raises(FileNotFoundError, match="missing feature files"):
            CSLRDataset(manifest_path, vocab, t_max=4)

    def test_cslr_dataset_waits_for_delayed_feature_visibility(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "manifest.jsonl"
        delayed_path = tmp_path / "delayed.npz"
        entries = [
            {
                "id": "delayed",
                "features_path": str(delayed_path),
                "glosses": ["HELLO"],
                "num_frames": 4,
                "split": "train",
                "dataset": "synthetic",
            }
        ]

        with open(manifest_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        monkeypatch.setattr(dataset_module, "PATH_VISIBILITY_TIMEOUT_SEC", 0.5)
        monkeypatch.setattr(dataset_module, "PATH_VISIBILITY_POLL_SEC", 0.02)

        def _materialize():
            time.sleep(0.05)
            np.savez_compressed(
                delayed_path,
                X=np.random.randn(4, FEATURE_DIM).astype(np.float32),
            )

        writer = threading.Thread(target=_materialize, daemon=True)
        writer.start()

        vocab = build_vocab(["HELLO"])
        dataset = CSLRDataset(manifest_path, vocab, t_max=4)

        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["id"] == "delayed"

    def test_cslr_dataset_applies_frame_stride_before_motion_features(self, tmp_path):
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        manifest_path = tmp_path / "manifest.jsonl"
        sample_path = npz_dir / "sample.npz"

        X = np.random.randn(6, FEATURE_DIM).astype(np.float32)
        X_vel = np.random.randn(6, COORD_FEATURE_DIM).astype(np.float32)
        np.savez_compressed(sample_path, X=X, X_vel=X_vel, schema_version=2)

        entry = {
            "id": "stride_sample",
            "features_path": str(sample_path),
            "glosses": ["HELLO", "PLEASE"],
            "num_frames": 6,
            "split": "train",
            "dataset": "synthetic",
        }
        with open(manifest_path, "w") as f:
            f.write(json.dumps(entry) + "\n")

        vocab = build_vocab(["HELLO", "PLEASE"])
        dataset = CSLRDataset(
            manifest_path,
            vocab,
            t_max=8,
            use_motion=True,
            frame_stride=2,
        )

        sample = dataset[0]
        assert sample["input_length"] == 3
        assert sample["features"].shape == (3, FEATURE_DIM + COORD_FEATURE_DIM)


class TestPilotManifestSelection:
    """Tests for deterministic pilot split capping."""

    def test_balanced_subset_preserves_label_coverage(self):
        entries = [
            {"id": "001", "split": "train", "glosses": ["LIKE"]},
            {"id": "002", "split": "train", "glosses": ["LIKE"]},
            {"id": "003", "split": "train", "glosses": ["LIKE"]},
            {"id": "004", "split": "train", "glosses": ["NOW"]},
            {"id": "005", "split": "train", "glosses": ["NOW"]},
            {"id": "006", "split": "train", "glosses": ["HOT"]},
        ]

        selected = select_balanced_subset(entries, max_samples=3)
        selected_labels = {entry["glosses"][0] for entry in selected}

        assert len(selected) == 3
        assert selected_labels == {"LIKE", "NOW", "HOT"}


class TestDemoDatasetBuilder:
    def test_synthesize_cslr_split_can_repeat_adjacent_glosses(self, tmp_path):
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        output_dir = tmp_path / "synthetic"

        entries = []
        for idx, gloss in enumerate(["HELLO", "PLEASE"]):
            path = npz_dir / f"{gloss.lower()}.npz"
            X = np.random.randn(12 + idx, FEATURE_DIM).astype(np.float32)
            X_vel = np.zeros((X.shape[0], COORD_FEATURE_DIM), dtype=np.float32)
            np.savez_compressed(path, X=X, X_vel=X_vel)
            entries.append(
                {
                    "id": gloss.lower(),
                    "features_path": str(path),
                    "glosses": [gloss],
                    "num_frames": int(X.shape[0]),
                    "split": "train",
                    "dataset": "synthetic",
                }
            )

        manifests = synthesize_cslr_split(
            source_entries=entries,
            output_dir=output_dir,
            split="train",
            num_sequences=8,
            seed=7,
            min_sequence_len=2,
            max_sequence_len=3,
            repeat_gloss_probability=1.0,
            transition_frames_min=0,
            transition_frames_max=0,
        )

        assert manifests
        assert any(
            any(prev == cur for prev, cur in zip(item["glosses"][:-1], item["glosses"][1:]))
            for item in manifests
        )

    def test_synthesize_cslr_split_guarantees_required_gloss_coverage(self, tmp_path):
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        output_dir = tmp_path / "synthetic"

        entries = []
        for idx, gloss in enumerate(["BOOK", "LIKE", "DRINK"]):
            path = npz_dir / f"{gloss.lower()}.npz"
            X = np.random.randn(10 + idx, FEATURE_DIM).astype(np.float32)
            X_vel = np.zeros((X.shape[0], COORD_FEATURE_DIM), dtype=np.float32)
            np.savez_compressed(path, X=X, X_vel=X_vel)
            entries.append(
                {
                    "id": gloss.lower(),
                    "features_path": str(path),
                    "glosses": [gloss],
                    "num_frames": int(X.shape[0]),
                    "split": "train",
                    "dataset": "synthetic",
                }
            )

        manifests = synthesize_cslr_split(
            source_entries=entries,
            output_dir=output_dir,
            split="train",
            num_sequences=2,
            seed=13,
            min_sequence_len=2,
            max_sequence_len=2,
            transition_frames_min=0,
            transition_frames_max=0,
            required_glosses=["BOOK", "LIKE", "DRINK"],
        )

        covered = {gloss for entry in manifests for gloss in entry["glosses"]}
        assert covered == {"BOOK", "LIKE", "DRINK"}

    def test_synthesize_cslr_split_preserves_binary_observation_masks(self, tmp_path):
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        output_dir = tmp_path / "synthetic"

        entries = []
        for idx, gloss in enumerate(["BOOK", "LIKE"]):
            path = npz_dir / f"{gloss.lower()}.npz"
            X = np.zeros((8 + idx, FEATURE_DIM), dtype=np.float32)
            for t in range(X.shape[0]):
                joints = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
                joints[0, 0] = 0.1 * t
                observed = np.ones(NUM_JOINTS, dtype=np.float32)
                observed[10 + idx] = 0.0 if t % 2 else 1.0
                X[t] = build_feature_frame(joints, observed)
            X_vel = np.zeros((X.shape[0], COORD_FEATURE_DIM), dtype=np.float32)
            np.savez_compressed(path, X=X, X_vel=X_vel)
            entries.append(
                {
                    "id": gloss.lower(),
                    "features_path": str(path),
                    "glosses": [gloss],
                    "num_frames": int(X.shape[0]),
                    "split": "train",
                    "dataset": "synthetic",
                }
            )

        manifests = synthesize_cslr_split(
            source_entries=entries,
            output_dir=output_dir,
            split="train",
            num_sequences=2,
            seed=17,
            min_sequence_len=2,
            max_sequence_len=2,
            transition_frames_min=2,
            transition_frames_max=2,
            speed_jitter=0.2,
        )

        assert manifests
        with np.load(manifests[0]["features_path"]) as data:
            masks = data["X"].reshape(data["X"].shape[0], NUM_JOINTS, 4)[..., 3]
        assert set(np.unique(masks).tolist()).issubset({0.0, 1.0})

    def test_transition_frames_keep_joint_masked_when_missing_in_either_clip(self):
        left_joints = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
        right_joints = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
        left_mask = np.ones(NUM_JOINTS, dtype=np.float32)
        right_mask = np.ones(NUM_JOINTS, dtype=np.float32)
        left_mask[10] = 0.0

        left_clip = np.stack([build_feature_frame(left_joints, left_mask)], axis=0)
        right_clip = np.stack([build_feature_frame(right_joints, right_mask)], axis=0)

        transition = _build_transition_frames(left_clip, right_clip, frames=2)

        assert transition is not None
        transition_masks = transition.reshape(transition.shape[0], NUM_JOINTS, 4)[..., 3]
        assert np.all(transition_masks[:, 10] == 0.0)
        assert np.all(transition_masks[:, 11] == 1.0)

    def test_synthesize_cslr_split_raises_when_required_coverage_is_impossible(self, tmp_path):
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        output_dir = tmp_path / "synthetic"

        entries = []
        for gloss in ["BOOK", "LIKE", "DRINK"]:
            path = npz_dir / f"{gloss.lower()}.npz"
            X = np.random.randn(10, FEATURE_DIM).astype(np.float32)
            X_vel = np.zeros((X.shape[0], COORD_FEATURE_DIM), dtype=np.float32)
            np.savez_compressed(path, X=X, X_vel=X_vel)
            entries.append(
                {
                    "id": gloss.lower(),
                    "features_path": str(path),
                    "glosses": [gloss],
                    "num_frames": int(X.shape[0]),
                    "split": "train",
                    "dataset": "synthetic",
                }
            )

        with pytest.raises(ValueError, match="Not enough sampled synthetic sequence capacity"):
            synthesize_cslr_split(
                source_entries=entries,
                output_dir=output_dir,
                split="train",
                num_sequences=1,
                seed=21,
                min_sequence_len=2,
                max_sequence_len=2,
                transition_frames_min=0,
                transition_frames_max=0,
                required_glosses=["BOOK", "LIKE", "DRINK"],
            )


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------

class TestAugmentation:
    """Tests for skeleton augmentation transforms."""

    def test_augmentor_preserves_shape_dims(self):
        """Augmented output should preserve feature width while T may change."""
        aug = SkeletonAugmentor(enabled=True)
        X = np.random.randn(30, FEATURE_DIM).astype(np.float32)
        out = aug(X)
        assert out.shape[1] == FEATURE_DIM
        assert out.shape[0] > 0

    def test_augmentor_disabled(self):
        """Disabled augmentor should be identity."""
        aug = SkeletonAugmentor(enabled=False)
        X = np.random.randn(30, FEATURE_DIM).astype(np.float32)
        out = aug(X)
        np.testing.assert_array_equal(X, out)

    def test_horizontal_flip_swaps_hands(self):
        """Flip should swap left hand (10-30) ↔ right hand (31-51)."""
        aug = SkeletonAugmentor(
            flip_prob=1.0,  # Always flip
            allow_horizontal_flip=True,
            spatial_jitter_std=0,
            scale_range=(1.0, 1.0),
            translate_range=0,
            temporal_crop_ratio=(1.0, 1.0),
            joint_dropout_prob=0,
            speed_perturb_range=(1.0, 1.0),
        )
        X = np.random.randn(10, FEATURE_DIM).astype(np.float32)
        out = aug(X)

        original = X.reshape(10, NUM_JOINTS, 4)
        flipped = out.reshape(10, NUM_JOINTS, 4)
        np.testing.assert_allclose(
            flipped[:, 31:52, 1:],
            original[:, 10:31, 1:],
        )

    def test_horizontal_flip_is_disabled_by_default(self):
        """Semantically unsafe mirroring should not happen unless explicitly enabled."""
        aug = SkeletonAugmentor(
            flip_prob=1.0,
            spatial_jitter_std=0,
            scale_range=(1.0, 1.0),
            translate_range=0,
            temporal_crop_ratio=(1.0, 1.0),
            joint_dropout_prob=0,
            speed_perturb_range=(1.0, 1.0),
        )
        X = np.random.randn(10, FEATURE_DIM).astype(np.float32)
        out = aug(X)
        np.testing.assert_array_equal(X, out)

    def test_joint_dropout_does_not_peek_into_future(self):
        """Frame 0 should stay untouched when dropout simulates tracker holds."""
        aug = SkeletonAugmentor(
            spatial_jitter_std=0,
            scale_range=(1.0, 1.0),
            rotation_range_deg=(0.0, 0.0),
            translate_range=0,
            temporal_crop_ratio=(1.0, 1.0),
            temporal_drop_ratio=0.0,
            joint_dropout_prob=1.0,
            speed_perturb_range=(1.0, 1.0),
        )
        X = np.arange(3 * FEATURE_DIM, dtype=np.float32).reshape(3, FEATURE_DIM)
        out = aug(X)
        np.testing.assert_array_equal(out[0], X[0])

    def test_view_rotation_uses_depth_dimension(self):
        """Yaw augmentation should rotate depth into x when z is present."""
        aug = SkeletonAugmentor(
            spatial_jitter_std=0.0,
            scale_range=(1.0, 1.0),
            rotation_range_deg=(0.0, 0.0),
            pitch_range_deg=(0.0, 0.0),
            yaw_range_deg=(90.0, 90.0),
            translate_range=0.0,
            temporal_crop_ratio=(1.0, 1.0),
            temporal_drop_ratio=0.0,
            joint_dropout_prob=0.0,
            hand_dropout_prob=0.0,
            pose_dropout_prob=0.0,
            speed_perturb_range=(1.0, 1.0),
        )
        X = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        packed = X.reshape(1, NUM_JOINTS, 4)
        packed[..., 3] = 1.0
        packed[0, 0, :3] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        out = aug(X).reshape(1, NUM_JOINTS, 4)

        assert out[0, 0, 0] > 0.99
        assert abs(out[0, 0, 2]) < 1e-4

    def test_hand_dropout_masks_an_entire_hand_span(self):
        """Sequence augmentation should simulate full-hand tracker loss."""
        np.random.seed(0)
        aug = SkeletonAugmentor(
            spatial_jitter_std=0.0,
            scale_range=(1.0, 1.0),
            rotation_range_deg=(0.0, 0.0),
            translate_range=0.0,
            temporal_crop_ratio=(1.0, 1.0),
            temporal_drop_ratio=0.0,
            joint_dropout_prob=0.0,
            hand_dropout_prob=1.0,
            hand_dropout_ratio=(0.5, 0.5),
            pose_dropout_prob=0.0,
            speed_perturb_range=(1.0, 1.0),
        )
        X = np.zeros((8, FEATURE_DIM), dtype=np.float32)
        packed = X.reshape(8, NUM_JOINTS, 4)
        packed[..., 3] = 1.0

        out = aug(X).reshape(8, NUM_JOINTS, 4)
        left_missing = (out[:, 10:31, 3].sum(axis=1) == 0).sum()
        right_missing = (out[:, 31:52, 3].sum(axis=1) == 0).sum()

        assert max(left_missing, right_missing) >= 4

    def test_pose_dropout_masks_pose_span(self):
        """Sequence augmentation should simulate pose-landmark outages."""
        np.random.seed(1)
        aug = SkeletonAugmentor(
            spatial_jitter_std=0.0,
            scale_range=(1.0, 1.0),
            rotation_range_deg=(0.0, 0.0),
            translate_range=0.0,
            temporal_crop_ratio=(1.0, 1.0),
            temporal_drop_ratio=0.0,
            joint_dropout_prob=0.0,
            hand_dropout_prob=0.0,
            pose_dropout_prob=1.0,
            pose_dropout_ratio=(0.5, 0.5),
            speed_perturb_range=(1.0, 1.0),
        )
        X = np.zeros((8, FEATURE_DIM), dtype=np.float32)
        packed = X.reshape(8, NUM_JOINTS, 4)
        packed[..., 3] = 1.0

        out = aug(X).reshape(8, NUM_JOINTS, 4)
        pose_missing = (out[:, :10, 3].sum(axis=1) == 0).sum()

        assert pose_missing >= 4


# ---------------------------------------------------------------------------
# Vocab tests
# ---------------------------------------------------------------------------

class TestVocab:
    """Tests for gloss vocabulary."""

    def test_vocab_size(self, vocab, synth_glosses):
        """Vocab size should be glosses + special tokens."""
        # Special: <blank>, <pad>, <bos>, <eos>, <unk>
        assert len(vocab) == len(synth_glosses) + 5

    def test_encode_decode(self, vocab):
        """Encode then decode should round-trip."""
        idx = vocab.encode("HELLO")
        decoded = vocab.decode(idx)
        assert decoded == "HELLO"

    def test_unknown_gloss(self, vocab):
        """Unknown gloss should map to <unk>."""
        idx = vocab.encode("NONEXISTENT_GLOSS")
        assert idx == vocab.unk_idx

    def test_encode_sequence(self, vocab):
        """Sequence encoding should produce correct IDs."""
        ids = vocab.encode_sequence(["HELLO", "THANK-YOU"])
        assert len(ids) == 2
        assert ids[0] == vocab.encode("HELLO")

    def test_save_load(self, vocab, tmp_path):
        """Vocab should survive save/load cycle."""
        path = tmp_path / "vocab.json"
        vocab.save(str(path))
        loaded = GlossVocab.load(str(path))
        assert len(loaded) == len(vocab)
        assert loaded.encode("HELLO") == vocab.encode("HELLO")


class TestCTCSpecialMasking:
    """Tests for suppressing invalid CTC specials."""

    def test_suppress_ctc_special_tokens_prefers_real_glosses(self, vocab):
        hello_idx = vocab.encode("HELLO")
        logits = torch.full((1, 3, len(vocab)), -10.0)
        logits[..., vocab.pad_idx] = 6.0
        logits[..., vocab.unk_idx] = 5.0
        logits[..., hello_idx] = 4.0

        log_probs = torch.log_softmax(logits, dim=-1)
        masked = suppress_ctc_special_tokens(
            log_probs,
            vocab.special_indices(include_blank=False),
        )

        preds = masked.argmax(dim=-1)
        assert preds.shape == (1, 3)
        assert preds.unique().tolist() == [hello_idx]


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestDatasets:
    """Tests for ISLR and CSLR datasets."""

    def test_islr_dataset_getitem(self, synth_data_dir, vocab):
        """ISLRDataset should return correctly shaped tensors."""
        ds = ISLRDataset(synth_data_dir["manifest_path"], vocab)
        sample = ds[0]

        assert sample["features"].ndim == 2
        assert sample["features"].shape[1] == FEATURE_DIM
        assert sample["label"].ndim == 0
        assert isinstance(sample["length"], int)

    def test_islr_collate(self, synth_data_dir, vocab):
        """ISLR collate should pad to batch max length."""
        ds = ISLRDataset(synth_data_dir["manifest_path"], vocab)
        batch = [ds[i] for i in range(min(4, len(ds)))]
        collated = islr_collate_fn(batch)

        B = len(batch)
        assert collated["features"].shape[0] == B
        assert collated["labels"].shape == (B,)
        assert collated["lengths"].shape == (B,)

    def test_cslr_dataset_getitem(self, synth_data_dir, vocab):
        """CSLRDataset should return features and label sequences."""
        ds = CSLRDataset(synth_data_dir["manifest_path"], vocab)
        sample = ds[0]

        assert sample["features"].ndim == 2
        assert sample["features"].shape[1] == FEATURE_DIM
        assert sample["labels"].ndim == 1
        assert sample["label_length"] == len(sample["labels"])

    def test_cslr_collate(self, synth_data_dir, vocab):
        """CSLR collate should produce concatenated labels for CTC."""
        ds = CSLRDataset(synth_data_dir["manifest_path"], vocab)
        batch = [ds[i] for i in range(min(4, len(ds)))]
        collated = cslr_collate_fn(batch)

        B = len(batch)
        assert collated["features"].shape[0] == B
        assert collated["input_lengths"].shape == (B,)
        assert collated["label_lengths"].shape == (B,)
        # Labels should be concatenated
        total_labels = sum(collated["label_lengths"].tolist())
        assert collated["labels"].shape[0] == total_labels

    def test_cslr_dual_stream_collate(self, synth_data_dir, vocab):
        """Dual-stream CSLR batches should pad pose and motion separately."""
        ds = CSLRDataset(
            synth_data_dir["manifest_path"],
            vocab,
            use_motion=True,
            dual_stream=True,
        )
        batch = [ds[i] for i in range(min(4, len(ds)))]
        collated = cslr_collate_fn(batch)

        B = len(batch)
        assert collated["features_pose"].shape[0] == B
        assert collated["features_motion"].shape[0] == B
        assert collated["features_pose"].shape[-1] == FEATURE_DIM
        assert collated["features_motion"].shape[-1] == COORD_FEATURE_DIM

    def test_dataset_with_augmentation(self, synth_data_dir, vocab):
        """Dataset with augmentor should work without errors."""
        aug = SkeletonAugmentor(enabled=True, spatial_jitter_std=0.005)
        ds = ISLRDataset(synth_data_dir["manifest_path"], vocab, augmentor=aug)
        sample = ds[0]
        assert sample["features"].shape[1] == FEATURE_DIM

    def test_dataset_recomputes_motion_after_augmentation(self, synth_data_dir, vocab):
        """Motion features should stay aligned after temporal augmentation."""
        aug = SkeletonAugmentor(
            enabled=True,
            spatial_jitter_std=0.0,
            scale_range=(1.0, 1.0),
            translate_range=0.0,
            temporal_crop_ratio=(0.6, 0.8),
            joint_dropout_prob=0.0,
            speed_perturb_range=(1.0, 1.0),
        )
        ds = ISLRDataset(
            synth_data_dir["manifest_path"],
            vocab,
            use_motion=True,
            augmentor=aug,
        )
        sample = ds[0]
        assert sample["features"].shape[1] == FEATURE_DIM + COORD_FEATURE_DIM

    def test_dataset_recomputes_incompatible_stored_motion_features(self, tmp_path, vocab):
        """Mismatched X/X_vel widths should trigger safe motion recomputation."""
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        manifest_path = tmp_path / "manifest.jsonl"

        X = np.random.randn(6, FEATURE_DIM).astype(np.float32)
        X[:, 0] = np.linspace(0.0, 0.5, 6, dtype=np.float32)
        wrong_vel = np.zeros((6, LEGACY_XY_COORD_DIM), dtype=np.float32)
        sample_path = npz_dir / "sample.npz"
        np.savez_compressed(sample_path, X=X, X_vel=wrong_vel)

        with open(manifest_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "id": "sample",
                        "features_path": str(sample_path),
                        "glosses": ["HELLO"],
                        "num_frames": 6,
                        "split": "train",
                        "dataset": "synthetic",
                    }
                )
                + "\n"
            )

        ds = ISLRDataset(manifest_path, vocab, use_motion=True)
        sample = ds[0]

        assert sample["features"].shape[1] == FEATURE_DIM + COORD_FEATURE_DIM
        assert not torch.allclose(sample["features"][:, FEATURE_DIM:], torch.zeros_like(sample["features"][:, FEATURE_DIM:]))

    def test_dataset_preserves_nonzero_z_coordinates(self, tmp_path, vocab):
        """Stored z values should survive loading through the dataset path."""
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        manifest_path = tmp_path / "manifest.jsonl"

        joints = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
        joints[0] = [0.1, 0.2, 0.3]
        X = np.stack(
            [build_feature_frame(joints, np.ones(NUM_JOINTS, dtype=np.float32))],
            axis=0,
        )
        sample_path = npz_dir / "sample_z.npz"
        np.savez_compressed(sample_path, X=X)

        with open(manifest_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "id": "sample_z",
                        "features_path": str(sample_path),
                        "glosses": ["HELLO"],
                        "num_frames": 1,
                        "split": "train",
                        "dataset": "synthetic",
                    }
                )
                + "\n"
            )

        dataset = ISLRDataset(manifest_path, vocab, use_motion=False)
        sample = dataset[0]
        frame0 = sample["features"][0].reshape(NUM_JOINTS, 4)

        assert frame0[0, 2].item() == pytest.approx(0.3, abs=1e-6)

    def test_dataset_rejects_legacy_schema_when_packed_features_required(self, tmp_path, vocab):
        """Strict packed-schema loaders should reject legacy coordinate-only artifacts."""
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        manifest_path = tmp_path / "manifest.jsonl"

        X = np.random.randn(6, LEGACY_XY_COORD_DIM).astype(np.float32)
        sample_path = npz_dir / "legacy_xy.npz"
        np.savez_compressed(sample_path, X=X, schema_version=np.array(1, dtype=np.int32))

        with open(manifest_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "id": "legacy_xy",
                        "features_path": str(sample_path),
                        "glosses": ["HELLO"],
                        "num_frames": 6,
                        "split": "train",
                        "dataset": "synthetic",
                    }
                )
                + "\n"
            )

        with pytest.raises(ValueError, match="expected 208|schema >= 2|failed feature-schema preflight"):
            ISLRDataset(
                manifest_path,
                vocab,
                expected_frame_feature_dim=FEATURE_DIM,
                required_schema_version=2,
            )

    def test_dataset_rejects_missing_schema_version_for_packed_features(self, tmp_path, vocab):
        """Strict packed-schema loaders should require schema_version metadata."""
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        manifest_path = tmp_path / "manifest.jsonl"

        X = np.random.randn(6, FEATURE_DIM).astype(np.float32)
        sample_path = npz_dir / "packed_missing_schema.npz"
        np.savez_compressed(sample_path, X=X)

        with open(manifest_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "id": "packed_missing_schema",
                        "features_path": str(sample_path),
                        "glosses": ["HELLO"],
                        "num_frames": 6,
                        "split": "train",
                        "dataset": "synthetic",
                    }
                )
                + "\n"
            )

        with pytest.raises(ValueError, match="missing schema_version"):
            CSLRDataset(
                manifest_path,
                vocab,
                expected_frame_feature_dim=FEATURE_DIM,
                required_schema_version=2,
            )

    def test_dataset_preflight_schema_validation_fails_at_init(self, tmp_path, vocab):
        """Strict schema validation should fail before training starts, not mid-epoch."""
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        manifest_path = tmp_path / "manifest.jsonl"

        good_path = npz_dir / "good.npz"
        bad_path = npz_dir / "bad.npz"
        np.savez_compressed(
            good_path,
            X=np.random.randn(4, FEATURE_DIM).astype(np.float32),
            schema_version=np.array(2, dtype=np.int32),
        )
        np.savez_compressed(
            bad_path,
            X=np.random.randn(4, COORD_FEATURE_DIM).astype(np.float32),
            schema_version=np.array(2, dtype=np.int32),
        )

        with open(manifest_path, "w", encoding="utf-8") as handle:
            for sample_id, path in (("good", good_path), ("bad", bad_path)):
                handle.write(
                    json.dumps(
                        {
                            "id": sample_id,
                            "features_path": str(path),
                            "glosses": ["HELLO"],
                            "num_frames": 4,
                            "split": "train",
                            "dataset": "synthetic",
                        }
                    )
                    + "\n"
                )

        with pytest.raises(ValueError, match="failed feature-schema preflight"):
            ISLRDataset(
                manifest_path,
                vocab,
                expected_frame_feature_dim=FEATURE_DIM,
                required_schema_version=2,
            )

    def test_dataset_rejects_out_of_vocab_glosses_at_init(self, tmp_path, vocab):
        """Manifest/vocab drift should fail fast instead of silently mapping to <unk>."""
        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        manifest_path = tmp_path / "manifest.jsonl"

        sample_path = npz_dir / "sample.npz"
        np.savez_compressed(
            sample_path,
            X=np.random.randn(4, FEATURE_DIM).astype(np.float32),
            schema_version=np.array(2, dtype=np.int32),
        )

        with open(manifest_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "id": "oov",
                        "features_path": str(sample_path),
                        "glosses": ["MISSING_GLOSS"],
                        "num_frames": 4,
                        "split": "train",
                        "dataset": "synthetic",
                    }
                )
                + "\n"
            )

        with pytest.raises(ValueError, match="out-of-vocab gloss"):
            CSLRDataset(
                manifest_path,
                vocab,
                expected_frame_feature_dim=FEATURE_DIM,
                required_schema_version=2,
            )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModels:
    """Tests for model forward passes."""

    def test_temporal_conv_forward(self):
        """TemporalConvEncoder should produce correct output shape."""
        model = TemporalConvEncoder(input_dim=FEATURE_DIM, conv_dim=128, num_layers=2)
        x = torch.randn(2, 30, FEATURE_DIM)
        out = model(x)
        assert out.shape == (2, 30, 128)

    def test_multiscale_conv_forward(self):
        """MultiScaleTemporalConv should produce correct output shape."""
        model = MultiScaleTemporalConv(input_dim=FEATURE_DIM, conv_dim=128, kernel_sizes=[3, 5, 7])
        x = torch.randn(2, 30, FEATURE_DIM)
        out = model(x)
        assert out.shape == (2, 30, 128)

    def test_bilstm_forward(self):
        """BiLSTMEncoder should produce correct output shape."""
        model = BiLSTMEncoder(input_dim=128, hidden_size=64, num_layers=1)
        x = torch.randn(2, 30, 128)
        lengths = torch.tensor([30, 20])
        out = model(x, lengths)
        assert out.shape == (2, 30, 128)  # 2*hidden_size

    def test_transformer_forward(self):
        """TransformerSequenceEncoder should produce correct output shape."""
        model = TransformerSequenceEncoder(input_dim=128, hidden_dim=128, num_heads=4, num_layers=2)
        x = torch.randn(2, 30, 128)
        out = model(x)
        assert out.shape == (2, 30, 128)

    def test_classification_head_forward(self):
        """ClassificationHead should produce (B, num_classes) logits."""
        head = ClassificationHead(input_dim=128, num_classes=10)
        x = torch.randn(2, 30, 128)
        lengths = torch.tensor([30, 20])
        out = head(x, lengths)
        assert out.shape == (2, 10)

    def test_ctc_head_forward(self):
        """CTCHead should produce (B, T, num_classes) log-probs."""
        head = CTCHead(input_dim=128, num_classes=10)
        x = torch.randn(2, 30, 128)
        out = head(x)
        assert out.shape == (2, 30, 10)

    def test_islr_model_full_forward(self, vocab):
        """Full ISLR model forward pass: features → logits."""
        model = ISLRModel(
            input_dim=FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
            fc_dropout=0.1,
        )
        x = torch.randn(2, 25, FEATURE_DIM)
        lengths = torch.tensor([25, 18])
        logits = model(x, lengths)
        assert logits.shape == (2, len(vocab))

    def test_cslr_model_full_forward(self, vocab):
        """Full CSLR model forward pass: features → log-probs."""
        model = CSLRModel(
            input_dim=FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
        )
        x = torch.randn(2, 25, FEATURE_DIM)
        lengths = torch.tensor([25, 18])
        log_probs = model(x, lengths)
        assert log_probs.shape[0] == 2
        assert log_probs.shape[2] == len(vocab)

    def test_cslr_load_backbone_returns_key_mismatches(self, vocab):
        """Transformer warm starts should report partial key mismatches."""
        source = CSLRModel(
            input_dim=FEATURE_DIM + COORD_FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
        )
        target = CSLRModel(
            input_dim=FEATURE_DIM + COORD_FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            encoder_type="transformer",
            transformer_hidden=64,
            transformer_layers=2,
            transformer_heads=4,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
        )

        missing, unexpected = target.load_backbone(source.state_dict(), strict=False)

        assert isinstance(missing, list)
        assert isinstance(unexpected, list)
        assert missing or unexpected

    def test_cslr_load_backbone_ignores_ctc_head_shape_mismatch(self):
        source = CSLRModel(
            input_dim=FEATURE_DIM + COORD_FEATURE_DIM,
            num_classes=7,
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
        )
        target = CSLRModel(
            input_dim=FEATURE_DIM + COORD_FEATURE_DIM,
            num_classes=11,
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
        )

        missing, unexpected = target.load_backbone(source.state_dict(), strict=False)

        assert isinstance(missing, list)
        assert isinstance(unexpected, list)
        assert len(unexpected) == 0
        assert any(key.startswith("ctc_head.") for key in missing)

    def test_cslr_load_backbone_can_seed_ctc_head_from_islr_classifier(self, vocab):
        source = ISLRModel(
            input_dim=FEATURE_DIM + COORD_FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
            fc_dropout=0.1,
            multi_scale=False,
        )
        target = CSLRModel(
            input_dim=FEATURE_DIM + COORD_FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
            multi_scale=False,
        )

        with torch.no_grad():
            source.head.fc.weight.fill_(0.25)
            source.head.fc.bias.fill_(0.5)

        missing, unexpected = target.load_backbone(source.state_dict(), strict=False)

        assert isinstance(missing, list)
        assert isinstance(unexpected, list)
        assert getattr(target, "_loaded_ctc_head_from_backbone", False) is True
        assert "ctc_head.fc.weight" not in missing
        assert "ctc_head.fc.bias" not in missing
        assert torch.allclose(target.ctc_head.fc.weight, source.head.fc.weight)
        assert torch.allclose(target.ctc_head.fc.bias, source.head.fc.bias)

    def test_dual_stream_backbone_warm_start(self, vocab):
        """Dual-stream CSLR should be able to load a single-stream backbone."""
        islr = ISLRModel(
            input_dim=FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
            fc_dropout=0.1,
            multi_scale=False,
        )
        dual = DualStreamCSLRModel(
            pose_dim=FEATURE_DIM,
            motion_dim=FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
            multi_scale=False,
        )

        dual.load_backbone(islr.get_backbone_state_dict(), strict=False)

        assert torch.allclose(
            dual.pose_conv.layers[0].conv.weight,
            islr.conv_encoder.layers[0].conv.weight,
        )
        assert torch.allclose(
            dual.motion_conv.layers[0].conv.weight,
            islr.conv_encoder.layers[0].conv.weight,
        )

    def test_dual_stream_backbone_warm_start_skips_mismatched_conv_inputs(self, vocab):
        """Warm start should not crash when pose/motion stream widths differ from ISLR input."""
        islr = ISLRModel(
            input_dim=FEATURE_DIM + COORD_FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
            fc_dropout=0.1,
            multi_scale=False,
        )
        dual = DualStreamCSLRModel(
            pose_dim=FEATURE_DIM,
            motion_dim=COORD_FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
            multi_scale=False,
        )

        before_pose_conv = dual.pose_conv.layers[0].conv.weight.detach().clone()
        before_motion_conv = dual.motion_conv.layers[0].conv.weight.detach().clone()

        dual.load_backbone(islr.get_backbone_state_dict(), strict=False)

        assert torch.allclose(dual.pose_conv.layers[0].conv.weight, before_pose_conv)
        assert torch.allclose(dual.motion_conv.layers[0].conv.weight, before_motion_conv)
        assert torch.allclose(
            dual.pose_lstm.lstm.weight_ih_l0,
            islr.seq_encoder.lstm.weight_ih_l0,
        )


# ---------------------------------------------------------------------------
# Manifest builder tests
# ---------------------------------------------------------------------------

class TestManifestBuilders:
    """Tests for deterministic task-manifest generation."""

    def test_select_shared_goal_glosses_filters_stopwords(self):
        wlasl_entries = [
            {"id": "w1", "glosses": ["BOOK"], "split": "train"},
            {"id": "w2", "glosses": ["BOOK"], "split": "train"},
            {"id": "w3", "glosses": ["BOOK"], "split": "val"},
            {"id": "w4", "glosses": ["NOW"], "split": "train"},
            {"id": "w5", "glosses": ["NOW"], "split": "train"},
            {"id": "w6", "glosses": ["YOU"], "split": "train"},
            {"id": "w7", "glosses": ["YOU"], "split": "val"},
        ]
        how2sign_entries = [
            {"id": "h1", "sentence": "book now you", "split": "train"},
            {"id": "h2", "sentence": "book now", "split": "train"},
            {"id": "h3", "sentence": "book now you", "split": "val"},
        ]

        goal_glosses = select_shared_goal_glosses(
            wlasl_entries,
            how2sign_entries,
            shared_vocab_size=5,
            min_wlasl_frequency=2,
            min_how2sign_frequency=2,
        )

        assert "BOOK" in goal_glosses
        assert "NOW" in goal_glosses
        assert "YOU" not in goal_glosses

    def test_select_shared_goal_glosses_balances_continuous_coverage(self):
        wlasl_entries = []
        for idx in range(7):
            wlasl_entries.append(
                {"id": f"rare_{idx}", "glosses": ["RARE"], "split": "train"}
            )
        for idx in range(6):
            wlasl_entries.append(
                {"id": f"common_{idx}", "glosses": ["COMMON"], "split": "train"}
            )

        how2sign_entries = (
            [{"id": "rare_h", "sentence": "rare " * 20, "split": "train"}]
            + [{"id": "common_h", "sentence": "common " * 1000, "split": "train"}]
        )

        goal_glosses = select_shared_goal_glosses(
            wlasl_entries,
            how2sign_entries,
            shared_vocab_size=2,
            min_wlasl_frequency=6,
            min_how2sign_frequency=15,
        )

        assert goal_glosses[:2] == ["COMMON", "RARE"]

    def test_build_task_split_entries(self):
        wlasl_entries = [
            {"id": "w1", "glosses": ["BOOK"], "split": "train"},
            {"id": "w2", "glosses": ["LIKE"], "split": "val"},
            {"id": "w3", "glosses": ["DRINK"], "split": "test"},
        ]
        how2sign_entries = [
            {
                "id": "h1",
                "sentence": "book like drink",
                "split": "train",
                "num_frames": 12,
            },
            {
                "id": "h2",
                "sentence": "you and book",
                "split": "val",
                "num_frames": 8,
            },
        ]

        islr = build_islr_split_entries(
            wlasl_entries,
            allowed_glosses={"BOOK", "LIKE"},
            selection_name="goal_shared_vocab",
        )
        cslr = build_cslr_split_entries(
            how2sign_entries,
            allowed_glosses={"BOOK", "LIKE", "DRINK"},
            min_glosses_per_sequence=2,
            selection_name="goal_shared_vocab",
            label_source="sentence_tokens_filtered",
        )

        assert [entry["id"] for entry in islr["train"]] == ["w1"]
        assert [entry["id"] for entry in islr["val"]] == ["w2"]
        assert not islr["test"]
        assert cslr["train"][0]["glosses"] == ["BOOK", "LIKE", "DRINK"]
        assert cslr["train"][0]["label_source"] == "sentence_tokens_filtered"
        assert cslr["train"][0]["selection"] == "goal_shared_vocab"
        assert cslr["val"] == []

    def test_stratify_islr_entries_by_gloss_guarantees_eval_coverage(self):
        entries = [
            {"id": "book_train_a", "glosses": ["BOOK"], "split": "train"},
            {"id": "book_train_b", "glosses": ["BOOK"], "split": "train"},
            {"id": "book_val", "glosses": ["BOOK"], "split": "val"},
            {"id": "like_train_a", "glosses": ["LIKE"], "split": "train"},
            {"id": "like_train_b", "glosses": ["LIKE"], "split": "train"},
            {"id": "like_test", "glosses": ["LIKE"], "split": "test"},
        ]

        stratified = stratify_islr_entries_by_gloss(
            entries,
            min_val_per_gloss=1,
            min_test_per_gloss=1,
        )

        train_glosses = [entry["glosses"][0] for entry in stratified["train"]]
        val_glosses = [entry["glosses"][0] for entry in stratified["val"]]
        test_glosses = [entry["glosses"][0] for entry in stratified["test"]]

        assert train_glosses.count("BOOK") == 1
        assert train_glosses.count("LIKE") == 1
        assert val_glosses.count("BOOK") == 1
        assert val_glosses.count("LIKE") == 1
        assert test_glosses.count("BOOK") == 1
        assert test_glosses.count("LIKE") == 1


# ---------------------------------------------------------------------------
# Training smoke test
# ---------------------------------------------------------------------------

class TestTraining:
    """Tests for loss computation and backward pass."""

    def test_loader_workers_fall_back_to_zero_on_mps(self):
        device = torch.device("mps")

        assert resolve_islr_loader_workers(device, 8) == 0
        assert resolve_cslr_loader_workers(device, 8) == 0

    def test_loader_workers_preserve_requested_count_off_mps(self):
        device = torch.device("cpu")

        assert resolve_islr_loader_workers(device, 4) == 4
        assert resolve_cslr_loader_workers(device, 4) == 4

    def test_cslr_balanced_sampler_upweights_rare_gloss_sequences(self):
        dataset = type(
            "DummyDataset",
            (),
            {
                "entries": [
                    {"glosses": ["LIKE"]},
                    {"glosses": ["LIKE", "NOW"]},
                    {"glosses": ["FORGET"]},
                ]
            },
        )()

        sampler = build_cslr_balanced_sampler(dataset)
        weights = sampler.weights.tolist()

        assert weights[2] > weights[0]
        assert weights[1] > weights[0]

    def test_ctc_bias_init_penalizes_blank_and_uses_gloss_priors(self):
        vocab = build_vocab(["LIKE", "NOW", "FORGET"])
        dataset = type(
            "DummyDataset",
            (),
            {
                "entries": [
                    {"glosses": ["LIKE", "NOW"]},
                    {"glosses": ["LIKE"]},
                    {"glosses": ["FORGET"]},
                ]
            },
        )()
        model = type("DummyModel", (), {"ctc_head": CTCHead(input_dim=4, num_classes=len(vocab))})()

        _initialize_ctc_head_biases(
            model,
            dataset,
            vocab,
            blank_bias=-2.0,
            smoothing=1.0,
            prior_scale=1.0,
        )

        bias = model.ctc_head.fc.bias.detach().cpu()
        assert bias[vocab.blank_idx].item() == pytest.approx(-2.0)
        assert bias[vocab.encode("LIKE")].item() > bias[vocab.encode("FORGET")].item()
        assert bias[vocab.encode("NOW")].item() > bias[vocab.blank_idx].item()

    def test_configure_ctc_blank_row_can_zero_blank_weight(self):
        vocab = build_vocab(["LIKE", "NOW"])
        model = type("DummyModel", (), {"ctc_head": CTCHead(input_dim=4, num_classes=len(vocab))})()

        with torch.no_grad():
            model.ctc_head.fc.weight.fill_(1.0)
            model.ctc_head.fc.bias.fill_(0.5)

        _configure_ctc_blank_row(
            model,
            vocab,
            blank_bias=-3.0,
            zero_blank_weight=True,
            special_bias=-4.0,
        )

        assert model.ctc_head.fc.bias[vocab.blank_idx].item() == pytest.approx(-3.0)
        assert torch.allclose(
            model.ctc_head.fc.weight[vocab.blank_idx],
            torch.zeros_like(model.ctc_head.fc.weight[vocab.blank_idx]),
        )
        for idx in vocab.special_indices(include_blank=False):
            assert model.ctc_head.fc.bias[idx].item() == pytest.approx(-4.0)

    def test_freeze_ctc_blank_gradients_only_zeroes_blank_row(self):
        vocab = build_vocab(["LIKE", "NOW"])
        model = type("DummyModel", (), {"ctc_head": CTCHead(input_dim=4, num_classes=len(vocab))})()
        model.ctc_head.fc.weight.grad = torch.ones_like(model.ctc_head.fc.weight)
        model.ctc_head.fc.bias.grad = torch.ones_like(model.ctc_head.fc.bias)

        _freeze_ctc_blank_gradients(model, vocab)

        assert torch.allclose(
            model.ctc_head.fc.weight.grad[vocab.blank_idx],
            torch.zeros_like(model.ctc_head.fc.weight.grad[vocab.blank_idx]),
        )
        assert model.ctc_head.fc.bias.grad[vocab.blank_idx].item() == pytest.approx(0.0)
        assert model.ctc_head.fc.bias.grad[vocab.encode("LIKE")].item() == pytest.approx(1.0)

    def test_islr_training_step(self, vocab):
        """Complete ISLR training step should not crash."""
        model = ISLRModel(
            input_dim=FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
            fc_dropout=0.1,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(4, 20, FEATURE_DIM)
        labels = torch.randint(0, len(vocab), (4,))
        lengths = torch.tensor([20, 15, 18, 10])

        model.train()
        logits = model(x, lengths)
        loss = criterion(logits, labels)

        assert torch.isfinite(loss), "Loss should be finite"

        loss.backward()
        optimizer.step()

        # Check grads exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"

    def test_cslr_training_step(self, vocab):
        """Complete CSLR training step with CTC loss."""
        model = CSLRModel(
            input_dim=FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ctc_loss = torch.nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)

        x = torch.randn(2, 30, FEATURE_DIM)
        input_lengths = torch.tensor([30, 25])
        # Short label sequences
        labels = torch.tensor([
            vocab.encode("HELLO"), vocab.encode("YES"),
            vocab.encode("NO"),
        ])
        label_lengths = torch.tensor([2, 1])

        model.train()
        log_probs = model(x, input_lengths)
        log_probs_t = log_probs.transpose(0, 1)  # (T, B, C)

        loss = ctc_loss(log_probs_t, labels, input_lengths, label_lengths)
        assert torch.isfinite(loss), "CTC loss should be finite"

        loss.backward()
        optimizer.step()

    def test_greedy_decode(self, vocab):
        """CTC greedy decode should return valid gloss ID sequences."""
        model = CSLRModel(
            input_dim=FEATURE_DIM,
            num_classes=len(vocab),
            conv_dim=64,
            conv_layers=2,
            conv_kernel_size=3,
            conv_dropout=0.1,
            lstm_hidden_size=32,
            lstm_layers=1,
            lstm_dropout=0.0,
        )
        model.eval()
        x = torch.randn(2, 20, FEATURE_DIM)
        lengths = torch.tensor([20, 15])

        with torch.no_grad():
            log_probs = model(x, lengths)
            decoded = model.greedy_decode(log_probs)

        assert len(decoded) == 2
        for seq in decoded:
            assert isinstance(seq, list)
            for idx in seq:
                assert 0 <= idx < len(vocab)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for evaluation metrics."""

    def test_top1_accuracy(self):
        """Top-1 accuracy with perfect predictions."""
        logits = torch.eye(5).float()
        labels = torch.arange(5)
        acc1, = compute_accuracy(logits, labels, topk=(1,))
        assert acc1 == 1.0

    def test_top5_accuracy(self):
        """Top-5 accuracy should be >= top-1."""
        logits = torch.randn(10, 20)
        labels = torch.randint(0, 20, (10,))
        acc1, acc5 = compute_accuracy(logits, labels, topk=(1, 5))
        assert acc5 >= acc1

    def test_wer_perfect(self):
        """WER with identical sequences should be 0."""
        refs = [[1, 2, 3], [4, 5]]
        hyps = [[1, 2, 3], [4, 5]]
        assert compute_wer(refs, hyps) == 0.0

    def test_wer_all_wrong(self):
        """WER with completely wrong predictions."""
        refs = [[1, 2, 3]]
        hyps = [[4, 5, 6]]
        assert compute_wer(refs, hyps) == 1.0  # 3 subs / 3 ref tokens

    def test_macro_accuracy(self):
        """Macro accuracy with 2 classes."""
        preds = [0, 0, 1, 1]
        labels = [0, 0, 1, 0]
        # Class 0: 2/2 = 1.0, Class 1: 1/1 = 1.0... wait
        # Actually: labels=[0,0,1,0], preds=[0,0,1,1]
        # Class 0: preds correct when label=0: [0,0,1] → 2 correct / 3 total
        # Class 1: preds correct when label=1: [1] → 1/1
        # Macro = (2/3 + 1.0) / 2 ≈ 0.833
        acc = macro_averaged_accuracy(preds, labels, num_classes=2)
        assert 0.8 < acc < 0.9


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------

class TestScheduler:
    """Tests for learning rate schedulers."""

    def test_cosine_scheduler_warmup(self):
        """LR should ramp up during warmup then decay."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = build_scheduler(optimizer, "cosine", epochs=20, warmup_epochs=5)

        lrs = []
        for _ in range(20):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # LR should increase during warmup (first 5 epochs)
        assert lrs[1] > lrs[0], "LR should increase during warmup"
        # LR should decrease after warmup
        assert lrs[-1] < lrs[5], "LR should decrease after warmup"


# ---------------------------------------------------------------------------
# Preprocessing function tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    """Tests for preprocessing helper functions."""

    def test_remap_openpose_body25_flat(self):
        """Remap from flat OpenPose (201,) format."""
        # Create fake body25 + hands with confidence
        raw = np.random.rand(201).astype(np.float32) * 0.5 + 0.25
        # Set confidences to high values (every 3rd element)
        raw[2::3] = 0.9
        joints = _remap_how2sign_keypoints(raw)
        assert joints.shape == (NUM_JOINTS, NUM_COORDS)
        # At least nose and shoulders should be mapped
        assert not np.isnan(joints[0]).any(), "Nose should be mapped"

    def test_remap_openpose_shaped(self):
        """Remap from shaped (67, 3) format."""
        raw = np.random.rand(67, 3).astype(np.float32) * 0.5 + 0.25
        raw[:, 2] = 0.9  # confidence
        joints = _remap_how2sign_keypoints(raw)
        assert joints.shape == (NUM_JOINTS, NUM_COORDS)
        assert not np.isnan(joints[0]).any()

    def test_remap_zero_confidence_produces_nan(self):
        """Keypoints with zero confidence should remain NaN."""
        raw = np.zeros((67, 3), dtype=np.float32)  # all zero conf
        joints = _remap_how2sign_keypoints(raw)
        # All should be NaN since confidence is 0
        assert np.isnan(joints).all()

    def test_remap_how2sign_mediapipe_visibility_threshold_respected(self):
        """MediaPipe-format How2Sign frames should honor the configured pose threshold."""
        raw = np.zeros(1662, dtype=np.float32)
        raw[0:4] = [0.5, 0.25, -0.1, 0.4]

        joints_low = _remap_how2sign_keypoints(raw, pose_visibility_threshold=0.5)
        joints_high = _remap_how2sign_keypoints(raw, pose_visibility_threshold=0.3)

        assert np.isnan(joints_low[0]).all()
        assert not np.isnan(joints_high[0]).any()

    def test_remap_how2sign_mediapipe_presence_threshold_uses_pose_quality_proxy(self):
        """MediaPipe-format How2Sign frames should treat pose visibility as the quality proxy."""
        raw = np.zeros(1662, dtype=np.float32)
        raw[0:4] = [0.5, 0.25, -0.1, 0.6]

        joints = _remap_how2sign_keypoints(
            raw,
            pose_visibility_threshold=0.5,
            pose_presence_threshold=0.7,
        )

        assert np.isnan(joints[0]).all()

    def test_rebuild_training_manifests_drops_unviable_seed_glosses(self, tmp_path):
        """Requested goal glosses should be filtered to split-viable overlap labels."""
        wlasl_manifest = tmp_path / "wlasl.jsonl"
        how2sign_manifest = tmp_path / "how2sign.jsonl"
        output_dir = tmp_path / "manifests"

        wlasl_entries = [
            {"id": "alpha_train", "glosses": ["ALPHA"], "split": "train"},
            {"id": "alpha_val", "glosses": ["ALPHA"], "split": "val"},
            {"id": "alpha_test", "glosses": ["ALPHA"], "split": "test"},
            {"id": "beta_train", "glosses": ["BETA"], "split": "train"},
            {"id": "beta_val", "glosses": ["BETA"], "split": "val"},
            {"id": "beta_test", "glosses": ["BETA"], "split": "test"},
            {"id": "bad_train", "glosses": ["BAD"], "split": "train"},
            {"id": "bad_val", "glosses": ["BAD"], "split": "val"},
        ]
        how2sign_entries = [
            {"id": "h1", "sentence": "alpha beta bad", "split": "train", "num_frames": 12},
            {"id": "h2", "sentence": "alpha beta bad", "split": "val", "num_frames": 12},
            {"id": "h3", "sentence": "alpha beta bad", "split": "test", "num_frames": 12},
        ]

        with open(wlasl_manifest, "w", encoding="utf-8") as handle:
            for entry in wlasl_entries:
                handle.write(json.dumps(entry) + "\n")
        with open(how2sign_manifest, "w", encoding="utf-8") as handle:
            for entry in how2sign_entries:
                handle.write(json.dumps(entry) + "\n")

        metadata = rebuild_training_manifests(
            wlasl_manifest_path=wlasl_manifest,
            how2sign_manifest_path=how2sign_manifest,
            output_dir=output_dir,
            goal_shared_vocab_size=2,
            goal_glosses=["BAD", "ALPHA"],
            goal_min_wlasl_frequency=1,
            goal_min_how2sign_frequency=1,
            goal_min_glosses_per_sequence=1,
            goal_split_strategy="stratified_per_gloss",
            goal_min_val_per_gloss=1,
            goal_min_test_per_gloss=1,
        )

        assert metadata["requested_goal_glosses"] == ["BAD", "ALPHA"]
        assert metadata["goal_glosses"] == ["ALPHA", "BETA"]
