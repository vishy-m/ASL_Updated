import os

import pytest
import numpy as np
import torch
from torch import nn

from asl_cslr.data.skeleton import (
    FEATURE_DIM,
    COORD_FEATURE_DIM,
    NUM_JOINTS,
    NUM_COORDS,
    build_feature_frame,
)
from asl_cslr.data.vocab import build_vocab
from asl_cslr.online.camera import WebcamCapture, smooth_joints
from asl_cslr.online.model_loader import load_online_cslr_model
from asl_cslr.models.cslr_model import CSLRModel
from asl_cslr.models.transformer import TransformerSequenceEncoder
from asl_cslr.online.pipeline import (
    compute_motion_energy,
    get_online_runtime_config,
    prepare_online_features,
    resolve_online_mode,
    SlidingWindowISLR,
    StreamingCSLR,
    suppress_islr_special_logits,
    validate_online_model_schema,
)
from asl_cslr.utils.device import setup_mps_fallback


@pytest.fixture
def vocab():
    return build_vocab(["HELLO", "THANK-YOU", "PLEASE", "YES", "NO"])


def _packed_buffer(xs: np.ndarray) -> np.ndarray:
    observed = np.ones(NUM_JOINTS, dtype=np.float32)
    frames = []
    for x in np.asarray(xs, dtype=np.float32):
        joints = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
        joints[0, 0] = x
        frames.append(build_feature_frame(joints, observed))
    return np.stack(frames, axis=0)


def test_resolve_online_mode_prefers_explicit_choice():
    config = {
        "default_mode": "islr",
        "islr": {"enabled": True},
        "cslr": {"enabled": True},
    }

    assert resolve_online_mode(config, "cslr") == "cslr"


def test_resolve_online_mode_uses_config_default():
    config = {
        "default_mode": "cslr",
        "islr": {"enabled": True},
        "cslr": {"enabled": True},
    }

    assert resolve_online_mode(config) == "cslr"


def test_resolve_online_mode_falls_back_to_cslr_first():
    config = {
        "islr": {"enabled": True},
        "cslr": {"enabled": True},
    }

    assert resolve_online_mode(config) == "cslr"


def test_online_runtime_config_uses_mode_specific_timing():
    config = {
        "camera": {"capture_fps": 30, "downsample_factor": 2},
        "islr": {
            "enabled": True,
            "window_duration_sec": 2.0,
            "hop_duration_sec": 0.5,
            "effective_fps": 15,
        },
        "cslr": {
            "enabled": True,
            "buffer_duration_sec": 3.0,
            "decode_interval_sec": 0.25,
        },
    }

    islr_cfg = get_online_runtime_config(config, "islr")
    cslr_cfg = get_online_runtime_config(config, "cslr")

    assert islr_cfg["hop_interval_sec"] == pytest.approx(0.5)
    assert islr_cfg["buffer_duration_sec"] == pytest.approx(2.5)
    assert islr_cfg["effective_fps"] == pytest.approx(15)
    assert cslr_cfg["hop_interval_sec"] == pytest.approx(0.25)
    assert cslr_cfg["buffer_duration_sec"] == pytest.approx(3.0)
    assert cslr_cfg["effective_fps"] == pytest.approx(15)


def test_setup_mps_fallback_is_idempotent(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    assert setup_mps_fallback() is True
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert setup_mps_fallback() is False


def test_smooth_joints_blends_previous_frame():
    current = np.ones((52, 2), dtype=np.float32)
    previous = np.zeros((52, 2), dtype=np.float32)

    blended = smooth_joints(current, previous, alpha=0.25)

    assert blended.shape == (52, 2)
    assert np.allclose(blended, 0.25)


def test_webcam_capture_rejects_invalid_downsample_factor():
    with pytest.raises(ValueError):
        WebcamCapture(downsample_factor=0)


def test_webcam_capture_start_clears_stale_buffers(monkeypatch):
    class _FakeCapture:
        def __init__(self, *_args, **_kwargs):
            self._opened = True

        def isOpened(self):
            return self._opened

        def set(self, *_args, **_kwargs):
            return True

        def release(self):
            self._opened = False

    class _FakeHolistic:
        def close(self):
            return None

    monkeypatch.setattr("asl_cslr.online.camera.cv2.VideoCapture", _FakeCapture)
    monkeypatch.setattr(
        "asl_cslr.online.camera.create_holistic_landmarker",
        lambda **_kwargs: _FakeHolistic(),
    )

    camera = WebcamCapture()
    camera.buffer.append(np.zeros(FEATURE_DIM, dtype=np.float32))
    camera.timestamps.append(1.0)

    camera.start()

    assert len(camera.buffer) == 0
    assert len(camera.timestamps) == 0

    camera.stop()


def test_webcam_capture_skips_mp_image_when_frame_is_not_submitted(monkeypatch):
    class _FakeCapture:
        def __init__(self, *_args, **_kwargs):
            self._opened = True
            self._frame = np.zeros((8, 8, 3), dtype=np.uint8)

        def isOpened(self):
            return self._opened

        def set(self, *_args, **_kwargs):
            return True

        def read(self):
            return True, self._frame.copy()

        def release(self):
            self._opened = False

    class _FakeHolistic:
        def close(self):
            return None

        def detect_async(self, *_args, **_kwargs):
            return None

    create_calls = []

    monkeypatch.setattr("asl_cslr.online.camera.cv2.VideoCapture", _FakeCapture)
    monkeypatch.setattr(
        "asl_cslr.online.camera.create_holistic_landmarker",
        lambda **_kwargs: _FakeHolistic(),
    )
    monkeypatch.setattr(
        "asl_cslr.online.camera.create_mp_image",
        lambda image: create_calls.append(image.shape) or object(),
    )

    camera = WebcamCapture(downsample_factor=2)
    camera.start()

    frame, raw_joints, skeleton, result_id = camera.read_frame()

    assert frame is not None
    assert raw_joints is None
    assert skeleton is None
    assert result_id == 0
    assert create_calls == []

    camera.stop()


def test_webcam_capture_does_not_reuse_stale_processed_result(monkeypatch):
    class _FakeCapture:
        def __init__(self, *_args, **_kwargs):
            self._opened = True
            self._frame = np.full((8, 8, 3), 17, dtype=np.uint8)

        def isOpened(self):
            return self._opened

        def set(self, *_args, **_kwargs):
            return True

        def read(self):
            return True, self._frame.copy()

        def release(self):
            self._opened = False

    class _FakeHolistic:
        def close(self):
            return None

        def detect_async(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr("asl_cslr.online.camera.cv2.VideoCapture", _FakeCapture)
    monkeypatch.setattr(
        "asl_cslr.online.camera.create_holistic_landmarker",
        lambda **_kwargs: _FakeHolistic(),
    )

    camera = WebcamCapture(downsample_factor=2)
    camera.start()
    camera._last_display_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    camera._last_raw_joints = np.zeros((52, 2), dtype=np.float32)
    camera._last_skeleton = np.zeros(FEATURE_DIM, dtype=np.float32)
    camera._last_result_id = 9
    camera._last_returned_result_id = 9

    frame, raw_joints, skeleton, result_id = camera.read_frame()

    assert result_id == 9
    assert raw_joints is None
    assert skeleton is None
    assert np.all(frame == 17)

    camera.stop()


def test_compute_motion_energy_rejects_static_buffer():
    static = _packed_buffer(np.zeros(8, dtype=np.float32))
    moving = _packed_buffer(np.linspace(0.0, 0.2, 8, dtype=np.float32))

    assert compute_motion_energy(static) == pytest.approx(0.0)
    assert compute_motion_energy(moving) > 0.0


def test_compute_motion_energy_ignores_imputed_gap_jumps():
    joints = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    observed = np.ones(NUM_JOINTS, dtype=np.float32)

    frame0 = build_feature_frame(joints, observed)
    frame1 = build_feature_frame(joints, observed)

    hidden = joints.copy()
    hidden[10, 0] = 1.0
    hidden_mask = observed.copy()
    hidden_mask[10] = 0.0
    frame2 = build_feature_frame(hidden, hidden_mask)

    reappeared = hidden.copy()
    frame3 = build_feature_frame(reappeared, observed)

    energy = compute_motion_energy(np.stack([frame0, frame1, frame2, frame3], axis=0))
    assert energy == pytest.approx(0.0)


def test_compute_motion_energy_requires_packed_frames():
    legacy = np.zeros((4, COORD_FEATURE_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match="packed 208-dim"):
        compute_motion_energy(legacy)


def test_prepare_online_features_adds_motion_when_requested():
    buffer = np.zeros((6, FEATURE_DIM), dtype=np.float32)
    observed = np.ones(NUM_JOINTS, dtype=np.float32)
    for t, x in enumerate(np.linspace(0.0, 0.2, 6, dtype=np.float32)):
        joints = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
        joints[0, 0] = x
        buffer[t] = build_feature_frame(joints, observed)

    pose_only = prepare_online_features(buffer, use_motion=False)
    pose_motion = prepare_online_features(buffer, use_motion=True)

    assert pose_only.shape == (6, FEATURE_DIM)
    assert pose_motion.shape == (6, FEATURE_DIM + COORD_FEATURE_DIM)
    assert not np.allclose(pose_motion[:, FEATURE_DIM:], 0.0)


def test_prepare_online_features_requires_packed_frames():
    legacy = np.zeros((4, COORD_FEATURE_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match="packed 208-dim"):
        prepare_online_features(legacy, use_motion=False)


def test_validate_online_model_schema_accepts_current_packed_contract():
    model = type(
        "Model",
        (),
        {
            "online_frame_feature_dim": FEATURE_DIM,
            "online_motion_dim": COORD_FEATURE_DIM,
            "online_use_motion": True,
        },
    )()

    validate_online_model_schema(model)


def test_validate_online_model_schema_rejects_unexpected_frame_dim():
    model = type(
        "Model",
        (),
        {
            "online_frame_feature_dim": COORD_FEATURE_DIM,
            "online_motion_dim": 0,
            "online_use_motion": False,
        },
    )()

    with pytest.raises(ValueError, match="frame_feature_dim"):
        validate_online_model_schema(model)


def test_validate_online_model_schema_rejects_unexpected_motion_dim():
    model = type(
        "Model",
        (),
        {
            "online_frame_feature_dim": FEATURE_DIM,
            "online_motion_dim": FEATURE_DIM,
            "online_use_motion": True,
        },
    )()

    with pytest.raises(ValueError, match="motion_dim"):
        validate_online_model_schema(model)


def test_load_online_cslr_model_rejects_dual_stream_checkpoint(tmp_path):
    vocab = build_vocab(["HELLO"])
    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)
    ckpt_path = tmp_path / "dual_stream.pt"
    torch.save(
        {
            "config": {
                "model": {
                    "type": "cslr",
                    "dual_stream": True,
                    "input_dim": FEATURE_DIM + COORD_FEATURE_DIM,
                    "frame_feature_dim": FEATURE_DIM,
                    "motion_dim": COORD_FEATURE_DIM,
                    "conv_dim": 32,
                    "conv_layers": 1,
                    "conv_kernel_size": 3,
                    "conv_dropout": 0.1,
                    "lstm_hidden_size": 16,
                    "lstm_layers": 1,
                    "lstm_dropout": 0.1,
                    "use_motion": True,
                }
            },
            "model_state_dict": {},
        },
        ckpt_path,
    )

    with pytest.raises(ValueError, match="dual-stream"):
        load_online_cslr_model(ckpt_path, vocab_path, torch.device("cpu"))


def test_loaded_online_cslr_model_exposes_motion_dim_from_checkpoint_config(tmp_path):
    vocab = build_vocab(["HELLO"])
    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)

    model = CSLRModel(
        input_dim=FEATURE_DIM + COORD_FEATURE_DIM,
        num_classes=len(vocab),
        conv_dim=16,
        conv_layers=1,
        conv_kernel_size=3,
        conv_dropout=0.0,
        lstm_hidden_size=8,
        lstm_layers=1,
        lstm_dropout=0.0,
    )
    ckpt_path = tmp_path / "bad_motion_dim.pt"
    torch.save(
        {
            "config": {
                "model": {
                    "type": "cslr",
                    "input_dim": FEATURE_DIM + COORD_FEATURE_DIM,
                    "frame_feature_dim": FEATURE_DIM,
                    "motion_dim": FEATURE_DIM,
                    "conv_dim": 16,
                    "conv_layers": 1,
                    "conv_kernel_size": 3,
                    "conv_dropout": 0.0,
                    "lstm_hidden_size": 8,
                    "lstm_layers": 1,
                    "lstm_dropout": 0.0,
                    "use_motion": True,
                }
            },
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    loaded_model, _vocab, _ckpt, _mcfg = load_online_cslr_model(
        ckpt_path,
        vocab_path,
        torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="motion_dim"):
        validate_online_model_schema(loaded_model)


def test_loaded_online_cslr_model_preserves_transformer_encoder(tmp_path):
    vocab = build_vocab(["HELLO"])
    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)

    model = CSLRModel(
        input_dim=FEATURE_DIM + COORD_FEATURE_DIM,
        num_classes=len(vocab),
        conv_dim=16,
        conv_layers=1,
        conv_kernel_size=3,
        conv_dropout=0.0,
        encoder_type="transformer",
        transformer_hidden=32,
        transformer_layers=2,
        transformer_heads=4,
        lstm_hidden_size=8,
        lstm_layers=1,
        lstm_dropout=0.1,
    )
    ckpt_path = tmp_path / "transformer.pt"
    torch.save(
        {
            "config": {
                "model": {
                    "type": "cslr",
                    "input_dim": FEATURE_DIM + COORD_FEATURE_DIM,
                    "frame_feature_dim": FEATURE_DIM,
                    "motion_dim": COORD_FEATURE_DIM,
                    "conv_dim": 16,
                    "conv_layers": 1,
                    "conv_kernel_size": 3,
                    "conv_dropout": 0.0,
                    "encoder_type": "transformer",
                    "transformer_hidden": 32,
                    "transformer_layers": 2,
                    "transformer_heads": 4,
                    "lstm_hidden_size": 8,
                    "lstm_layers": 1,
                    "lstm_dropout": 0.1,
                    "use_motion": True,
                }
            },
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    loaded_model, _vocab, _ckpt, mcfg = load_online_cslr_model(
        ckpt_path,
        vocab_path,
        torch.device("cpu"),
    )

    assert loaded_model.encoder_type == "transformer"
    assert isinstance(loaded_model.seq_encoder, TransformerSequenceEncoder)
    assert mcfg["encoder_type"] == "transformer"


def test_suppress_islr_special_logits_masks_reserved_classes(vocab):
    logits = torch.zeros((1, len(vocab)))
    logits[:, vocab.pad_idx] = 9.0
    logits[:, vocab.encode("HELLO")] = 3.0

    masked = suppress_islr_special_logits(logits, vocab)

    assert masked.argmax(dim=-1).item() == vocab.encode("HELLO")


class _DummyISLRModel(nn.Module):
    def __init__(self, num_classes: int, pred_idx: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes
        self.pred_idx = pred_idx

    def forward(self, x, lengths=None):
        logits = torch.full((x.shape[0], self.num_classes), -6.0, device=x.device)
        logits[:, self.pred_idx] = 6.0 + self.bias
        return logits


class _DummyCSLRModel(nn.Module):
    def __init__(self, num_classes: int, pred_idx: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes
        self.pred_idx = pred_idx

    def forward(self, x, lengths=None):
        logits = torch.full(
            (x.shape[0], x.shape[1], self.num_classes),
            -6.0,
            device=x.device,
        )
        logits[..., self.pred_idx] = 6.0 + self.bias
        return torch.log_softmax(logits, dim=-1)

    def greedy_decode(self, log_probs, lengths=None, ignore_ids=None):
        predictions = log_probs.argmax(dim=-1)
        decoded = []
        ignore_ids = {0} if ignore_ids is None else set(ignore_ids) | {0}
        for b in range(predictions.size(0)):
            seq = predictions[b]
            if lengths is not None:
                seq = seq[: lengths[b].item()]
            seq = seq.tolist()
            collapsed = []
            prev = -1
            for idx in seq:
                if idx != prev:
                    collapsed.append(idx)
                prev = idx
            collapsed = [idx for idx in collapsed if idx not in ignore_ids]
            decoded.append(collapsed)
        return decoded


class _QueuedCSLRModel(nn.Module):
    def __init__(self, num_classes: int, decoded_sequences: list[list[int]], blank_idx: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes
        self.decoded_sequences = [list(sequence) for sequence in decoded_sequences]
        self.blank_idx = blank_idx
        self._call_idx = -1

    def forward(self, x, lengths=None):
        self._call_idx = min(self._call_idx + 1, len(self.decoded_sequences) - 1)
        logits = torch.full(
            (x.shape[0], x.shape[1], self.num_classes),
            -6.0,
            device=x.device,
        )
        logits[..., self.blank_idx] = -4.0
        sequence = self.decoded_sequences[self._call_idx]
        if sequence:
            logits[..., sequence[0]] = 4.0 + self.bias
        return torch.log_softmax(logits, dim=-1)

    def greedy_decode(self, log_probs, lengths=None, ignore_ids=None):
        sequence = list(self.decoded_sequences[self._call_idx])
        return [sequence for _ in range(log_probs.shape[0])]


def test_sliding_window_islr_gates_static_buffer(vocab):
    pred_idx = vocab.encode("HELLO")
    model = _DummyISLRModel(len(vocab), pred_idx)
    pipeline = SlidingWindowISLR(
        model=model,
        vocab=vocab,
        effective_fps=15,
        window_duration_sec=1.0,
        hop_duration_sec=0.5,
        stability_windows=1,
        confidence_threshold=0.5,
        confidence_margin_threshold=0.1,
        motion_energy_threshold=0.01,
        min_buffer_frames=4,
    )

    static = _packed_buffer(np.zeros(12, dtype=np.float32))
    moving = _packed_buffer(np.linspace(0.0, 0.25, 12, dtype=np.float32))

    assert pipeline.process_buffer(static) is None
    assert pipeline.process_buffer(moving) == "HELLO"


def test_streaming_cslr_gates_static_buffer(vocab):
    pred_idx = vocab.encode("HELLO")
    model = _DummyCSLRModel(len(vocab), pred_idx)
    pipeline = StreamingCSLR(
        model=model,
        vocab=vocab,
        decode_interval_sec=0.0,
        effective_fps=15,
        stability_windows=1,
        motion_energy_threshold=0.01,
        blank_rejection_threshold=0.9,
        min_buffer_frames=4,
    )

    static = _packed_buffer(np.zeros(12, dtype=np.float32))
    moving = _packed_buffer(np.linspace(0.0, 0.25, 12, dtype=np.float32))

    assert pipeline.process_buffer(static) is None
    assert pipeline.process_buffer(moving) == ["HELLO"]


def test_streaming_cslr_commits_stable_multiword_prefix(vocab):
    hello = vocab.encode("HELLO")
    please = vocab.encode("PLEASE")
    model = _QueuedCSLRModel(
        len(vocab),
        decoded_sequences=[
            [hello],
            [hello, please],
            [hello, please],
        ],
        blank_idx=vocab.blank_idx,
    )
    pipeline = StreamingCSLR(
        model=model,
        vocab=vocab,
        decode_interval_sec=0.0,
        effective_fps=15,
        stability_windows=2,
        history_size=4,
        motion_energy_threshold=0.01,
        blank_rejection_threshold=0.9,
        min_buffer_frames=4,
    )

    moving = _packed_buffer(np.linspace(0.0, 0.25, 12, dtype=np.float32))

    assert pipeline.process_buffer(moving) is None
    assert pipeline.get_output() == []
    assert pipeline.process_buffer(moving) == ["HELLO"]
    assert pipeline.get_output() == ["HELLO"]
    assert pipeline.process_buffer(moving) == ["HELLO", "PLEASE"]
    assert pipeline.get_output() == ["HELLO", "PLEASE"]


def test_streaming_cslr_merges_committed_prefix_when_buffer_rolls(vocab):
    hello = vocab.encode("HELLO")
    please = vocab.encode("PLEASE")
    model = _QueuedCSLRModel(
        len(vocab),
        decoded_sequences=[
            [hello],
            [please],
            [please],
        ],
        blank_idx=vocab.blank_idx,
    )
    pipeline = StreamingCSLR(
        model=model,
        vocab=vocab,
        decode_interval_sec=0.0,
        effective_fps=15,
        stability_windows=2,
        history_size=4,
        motion_energy_threshold=0.01,
        blank_rejection_threshold=0.9,
        min_buffer_frames=4,
    )

    moving = _packed_buffer(np.linspace(0.0, 0.25, 12, dtype=np.float32))

    assert pipeline.process_buffer(moving) is None
    assert pipeline.process_buffer(moving) == ["HELLO"]
    assert pipeline.process_buffer(moving) == ["HELLO", "PLEASE"]
    assert pipeline.get_output() == ["HELLO", "PLEASE"]


def test_streaming_cslr_does_not_keep_one_off_middle_word(vocab):
    hello = vocab.encode("HELLO")
    drink = vocab.encode("THANK-YOU")
    like = vocab.encode("PLEASE")
    model = _QueuedCSLRModel(
        len(vocab),
        decoded_sequences=[
            [hello],
            [hello],
            [drink],
            [like],
            [like],
            [like],
        ],
        blank_idx=vocab.blank_idx,
    )
    pipeline = StreamingCSLR(
        model=model,
        vocab=vocab,
        decode_interval_sec=0.0,
        effective_fps=15,
        stability_windows=2,
        history_size=4,
        motion_energy_threshold=0.01,
        blank_rejection_threshold=0.9,
        min_buffer_frames=4,
    )

    moving = _packed_buffer(np.linspace(0.0, 0.25, 12, dtype=np.float32))

    assert pipeline.process_buffer(moving) is None
    assert pipeline.process_buffer(moving) == ["HELLO"]
    assert pipeline.process_buffer(moving) is None
    assert pipeline.get_output() == ["HELLO"]
    assert pipeline.process_buffer(moving) is None
    assert pipeline.process_buffer(moving) == ["HELLO", "PLEASE"]
    assert pipeline.get_output() == ["HELLO", "PLEASE"]


def test_streaming_cslr_flushes_tail_after_pause(vocab):
    hello = vocab.encode("HELLO")
    model = _QueuedCSLRModel(
        len(vocab),
        decoded_sequences=[
            [hello],
            [hello],
        ],
        blank_idx=vocab.blank_idx,
    )
    pipeline = StreamingCSLR(
        model=model,
        vocab=vocab,
        decode_interval_sec=0.0,
        effective_fps=15,
        stability_windows=3,
        history_size=4,
        motion_energy_threshold=0.01,
        blank_rejection_threshold=0.9,
        min_buffer_frames=4,
        inactivity_reset_windows=2,
        pause_commit_windows=2,
    )

    moving = _packed_buffer(np.linspace(0.0, 0.25, 12, dtype=np.float32))
    static = _packed_buffer(np.zeros(12, dtype=np.float32))

    assert pipeline.process_buffer(moving) is None
    assert pipeline.process_buffer(moving) is None
    assert pipeline.get_output() == []
    assert pipeline.process_buffer(static) is None
    assert pipeline.process_buffer(static) is None
    assert pipeline.get_output() == ["HELLO"]


def test_streaming_cslr_stable_buffer_mode_replaces_sequence_without_duplicates(vocab):
    hello = vocab.encode("HELLO")
    please = vocab.encode("PLEASE")
    model = _QueuedCSLRModel(
        len(vocab),
        decoded_sequences=[
            [hello],
            [hello],
            [hello, please],
            [hello, please],
            [please],
            [please],
        ],
        blank_idx=vocab.blank_idx,
    )
    pipeline = StreamingCSLR(
        model=model,
        vocab=vocab,
        decode_interval_sec=0.0,
        effective_fps=15,
        stability_windows=2,
        history_size=4,
        motion_energy_threshold=0.01,
        blank_rejection_threshold=0.9,
        min_buffer_frames=4,
        cumulative_commits=False,
    )

    moving = _packed_buffer(np.linspace(0.0, 0.25, 12, dtype=np.float32))

    assert pipeline.process_buffer(moving) == ["HELLO"]
    assert pipeline.process_buffer(moving) is None
    assert pipeline.process_buffer(moving) is None
    assert pipeline.process_buffer(moving) == ["HELLO", "PLEASE"]
    assert pipeline.process_buffer(moving) is None
    assert pipeline.process_buffer(moving) == ["PLEASE"]
    assert pipeline.get_output() == ["PLEASE"]
