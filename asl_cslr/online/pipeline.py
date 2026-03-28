"""
Online inference pipelines (§9.2, §9.3).

SlidingWindowISLR: sliding-window ISLR with stability voting.
StreamingCSLR: rolling-buffer CSLR with streaming CTC decode.
"""

import time
import logging
from collections import deque

import numpy as np
import torch

from asl_cslr.data.skeleton import (
    COORD_FEATURE_DIM,
    FEATURE_DIM,
    NUM_JOINTS,
    NUM_COORDS,
    JOINT_FEATURES,
    compute_motion_features,
)
from asl_cslr.data.vocab import GlossVocab
from asl_cslr.models.cslr_model import suppress_ctc_special_tokens
from asl_cslr.utils.device import get_autocast_context

logger = logging.getLogger(__name__)

ONLINE_MODES = ("cslr", "islr")


def _require_packed_online_buffer(buffer: np.ndarray) -> np.ndarray:
    """Validate that the live runtime buffer uses packed xyz+mask frames."""
    array = np.asarray(buffer, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D online buffer, got shape={array.shape}")
    if array.shape[1] != FEATURE_DIM:
        raise ValueError(
            "Online runtime expects packed 208-dim xyz+mask frames; "
            f"got width={array.shape[1]}"
        )
    return array


def validate_online_model_schema(model) -> None:
    """Fail fast when a checkpoint declares an unsupported live feature schema."""
    expected_frame_dim = int(getattr(model, "online_frame_feature_dim", FEATURE_DIM))
    if expected_frame_dim != FEATURE_DIM:
        raise ValueError(
            "Online runtime currently supports packed 208-dim xyz+mask frames only; "
            f"checkpoint expects frame_feature_dim={expected_frame_dim}"
        )

    use_motion = bool(getattr(model, "online_use_motion", False))
    expected_motion_dim = int(
        getattr(
            model,
            "online_motion_dim",
            COORD_FEATURE_DIM if use_motion else 0,
        )
    )
    if use_motion and expected_motion_dim != COORD_FEATURE_DIM:
        raise ValueError(
            "Online runtime currently supports computed 156-dim coordinate velocity "
            f"only; checkpoint expects motion_dim={expected_motion_dim}"
        )


def suppress_islr_special_logits(
    logits: torch.Tensor,
    vocab: GlossVocab,
) -> torch.Tensor:
    """Exclude reserved vocabulary entries from live ISLR classification."""
    masked = logits.clone()
    special_indices = vocab.special_indices(include_blank=True)
    if special_indices:
        masked[:, special_indices] = torch.finfo(masked.dtype).min
    return masked


def compute_motion_energy(buffer: np.ndarray, window: int = 8) -> float:
    """Measure frame-to-frame motion in the last portion of a skeleton buffer."""
    if buffer is None or len(buffer) < 2:
        return 0.0

    recent = _require_packed_online_buffer(buffer[-max(2, window):])
    velocity = compute_motion_features(recent)["velocity"]
    if velocity.shape[0] < 2:
        return 0.0

    flat = velocity[1:].reshape(velocity.shape[0] - 1, -1)
    return float(np.mean(np.linalg.norm(flat, axis=1)))


def suppress_idle_hand(
    buffer: np.ndarray,
    motion_threshold: float = 0.003,
) -> np.ndarray:
    """Zero out hand features when the hand has very low motion energy.

    In training data, one-handed signs often have the non-signing hand as all
    zeros (undetected by MediaPipe). During live inference both hands are
    visible, so the idle hand has non-zero coordinates. This function detects
    idle hands and zeros them out (coords and observation mask) so the model
    sees the same distribution it was trained on.

    Args:
        buffer: (T, 208) packed xyz+mask frames.
        motion_threshold: Per-joint mean velocity magnitude below which a hand
            is considered idle.

    Returns:
        Buffer with idle hand joints zeroed out.
    """
    if buffer is None or len(buffer) < 3:
        return buffer

    buf = _require_packed_online_buffer(buffer)
    T = buf.shape[0]
    packed = buf.reshape(T, NUM_JOINTS, JOINT_FEATURES)

    # Extract per-hand coordinate slices: left hand joints 10-30, right 31-51
    left_coords = packed[:, 10:31, :NUM_COORDS]   # (T, 21, 3)
    right_coords = packed[:, 31:52, :NUM_COORDS]   # (T, 21, 3)

    def _hand_motion_energy(coords: np.ndarray) -> float:
        if coords.shape[0] < 2:
            return float("inf")
        velocity = np.diff(coords, axis=0)  # (T-1, 21, 3)
        per_joint_speed = np.linalg.norm(velocity, axis=2)  # (T-1, 21)
        return float(np.mean(per_joint_speed))

    left_energy = _hand_motion_energy(left_coords)
    right_energy = _hand_motion_energy(right_coords)

    # Only suppress if one hand is idle and the other is active
    left_idle = left_energy < motion_threshold
    right_idle = right_energy < motion_threshold

    if left_idle == right_idle:
        # Both idle or both active: don't suppress
        return buffer

    result = buf.copy()
    packed_out = result.reshape(T, NUM_JOINTS, JOINT_FEATURES)

    if left_idle:
        packed_out[:, 10:31, :] = 0.0
    if right_idle:
        packed_out[:, 31:52, :] = 0.0

    return packed_out.reshape(T, FEATURE_DIM)


def longest_common_gloss_prefix(sequences: list[list[str]]) -> list[str]:
    """Return the longest shared prefix across decoded gloss sequences."""
    if not sequences:
        return []

    prefix = list(sequences[0])
    for sequence in sequences[1:]:
        shared = 0
        limit = min(len(prefix), len(sequence))
        while shared < limit and prefix[shared] == sequence[shared]:
            shared += 1
        prefix = prefix[:shared]
        if not prefix:
            break

    return prefix


def merge_committed_glosses(
    committed: list[str],
    decoded: list[str],
) -> list[str]:
    """Merge a rolling-buffer decode with the already committed prefix."""
    if not committed:
        return list(decoded)
    if not decoded:
        return list(committed)

    max_overlap = min(len(committed), len(decoded))
    for overlap in range(max_overlap, 0, -1):
        if committed[-overlap:] == decoded[:overlap]:
            return list(committed) + list(decoded[overlap:])

    return list(committed) + list(decoded)


def prepare_online_features(
    buffer: np.ndarray,
    use_motion: bool,
    idle_hand_suppression: bool = True,
    idle_hand_threshold: float = 0.003,
) -> np.ndarray:
    """Match live features to the checkpoint's expected input representation."""
    sequence = _require_packed_online_buffer(buffer)

    if idle_hand_suppression:
        sequence = suppress_idle_hand(sequence, motion_threshold=idle_hand_threshold)

    if not use_motion:
        return sequence

    motion = compute_motion_features(sequence)
    return np.concatenate([sequence, motion["velocity"].astype(np.float32)], axis=1)


def resolve_online_mode(config: dict, requested_mode: str | None = None) -> str:
    """Resolve the online inference mode from CLI input and config defaults."""
    if requested_mode is not None:
        if requested_mode not in ONLINE_MODES:
            raise ValueError(f"Unsupported online mode: {requested_mode}")
        return requested_mode

    default_mode = config.get("default_mode")
    if default_mode in ONLINE_MODES:
        return default_mode

    enabled_modes = [
        mode for mode in ONLINE_MODES
        if config.get(mode, {}).get("enabled", False)
    ]
    if not enabled_modes:
        raise ValueError("No online inference mode is enabled in the config")

    return enabled_modes[0]


def get_online_runtime_config(config: dict, mode: str) -> dict:
    """Compute runtime settings for the selected online mode."""
    if mode not in ONLINE_MODES:
        raise ValueError(f"Unsupported online mode: {mode}")

    camera_cfg = config.get("camera", {})
    mode_cfg = config.get(mode, {})
    effective_fps = mode_cfg.get(
        "effective_fps",
        camera_cfg.get("capture_fps", 30) / max(camera_cfg.get("downsample_factor", 1), 1),
    )

    if mode == "islr":
        hop_interval_sec = mode_cfg.get("hop_duration_sec", 0.5)
        buffer_duration_sec = mode_cfg.get(
            "buffer_duration_sec",
            mode_cfg.get("window_duration_sec", 2.0) + hop_interval_sec,
        )
    else:
        hop_interval_sec = mode_cfg.get("decode_interval_sec", 0.5)
        buffer_duration_sec = mode_cfg.get("buffer_duration_sec", 4.0)

    return {
        "effective_fps": effective_fps,
        "hop_interval_sec": hop_interval_sec,
        "buffer_duration_sec": buffer_duration_sec,
    }


class SlidingWindowISLR:
    """Sliding-window online recognition using ISLR model (§9.2).

    Feeds overlapping windows of skeleton frames to the ISLR model,
    accumulates predictions, and emits stable glosses.

    Args:
        model: Loaded ISLRModel in eval mode.
        vocab: GlossVocab instance.
        effective_fps: Effective frame rate after downsampling.
        window_duration_sec: Window size in seconds.
        hop_duration_sec: Hop between windows in seconds.
        stability_windows: Consecutive windows required for stable prediction.
        confidence_threshold: Minimum softmax confidence to consider.
    """

    def __init__(
        self,
        model,
        vocab: GlossVocab,
        effective_fps: float = 15.0,
        window_duration_sec: float = 2.0,
        hop_duration_sec: float = 0.5,
        stability_windows: int = 2,
        confidence_threshold: float = 0.6,
        confidence_margin_threshold: float = 0.08,
        motion_energy_threshold: float = 0.01,
        min_buffer_frames: int = 8,
    ):
        self.model = model
        self.vocab = vocab
        self.device = next(model.parameters()).device

        self.window_frames = max(1, int(round(effective_fps * window_duration_sec)))
        self.hop_frames = max(1, int(round(effective_fps * hop_duration_sec)))
        self.stability_windows = stability_windows
        self.confidence_threshold = confidence_threshold
        self.confidence_margin_threshold = confidence_margin_threshold
        self.motion_energy_threshold = motion_energy_threshold
        self.min_buffer_frames = min_buffer_frames

        # Prediction history
        self.predictions = deque(maxlen=20)
        self.emitted_glosses = []
        self._frame_count = 0
        self._last_emitted = None

    def process_buffer(self, buffer: np.ndarray) -> str | None:
        """Process the current skeleton buffer and maybe emit a gloss.

        Should be called at each hop interval with the latest buffer.

        Args:
            buffer: (T, 104) skeleton frames from the webcam buffer.

        Returns:
            Emitted gloss string if stable prediction reached, else None.
        """
        if buffer is None or len(buffer) < max(self.window_frames // 2, self.min_buffer_frames):
            return None

        # Extract window
        window = buffer[-self.window_frames:]
        if compute_motion_energy(window) < self.motion_energy_threshold:
            return None

        features = prepare_online_features(
            window,
            use_motion=bool(getattr(self.model, "online_use_motion", False)),
        )
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        lengths = torch.tensor([window.shape[0]], dtype=torch.long).to(self.device)

        # Inference
        self.model.eval()
        with torch.no_grad(), get_autocast_context(self.device, enabled=self.device.type != "cpu"):
            logits = self.model(x, lengths)
            logits = suppress_islr_special_logits(logits, self.vocab)
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_idx = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
            confidence = top_probs[..., 0]
            pred_idx = top_idx[..., 0]
            confidence_margin = (
                top_probs[..., 0] - top_probs[..., 1]
                if top_probs.shape[-1] > 1
                else top_probs[..., 0]
            )

        pred_gloss = self.vocab.decode(pred_idx.item())
        conf = confidence.item()
        margin = confidence_margin.item()

        self.predictions.append((pred_gloss, conf, margin))

        # Check stability
        if conf >= self.confidence_threshold and margin >= self.confidence_margin_threshold:
            recent = list(self.predictions)[-self.stability_windows:]
            if len(recent) >= self.stability_windows:
                dominant = all(p[0] == pred_gloss for p in recent)
                high_conf = all(
                    p[1] >= self.confidence_threshold
                    and p[2] >= self.confidence_margin_threshold
                    for p in recent
                )

                if dominant and high_conf and pred_gloss != self._last_emitted:
                    self._last_emitted = pred_gloss
                    self.emitted_glosses.append(pred_gloss)
                    logger.info(
                        "Emitted: %s (conf=%.3f, margin=%.3f, motion=%.4f)",
                        pred_gloss,
                        conf,
                        margin,
                        compute_motion_energy(window),
                    )
                    return pred_gloss

        return None

    def get_output(self) -> list[str]:
        """Get all emitted glosses so far."""
        return list(self.emitted_glosses)

    def reset(self):
        """Reset prediction state."""
        self.predictions.clear()
        self.emitted_glosses.clear()
        self._last_emitted = None


class StreamingCSLR:
    """Streaming CSLR with rolling buffer and periodic CTC decode (§9.3).

    Periodically feeds the entire rolling buffer to the CSLR model and
    tracks changes in the decoded gloss sequence.

    Args:
        model: Loaded CSLRModel in eval mode.
        vocab: GlossVocab instance.
        decode_interval_sec: How often to run CTC decode.
        effective_fps: Effective frame rate.
    """

    def __init__(
        self,
        model,
        vocab: GlossVocab,
        decode_interval_sec: float = 0.5,
        effective_fps: float = 15.0,
        stability_windows: int = 3,
        history_size: int = 6,
        motion_energy_threshold: float = 0.008,
        blank_rejection_threshold: float = 0.88,
        min_buffer_frames: int = 8,
        inactivity_reset_windows: int = 3,
        pause_commit_windows: int | None = None,
        cumulative_commits: bool = True,
    ):
        self.model = model
        self.vocab = vocab
        self.device = next(model.parameters()).device

        self.decode_interval = decode_interval_sec
        self.effective_fps = effective_fps
        self.stability_windows = max(1, stability_windows)
        self.history_size = max(self.stability_windows, history_size)
        self.motion_energy_threshold = motion_energy_threshold
        self.blank_rejection_threshold = blank_rejection_threshold
        self.min_buffer_frames = min_buffer_frames
        self.inactivity_reset_windows = max(1, inactivity_reset_windows)
        self.cumulative_commits = bool(cumulative_commits)
        self.pause_commit_windows = max(
            1,
            pause_commit_windows
            if pause_commit_windows is not None
            else max(1, self.stability_windows - 1),
        )

        self._last_decode_time = 0.0
        self._inactive_windows = 0
        self.decode_history = deque(maxlen=self.history_size)
        self.committed_sequence = []
        self.current_sequence = []

    def _finalize_unstable_tail(self):
        """Drop unstable history after a confident pause while keeping commits."""
        if not self.cumulative_commits:
            if len(self.decode_history) >= self.pause_commit_windows:
                recent = list(self.decode_history)[-self.pause_commit_windows:]
                prefix = longest_common_gloss_prefix(recent)
                if prefix:
                    self.current_sequence = list(prefix)
            self.decode_history.clear()
            return

        if len(self.decode_history) >= self.pause_commit_windows:
            recent = list(self.decode_history)[-self.pause_commit_windows:]
            prefix = longest_common_gloss_prefix(recent)
            if len(prefix) > len(self.committed_sequence):
                self.committed_sequence = list(prefix)
        self.decode_history.clear()
        self.current_sequence = list(self.committed_sequence)

    def process_buffer(self, buffer: np.ndarray) -> list[str] | None:
        """Process the buffer and return updated gloss sequence if changed.

        Args:
            buffer: (T, 104) full rolling buffer.

        Returns:
            Updated gloss sequence if changed, else None.
        """
        if buffer is None or len(buffer) < self.min_buffer_frames:
            return None

        motion_energy = compute_motion_energy(buffer)
        if motion_energy < self.motion_energy_threshold:
            self._inactive_windows += 1
            if self._inactive_windows >= self.inactivity_reset_windows:
                self._finalize_unstable_tail()
            return None
        self._inactive_windows = 0

        now = time.time()
        if now - self._last_decode_time < self.decode_interval:
            return None

        self._last_decode_time = now

        # Run CSLR model
        features = prepare_online_features(
            buffer,
            use_motion=bool(getattr(self.model, "online_use_motion", False)),
        )
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        lengths = torch.tensor([buffer.shape[0]], dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad(), get_autocast_context(self.device, enabled=self.device.type != "cpu"):
            log_probs = self.model(x, lengths)
        log_probs = suppress_ctc_special_tokens(
            log_probs,
            self.vocab.special_indices(include_blank=False),
        )
        probs = torch.softmax(log_probs, dim=-1)
        blank_prob = probs[..., self.vocab.blank_idx].mean().item()

        # Greedy CTC decode
        ignore_ids = set(self.vocab.special_indices(include_blank=False))
        decoded_ids = self.model.greedy_decode(
            log_probs,
            lengths=lengths,
            ignore_ids=ignore_ids,
        )[0]
        decoded_glosses = self.vocab.decode_sequence(
            decoded_ids,
            skip_special=True,
        )

        if not decoded_glosses:
            if blank_prob >= self.blank_rejection_threshold:
                self._inactive_windows += 1
                if self._inactive_windows >= self.inactivity_reset_windows:
                    self._finalize_unstable_tail()
                return None
            return None
        self._inactive_windows = 0

        if not self.cumulative_commits:
            self.decode_history.append(list(decoded_glosses))
            if len(self.decode_history) >= self.stability_windows:
                recent = list(self.decode_history)[-self.stability_windows:]
                stable_sequence = longest_common_gloss_prefix(recent)
            else:
                stable_sequence = list(decoded_glosses)

            if not stable_sequence and self.current_sequence:
                return None

            if stable_sequence != self.current_sequence:
                self.current_sequence = list(stable_sequence)
                logger.info(
                    "CSLR stable buffer sequence: %s (blank=%.3f, motion=%.4f)",
                    " ".join(stable_sequence) if stable_sequence else "<none>",
                    blank_prob,
                    motion_energy,
                )
                return list(self.current_sequence)
            return None

        if self.committed_sequence:
            base_sequence = list(self.committed_sequence)
        elif self.decode_history:
            # Bootstrap the first committed word across the initial rolling decodes.
            base_sequence = list(self.decode_history[-1])
        else:
            base_sequence = []
        merged_sequence = merge_committed_glosses(
            base_sequence,
            decoded_glosses,
        )
        self.decode_history.append(merged_sequence)

        stable_sequence = list(self.committed_sequence)
        if len(self.decode_history) >= self.stability_windows:
            recent = list(self.decode_history)[-self.stability_windows:]
            prefix = longest_common_gloss_prefix(recent)
            if len(prefix) >= len(self.committed_sequence):
                stable_sequence = prefix

        if stable_sequence != self.current_sequence:
            self.current_sequence = list(stable_sequence)
            logger.info(
                "CSLR stable sequence: %s (blank=%.3f, motion=%.4f)",
                " ".join(stable_sequence) if stable_sequence else "<none>",
                blank_prob,
                motion_energy,
            )

        if len(stable_sequence) > len(self.committed_sequence):
            new_words = stable_sequence[len(self.committed_sequence):]
            self.committed_sequence = list(stable_sequence)
            logger.info(
                "CSLR committed: %s",
                " ".join(new_words),
            )
            return list(self.current_sequence)

        return None

    def get_output(self) -> list[str]:
        """Get the current decoded gloss sequence."""
        return list(self.current_sequence)

    def reset(self):
        """Reset state."""
        self.decode_history.clear()
        self.committed_sequence = []
        self.current_sequence = []
        self._last_decode_time = 0.0
        self._inactive_windows = 0
