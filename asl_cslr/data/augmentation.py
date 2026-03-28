"""
Skeleton-level data augmentation for training robustness.

All transforms operate on per-frame skeleton features and preserve the explicit
observed/imputed joint mask channel when present.
They are applied at training time only.
"""

import numpy as np

from .skeleton import (
    NUM_JOINTS,
    NUM_COORDS,
    FEATURE_DIM,
    COORD_FEATURE_DIM,
    LEGACY_XY_COORD_DIM,
    JOINT_FEATURES,
)


class SkeletonAugmentor:
    """Composable skeleton augmentation pipeline.

    Applies a sequence of random transforms to skeleton sequences.
    All transforms accept and return arrays of shape (T, D).

    Args:
        spatial_jitter_std: Std of Gaussian noise added to joint coords.
        scale_range: (min, max) random scale perturbation.
        rotation_range_deg: (min, max) roll rotation around the camera axis.
        pitch_range_deg: (min, max) 3D pitch rotation in degrees.
        yaw_range_deg: (min, max) 3D yaw rotation in degrees.
        translate_range: Max random translation magnitude.
        temporal_crop_ratio: (min, max) fraction of sequence to keep.
        temporal_drop_ratio: Fraction of frames to drop uniformly at random.
        flip_prob: Probability of horizontal (left-right) flip.
        allow_horizontal_flip: Whether horizontal mirroring is semantically safe.
        joint_dropout_prob: Probability that each joint briefly reuses the
            previous tracked position per frame.
        hand_dropout_prob: Probability of hiding one or both hands for a
            contiguous time span.
        hand_dropout_ratio: Fraction of the sequence affected when hand dropout
            triggers.
        pose_dropout_prob: Probability of hiding the pose joints for a
            contiguous time span.
        pose_dropout_ratio: Fraction of the sequence affected when pose dropout
            triggers.
        speed_perturb_range: (min, max) speed factor for temporal stretching.
        idle_hand_inject_prob: Probability of injecting a plausible idle hand
            when the sequence has one hand entirely undetected. This bridges the
            gap between training data (where the non-signing hand is missing) and
            live inference (where both hands are visible but one is idle).
        enabled: Whether augmentation is active (set to False for eval).
    """

    def __init__(
        self,
        spatial_jitter_std: float = 0.01,
        scale_range: tuple[float, float] = (0.9, 1.1),
        rotation_range_deg: tuple[float, float] = (0.0, 0.0),
        pitch_range_deg: tuple[float, float] = (0.0, 0.0),
        yaw_range_deg: tuple[float, float] = (0.0, 0.0),
        translate_range: float = 0.05,
        temporal_crop_ratio: tuple[float, float] = (0.8, 1.0),
        temporal_drop_ratio: float = 0.0,
        flip_prob: float = 0.0,
        allow_horizontal_flip: bool = False,
        joint_dropout_prob: float = 0.05,
        hand_dropout_prob: float = 0.0,
        hand_dropout_ratio: tuple[float, float] = (0.08, 0.25),
        pose_dropout_prob: float = 0.0,
        pose_dropout_ratio: tuple[float, float] = (0.05, 0.18),
        speed_perturb_range: tuple[float, float] = (0.8, 1.2),
        idle_hand_inject_prob: float = 0.0,
        enabled: bool = True,
    ):
        self.spatial_jitter_std = spatial_jitter_std
        self.scale_range = scale_range
        self.rotation_range_deg = rotation_range_deg
        self.pitch_range_deg = pitch_range_deg
        self.yaw_range_deg = yaw_range_deg
        self.translate_range = translate_range
        self.temporal_crop_ratio = temporal_crop_ratio
        self.temporal_drop_ratio = temporal_drop_ratio
        self.flip_prob = flip_prob
        self.allow_horizontal_flip = allow_horizontal_flip
        self.joint_dropout_prob = joint_dropout_prob
        self.hand_dropout_prob = hand_dropout_prob
        self.hand_dropout_ratio = hand_dropout_ratio
        self.pose_dropout_prob = pose_dropout_prob
        self.pose_dropout_ratio = pose_dropout_ratio
        self.speed_perturb_range = speed_perturb_range
        self.idle_hand_inject_prob = idle_hand_inject_prob
        self.enabled = enabled

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline.

        Args:
            sequence: Skeleton sequence of shape (T, D).

        Returns:
            Augmented sequence of shape (T', D).
        """
        if not self.enabled or sequence.shape[0] == 0:
            return sequence

        coords, observed_mask, layout = self._split_sequence(sequence)

        # Order matters: spatial ops on coords, then temporal ops
        coords, observed_mask = self._spatial_jitter(coords, observed_mask)
        coords, observed_mask = self._random_scale(coords, observed_mask)
        coords, observed_mask = self._random_view_rotate(coords, observed_mask)
        coords, observed_mask = self._random_translate(coords, observed_mask)
        if self.allow_horizontal_flip:
            coords, observed_mask = self._horizontal_flip(coords, observed_mask)
        coords, observed_mask = self._temporal_crop(coords, observed_mask)
        coords, observed_mask = self._temporal_drop(coords, observed_mask)
        coords, observed_mask = self._speed_perturb(coords, observed_mask)
        coords, observed_mask = self._joint_dropout(coords, observed_mask)
        coords, observed_mask = self._hand_dropout(coords, observed_mask)
        coords, observed_mask = self._pose_dropout(coords, observed_mask)
        coords, observed_mask = self._idle_hand_inject(coords, observed_mask)

        return self._pack_sequence(coords, observed_mask, layout)

    def _split_sequence(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Split a sequence into coordinates and observed-mask channels."""
        x = np.asarray(x, dtype=np.float32)
        T = x.shape[0]
        if x.shape[1] == FEATURE_DIM:
            packed = x.reshape(T, NUM_JOINTS, JOINT_FEATURES)
            return (
                packed[..., :NUM_COORDS].copy(),
                packed[..., NUM_COORDS].copy(),
                "packed",
            )
        if x.shape[1] == COORD_FEATURE_DIM:
            return (
                x.reshape(T, NUM_JOINTS, NUM_COORDS).copy(),
                np.ones((T, NUM_JOINTS), dtype=np.float32),
                "coords3d",
            )
        if x.shape[1] == LEGACY_XY_COORD_DIM:
            coords = np.zeros((T, NUM_JOINTS, NUM_COORDS), dtype=np.float32)
            coords[..., :2] = x.reshape(T, NUM_JOINTS, 2)
            return coords, np.ones((T, NUM_JOINTS), dtype=np.float32), "legacy_xy"
        raise ValueError(f"Unsupported feature width for augmentation: {x.shape[1]}")

    def _pack_sequence(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
        layout: str,
    ) -> np.ndarray:
        """Restore a sequence to its original feature layout."""
        if layout == "packed":
            packed = np.concatenate(
                [coords, observed_mask[..., None]],
                axis=2,
            )
            return packed.reshape(coords.shape[0], FEATURE_DIM).astype(np.float32, copy=False)
        if layout == "coords3d":
            return coords.reshape(coords.shape[0], COORD_FEATURE_DIM).astype(np.float32, copy=False)
        if layout == "legacy_xy":
            return coords[..., :2].reshape(coords.shape[0], LEGACY_XY_COORD_DIM).astype(np.float32, copy=False)
        raise ValueError(f"Unknown augmentation layout: {layout}")

    def _spatial_jitter(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Add small Gaussian noise to all joint coordinates."""
        if self.spatial_jitter_std > 0:
            noise = np.random.randn(*coords.shape).astype(np.float32)
            coords = coords + noise * self.spatial_jitter_std
        return coords, observed_mask

    def _random_scale(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly scale all coords by a uniform factor."""
        lo, hi = self.scale_range
        if lo < hi:
            scale = np.random.uniform(lo, hi)
            coords = coords * scale
        return coords, observed_mask

    def _random_translate(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly translate all joints by a small offset.

        Applies the same offset to all frames (global shift).
        """
        if self.translate_range > 0:
            offset_xy = np.random.uniform(
                -self.translate_range, self.translate_range, size=2
            ).astype(np.float32)
            coords[..., 0] += offset_xy[0]
            coords[..., 1] += offset_xy[1]
        return coords, observed_mask

    def _random_view_rotate(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply small 3D camera/viewpoint rotations around the origin."""
        roll_lo, roll_hi = self.rotation_range_deg
        pitch_lo, pitch_hi = self.pitch_range_deg
        yaw_lo, yaw_hi = self.yaw_range_deg
        if (
            roll_lo == roll_hi == 0
            and pitch_lo == pitch_hi == 0
            and yaw_lo == yaw_hi == 0
        ):
            return coords, observed_mask

        roll = np.deg2rad(np.random.uniform(roll_lo, roll_hi))
        pitch = np.deg2rad(np.random.uniform(pitch_lo, pitch_hi))
        yaw = np.deg2rad(np.random.uniform(yaw_lo, yaw_hi))
        cx, sx = np.float32(np.cos(pitch)), np.float32(np.sin(pitch))
        cy, sy = np.float32(np.cos(yaw)), np.float32(np.sin(yaw))
        cz, sz = np.float32(np.cos(roll)), np.float32(np.sin(roll))

        rot_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
            dtype=np.float32,
        )
        rot_y = np.array(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
            dtype=np.float32,
        )
        rot_z = np.array(
            [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        rotation = rot_z @ rot_y @ rot_x
        coords = coords @ rotation.T
        return coords, observed_mask

    def _horizontal_flip(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly mirror left↔right by negating x-coords and swapping hands.

        Our canonical layout:
          0-9:     body pose joints
          10-30:   left hand (21 joints)
          31-51:   right hand (21 joints)

        Flipping means:
          1. Negate all x-coordinates (every even index in the flat 104 vector)
          2. Swap left↔right shoulder, elbow, wrist, hip
          3. Swap left hand ↔ right hand
        """
        if np.random.rand() > self.flip_prob:
            return coords, observed_mask

        coords = coords.copy()
        observed_mask = observed_mask.copy()

        # 1. Negate x-coordinates only.
        coords[..., 0] = -coords[..., 0]

        # 2. Swap pose joints: left↔right (shoulder, elbow, wrist, hip)
        swap_pairs = [
            (1, 2),   # left_shoulder ↔ right_shoulder
            (3, 4),   # left_elbow ↔ right_elbow
            (5, 6),   # left_wrist ↔ right_wrist
            (7, 8),   # left_hip ↔ right_hip
        ]
        for a, b in swap_pairs:
            coords[:, a], coords[:, b] = (
                coords[:, b].copy(),
                coords[:, a].copy(),
            )
            observed_mask[:, a], observed_mask[:, b] = (
                observed_mask[:, b].copy(),
                observed_mask[:, a].copy(),
            )

        coords[:, 10:31], coords[:, 31:52] = (
            coords[:, 31:52].copy(),
            coords[:, 10:31].copy(),
        )
        observed_mask[:, 10:31], observed_mask[:, 31:52] = (
            observed_mask[:, 31:52].copy(),
            observed_mask[:, 10:31].copy(),
        )

        return coords, observed_mask

    def _joint_dropout(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly hold joints at their previous tracked position.

        This better approximates landmark tracker dropouts than snapping
        joints to the normalized origin.
        """
        if self.joint_dropout_prob > 0:
            T = coords.shape[0]
            coords = coords.copy()
            observed_mask = observed_mask.copy()
            mask = np.random.rand(T, NUM_JOINTS) < self.joint_dropout_prob

            for t in range(T):
                if not mask[t].any():
                    continue
                if t == 0:
                    continue
                coords[t, mask[t]] = coords[t - 1, mask[t]]
                observed_mask[t, mask[t]] = 0.0

        return coords, observed_mask

    def _sample_contiguous_span(
        self,
        length: int,
        ratio_range: tuple[float, float],
    ) -> tuple[int, int]:
        """Sample a contiguous temporal span inside a sequence."""
        if length <= 1:
            return 0, length

        lo, hi = ratio_range
        lo = float(np.clip(lo, 0.0, 1.0))
        hi = float(np.clip(hi, lo, 1.0))
        ratio = np.random.uniform(lo, hi)
        span = max(1, min(length, int(round(length * ratio))))
        start = 0 if span >= length else int(np.random.randint(0, length - span + 1))
        return start, start + span

    def _apply_group_dropout(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
        joint_indices: np.ndarray,
        ratio_range: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mask a joint group for a contiguous span while holding coordinates."""
        T = coords.shape[0]
        if T == 0 or joint_indices.size == 0:
            return coords, observed_mask

        start, end = self._sample_contiguous_span(T, ratio_range)
        coords = coords.copy()
        observed_mask = observed_mask.copy()
        observed_mask[start:end, joint_indices] = 0.0

        for t in range(start, end):
            if t == 0:
                continue
            coords[t, joint_indices] = coords[t - 1, joint_indices]

        return coords, observed_mask

    def _hand_dropout(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drop one or both hands for a contiguous span."""
        if self.hand_dropout_prob <= 0 or coords.shape[0] == 0:
            return coords, observed_mask
        if np.random.rand() > self.hand_dropout_prob:
            return coords, observed_mask

        left = np.arange(10, 31, dtype=np.int64)
        right = np.arange(31, 52, dtype=np.int64)
        mode = np.random.choice(["left", "right", "both"], p=[0.4, 0.4, 0.2])
        if mode == "left":
            joint_indices = left
        elif mode == "right":
            joint_indices = right
        else:
            joint_indices = np.concatenate([left, right])

        return self._apply_group_dropout(
            coords,
            observed_mask,
            joint_indices,
            self.hand_dropout_ratio,
        )

    def _pose_dropout(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drop the upper-body pose joints for a contiguous span."""
        if self.pose_dropout_prob <= 0 or coords.shape[0] == 0:
            return coords, observed_mask
        if np.random.rand() > self.pose_dropout_prob:
            return coords, observed_mask

        pose = np.arange(0, 10, dtype=np.int64)
        return self._apply_group_dropout(
            coords,
            observed_mask,
            pose,
            self.pose_dropout_ratio,
        )

    def _idle_hand_inject(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inject a plausible idle hand when one hand is entirely undetected.

        In training data, one-handed signs often have the non-signing hand as
        all zeros (MediaPipe didn't see it). At live inference time, both hands
        are visible — the idle hand sits near the hip/lap with low motion.

        This augmentation randomly fills in a missing hand with a static idle
        position plus small per-frame noise, so the model learns to ignore the
        presence of an idle hand.

        Joint layout:
          - Left hand:  joints 10-30 (21 joints)
          - Right hand: joints 31-51 (21 joints)
          - Left hip:   joint 7
          - Right hip:  joint 8
          - Left wrist:  joint 5
          - Right wrist: joint 6
        """
        if self.idle_hand_inject_prob <= 0 or coords.shape[0] == 0:
            return coords, observed_mask
        if np.random.rand() > self.idle_hand_inject_prob:
            return coords, observed_mask

        T = coords.shape[0]
        left_hand = slice(10, 31)
        right_hand = slice(31, 52)

        left_missing = (observed_mask[:, left_hand].sum() == 0.0)
        right_missing = (observed_mask[:, right_hand].sum() == 0.0)

        if not left_missing and not right_missing:
            return coords, observed_mask

        coords = coords.copy()
        observed_mask = observed_mask.copy()

        def _generate_idle_hand(hip_idx: int, wrist_idx: int) -> np.ndarray:
            """Generate 21 idle hand joint positions near the hip/wrist area."""
            # Anchor: midpoint between hip and wrist, shifted slightly toward hip
            hip_pos = coords[:, hip_idx]   # (T, 3)
            wrist_pos = coords[:, wrist_idx]  # (T, 3)
            # Use hip if wrist is unobserved; otherwise blend
            hip_obs = observed_mask[:, hip_idx] > 0.5
            wrist_obs = observed_mask[:, wrist_idx] > 0.5
            anchor = np.zeros((T, NUM_COORDS), dtype=np.float32)
            for t in range(T):
                if hip_obs[t] and wrist_obs[t]:
                    anchor[t] = 0.7 * hip_pos[t] + 0.3 * wrist_pos[t]
                elif hip_obs[t]:
                    anchor[t] = hip_pos[t]
                elif wrist_obs[t]:
                    anchor[t] = wrist_pos[t]
                # else stays at origin (0, 0, 0)

            # All 21 hand joints cluster around the anchor with small offsets
            # Finger joints fan out slightly from wrist
            hand = np.zeros((T, 21, NUM_COORDS), dtype=np.float32)
            for j in range(21):
                # Small deterministic spread: fingertip joints are farther out
                spread = 0.02 * (j % 5) / 4.0  # tips spread more
                offset = np.array([spread, spread * 0.5, 0.0], dtype=np.float32)
                hand[:, j] = anchor + offset
            # Add per-frame jitter to simulate slight natural idle movement
            jitter = np.random.randn(T, 21, NUM_COORDS).astype(np.float32) * 0.008
            hand += jitter
            return hand

        if left_missing:
            idle_left = _generate_idle_hand(hip_idx=7, wrist_idx=5)
            coords[:, left_hand] = idle_left
            observed_mask[:, left_hand] = 1.0

        if right_missing:
            idle_right = _generate_idle_hand(hip_idx=8, wrist_idx=6)
            coords[:, right_hand] = idle_right
            observed_mask[:, right_hand] = 1.0

        return coords, observed_mask

    def _temporal_crop(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly crop a contiguous temporal segment."""
        T = coords.shape[0]
        if T <= 2:
            return coords, observed_mask

        lo, hi = self.temporal_crop_ratio
        ratio = np.random.uniform(lo, hi)
        crop_len = max(2, int(T * ratio))

        if crop_len >= T:
            return coords, observed_mask

        start = np.random.randint(0, T - crop_len)
        end = start + crop_len
        return coords[start:end], observed_mask[start:end]

    def _temporal_drop(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly drop a subset of frames while preserving order."""
        T = coords.shape[0]
        if T <= 2 or self.temporal_drop_ratio <= 0:
            return coords, observed_mask

        drop_count = int(round(T * self.temporal_drop_ratio))
        drop_count = min(max(drop_count, 0), T - 2)
        if drop_count == 0:
            return coords, observed_mask

        keep_count = T - drop_count
        keep_indices = np.sort(
            np.random.choice(T, size=keep_count, replace=False)
        )
        return coords[keep_indices], observed_mask[keep_indices]

    def _speed_perturb(
        self,
        coords: np.ndarray,
        observed_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly stretch or compress the temporal axis.

        Resamples the sequence to a new length using linear interpolation.
        """
        T = coords.shape[0]
        if T <= 2:
            return coords, observed_mask

        lo, hi = self.speed_perturb_range
        speed = np.random.uniform(lo, hi)
        new_len = max(2, int(T / speed))

        if new_len == T:
            return coords, observed_mask

        # Linear interpolation along time axis
        old_t = np.linspace(0, 1, T)
        new_t = np.linspace(0, 1, new_len)

        from scipy.interpolate import interp1d
        coords_flat = coords.reshape(T, COORD_FEATURE_DIM)
        interp_fn = interp1d(
            old_t,
            coords_flat,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        coords_resampled = interp_fn(new_t).astype(np.float32).reshape(
            new_len,
            NUM_JOINTS,
            NUM_COORDS,
        )

        mask_interp = interp1d(
            old_t,
            observed_mask,
            axis=0,
            kind="nearest",
            fill_value="extrapolate",
        )
        mask_resampled = np.clip(mask_interp(new_t), 0.0, 1.0).astype(np.float32)
        return coords_resampled, mask_resampled
