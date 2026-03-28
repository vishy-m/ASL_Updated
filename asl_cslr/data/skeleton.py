"""
Canonical 52-joint skeleton representation and normalization.

Implements the shared joint layout, coordinate normalization, observation-mask
packing, and motion features used across preprocessing, training, and online
inference.

Joint layout (52 joints total):
  - 10 Pose joints (from MediaPipe Pose, incl. synthetic MID_SHOULDERS)
  - 21 Left hand joints (from MediaPipe Hands)
  - 21 Right hand joints (from MediaPipe Hands)

Per-joint channels in the current feature schema:
  - x, y, z normalized coordinates
  - observed mask (1.0 if detected this frame, 0.0 if imputed)

This produces:
  - coordinate-only frame: 52 × 3 = 156 dims
  - packed frame feature: 52 × (3 + 1) = 208 dims
"""

import numpy as np

# ---------------------------------------------------------------------------
# Joint definitions
# ---------------------------------------------------------------------------

# Pose joints (10) — indices into this canonical 52-joint layout
POSE_JOINT_NAMES = [
    "NOSE",              # 0
    "LEFT_SHOULDER",     # 1
    "RIGHT_SHOULDER",    # 2
    "LEFT_ELBOW",        # 3
    "RIGHT_ELBOW",       # 4
    "LEFT_WRIST",        # 5
    "RIGHT_WRIST",       # 6
    "LEFT_HIP",          # 7
    "RIGHT_HIP",         # 8
    "MID_SHOULDERS",     # 9 (synthetic)
]

# MediaPipe Pose landmark indices for the 9 real pose joints
# (MID_SHOULDERS is computed, not directly from MediaPipe)
MEDIAPIPE_POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]

# Left hand joints (21) — standard MediaPipe Hands ordering (0-20)
LEFT_HAND_JOINT_NAMES = [
    "LH_WRIST",         # 10
    "LH_THUMB_CMC",     # 11
    "LH_THUMB_MCP",     # 12
    "LH_THUMB_IP",      # 13
    "LH_THUMB_TIP",     # 14
    "LH_INDEX_MCP",     # 15
    "LH_INDEX_PIP",     # 16
    "LH_INDEX_DIP",     # 17
    "LH_INDEX_TIP",     # 18
    "LH_MIDDLE_MCP",    # 19
    "LH_MIDDLE_PIP",    # 20
    "LH_MIDDLE_DIP",    # 21
    "LH_MIDDLE_TIP",    # 22
    "LH_RING_MCP",      # 23
    "LH_RING_PIP",      # 24
    "LH_RING_DIP",      # 25
    "LH_RING_TIP",      # 26
    "LH_PINKY_MCP",     # 27
    "LH_PINKY_PIP",     # 28
    "LH_PINKY_DIP",     # 29
    "LH_PINKY_TIP",     # 30
]

# Right hand joints (21) — same ordering as left
RIGHT_HAND_JOINT_NAMES = [
    "RH_WRIST",         # 31
    "RH_THUMB_CMC",     # 32
    "RH_THUMB_MCP",     # 33
    "RH_THUMB_IP",      # 34
    "RH_THUMB_TIP",     # 35
    "RH_INDEX_MCP",     # 36
    "RH_INDEX_PIP",     # 37
    "RH_INDEX_DIP",     # 38
    "RH_INDEX_TIP",     # 39
    "RH_MIDDLE_MCP",    # 40
    "RH_MIDDLE_PIP",    # 41
    "RH_MIDDLE_DIP",    # 42
    "RH_MIDDLE_TIP",    # 43
    "RH_RING_MCP",      # 44
    "RH_RING_PIP",      # 45
    "RH_RING_DIP",      # 46
    "RH_RING_TIP",      # 47
    "RH_PINKY_MCP",     # 48
    "RH_PINKY_PIP",     # 49
    "RH_PINKY_DIP",     # 50
    "RH_PINKY_TIP",     # 51
]

# Complete ordered list of all 52 joint names
JOINT_NAMES = POSE_JOINT_NAMES + LEFT_HAND_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

NUM_JOINTS = len(JOINT_NAMES)          # 52
NUM_COORDS = 3                          # x, y, z
OBSERVATION_DIM_PER_JOINT = 1           # observed vs imputed
JOINT_FEATURES = NUM_COORDS + OBSERVATION_DIM_PER_JOINT
COORD_FEATURE_DIM = NUM_JOINTS * NUM_COORDS
OBSERVATION_MASK_DIM = NUM_JOINTS
FEATURE_DIM = NUM_JOINTS * JOINT_FEATURES
LEGACY_XY_COORD_DIM = NUM_JOINTS * 2

# Canonical indices for reference joints used in normalization
IDX_LEFT_SHOULDER = 1
IDX_RIGHT_SHOULDER = 2
IDX_MID_SHOULDERS = 9

# Default epsilon for scale normalization (avoid division by zero)
SCALE_EPSILON = 1e-3


# ---------------------------------------------------------------------------
# Coordinate normalization (per frame) — §3.3
# ---------------------------------------------------------------------------

def normalize_frame(
    joints_xyz: np.ndarray,
    observed_mask: np.ndarray | None = None,
    epsilon: float = SCALE_EPSILON,
) -> np.ndarray:
    """Normalize a single frame's joint coordinates.

    Applies shoulder-based translation and scale normalization to make the
    skeleton representation invariant to position and body size.

    Args:
        joints_xyz: Array of shape (52, 3) — raw (x, y, z) for each canonical joint.
        observed_mask: Optional (52,) array with 1.0 for currently observed
            joints and 0.0 for imputed joints. When provided, normalization
            prefers observed joints when choosing the reference point and scale.
        epsilon: Minimum scale to avoid division by zero.

    Returns:
        Normalized array of shape (52, 3). MID_SHOULDERS will be (0, 0, 0).
    """
    joints = joints_xyz.copy()
    observed = None
    if observed_mask is not None:
        observed = np.asarray(observed_mask, dtype=np.float32).reshape(-1) > 0.5
        if observed.shape[0] != NUM_JOINTS:
            raise ValueError(
                f"Expected observed_mask with {NUM_JOINTS} entries, got {observed.shape[0]}"
            )

    finite_mask = np.isfinite(joints).all(axis=1)
    observed_finite = finite_mask if observed is None else (finite_mask & observed)

    # Reference point: midpoint of left and right shoulders
    left_shoulder = joints[IDX_LEFT_SHOULDER]
    right_shoulder = joints[IDX_RIGHT_SHOULDER]
    left_valid = observed_finite[IDX_LEFT_SHOULDER]
    right_valid = observed_finite[IDX_RIGHT_SHOULDER]

    if left_valid and right_valid:
        ref = (left_shoulder + right_shoulder) / 2.0
    else:
        reference_mask = observed_finite
        if not reference_mask.any():
            reference_mask = finite_mask
        if reference_mask.any():
            ref = joints[reference_mask].mean(axis=0)
        else:
            ref = np.zeros(NUM_COORDS, dtype=joints.dtype)

    # Scale: shoulder span when available, otherwise fall back to the median
    # non-zero joint distance from the reference point using observed joints
    # when available, so stale forward-filled joints do not define scale.
    if left_valid and right_valid:
        d_shoulder = np.linalg.norm(left_shoulder - right_shoulder)
    else:
        d_shoulder = np.nan

    if np.isfinite(d_shoulder) and d_shoulder >= epsilon:
        s_ref = d_shoulder
    else:
        scale_mask = observed_finite
        if not scale_mask.any():
            scale_mask = finite_mask
        distances = np.linalg.norm(joints - ref, axis=1)
        distances = distances[scale_mask]
        valid_distances = distances[np.isfinite(distances) & (distances > epsilon)]
        s_ref = max(float(np.median(valid_distances)), epsilon) if valid_distances.size else epsilon

    # Normalize all joints
    joints = (joints - ref) / s_ref

    # MID_SHOULDERS is always (0, 0, 0) by construction
    joints[IDX_MID_SHOULDERS] = np.array([0.0, 0.0, 0.0], dtype=joints.dtype)

    return joints


def build_feature_frame(
    normalized_joints_xyz: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    """Pack normalized coordinates plus the observed/imputed mask.

    Args:
        normalized_joints_xyz: Array of shape (52, 3).
        observed_mask: Array of shape (52,) with 1.0 for observed joints and
            0.0 for imputed joints.

    Returns:
        Flattened feature frame of shape (208,).
    """
    features = np.concatenate(
        [
            normalized_joints_xyz.astype(np.float32, copy=False),
            observed_mask.astype(np.float32, copy=False).reshape(NUM_JOINTS, 1),
        ],
        axis=1,
    )
    return features.reshape(FEATURE_DIM).astype(np.float32, copy=False)


def split_feature_frame(
    frame_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one feature frame into coordinates and observation mask.

    Supports the current packed schema `(x, y, z, mask)` and legacy xy-only
    feature frames for backward compatibility in tests and old artifacts.
    """
    frame = np.asarray(frame_features, dtype=np.float32).reshape(-1)
    if frame.size == FEATURE_DIM:
        reshaped = frame.reshape(NUM_JOINTS, JOINT_FEATURES)
        return reshaped[:, :NUM_COORDS], reshaped[:, NUM_COORDS]
    if frame.size == COORD_FEATURE_DIM:
        coords = frame.reshape(NUM_JOINTS, NUM_COORDS)
        mask = np.ones(NUM_JOINTS, dtype=np.float32)
        return coords, mask
    if frame.size == LEGACY_XY_COORD_DIM:
        coords_xy = frame.reshape(NUM_JOINTS, 2)
        coords = np.zeros((NUM_JOINTS, NUM_COORDS), dtype=np.float32)
        coords[:, :2] = coords_xy
        mask = np.ones(NUM_JOINTS, dtype=np.float32)
        return coords, mask
    raise ValueError(f"Unsupported frame feature width: {frame.size}")


def extract_coordinate_features(sequence: np.ndarray) -> np.ndarray:
    """Return the coordinate-only representation for a feature sequence."""
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature sequence, got shape={array.shape}")

    if array.shape[1] == FEATURE_DIM:
        coords = array.reshape(array.shape[0], NUM_JOINTS, JOINT_FEATURES)[..., :NUM_COORDS]
        return coords.reshape(array.shape[0], COORD_FEATURE_DIM).astype(np.float32, copy=False)
    if array.shape[1] == COORD_FEATURE_DIM:
        return array.astype(np.float32, copy=False)
    if array.shape[1] == LEGACY_XY_COORD_DIM:
        legacy = array.reshape(array.shape[0], NUM_JOINTS, 2)
        coords = np.zeros((array.shape[0], NUM_JOINTS, NUM_COORDS), dtype=np.float32)
        coords[..., :2] = legacy
        return coords.reshape(array.shape[0], COORD_FEATURE_DIM)
    raise ValueError(f"Unsupported feature width for coordinate extraction: {array.shape[1]}")


def extract_observation_mask(sequence: np.ndarray) -> np.ndarray | None:
    """Return the observed/imputed joint mask for a feature sequence when present."""
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature sequence, got shape={array.shape}")

    if array.shape[1] == FEATURE_DIM:
        packed = array.reshape(array.shape[0], NUM_JOINTS, JOINT_FEATURES)
        return packed[..., NUM_COORDS].astype(np.float32, copy=False)
    if array.shape[1] in {COORD_FEATURE_DIM, LEGACY_XY_COORD_DIM}:
        return None
    raise ValueError(f"Unsupported feature width for mask extraction: {array.shape[1]}")


def normalize_frame_flat(
    joints_flat: np.ndarray,
    epsilon: float = SCALE_EPSILON,
) -> np.ndarray:
    """Normalize a flattened coordinate frame.

    Args:
        joints_flat: Flattened coordinate frame.
        epsilon: Minimum scale.

    Returns:
        Normalized flat coordinate array.
    """
    joints_xyz, _mask = split_feature_frame(joints_flat)
    normalized = normalize_frame(joints_xyz, epsilon)
    return normalized.flatten()


# ---------------------------------------------------------------------------
# Motion features — §3.4
# ---------------------------------------------------------------------------

def compute_motion_features(
    sequence: np.ndarray,
    compute_acceleration: bool = False,
) -> dict:
    """Compute velocity (and optionally acceleration) from a skeleton sequence.

    Args:
        sequence: Array of shape (T, D) — skeleton features.
        compute_acceleration: Whether to also compute acceleration.

    Returns:
        Dictionary with:
            - 'velocity': shape (T, C), zero-padded at t=0
            - 'acceleration': shape (T, C), if requested, zero-padded at t=0,1
    """
    coords = extract_coordinate_features(sequence)
    observed_mask = extract_observation_mask(sequence)
    T = coords.shape[0]
    velocity = np.zeros_like(coords)
    if T > 1:
        velocity[1:] = coords[1:] - coords[:-1]
        if observed_mask is not None:
            valid_pairs = (observed_mask[1:] > 0.5) & (observed_mask[:-1] > 0.5)
            valid_pairs = np.repeat(valid_pairs, NUM_COORDS, axis=1)
            velocity[1:][~valid_pairs] = 0.0

    result = {"velocity": velocity}

    if compute_acceleration:
        acceleration = np.zeros_like(coords)
        if T > 2:
            acceleration[2:] = velocity[2:] - velocity[1:-1]
            if observed_mask is not None:
                valid_triplets = (
                    (observed_mask[2:] > 0.5)
                    & (observed_mask[1:-1] > 0.5)
                    & (observed_mask[:-2] > 0.5)
                )
                valid_triplets = np.repeat(valid_triplets, NUM_COORDS, axis=1)
                acceleration[2:][~valid_triplets] = 0.0
        result["acceleration"] = acceleration

    return result


# ---------------------------------------------------------------------------
# MediaPipe extraction — converting raw landmarks to canonical layout
# ---------------------------------------------------------------------------

def fill_missing_joints(
    joints_xyz: np.ndarray,
    prev_joints_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Fill missing joints (NaN values) using forward-fill or a local reference.

    Args:
        joints_xyz: Array of shape (52, 3), may contain NaN for undetected joints.
        prev_joints_xy: Previous frame's joints for forward-fill. None for t=0.

    Returns:
        Array of shape (52, 2) with NaN values replaced.
    """
    nan_mask = np.isnan(joints_xyz).any(axis=1)

    if nan_mask.any():
        if prev_joints_xy is not None:
            # Forward-fill from previous frame
            joints_xyz[nan_mask] = prev_joints_xy[nan_mask]
        else:
            # On the first frame, fill missing points at the current frame's
            # reference location so they normalize to (0, 0) instead of the
            # image origin.
            ref = np.zeros(NUM_COORDS, dtype=joints_xyz.dtype)
            left_shoulder = joints_xyz[IDX_LEFT_SHOULDER]
            right_shoulder = joints_xyz[IDX_RIGHT_SHOULDER]

            left_valid = not np.isnan(left_shoulder).any()
            right_valid = not np.isnan(right_shoulder).any()

            if left_valid and right_valid:
                ref = (left_shoulder + right_shoulder) / 2.0
            elif left_valid:
                ref = left_shoulder
            elif right_valid:
                ref = right_shoulder
            else:
                observed = joints_xyz[~nan_mask]
                if observed.size:
                    ref = np.nanmean(observed, axis=0).astype(joints_xyz.dtype, copy=False)

            joints_xyz[nan_mask] = ref

    return joints_xyz


def _landmark_is_observed(
    landmark,
    *,
    visibility_threshold: float | None = None,
    presence_threshold: float | None = None,
) -> bool:
    """Return True when a MediaPipe landmark should be treated as observed."""
    if landmark is None:
        return False

    values = [getattr(landmark, axis, np.nan) for axis in ("x", "y", "z")]
    if not np.isfinite(values).all():
        return False

    visibility = getattr(landmark, "visibility", None)
    if visibility is not None and visibility_threshold is not None and visibility < visibility_threshold:
        return False

    presence = getattr(landmark, "presence", None)
    if presence is not None and presence_threshold is not None and presence < presence_threshold:
        return False

    return True


def extract_skeleton_from_mediapipe(
    pose_landmarks,
    left_hand_landmarks,
    right_hand_landmarks,
    prev_joints: np.ndarray | None = None,
    fill: bool = True,
    *,
    pose_visibility_threshold: float | None = 0.5,
    pose_presence_threshold: float | None = 0.5,
    hand_visibility_threshold: float | None = None,
    hand_presence_threshold: float | None = None,
    return_observed_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Extract canonical 52-joint skeleton from MediaPipe landmarks.

    Args:
        pose_landmarks: MediaPipe pose landmarks (33 landmarks) or None.
        left_hand_landmarks: MediaPipe left hand landmarks (21) or None.
        right_hand_landmarks: MediaPipe right hand landmarks (21) or None.
        prev_joints: Previous frame's (52, 3) joints for forward-fill.
        fill: Whether to run the NaN-filling logic.
        pose_visibility_threshold: Minimum visibility score for pose joints.
        pose_presence_threshold: Minimum presence score for pose joints.
        hand_visibility_threshold: Minimum visibility score for hand joints.
        hand_presence_threshold: Minimum presence score for hand joints.
        return_observed_mask: Whether to also return the 52-element observed mask.

    Returns:
        Array of shape (52, 3) — raw (x, y, z) coords for canonical joints.
    """
    joints = np.full((NUM_JOINTS, NUM_COORDS), np.nan, dtype=np.float32)
    observed_mask = np.zeros(NUM_JOINTS, dtype=np.float32)

    # Extract pose joints (9 real + 1 synthetic)
    if pose_landmarks is not None:
        for canon_idx, mp_idx in enumerate(MEDIAPIPE_POSE_INDICES):
            lm = pose_landmarks[mp_idx]
            if _landmark_is_observed(
                lm,
                visibility_threshold=pose_visibility_threshold,
                presence_threshold=pose_presence_threshold,
            ):
                joints[canon_idx] = [lm.x, lm.y, lm.z]
                observed_mask[canon_idx] = 1.0

        # Synthetic MID_SHOULDERS
        if observed_mask[IDX_LEFT_SHOULDER] and observed_mask[IDX_RIGHT_SHOULDER]:
            joints[IDX_MID_SHOULDERS] = (
                joints[IDX_LEFT_SHOULDER] + joints[IDX_RIGHT_SHOULDER]
            ) / 2.0
            observed_mask[IDX_MID_SHOULDERS] = 1.0

    # Extract left hand joints (21)
    if left_hand_landmarks is not None:
        for i in range(21):
            lm = left_hand_landmarks[i]
            if _landmark_is_observed(
                lm,
                visibility_threshold=hand_visibility_threshold,
                presence_threshold=hand_presence_threshold,
            ):
                joints[10 + i] = [lm.x, lm.y, lm.z]
                observed_mask[10 + i] = 1.0

    # Extract right hand joints (21)
    if right_hand_landmarks is not None:
        for i in range(21):
            lm = right_hand_landmarks[i]
            if _landmark_is_observed(
                lm,
                visibility_threshold=hand_visibility_threshold,
                presence_threshold=hand_presence_threshold,
            ):
                joints[31 + i] = [lm.x, lm.y, lm.z]
                observed_mask[31 + i] = 1.0

    # Handle missing joints
    if fill:
        joints = fill_missing_joints(joints, prev_joints)

    if return_observed_mask:
        return joints, observed_mask
    return joints


def extract_skeleton_from_holistic_result(
    holistic_result,
    prev_joints: np.ndarray | None = None,
    fill: bool = True,
    *,
    pose_visibility_threshold: float | None = 0.5,
    pose_presence_threshold: float | None = 0.5,
    hand_visibility_threshold: float | None = None,
    hand_presence_threshold: float | None = None,
    return_observed_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Extract the canonical 52-joint skeleton from a Holistic result."""
    pose_landmarks = getattr(holistic_result, "pose_landmarks", None) or None
    left_hand_landmarks = (
        getattr(holistic_result, "left_hand_landmarks", None) or None
    )
    right_hand_landmarks = (
        getattr(holistic_result, "right_hand_landmarks", None) or None
    )
    return extract_skeleton_from_mediapipe(
        pose_landmarks,
        left_hand_landmarks,
        right_hand_landmarks,
        prev_joints=prev_joints,
        fill=fill,
        pose_visibility_threshold=pose_visibility_threshold,
        pose_presence_threshold=pose_presence_threshold,
        hand_visibility_threshold=hand_visibility_threshold,
        hand_presence_threshold=hand_presence_threshold,
        return_observed_mask=return_observed_mask,
    )


def extract_and_normalize(
    pose_landmarks,
    left_hand_landmarks,
    right_hand_landmarks,
    prev_joints: np.ndarray | None = None,
    epsilon: float = SCALE_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract skeleton and normalize in one step.

    Args:
        pose_landmarks: MediaPipe pose landmarks or None.
        left_hand_landmarks: MediaPipe left hand landmarks or None.
        right_hand_landmarks: MediaPipe right hand landmarks or None.
        prev_joints: Previous frame's raw (52, 2) joints for forward-fill.
        epsilon: Minimum normalization scale.

    Returns:
        Tuple of:
            - raw_joints: (52, 3) raw extracted joints (for next frame's forward-fill)
            - packed_flat: (208,) normalized feature vector plus observation mask
    """
    raw_joints, observed_mask = extract_skeleton_from_mediapipe(
        pose_landmarks,
        left_hand_landmarks,
        right_hand_landmarks,
        prev_joints,
        return_observed_mask=True,
    )
    normalized = normalize_frame(raw_joints, observed_mask=observed_mask, epsilon=epsilon)
    return raw_joints, build_feature_frame(normalized, observed_mask)
