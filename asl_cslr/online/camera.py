"""Webcam capture with temporal downsampling and stabilized landmark tracking."""

import logging
import math
import time
from collections import deque
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np

from asl_cslr.data.mediapipe_tasks import (
    create_holistic_landmarker,
    create_mp_image,
)
from asl_cslr.data.skeleton import (
    build_feature_frame,
    extract_skeleton_from_holistic_result,
    normalize_frame,
)

logger = logging.getLogger(__name__)


def smooth_joints(
    current: np.ndarray,
    previous: np.ndarray | None,
    alpha: float = 0.35,
) -> np.ndarray:
    """Blend current joints with the previous frame to reduce jitter."""
    current = current.astype(np.float32, copy=False)
    if previous is None:
        return current.astype(np.float32, copy=True)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * current + (1.0 - alpha) * previous.astype(np.float32, copy=False)


class WebcamCapture:
    """Captures webcam frames and extracts skeletons in real time.

    Handles temporal downsampling and maintains a rolling buffer of
    normalized skeleton frames for downstream inference.

    Args:
        device_id: Camera device index.
        capture_fps: Raw capture FPS.
        downsample_factor: Process every n-th frame.
        buffer_duration_sec: Max seconds of frames to keep in buffer.
    """

    def __init__(
        self,
        device_id: int = 0,
        capture_fps: int = 30,
        downsample_factor: int = 2,
        buffer_duration_sec: float = 4.0,
        smoothing_alpha: float = 0.35,
        capture_width: int | None = None,
        capture_height: int | None = None,
        inference_width: int | None = None,
        inference_height: int | None = None,
        holistic_model_path: str | Path | None = None,
        min_face_detection_confidence: float = 0.5,
        min_face_landmarks_confidence: float = 0.5,
        min_pose_detection_confidence: float = 0.5,
        min_pose_landmarks_confidence: float = 0.5,
        min_hand_landmarks_confidence: float = 0.5,
        min_face_suppression_threshold: float = 0.5,
        min_pose_suppression_threshold: float = 0.5,
        pose_visibility_threshold: float = 0.5,
        pose_presence_threshold: float = 0.5,
        hand_visibility_threshold: float | None = None,
        hand_presence_threshold: float | None = None,
        max_pending_timestamps: int = 2,
    ):
        if capture_fps <= 0:
            raise ValueError("capture_fps must be positive")
        if downsample_factor <= 0:
            raise ValueError("downsample_factor must be positive")
        if max_pending_timestamps <= 0:
            raise ValueError("max_pending_timestamps must be positive")

        self.device_id = device_id
        self.capture_fps = capture_fps
        self.downsample_factor = downsample_factor
        self.smoothing_alpha = smoothing_alpha
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.inference_width = inference_width
        self.inference_height = inference_height
        self.holistic_model_path = holistic_model_path
        self.min_face_detection_confidence = min_face_detection_confidence
        self.min_face_landmarks_confidence = min_face_landmarks_confidence
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_landmarks_confidence = min_pose_landmarks_confidence
        self.min_hand_landmarks_confidence = min_hand_landmarks_confidence
        self.min_face_suppression_threshold = min_face_suppression_threshold
        self.min_pose_suppression_threshold = min_pose_suppression_threshold
        self.pose_visibility_threshold = pose_visibility_threshold
        self.pose_presence_threshold = pose_presence_threshold
        self.hand_visibility_threshold = hand_visibility_threshold
        self.hand_presence_threshold = hand_presence_threshold
        self.max_pending_timestamps = max_pending_timestamps

        self.effective_fps = capture_fps / downsample_factor
        buffer_size = max(1, math.ceil(self.effective_fps * buffer_duration_sec))
        self.buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)

        self._cap = None
        self._frame_count = 0
        self._prev_inference_joints = None
        self._display_joints = None
        self._last_timestamp_ms = 0
        self._start_time = 0.0
        self._state_lock = Lock()
        self._submission_frame_indices: dict[int, int] = {}
        self._last_result_id = 0
        self._last_returned_result_id = 0

        # MediaPipe (lazy init)
        self._holistic = None
        self._last_raw_joints = None
        self._last_skeleton = None
        self._last_display_frame = None

    def start(self):
        """Open the webcam and initialize MediaPipe Holistic Tasks API."""
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open camera {self.device_id}")

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FPS, self.capture_fps)
        if self.capture_width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.capture_width))
        if self.capture_height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.capture_height))
        self._frame_count = 0
        self.buffer.clear()
        self.timestamps.clear()
        self._prev_inference_joints = None
        self._display_joints = None
        self._last_raw_joints = None
        self._last_skeleton = None
        self._last_display_frame = None
        self._last_result_id = 0
        self._last_returned_result_id = 0
        self._submission_frame_indices.clear()
        self._start_time = time.monotonic()
        self._last_timestamp_ms = 0
        self._holistic = create_holistic_landmarker(
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            model_path=self.holistic_model_path,
            result_callback=self._on_landmarks,
            min_face_detection_confidence=self.min_face_detection_confidence,
            min_face_suppression_threshold=self.min_face_suppression_threshold,
            min_face_landmarks_confidence=self.min_face_landmarks_confidence,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_suppression_threshold=self.min_pose_suppression_threshold,
            min_pose_landmarks_confidence=self.min_pose_landmarks_confidence,
            min_hand_landmarks_confidence=self.min_hand_landmarks_confidence,
        )
        logger.info(
            f"Camera started: device={self.device_id}, "
            f"fps={self.capture_fps}, downsample={self.downsample_factor}"
        )

    def stop(self):
        """Release camera and MediaPipe resources."""
        if self._cap:
            self._cap.release()
        if self._holistic:
            self._holistic.close()
        logger.info("Camera stopped")

    def _prune_pending_timestamps(self):
        """Bound bookkeeping for callbacks that were dropped by detect_async."""
        if len(self._submission_frame_indices) <= self.max_pending_timestamps:
            return
        stale = list(self._submission_frame_indices)[:-self.max_pending_timestamps]
        for timestamp_ms in stale:
            self._submission_frame_indices.pop(timestamp_ms, None)

    def _on_landmarks(self, result, output_image, timestamp_ms: int):
        """MediaPipe live-stream callback with the latest accepted landmarks."""
        try:
            with self._state_lock:
                frame_index = self._submission_frame_indices.pop(timestamp_ms, None)
                prev_inference_joints = self._prev_inference_joints
                display_joints = self._display_joints

            if frame_index is None:
                return

            frame_rgb = output_image.numpy_view().copy()
            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            raw_joints, observed_mask = extract_skeleton_from_holistic_result(
                result,
                prev_joints=prev_inference_joints,
                fill=True,
                pose_visibility_threshold=self.pose_visibility_threshold,
                pose_presence_threshold=self.pose_presence_threshold,
                hand_visibility_threshold=self.hand_visibility_threshold,
                hand_presence_threshold=self.hand_presence_threshold,
                return_observed_mask=True,
            )
            rendered_joints = smooth_joints(
                raw_joints,
                display_joints,
                alpha=self.smoothing_alpha,
            )
            display_joints_xy = rendered_joints[:, :2].copy()
            display_joints_xy[observed_mask < 0.5] = np.nan
            skeleton = build_feature_frame(
                normalize_frame(raw_joints, observed_mask=observed_mask),
                observed_mask,
            )

            with self._state_lock:
                self._prev_inference_joints = raw_joints
                self._display_joints = rendered_joints
                self._last_raw_joints = display_joints_xy
                self._last_skeleton = skeleton
                self._last_display_frame = display_frame
                self._last_result_id = timestamp_ms
                self.buffer.append(skeleton)
                self.timestamps.append(timestamp_ms / 1000.0)
        except Exception:
            logger.exception("MediaPipe Holistic callback failed")

    def read_frame(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
        """Read one frame and submit it to MediaPipe live-stream processing.

        Returns:
            Tuple of (frame_bgr, raw_joints, skeleton_flat, result_id):
                - frame_bgr: Latest raw BGR frame, or the synchronized processed
                  frame when a fresh callback result is available.
                - raw_joints: (52, 2) coords in [0, 1] for a fresh processed
                  result only; otherwise None.
                - skeleton_flat: normalized per-frame features for a fresh
                  processed result only; otherwise None.
                - result_id: Monotonic id for the latest processed callback result.
        """
        if not self._cap or not self._cap.isOpened():
            return None, None, None, 0

        ret, frame = self._cap.read()
        if not ret:
            return None, None, None, 0

        self._frame_count += 1
        should_submit = (self._frame_count % self.downsample_factor) == 0
        computed_timestamp_ms = int(round((time.monotonic() - self._start_time) * 1000.0))
        timestamp_ms = max(self._last_timestamp_ms + 1, computed_timestamp_ms)
        self._last_timestamp_ms = timestamp_ms

        with self._state_lock:
            can_submit = should_submit and (
                len(self._submission_frame_indices) < self.max_pending_timestamps
            )
            if can_submit:
                self._submission_frame_indices[timestamp_ms] = self._frame_count
                self._prune_pending_timestamps()
            raw_joints = (
                None if self._last_raw_joints is None else self._last_raw_joints.copy()
            )
            skeleton = (
                None if self._last_skeleton is None else self._last_skeleton.copy()
            )
            display_frame = (
                None
                if self._last_display_frame is None
                else self._last_display_frame.copy()
            )
            result_id = int(self._last_result_id)
            has_fresh_result = result_id > self._last_returned_result_id
            if has_fresh_result:
                self._last_returned_result_id = result_id

        if can_submit:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.inference_width is not None and self.inference_height is not None:
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (int(self.inference_width), int(self.inference_height)),
                    interpolation=cv2.INTER_LINEAR,
                )
            mp_image = create_mp_image(frame_rgb)
            try:
                self._holistic.detect_async(mp_image, timestamp_ms)
            except ValueError:
                with self._state_lock:
                    self._submission_frame_indices.pop(timestamp_ms, None)
                logger.exception("Rejected non-monotonic MediaPipe timestamp")
            except Exception:
                with self._state_lock:
                    self._submission_frame_indices.pop(timestamp_ms, None)
                logger.exception("MediaPipe Holistic live-stream submission failed")

        if has_fresh_result and display_frame is not None:
            return display_frame, raw_joints, skeleton, result_id
        return frame, None, None, result_id

    def get_buffer_window(self, num_frames: int) -> np.ndarray | None:
        """Get the last N frames from the buffer as a `(N, D)` feature array.

        Args:
            num_frames: Number of frames to retrieve.

        Returns:
            Array of shape `(min(available, num_frames), D)`, or None if empty.
        """
        if not self.buffer:
            return None

        frames = list(self.buffer)[-num_frames:]
        return np.array(frames, dtype=np.float32)

    def get_full_buffer(self) -> np.ndarray | None:
        """Get the entire buffer as a `(T, D)` feature array."""
        if not self.buffer:
            return None
        return np.array(list(self.buffer), dtype=np.float32)
