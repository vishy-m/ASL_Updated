"""Shared MediaPipe Tasks helpers for online and offline landmark extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import mediapipe as mp
import numpy as np

DEFAULT_MEDIAPIPE_MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "mediapipe"
DEFAULT_HOLISTIC_MODEL_PATH = DEFAULT_MEDIAPIPE_MODELS_DIR / "holistic_landmarker.task"
DEFAULT_HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)


def create_mp_image(frame_rgb: np.ndarray) -> mp.Image:
    """Convert an RGB numpy frame into a MediaPipe Image."""
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def resolve_holistic_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve the Holistic Landmarker model path and fail clearly if missing."""
    path = Path(model_path) if model_path is not None else DEFAULT_HOLISTIC_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Missing MediaPipe Holistic model at "
            f"{path}. Download it from {DEFAULT_HOLISTIC_MODEL_URL}."
        )
    return path


def create_holistic_landmarker(
    *,
    running_mode: mp.tasks.vision.RunningMode,
    model_path: str | Path | None = None,
    result_callback: Callable | None = None,
    min_face_detection_confidence: float = 0.5,
    min_face_suppression_threshold: float = 0.5,
    min_face_landmarks_confidence: float = 0.5,
    min_pose_detection_confidence: float = 0.5,
    min_pose_suppression_threshold: float = 0.5,
    min_pose_landmarks_confidence: float = 0.5,
    min_hand_landmarks_confidence: float = 0.5,
    output_face_blendshapes: bool = False,
    output_segmentation_mask: bool = False,
) -> mp.tasks.vision.HolisticLandmarker:
    """Create a Holistic Landmarker with the current Tasks API."""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    holistic_model_path = resolve_holistic_model_path(model_path)
    options = vision.HolisticLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(holistic_model_path)),
        running_mode=running_mode,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_suppression_threshold=min_face_suppression_threshold,
        min_face_landmarks_confidence=min_face_landmarks_confidence,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_suppression_threshold=min_pose_suppression_threshold,
        min_pose_landmarks_confidence=min_pose_landmarks_confidence,
        min_hand_landmarks_confidence=min_hand_landmarks_confidence,
        output_face_blendshapes=output_face_blendshapes,
        output_segmentation_mask=output_segmentation_mask,
        result_callback=result_callback,
    )
    return vision.HolisticLandmarker.create_from_options(options)

