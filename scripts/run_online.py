#!/usr/bin/env python3
"""
CLI: Start the real-time webcam ASL recognition pipeline.

Usage:
    python scripts/run_online.py --config configs/online.yaml
    python scripts/run_online.py --config configs/online.yaml --mode islr
"""

import argparse
import logging
import time

import cv2

from asl_cslr.utils.logging import setup_logging
from asl_cslr.utils.io import load_yaml_config
from asl_cslr.utils.device import get_device
from asl_cslr.online.camera import WebcamCapture
from asl_cslr.online.model_loader import (
    load_online_cslr_model,
    load_online_islr_model,
)
from asl_cslr.online.pipeline import (
    SlidingWindowISLR,
    StreamingCSLR,
    get_online_runtime_config,
    resolve_online_mode,
    validate_online_model_schema,
)
from asl_cslr.online.visualizer import draw_skeleton, draw_glosses

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time ASL recognition from webcam."
    )
    parser.add_argument(
        "--config",
        default="configs/online.yaml",
        help="Online inference config YAML.",
    )
    parser.add_argument(
        "--mode",
        choices=["islr", "cslr"],
        help="Inference mode (default: auto from config).",
    )
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level))

    config = load_yaml_config(args.config)

    try:
        mode = resolve_online_mode(config, args.mode)
    except ValueError as exc:
        logger.error(str(exc))
        return

    logger.info(f"Starting online recognition — mode: {mode}")
    runtime_cfg = get_online_runtime_config(config, mode)
    device = get_device()

    # Load vocab and model
    if mode == "islr":
        model, vocab, _ckpt, _mcfg = load_online_islr_model(
            config["islr"]["checkpoint"],
            config["islr"]["vocab_path"],
            device,
            model_overrides=config["islr"].get("model_overrides"),
        )
        validate_online_model_schema(model)

        pipeline = SlidingWindowISLR(
            model=model,
            vocab=vocab,
            effective_fps=runtime_cfg["effective_fps"],
            window_duration_sec=config["islr"]["window_duration_sec"],
            hop_duration_sec=config["islr"]["hop_duration_sec"],
            stability_windows=config["islr"]["stability_windows"],
            confidence_threshold=config["islr"]["confidence_threshold"],
            confidence_margin_threshold=config["islr"].get(
                "confidence_margin_threshold", 0.08
            ),
            motion_energy_threshold=config["islr"].get(
                "motion_energy_threshold", 0.01
            ),
            min_buffer_frames=config["islr"].get("min_buffer_frames", 8),
        )

    elif mode == "cslr":
        model, vocab, _ckpt, _mcfg = load_online_cslr_model(
            config["cslr"]["checkpoint"],
            config["cslr"]["vocab_path"],
            device,
            model_overrides=config["cslr"].get("model_overrides"),
        )
        validate_online_model_schema(model)

        pipeline = StreamingCSLR(
            model=model,
            vocab=vocab,
            decode_interval_sec=config["cslr"]["decode_interval_sec"],
            effective_fps=runtime_cfg["effective_fps"],
            stability_windows=config["cslr"].get("stability_windows", 3),
            history_size=config["cslr"].get("history_size", 6),
            motion_energy_threshold=config["cslr"].get(
                "motion_energy_threshold", 0.008
            ),
            blank_rejection_threshold=config["cslr"].get(
                "blank_rejection_threshold", 0.88
            ),
            min_buffer_frames=config["cslr"].get("min_buffer_frames", 8),
            inactivity_reset_windows=config["cslr"].get(
                "inactivity_reset_windows", 3
            ),
            pause_commit_windows=config["cslr"].get("pause_commit_windows"),
            cumulative_commits=config["cslr"].get("cumulative_commits", True),
        )

    # Camera
    cam = WebcamCapture(
        device_id=config["camera"]["device_id"],
        capture_fps=config["camera"]["capture_fps"],
        capture_width=config["camera"].get("capture_width"),
        capture_height=config["camera"].get("capture_height"),
        inference_width=config["camera"].get("inference_width"),
        inference_height=config["camera"].get("inference_height"),
        downsample_factor=config["camera"]["downsample_factor"],
        buffer_duration_sec=runtime_cfg["buffer_duration_sec"],
        smoothing_alpha=config["camera"].get("smoothing_alpha", 0.35),
        max_pending_timestamps=config["camera"].get("max_pending_timestamps", 2),
        holistic_model_path=config["camera"].get("holistic_model_path"),
        min_face_detection_confidence=config["camera"].get(
            "min_face_detection_confidence", 0.5
        ),
        min_face_landmarks_confidence=config["camera"].get(
            "min_face_landmarks_confidence", 0.5
        ),
        min_pose_detection_confidence=config["camera"].get(
            "min_pose_detection_confidence", 0.5
        ),
        min_pose_landmarks_confidence=config["camera"].get(
            "min_pose_landmarks_confidence", 0.5
        ),
        min_hand_landmarks_confidence=config["camera"].get(
            "min_hand_landmarks_confidence", 0.5
        ),
        min_face_suppression_threshold=config["camera"].get(
            "min_face_suppression_threshold", 0.5
        ),
        min_pose_suppression_threshold=config["camera"].get(
            "min_pose_suppression_threshold", 0.5
        ),
        pose_visibility_threshold=config["camera"].get(
            "pose_visibility_threshold", 0.5
        ),
        pose_presence_threshold=config["camera"].get(
            "pose_presence_threshold", 0.5
        ),
        hand_visibility_threshold=config["camera"].get(
            "hand_visibility_threshold"
        ),
        hand_presence_threshold=config["camera"].get(
            "hand_presence_threshold"
        ),
    )

    show_skeleton = config.get("display", {}).get("show_skeleton", True)
    show_glosses = config.get("display", {}).get("show_glosses", True)
    win_w = config.get("display", {}).get("window_width", 1280)
    win_h = config.get("display", {}).get("window_height", 720)

    cam.start()
    logger.info("Press 'q' to quit, 'r' to reset")

    try:
        hop_interval = runtime_cfg["hop_interval_sec"]
        last_hop_time = time.time()
        last_processed_result_id = -1

        while True:
            frame, raw_joints, _skeleton, _result_id = cam.read_frame()
            if frame is None:
                break

            # Draw skeleton overlay
            if show_skeleton and raw_joints is not None:
                draw_skeleton(frame, raw_joints)

            # Run inference at hop intervals
            now = time.time()
            if now - last_hop_time >= hop_interval and _result_id > last_processed_result_id:
                buffer = cam.get_full_buffer()
                if buffer is not None and len(buffer) > 0:
                    pipeline.process_buffer(buffer)
                    last_processed_result_id = _result_id
                last_hop_time = now

            # Draw recognized glosses
            if show_glosses:
                draw_glosses(frame, pipeline.get_output())

            # Display
            frame_display = cv2.resize(frame, (win_w, win_h))
            cv2.imshow("ASL Recognition", frame_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                pipeline.reset()
                logger.info("Pipeline reset")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
