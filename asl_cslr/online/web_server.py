import os
import time
import base64
import logging
import threading
import cv2
import torch

from flask import Flask, render_template, jsonify, send_file, abort, url_for, request
from flask_socketio import SocketIO

from asl_cslr.utils.device import get_device
from asl_cslr.online.camera import WebcamCapture
from asl_cslr.online.model_loader import (
    load_online_cslr_model,
    load_online_islr_model,
)
from asl_cslr.online.sign_examples import WlaslExampleIndex
from asl_cslr.online.pipeline import (
    SlidingWindowISLR,
    StreamingCSLR,
    get_online_runtime_config,
    validate_online_model_schema,
)
from asl_cslr.online.visualizer import draw_skeleton

logger = logging.getLogger(__name__)

class ASLWebServer:
    def __init__(self, config, mode="cslr", port=5050, load_models=True, example_index=None):
        self.config = config
        self.mode = mode
        self.port = port
        self.load_models = load_models

        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.app = Flask(__name__, template_folder=template_dir)
        # Use native threading to prevent OpenCV blocking the async loop
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")

        self.device = get_device()
        self.cam = None
        self.pipeline = None
        self.runtime_cfg = get_online_runtime_config(self.config, self.mode)
        self.example_index = example_index or WlaslExampleIndex()

        self._running = False
        self._camera_thread = None
        self._inference_thread = None
        self._clients: set[str] = set()
        self._controller_sid: str | None = None

        # Shared state between camera and inference threads
        self._latest_glosses_lock = threading.Lock()
        self._latest_glosses: list[str] = []

        self.setup_routes()
        if self.load_models:
            self.setup_models()

    def setup_routes(self):
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule(
            "/api/sign-examples/<gloss>",
            "api_sign_examples",
            self.api_sign_examples,
        )
        self.app.add_url_rule(
            "/api/vocabulary",
            "api_vocabulary",
            self.api_vocabulary,
        )
        self.app.add_url_rule(
            "/media/wlasl/<video_id>",
            "serve_wlasl_media",
            self.serve_wlasl_media,
        )
        self.socketio.on_event("connect", self.on_connect)
        self.socketio.on_event("disconnect", self.on_disconnect)
        self.socketio.on_event("start_camera", self.on_start_camera)
        self.socketio.on_event("stop_camera", self.on_stop_camera)
        self.socketio.on_event("reset", self.on_reset)

    def index(self):
        return render_template("index.html", mode=self.mode)

    def api_sign_examples(self, gloss):
        examples = self.example_index.list_examples(gloss)
        payload = {
            "gloss": gloss.upper(),
            "canonical_gloss": examples[0].gloss if examples else gloss.upper(),
            "count": len(examples),
            "examples": [
                {
                    "video_id": ex.video_id,
                    "gloss": ex.gloss,
                    "split": ex.split,
                    "signer_id": ex.signer_id,
                    "video_url": url_for("serve_wlasl_media", video_id=ex.video_id),
                }
                for ex in examples
            ],
        }
        return jsonify(payload)

    def api_vocabulary(self):
        """Return the list of glosses the current model can detect."""
        glosses = []
        if self.pipeline is not None:
            vocab = self.pipeline.vocab
            special = set(vocab.special_indices(include_blank=True))
            for idx in range(len(vocab)):
                if idx not in special:
                    glosses.append(vocab.decode(idx))
        glosses.sort()
        return jsonify({"glosses": glosses, "count": len(glosses)})

    def serve_wlasl_media(self, video_id):
        video_path = self.example_index.resolve_video_path(video_id)
        if video_path is None or not video_path.exists():
            abort(404)
        return send_file(video_path, mimetype="video/mp4", conditional=True)

    def on_connect(self):
        self._clients.add(request.sid)
        if self._controller_sid is None:
            self._controller_sid = request.sid
        logger.info("Client connected")
        self.socketio.emit("status", {"message": "Connected", "mode": self.mode})

    def on_disconnect(self):
        self._clients.discard(request.sid)
        if self._controller_sid == request.sid:
            self._controller_sid = next(iter(self._clients), None)
        logger.info("Client disconnected")
        if not self._clients:
            self.stop_processing()

    def on_start_camera(self):
        if not self._running:
            if not self.load_models or self.pipeline is None:
                self.socketio.emit("status", {"message": "Camera unavailable: models are not loaded"})
                return
            self._controller_sid = request.sid
            self._running = True
            logger.info("Starting camera and inference threads")
            self._camera_thread = self.socketio.start_background_task(self._camera_loop)
            self._inference_thread = self.socketio.start_background_task(self._inference_loop)
            self.socketio.emit("status", {"message": "Camera started"})

    def on_stop_camera(self):
        if self._controller_sid is not None and request.sid != self._controller_sid:
            self.socketio.emit("status", {"message": "Camera control is owned by another client"})
            return
        self.stop_processing()
        self.socketio.emit("status", {"message": "Camera stopped"})

    def on_reset(self):
        if self._controller_sid is not None and request.sid != self._controller_sid:
            self.socketio.emit("status", {"message": "Camera control is owned by another client"})
            return
        if self.pipeline:
            self.pipeline.reset()
            self.socketio.emit("glosses", {"sequence": []})
            logger.info("Pipeline reset")

    def stop_processing(self):
        self._running = False
        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)
            self._inference_thread = None
        if self._camera_thread:
            self._camera_thread.join(timeout=2.0)
            self._camera_thread = None

    def setup_models(self):
        logger.info(f"Setting up models for mode: {self.mode}")
        if self.mode == "islr":
            cfg = self.config["islr"]
            model, vocab, _ckpt, _mcfg = load_online_islr_model(
                cfg["checkpoint"],
                cfg["vocab_path"],
                self.device,
                model_overrides=cfg.get("model_overrides"),
            )
            validate_online_model_schema(model)

            self.pipeline = SlidingWindowISLR(
                model=model,
                vocab=vocab,
                effective_fps=self.runtime_cfg["effective_fps"],
                window_duration_sec=cfg["window_duration_sec"],
                hop_duration_sec=cfg["hop_duration_sec"],
                stability_windows=cfg["stability_windows"],
                confidence_threshold=cfg["confidence_threshold"],
                confidence_margin_threshold=cfg.get(
                    "confidence_margin_threshold", 0.08
                ),
                motion_energy_threshold=cfg.get("motion_energy_threshold", 0.01),
                min_buffer_frames=cfg.get("min_buffer_frames", 8),
            )
        elif self.mode == "cslr":
            cfg = self.config["cslr"]
            model, vocab, _ckpt, _mcfg = load_online_cslr_model(
                cfg["checkpoint"],
                cfg["vocab_path"],
                self.device,
                model_overrides=cfg.get("model_overrides"),
            )
            validate_online_model_schema(model)

            self.pipeline = StreamingCSLR(
                model=model,
                vocab=vocab,
                decode_interval_sec=cfg["decode_interval_sec"],
                effective_fps=self.runtime_cfg["effective_fps"],
                stability_windows=cfg.get("stability_windows", 3),
                history_size=cfg.get("history_size", 6),
                motion_energy_threshold=cfg.get("motion_energy_threshold", 0.008),
                blank_rejection_threshold=cfg.get(
                    "blank_rejection_threshold", 0.88
                ),
                min_buffer_frames=cfg.get("min_buffer_frames", 8),
                inactivity_reset_windows=cfg.get("inactivity_reset_windows", 3),
                pause_commit_windows=cfg.get("pause_commit_windows"),
                cumulative_commits=cfg.get("cumulative_commits", True),
            )

    def _init_camera(self):
        """Create and start the webcam capture instance."""
        c_cfg = self.config["camera"]
        self.cam = WebcamCapture(
            device_id=c_cfg["device_id"],
            capture_fps=c_cfg["capture_fps"],
            capture_width=c_cfg.get("capture_width"),
            capture_height=c_cfg.get("capture_height"),
            inference_width=c_cfg.get("inference_width"),
            inference_height=c_cfg.get("inference_height"),
            downsample_factor=c_cfg["downsample_factor"],
            buffer_duration_sec=self.runtime_cfg["buffer_duration_sec"],
            smoothing_alpha=c_cfg.get("smoothing_alpha", 0.35),
            max_pending_timestamps=c_cfg.get("max_pending_timestamps", 2),
            holistic_model_path=c_cfg.get("holistic_model_path"),
            min_face_detection_confidence=c_cfg.get(
                "min_face_detection_confidence", 0.5
            ),
            min_face_landmarks_confidence=c_cfg.get(
                "min_face_landmarks_confidence", 0.5
            ),
            min_pose_detection_confidence=c_cfg.get(
                "min_pose_detection_confidence", 0.5
            ),
            min_pose_landmarks_confidence=c_cfg.get(
                "min_pose_landmarks_confidence", 0.5
            ),
            min_hand_landmarks_confidence=c_cfg.get(
                "min_hand_landmarks_confidence", 0.5
            ),
            min_face_suppression_threshold=c_cfg.get(
                "min_face_suppression_threshold", 0.5
            ),
            min_pose_suppression_threshold=c_cfg.get(
                "min_pose_suppression_threshold", 0.5
            ),
            pose_visibility_threshold=c_cfg.get(
                "pose_visibility_threshold", 0.5
            ),
            pose_presence_threshold=c_cfg.get(
                "pose_presence_threshold", 0.5
            ),
            hand_visibility_threshold=c_cfg.get(
                "hand_visibility_threshold"
            ),
            hand_presence_threshold=c_cfg.get(
                "hand_presence_threshold"
            ),
        )
        self.cam.start()

    def _camera_loop(self):
        """Thread 1: Capture frames, run MediaPipe, and stream video to browser.

        This thread is decoupled from inference so that frame delivery to the
        browser is never blocked by model forward passes.
        """
        try:
            self._init_camera()
        except Exception as e:
            logger.error(f"Failed to open camera: {e}")
            self.socketio.emit("status", {"message": f"Camera Error: {e}"})
            self._running = False
            return

        c_cfg = self.config["camera"]
        target_fps = c_cfg["capture_fps"]
        frame_time = 1.0 / target_fps
        last_emitted_result_id = -1

        win_w = self.config.get("display", {}).get("window_width", 1280)
        win_h = self.config.get("display", {}).get("window_height", 720)
        show_skeleton = self.config.get("display", {}).get("show_skeleton", True)

        try:
            while self._running:
                loop_start = time.time()

                frame, raw_joints, _skeleton, result_id = self.cam.read_frame()

                has_new = result_id > 0 and result_id != last_emitted_result_id
                if frame is not None and (result_id == 0 or has_new):
                    frame = cv2.flip(frame, 1)

                    if show_skeleton and raw_joints is not None:
                        raw_joints_flipped = raw_joints.copy()
                        raw_joints_flipped[:, 0] = 1.0 - raw_joints[:, 0]
                        draw_skeleton(frame, raw_joints_flipped)

                    frame = cv2.resize(frame, (win_w, win_h))

                    ret, buf = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    )
                    if ret:
                        b64 = base64.b64encode(buf).decode("utf-8")
                        self.socketio.emit(
                            "camera_frame",
                            {"image": f"data:image/jpeg;base64,{b64}"},
                        )
                        if result_id > 0:
                            last_emitted_result_id = result_id

                elapsed = time.time() - loop_start
                self.socketio.sleep(max(0.0, frame_time - elapsed))

        except Exception as e:
            logger.error(f"Error in camera loop: {e}", exc_info=True)
        finally:
            if self.cam:
                self.cam.stop()
                self.cam = None
            self._running = False
            self.socketio.emit("status", {"message": "Camera stopped"})

    def _inference_loop(self):
        """Thread 2: Periodically run model inference on the skeleton buffer.

        Runs independently of camera frame delivery so that model latency
        does not cause visible lag in the video stream.
        """
        hop_interval = self.runtime_cfg["hop_interval_sec"]
        last_processed_result_id = -1
        last_output: list[str] = []

        try:
            while self._running:
                # Wait until camera is initialized
                if self.cam is None:
                    self.socketio.sleep(0.05)
                    continue

                with self.cam._state_lock:
                    result_id = self.cam._last_result_id
                if result_id <= last_processed_result_id:
                    self.socketio.sleep(hop_interval * 0.25)
                    continue

                buf = self.cam.get_full_buffer()
                if buf is not None and len(buf) > 0:
                    self.pipeline.process_buffer(buf)
                    output = self.pipeline.get_output()
                    if output != last_output:
                        self.socketio.emit("glosses", {"sequence": output})
                        with self._latest_glosses_lock:
                            self._latest_glosses = list(output)
                        last_output = list(output)
                    last_processed_result_id = result_id

                self.socketio.sleep(hop_interval)

        except Exception as e:
            logger.error(f"Error in inference loop: {e}", exc_info=True)

    def run(self):
        logger.info(f"Starting ASL Web Server on http://0.0.0.0:{self.port} in mode {self.mode}")
        self.socketio.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            debug=False,
            allow_unsafe_werkzeug=True,
        )
