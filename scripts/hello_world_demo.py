"""
Hello-world demo for the ASL CSLR development environment.

Exercises the full planned pipeline end-to-end:
  1. MediaPipe Pose + Hands skeleton extraction on a synthetic image
  2. Normalization to the canonical 52-joint layout (T, 104) feature vectors
  3. A minimal Conv1d + BiLSTM + CTC model forward pass on CPU
"""

import os

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # 9 real pose joints


# ---------------------------------------------------------------------------
# 1. Skeleton extraction with MediaPipe Tasks API
# ---------------------------------------------------------------------------

def extract_skeleton_from_image(image_rgb: np.ndarray):
    """Run MediaPipe Pose + Hands on a single RGB image and return raw landmarks."""
    results = {"pose": None, "hands": [], "handedness": []}

    pose_model = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
    hand_model = os.path.join(MODELS_DIR, "hand_landmarker.task")

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=pose_model),
        num_poses=1,
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose:
        pose_result = pose.detect(mp_image)
        if pose_result.pose_landmarks:
            results["pose"] = pose_result.pose_landmarks[0]

    hand_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=hand_model),
        num_hands=2,
    )
    with mp.tasks.vision.HandLandmarker.create_from_options(hand_options) as hands:
        hand_result = hands.detect(mp_image)
        if hand_result.hand_landmarks:
            results["hands"] = hand_result.hand_landmarks
            results["handedness"] = hand_result.handedness

    return results


# ---------------------------------------------------------------------------
# 2. 52-joint normalization (matching the plan exactly)
# ---------------------------------------------------------------------------

def landmarks_to_52joint_vector(results: dict, img_h: int, img_w: int) -> np.ndarray:
    """Convert MediaPipe results to a 104-dim normalized feature vector."""
    joints = np.zeros((52, 2), dtype=np.float32)

    if results["pose"] is not None:
        plm = results["pose"]
        for i, idx in enumerate(POSE_INDICES):
            joints[i] = [plm[idx].x, plm[idx].y]
        joints[9] = (joints[1] + joints[2]) / 2  # MID_SHOULDERS

    left_hand, right_hand = None, None
    for hand_lm, hand_cls in zip(results["hands"], results["handedness"]):
        label = hand_cls[0].category_name
        if label == "Left":
            left_hand = hand_lm
        else:
            right_hand = hand_lm

    if left_hand is not None:
        for j in range(21):
            joints[10 + j] = [left_hand[j].x, left_hand[j].y]

    if right_hand is not None:
        for j in range(21):
            joints[31 + j] = [right_hand[j].x, right_hand[j].y]

    l_shoulder = joints[1]
    r_shoulder = joints[2]
    ref = (l_shoulder + r_shoulder) / 2
    d_shoulder = np.linalg.norm(l_shoulder - r_shoulder)
    scale = max(d_shoulder, 1e-3)

    normalized = (joints - ref) / scale
    return normalized.flatten()  # (104,)


# ---------------------------------------------------------------------------
# 3. Minimal Conv1d + BiLSTM + CTC model (matching the plan's Family A)
# ---------------------------------------------------------------------------

class MiniCSlrModel(nn.Module):
    """Tiny skeleton-based CSLR model for environment validation."""

    def __init__(self, input_dim=104, conv_dim=64, hidden_size=64,
                 num_layers=1, vocab_size=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(conv_dim),
        )
        self.lstm = nn.LSTM(
            conv_dim, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, vocab_size + 1)  # +1 for CTC blank

    def forward(self, x):
        # x: (B, T, 104)
        x = x.permute(0, 2, 1)       # (B, 104, T)
        x = self.conv(x)              # (B, conv_dim, T)
        x = x.permute(0, 2, 1)       # (B, T, conv_dim)
        x, _ = self.lstm(x)           # (B, T, 2*hidden)
        logits = self.fc(x)           # (B, T, vocab+1)
        return logits


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ASL CSLR Development Environment – Hello World Demo")
    print("=" * 60)

    # --- Step 1: Generate synthetic test image and extract skeleton ---
    print("\n[1/4] Generating synthetic test image (480x640)…")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 120), 40, (200, 180, 160), -1)  # head
    cv2.line(img, (320, 160), (320, 320), (200, 180, 160), 8)  # torso
    cv2.line(img, (320, 200), (220, 280), (200, 180, 160), 6)  # left arm
    cv2.line(img, (320, 200), (420, 280), (200, 180, 160), 6)  # right arm
    cv2.line(img, (320, 320), (280, 440), (200, 180, 160), 6)  # left leg
    cv2.line(img, (320, 320), (360, 440), (200, 180, 160), 6)  # right leg
    print(f"     Image shape: {img.shape}")

    print("\n[2/4] Running MediaPipe Pose + Hands (Tasks API)…")
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = extract_skeleton_from_image(image_rgb)
    pose_detected = results["pose"] is not None
    hands_detected = len(results["hands"]) > 0
    print(f"     Pose detected:  {pose_detected}")
    print(f"     Hands detected: {hands_detected}")

    # --- Step 2: Build a synthetic skeleton sequence ---
    print("\n[3/4] Building normalized skeleton sequence…")
    T = 30  # 30 frames (~1 second at 30 fps)
    if pose_detected:
        frame_vec = landmarks_to_52joint_vector(results, 480, 640)
        sequence = np.stack([
            frame_vec + np.random.randn(104).astype(np.float32) * 0.02
            for _ in range(T)
        ])
        print("     Using real MediaPipe landmarks + noise")
    else:
        print("     (No pose detected on synthetic stick-figure – using random skeleton data)")
        sequence = np.random.randn(T, 104).astype(np.float32) * 0.1

    print(f"     Sequence shape: {sequence.shape}  (T={T}, D=104)")

    # --- Step 3: Run through PyTorch model ---
    print("\n[4/4] Running Conv1d+BiLSTM+CTC model forward pass on CPU…")
    vocab_size = 10
    model = MiniCSlrModel(vocab_size=vocab_size)
    model.eval()

    x = torch.from_numpy(sequence).unsqueeze(0)  # (1, T, 104)
    with torch.no_grad():
        logits = model(x)  # (1, T, vocab+1)

    print(f"     Input tensor:  {tuple(x.shape)}")
    print(f"     Output logits: {tuple(logits.shape)}")

    # Greedy CTC decode
    preds = logits.argmax(dim=-1).squeeze(0)  # (T,)
    decoded = []
    prev = -1
    for p in preds.tolist():
        if p != 0 and p != prev:  # skip blank (0) and repeats
            decoded.append(p)
        prev = p
    print(f"     Greedy CTC decode → gloss IDs: {decoded}")

    # CTC loss sanity check
    log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, B, C)
    target = torch.tensor([[1, 3, 5]])
    input_lengths = torch.tensor([T])
    target_lengths = torch.tensor([3])
    ctc_loss = nn.CTCLoss(blank=0)(log_probs, target, input_lengths, target_lengths)
    print(f"     CTC loss (random weights): {ctc_loss.item():.4f}")

    print("\n" + "=" * 60)
    print("✅ All pipeline stages completed successfully!")
    print("   Environment is ready for ASL CSLR development.")
    print("=" * 60)


if __name__ == "__main__":
    main()
