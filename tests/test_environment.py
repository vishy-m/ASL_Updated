"""Smoke tests verifying the ASL CSLR development environment is functional."""

import numpy as np


def test_pytorch_imports_and_cpu():
    import torch

    assert torch.__version__
    x = torch.randn(2, 3)
    assert x.shape == (2, 3)
    assert x.device.type == "cpu"


def test_pytorch_conv1d_bilstm():
    import torch
    import torch.nn as nn

    conv = nn.Conv1d(104, 64, kernel_size=5, padding=2)
    lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
    x = torch.randn(1, 30, 104)
    x = conv(x.permute(0, 2, 1)).permute(0, 2, 1)
    out, _ = lstm(x)
    assert out.shape == (1, 30, 128)


def test_ctc_loss():
    import torch
    import torch.nn as nn

    T, B, C = 30, 1, 11
    log_probs = torch.randn(T, B, C).log_softmax(dim=-1)
    targets = torch.tensor([[1, 3, 5]])
    input_lengths = torch.tensor([T])
    target_lengths = torch.tensor([3])
    loss = nn.CTCLoss(blank=0)(log_probs, targets, input_lengths, target_lengths)
    assert loss.item() > 0
    assert not np.isnan(loss.item())


def test_mediapipe_loads():
    import mediapipe as mp

    assert hasattr(mp.tasks, "vision")
    assert hasattr(mp.tasks.vision, "PoseLandmarker")
    assert hasattr(mp.tasks.vision, "HandLandmarker")


def test_opencv_image_ops():
    import cv2

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (100, 100)


def test_numpy_skeleton_normalization():
    joints = np.random.rand(52, 2).astype(np.float32)
    l_shoulder = joints[1]
    r_shoulder = joints[2]
    ref = (l_shoulder + r_shoulder) / 2
    d = np.linalg.norm(l_shoulder - r_shoulder)
    scale = max(d, 1e-3)
    normalized = (joints - ref) / scale
    vec = normalized.flatten()
    assert vec.shape == (104,)


def test_h5py_roundtrip(tmp_path):
    import h5py

    data = np.random.randn(30, 104).astype(np.float32)
    path = str(tmp_path / "test.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("skeleton", data=data)
    with h5py.File(path, "r") as f:
        loaded = f["skeleton"][:]
    np.testing.assert_array_equal(data, loaded)


def test_pandas_manifest():
    import pandas as pd

    df = pd.DataFrame({
        "id": ["clip_001", "clip_002"],
        "gloss": [["HELLO", "WORLD"], ["THANK", "YOU"]],
        "split": ["train", "val"],
    })
    assert len(df) == 2
    assert df.iloc[0]["id"] == "clip_001"
