"""
I/O utilities for data files and configurations.
"""

import json
import logging
from pathlib import Path

import numpy as np
import yaml

from asl_cslr.data.skeleton import NUM_JOINTS, NUM_COORDS, extract_coordinate_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json_config(path: str | Path) -> dict:
    """Load a JSON configuration file."""
    with open(path, "r") as f:
        return json.load(f)


def load_config(path: str | Path) -> dict:
    """Load config from YAML or JSON based on extension."""
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        return load_yaml_config(path)
    elif path.suffix == ".json":
        return load_json_config(path)
    else:
        raise ValueError(f"Unknown config format: {path.suffix}")


# ---------------------------------------------------------------------------
# JSONL manifest I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    """Read a .jsonl file (one JSON object per line)."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_jsonl(entries: list[dict], path: str | Path):
    """Write entries to a .jsonl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def append_jsonl(entry: dict, path: str | Path):
    """Append a single entry to a .jsonl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# NPZ I/O
# ---------------------------------------------------------------------------

def load_skeleton(path: str | Path) -> dict[str, np.ndarray]:
    """Load a skeleton .npz file.

    Returns dict with 'X' and optionally 'X_vel', 'X_acc'.
    """
    data = np.load(str(path))
    return {key: data[key] for key in data.files}


def save_skeleton(
    path: str | Path,
    X: np.ndarray,
    X_vel: np.ndarray | None = None,
    X_acc: np.ndarray | None = None,
    compress: bool = True,
):
    """Save skeleton arrays to .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {"X": X.astype(np.float32)}
    save_dict["schema_version"] = np.array(2, dtype=np.int32)
    save_dict["num_joints"] = np.array(NUM_JOINTS, dtype=np.int32)
    save_dict["num_coords"] = np.array(NUM_COORDS, dtype=np.int32)
    save_dict["coord_feature_dim"] = np.array(
        X_vel.shape[1] if X_vel is not None else extract_coordinate_features(X[:1]).shape[1],
        dtype=np.int32,
    )
    save_dict["frame_feature_dim"] = np.array(X.shape[1], dtype=np.int32)
    if X_vel is not None:
        save_dict["X_vel"] = X_vel.astype(np.float32)
    if X_acc is not None:
        save_dict["X_acc"] = X_acc.astype(np.float32)

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(str(path), **save_dict)
