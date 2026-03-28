# ASL Continuous Sign Language Recognition (CSLR)

Skeleton-based ASL recognition system designed for real-time inference on Apple Silicon (M4 Pro). Uses MediaPipe pose+hand keypoints with temporal conv + BiLSTM/Transformer models trained via CTC.

## Architecture

```
Webcam → MediaPipe Pose+Hands → 52-joint skeleton (ℝ^104) → Model → Gloss sequence
```

## Project Structure

```
asl_cslr/            # Core Python package
├── data/            # Skeleton extraction, preprocessing, datasets, vocab
├── models/          # Model architectures (conv+BiLSTM, dual-stream, Transformer)
├── training/        # Training loops, metrics, schedulers
├── online/          # Real-time webcam inference pipeline
└── utils/           # Device management, I/O, logging

configs/             # Training and inference configuration files
scripts/             # CLI entry points (preprocess, train, evaluate, run_online)
data/                # Dataset storage (raw + processed)
```

## Setup

```bash
# 1. Create conda environment
conda create -n asl python=3.11 -y
conda activate asl

# 2. Install PyTorch with MPS support
pip install torch torchvision

# 3. Install project in editable mode
pip install -e .

# 4. Set MPS fallback for unsupported ops
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Pipeline Stages

### 1. Canonical Preprocessing
```bash
python scripts/preprocess.py --dataset wlasl --config configs/preprocessing.json --clean-output
python scripts/preprocess.py --dataset how2sign --config configs/preprocessing.json --clean-output
```

### 2. Rebuild Task Manifests
```bash
python scripts/build_training_manifests.py --config configs/preprocessing.json
python scripts/build_synthetic_cslr_manifests.py --manifest-dir data/processed/manifests
```

This writes:
- `data/processed/manifests/islr_*.jsonl` for the full WLASL backbone.
- `data/processed/manifests/islr_goal_*.jsonl` for the focused live CSLR vocabulary.
- `data/processed/manifests/cslr_*.jsonl` for the weakly supervised How2Sign goal subset.
- `data/processed/manifests/cslr_full_*.jsonl` for the broader pseudo-labeled How2Sign corpus.
- `data/processed/manifests/cslr_synthetic_*.jsonl` for exact-label multi-word CSLR sequences synthesized from goal-vocabulary WLASL clips.

The default focused goal vocabulary is the documented top-10 overlap set:
`BOOK`, `LIKE`, `DRINK`, `WRONG`, `FORGET`, `NOW`, `NEED`, `COLOR`, `HOT`, `FINISH`.
Goal ISLR manifests are stratified so every goal gloss appears in both `val` and `test`.

### 3. Training
```bash
# Stage 1a: full WLASL backbone
python scripts/train.py --config configs/islr_train.yaml

# Stage 1b: goal-vocabulary adaptation
python scripts/train.py --config configs/islr_goal_train.yaml

# Stage 2a: exact-label synthetic CSLR (recommended live path)
python scripts/train.py --config configs/cslr_synthetic_train.yaml

# Stage 2b: weakly supervised How2Sign CSLR
python scripts/train.py --config configs/cslr_train.yaml
```

Optional broader pseudo-labeled CSLR experiment:
```bash
python scripts/train.py --config configs/cslr_full_train.yaml
```

### 4. Evaluation
```bash
python scripts/evaluate.py --checkpoint checkpoints/islr_goal/best.pt --mode islr --split test
python scripts/evaluate.py --checkpoint checkpoints/cslr_synthetic/best.pt --mode cslr --split test
python scripts/evaluate.py --checkpoint checkpoints/cslr/best.pt --mode cslr --split test
```

### 5. Real-time Inference
```bash
python scripts/run_web.py --config configs/online.yaml --mode cslr
```

## Datasets

| Dataset | Type | Usage |
|---------|------|-------|
| How2Sign | Continuous | Primary CSLR training |
| ASLLVD | Lexicon (isolated) | ISLR training |
| NCSLGR/BU | Continuous | CSLR training |
| WLASL | Word-level (isolated) | ISLR training |

See `data/README.md` for download and placement instructions. In this workspace,
How2Sign ships with 2D keypoints and English sentence text; unless gold gloss
annotations are added, the CSLR manifests are built from deterministic
sentence-token pseudo labels. The official How2Sign download script also still
lists gloss annotations as a future/TODO modality, so the exact-label live CSLR
path in this repo is currently the synthetic `cslr_synthetic_*` pipeline built
from isolated WLASL clips.

## Model Families

- **Family A**: Temporal Conv + BiLSTM (baseline)
- **Family B**: Multi-scale/Dual-stream Conv + BiLSTM
- **Family C**: Temporal Conv + Transformer (optional)

## Hardware

Optimized for **M4 Pro MacBook Pro** using:
- PyTorch MPS backend
- Mixed precision (`torch.float16`)
- Compact skeleton-only input (no RGB storage)
