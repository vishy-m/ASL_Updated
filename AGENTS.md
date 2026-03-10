# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a **greenfield ASL (American Sign Language) Continuous Sign Language Recognition** project. The architecture plan lives in `Entire Plan.md`. The tech stack is Python 3.12 + PyTorch + MediaPipe + OpenCV.

### Virtual environment

All commands should run inside the venv at `.venv/`:

```bash
source .venv/bin/activate
```

### Key commands

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Lint | `ruff check .` |
| Lint (auto-fix) | `ruff check --fix .` |
| Tests | `pytest -v` |
| Demo | `python scripts/hello_world_demo.py` |

### Gotchas

- **MediaPipe Tasks API**: MediaPipe >=0.10.14 dropped `mp.solutions.*`. Use the `mp.tasks.vision` API instead (e.g. `PoseLandmarker`, `HandLandmarker`). Model `.task` files must be downloaded to `models/` — they are git-ignored.
- **No GPU in cloud VM**: PyTorch installs with CUDA stubs but `torch.cuda.is_available()` returns `False`. All training/inference runs on CPU here. The plan targets Apple MPS on M4 Pro, which is irrelevant in the cloud environment.
- **MediaPipe pose detection on synthetic images**: MediaPipe will not detect landmarks on simple stick-figure drawings. Use real person photographs or webcam frames to test the full skeleton extraction pipeline.
