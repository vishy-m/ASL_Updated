# Full Project Context Handoff

Last updated: 2026-03-27

This document is meant to be fed to a fresh model that has no prior context about this repo or this thread. It is intentionally detailed and should supersede any older shorter context notes.

## 1. What This Repo Is

This repository implements a skeleton-based ASL recognition system with:

- offline preprocessing from video or landmark/keypoint sources
- Stage 1 ISLR training for isolated signs
- Stage 2 CSLR training for continuous multi-word sign sequences
- live webcam inference
- a Flask web UI that shows the camera feed and a video example for the current predicted sign

The repo root is:

- `/Volumes/T7/ASL`

The user’s current product goal is CSLR-first, not ISLR-first:

- the user wants people to sign multiple words back to back
- the model should detect the sequence accurately
- the web UI should remain stable and easy to use
- ISLR is still used as a transfer-learning backbone, but it is not the primary product mode

## 2. Current Product Goal

The current goal is:

- robust CSLR on a focused but expanding ASL vocabulary
- broader word coverage than the older 4-word / 10-word pilots
- stable live webcam pipeline using MediaPipe Tasks, not the older solutions API
- web UI showing a single dedicated example video area instead of repeatedly opening popups

Concretely, the desired live behavior is:

1. the webcam captures the signer
2. MediaPipe Holistic Tasks extracts landmarks
3. the repo converts those into the project’s canonical skeleton feature representation
4. the CSLR model predicts a gloss sequence
5. the web UI shows the current guess and plays an example sign video from dataset media

## 3. Important High-Level Truths

These are the most important reality checks:

- The repo is now much healthier structurally than it was earlier in the project.
- The current best broader-vocabulary CSLR result is strong on the rebuilt synthetic held-out test set.
- That test set is still synthetic continuous signing assembled from real isolated clips, not a real continuous webcam benchmark.
- So the current best checkpoint is strong evidence that the pipeline is working, but not proof that live 27-word webcam CSLR is solved.

## 4. Current Best Artifacts

### Best broader-vocabulary Kaggle-expanded CSLR checkpoint

- Checkpoint:
  - `/Volumes/T7/ASL/checkpoints/cslr_kaggle_expanded30_short_motion/best.pt`
- Training config:
  - `/Volumes/T7/ASL/configs/cslr_kaggle_expanded30_short_motion_train.yaml`
- Vocabulary:
  - `/Volumes/T7/ASL/data/processed/kaggle_expanded30_short/manifests/demo_vocab.json`
- Best validation result:
  - epoch `6`
  - val WER `0.5332`
  - val CER `0.5332`
- Held-out test result:
  - WER `0.0837`
  - CER `0.0837`
  - sentences `1024`

### Stage 1 backbone used for warm start

- Checkpoint:
  - `/Volumes/T7/ASL/checkpoints/islr_kaggle_expanded30/best.pt`
- Role:
  - used as Stage 1 transfer backbone for the current Kaggle-expanded CSLR run
- Held-out test result:
  - top-1 `0.5357`
  - top-5 `0.9226`
  - macro `0.5427`
  - samples `504`

### Broader live config created for the new model

- Config:
  - `/Volumes/T7/ASL/configs/online_kaggle_expanded30.yaml`

Important:

- I did **not** replace `/Volumes/T7/ASL/configs/online.yaml` as the default live config.
- The new Kaggle-expanded config was smoke-tested successfully, but the new checkpoint is still only strongly validated on synthetic continuous data.
- Keeping the old default avoids silently replacing the established smaller live path with a broader but less live-proven model.

## 5. Current Vocabulary of the New Broader Model

The current Kaggle-expanded 27-gloss vocabulary is:

- `ABOUT`
- `APPLE`
- `BED`
- `BELIEVE`
- `BROTHER`
- `CITY`
- `DEAF`
- `DECIDE`
- `DOCTOR`
- `DOG`
- `EAT`
- `FINE`
- `FINISH`
- `HOME`
- `HOW`
- `KNOW`
- `LATER`
- `MONEY`
- `MOTHER`
- `NAME`
- `NEW`
- `PLAY`
- `RED`
- `RIGHT`
- `WALK`
- `WANT`
- `WORK`

The vocabulary file also contains reserved tokens:

- `<blank>`
- `<pad>`
- `<bos>`
- `<eos>`
- `<unk>`

Those reserved indices live in:

- `/Volumes/T7/ASL/asl_cslr/data/vocab.py`

## 6. Data Inventory on Disk

### Kaggle datasets downloaded and usable

#### ASL Citizen keypoints

- Path:
  - `/Volumes/T7/ASL/data/raw/kaggle/asl_citizen_keypoints`
- Key content:
  - `keypoints-400`
- Count:
  - `19,779` `.pkl` files
- Fit with architecture:
  - very good
  - these are already landmark/keypoint style artifacts that can be remapped into the canonical skeleton representation

#### Kaggle WLASL processed videos

- Path:
  - `/Volumes/T7/ASL/data/raw/kaggle/wlasl_processed`
- Key content:
  - `WLASL_v0.3.json`
  - `videos/`
- Count:
  - `11,980` `.mp4` files
- Fit with architecture:
  - very good
  - can be run through the repo’s MediaPipe Tasks preprocessing

#### Kaggle WLASL holistic keypoints

- Path:
  - `/Volumes/T7/ASL/data/raw/kaggle/wlasl_keypoints`
- Key content:
  - `output_V_WLASL/*.npy`
- Count:
  - `11,978` `.npy` files
- Sample shape:
  - `(30, 1662)` float-like arrays
- Fit with architecture:
  - excellent
  - these are easy to remap into the repo’s canonical 52-joint representation without rerunning MediaPipe

### Processed manifest / dataset artifacts relevant to the current best path

- Full processed WLASL keypoint manifest:
  - `/Volumes/T7/ASL/data/processed/manifests/wlasl_kaggle_keypoints.jsonl`
- ASL Citizen processed manifest:
  - `/Volumes/T7/ASL/data/processed/manifests/asl_citizen.jsonl`
- Kaggle-expanded short continuous CSLR set:
  - `/Volumes/T7/ASL/data/processed/kaggle_expanded30_short`

Current synthetic CSLR split sizes for the best run:

- train:
  - `/Volumes/T7/ASL/data/processed/kaggle_expanded30_short/manifests/cslr_demo_train.jsonl`
  - `8192` samples
- val:
  - `/Volumes/T7/ASL/data/processed/kaggle_expanded30_short/manifests/cslr_demo_val.jsonl`
  - `1024` samples
- test:
  - `/Volumes/T7/ASL/data/processed/kaggle_expanded30_short/manifests/cslr_demo_test.jsonl`
  - `1024` samples

## 7. Representation / Feature Contract

The current main path uses a packed canonical skeleton representation with explicit observed-mask channels.

### Skeleton layout

The learned representation uses:

- 52 joints total
- 3 coordinates per joint: `x, y, z`
- 1 observation/imputation mask channel per joint

This makes:

- frame feature dim = `208`
  - `52 * 4`
- motion feature dim = `156`
  - `52 * 3`
- single-stream CSLR input dim = `364`
  - `208 + 156`

Relevant files:

- `/Volumes/T7/ASL/asl_cslr/data/skeleton.py`
- `/Volumes/T7/ASL/asl_cslr/data/dataset.py`
- `/Volumes/T7/ASL/asl_cslr/utils/model_config.py`

### Missing-joint behavior

Current behavior:

- undetected joints are initially marked as missing
- missing joints are filled forward from previous valid frames when needed
- if missing on the first frame, they are anchored around the shoulder reference rather than the image origin
- the observed/imputed state is carried as a mask channel

This was introduced so the model can distinguish:

- real observed landmarks
- imputed / forward-filled landmarks

### Z values

Z is part of the representation now and is preserved end to end in the current packed schema.

This matters because:

- some signs that look similar in `x/y` differ in depth motion
- the user explicitly asked to consider `z`

## 8. MediaPipe Status

The repo was moved away from the old MediaPipe solutions path and toward the newer MediaPipe Tasks path.

Relevant files:

- `/Volumes/T7/ASL/asl_cslr/data/mediapipe_tasks.py`
- `/Volumes/T7/ASL/asl_cslr/data/preprocessing.py`
- `/Volumes/T7/ASL/asl_cslr/online/camera.py`

Important details:

- preprocessing and live webcam now use the Tasks-based Holistic landmarker path
- the camera code includes smoothing / stale-result control
- visibility-aware filtering exists in the live path for pose confidence thresholds

## 9. Current CSLR Architecture

The current best broader model is a single-stream motion-enabled CSLR model:

- type: CSLR
- encoder type: `bilstm`
- multi-scale temporal conv: enabled
- kernels: `[3, 5, 9]`
- conv dim: `256`
- conv layers: `3`
- BiLSTM hidden size: `256`
- BiLSTM layers: `2`
- dropout:
  - conv `0.1`
  - lstm `0.2`

Source config:

- `/Volumes/T7/ASL/configs/cslr_kaggle_expanded30_short_motion_train.yaml`

Important note:

- The transformer branch was wired and tested in earlier work.
- It did not outperform the best BiLSTM-based live-focused model.
- The current best Kaggle-expanded result is therefore BiLSTM-based, not transformer-based.

## 10. Why Earlier CSLR Runs Failed

This is important context for any future model work.

### Earlier failure mode

Earlier broader CSLR runs often failed with:

- validation WER `1.0`
- empty decoded sequences
- frame-level blank domination

This happened even after:

- more data
- better warm starts
- stronger augmentation

### Root cause diagnosis

The key diagnosis was:

- the pure ISLR warm start itself was **not** blank-collapsed
- before CTC training, the warm-started model emitted many glosses across frames
- after the first CTC epoch, the model often learned that blank was the cheapest path

That meant the problem was not:

- “the backbone is dead”
- “the data cannot separate classes”

It was much more:

- long input sequences relative to label lengths
- CTC training dynamics favoring blank too aggressively

### Quantitative clue

Before the most recent fix, the short 2-3 word synthetic CSLR set had:

- average frames per sequence around `100`
- average label length around `2.5`
- average frame-to-label ratio around `38`

With no temporal downsampling in the model, that made blank-heavy CTC alignments too easy.

## 11. What Fixed the Latest Training Run

The biggest gains in the current best Kaggle-expanded CSLR run came from three changes.

### 1. Stage 1 warm start was aligned correctly

The broader CSLR config now warm-starts from:

- `/Volumes/T7/ASL/checkpoints/islr_kaggle_expanded30/best.pt`

and the architecture was matched so that the warm start actually loads cleanly.

Relevant file:

- `/Volumes/T7/ASL/asl_cslr/models/cslr_model.py`

There is now support for seeding the CTC head from the ISLR classifier head when shapes match.

### 2. CSLR dataset now supports temporal subsampling

Relevant file:

- `/Volumes/T7/ASL/asl_cslr/data/dataset.py`

New important behavior:

- `frame_stride` can now be set for CSLR datasets
- the current best run uses `frame_stride: 2`
- motion features are recomputed when stride changes so pose and motion stay aligned

This reduced effective time length and made CTC alignment much less hostile.

### 3. Blank row control was added during early CTC training

Relevant file:

- `/Volumes/T7/ASL/asl_cslr/training/train_cslr.py`

New helper behavior includes:

- `_configure_ctc_blank_row(...)`
  - allows explicit blank-bias override
  - can zero the blank-row weight on init
  - can push non-blank special-token rows down
- `_freeze_ctc_blank_gradients(...)`
  - can freeze blank-row gradients for early epochs

The current best config uses:

- `ctc_blank_bias_init: -2.5`
- `ctc_zero_blank_weight_on_init: true`
- `ctc_special_bias_init: -4.0`
- `freeze_blank_epochs: 1`

That change prevented immediate epoch-1 blank collapse and gave the model room to learn segmentation more gradually.

## 12. Latest Best Training Run Details

### Config

- `/Volumes/T7/ASL/configs/cslr_kaggle_expanded30_short_motion_train.yaml`

Key settings:

- epochs: `6`
- batch size: `8`
- learning rate: `5e-5`
- cosine scheduler
- warmup epochs: `2`
- balanced sampling: enabled
- frame stride: `2`
- safe augmentation only
- horizontal flip disabled

### Augmentation policy

The user explicitly wanted proper augmentation only, especially because left/right flips can invalidate ASL semantics.

So the current best path uses:

- spatial jitter
- scale
- rotation
- pitch / yaw / roll-like viewpoint perturbation
- translation
- temporal crop
- temporal drop
- speed perturbation
- joint dropout
- hand dropout
- pose dropout

But:

- `flip_prob: 0.0`
- `allow_horizontal_flip: false`

### Validation curve across epochs

For the best Kaggle-expanded 27-gloss short-sequence run:

- epoch 1:
  - val WER `1.2570`
  - nonblank but over-emitting
- epoch 2:
  - val WER `0.9996`
  - model started swinging back toward blank-heavy behavior
- epoch 3:
  - val WER `0.8719`
- epoch 4:
  - val WER `0.6853`
- epoch 5:
  - val WER `0.5870`
- epoch 6:
  - val WER `0.5332`

Final summary:

- train loss at epoch 6: `0.7255`
- best val WER: `0.5332`
- best val CER: `0.5332`

## 13. Final Held-Out Test Result for the Current Best Model

Command used:

```bash
python3 scripts/evaluate.py \
  --checkpoint checkpoints/cslr_kaggle_expanded30_short_motion/best.pt \
  --mode cslr \
  --split test \
  --log-level INFO
```

Result:

- WER: `0.0837`
- CER: `0.0837`
- Sentences: `1024`

This is the strongest broader-vocabulary CSLR result so far in this repo.

Again, important caveat:

- this is on the rebuilt synthetic continuous test split assembled from real isolated clips
- it is not yet a webcam/live signer benchmark

## 14. Error Analysis of the Current Best Model

Command used:

```bash
python3 scripts/analyze_cslr_errors.py \
  --checkpoint checkpoints/cslr_kaggle_expanded30_short_motion/best.pt \
  --split test \
  --top-k 15 \
  --log-level INFO
```

Headline result:

- WER `0.0837`
- CER `0.0837`
- blank hypotheses `0`

Weakest gloss recall:

- `MOTHER` recall `0.705`
- `RED` recall `0.709`
- `RIGHT` recall `0.713`
- `APPLE` recall `0.821`
- `FINE` recall `0.882`

Top substitutions:

- `RED -> FINE` : `26`
- `RIGHT -> DOG` : `13`
- `APPLE -> DEAF` : `12`
- `RIGHT -> RED` : `9`
- `WORK -> HOW` : `7`
- `MOTHER -> RED` : `7`
- `MOTHER -> APPLE` : `6`
- `MOTHER -> DEAF` : `6`
- `FINE -> RED` : `6`
- `MONEY -> HOW` : `6`

Top deletions:

- `RIGHT` : `6`
- `EAT` : `4`
- `APPLE` : `4`
- `DECIDE` : `4`

Interpretation:

- the system is no longer structurally broken
- the remaining errors are now specific sign confusions
- color / handshape / motion-near neighbors still need attention

## 15. Online / Web State

### Current UI behavior

The web UI was previously refactored so that:

- there is a dedicated example video area
- it updates in place instead of opening a new popup for each sign
- the example video autoplay behavior is enabled
- the old “live translation” box was removed

Relevant files:

- `/Volumes/T7/ASL/asl_cslr/online/templates/index.html`
- `/Volumes/T7/ASL/asl_cslr/online/web_server.py`
- `/Volumes/T7/ASL/asl_cslr/online/sign_examples.py`

### New broader-vocabulary live config

- `/Volumes/T7/ASL/configs/online_kaggle_expanded30.yaml`

Important alignment details:

- camera `downsample_factor: 2`
- CSLR training `frame_stride: 2`

This matters because the live path should roughly match the temporal density used during training.

### Smoke test completed

Command used:

```bash
python3 scripts/run_web.py \
  --config configs/online_kaggle_expanded30.yaml \
  --mode cslr \
  --port 5061
```

Verified:

- `/` returned `200`
- `/api/sign-examples/APPLE` returned `200`
- example data was returned correctly from WLASL media mappings

## 16. Files That Matter Most Right Now

If a fresh model needs to understand the repo quickly, these are the highest-value files.

### Core data / schema

- `/Volumes/T7/ASL/asl_cslr/data/skeleton.py`
  - canonical joint layout
  - normalization
  - missing-joint handling
  - motion features
- `/Volumes/T7/ASL/asl_cslr/data/dataset.py`
  - ISLR / CSLR dataset loading
  - schema validation
  - frame stride logic
  - motion recompute rules
- `/Volumes/T7/ASL/asl_cslr/data/preprocessing.py`
  - preprocessing pipelines for different raw data sources
- `/Volumes/T7/ASL/asl_cslr/data/label_maps.py`
  - gloss cleanup / canonicalization

### Models / training

- `/Volumes/T7/ASL/asl_cslr/models/cslr_model.py`
  - CSLR model
  - backbone warm start
  - classifier-head-to-CTC-head seeding
- `/Volumes/T7/ASL/asl_cslr/models/islr_model.py`
  - Stage 1 isolated-sign model
- `/Volumes/T7/ASL/asl_cslr/training/train_cslr.py`
  - CTC training loop
  - blank-row control
  - balanced sampling
- `/Volumes/T7/ASL/asl_cslr/training/train_islr.py`
  - Stage 1 training loop
- `/Volumes/T7/ASL/scripts/evaluate.py`
  - official eval entrypoint
- `/Volumes/T7/ASL/scripts/analyze_cslr_errors.py`
  - confusion / deletion / insertion analysis

### Kaggle-related processing utilities

- `/Volumes/T7/ASL/scripts/preprocess_wlasl_keypoints.py`
  - preprocess full WLASL keypoint corpus
- `/Volumes/T7/ASL/scripts/preprocess_wlasl_batch.py`
  - batch WLASL video preprocessing
- `/Volumes/T7/ASL/scripts/build_shared_isolated_cslr_dataset.py`
  - combined shared-gloss dataset building
- `/Volumes/T7/ASL/scripts/fetch_wlasl_glosses.py`
  - targeted WLASL acquisition support

### Online inference

- `/Volumes/T7/ASL/asl_cslr/online/camera.py`
  - live MediaPipe Tasks capture
  - smoothing / pending result handling
- `/Volumes/T7/ASL/asl_cslr/online/pipeline.py`
  - online decode logic
- `/Volumes/T7/ASL/asl_cslr/online/model_loader.py`
  - runtime model loading
- `/Volumes/T7/ASL/asl_cslr/online/web_server.py`
  - Flask web server
- `/Volumes/T7/ASL/scripts/run_web.py`
  - web entrypoint

## 17. Important Historical Milestones

This section is here so a fresh model understands how the repo got here.

### Earlier phases

The repo started with:

- smaller pilot vocabularies
- ISLR-first experiments
- CSLR demos built from narrower synthetic data
- multiple live/demo configs

Earlier checkpoints still on disk include things like:

- `cslr_demo`
- `cslr_demo_holistic`
- `cslr_live_focus`
- `cslr_live_focus_motion`
- `cslr_live_product_motion`
- `cslr_synthetic`

Those are still useful as references, but they are not the best broader-vocabulary result anymore.

### Prior live-focused diagnosis

Earlier work identified:

- overfitting
- low-data regimes
- confusion on similar-looking signs
- live instability from MediaPipe / online decode behavior

Those led to:

- stronger augmentation
- safer camera smoothing
- improved runtime gating
- persistent example-video UI

### Kaggle expansion phase

The user later provided Kaggle credentials and asked for:

- broader ASL dataset coverage
- as much useful data as possible
- processing and retraining end to end

That led to:

- downloading ASL Citizen keypoints
- downloading Kaggle WLASL processed videos
- downloading Kaggle WLASL keypoints
- processing those into the canonical repo schema
- building broader shared / expanded datasets
- retraining ISLR and CSLR on the expanded vocabulary

## 18. Things a Fresh Model Should Not Misunderstand

These are common traps.

### Trap 1: thinking the old short context file is current

It is not.

This file supersedes the older short note that mentioned:

- stale synthetic checkpoint numbers
- a stale rollback request

The user later said the rollback concern was not needed.

### Trap 2: thinking ISLR is the user’s product goal

It is not.

ISLR still matters because:

- it improves Stage 2 warm start

But product success is judged on CSLR.

### Trap 3: assuming the new 27-word result is already a proven live-webcam benchmark

It is not.

It is a strong synthetic continuous result built from real isolated clips.

### Trap 4: re-enabling horizontal flip

Do not do this casually.

The user explicitly wanted only semantically safe augmentation.

### Trap 5: forgetting stride alignment

The current best broader model was trained with:

- dataset `frame_stride: 2`

The matching live config uses:

- camera `downsample_factor: 2`

If those drift too far apart, live behavior may degrade.

### Trap 6: assuming dataloader workers are actually 8 on MPS

They are not.

The config requests `8`, but the training code deliberately resolves them to `0` on MPS for stability.

This is enforced in:

- `/Volumes/T7/ASL/asl_cslr/training/train_cslr.py`
- `/Volumes/T7/ASL/asl_cslr/training/train_islr.py`

## 19. Validation State

Latest repo-wide test result:

```bash
python3 -m pytest /Volumes/T7/ASL/tests -q
```

Result:

- `127 passed, 4 warnings`

Warnings were non-blocking and mostly about:

- Transformer nested tensor warning
- known scheduler warmup warning in a test

## 20. Kaggle Credential Caveat

The file:

- `/Volumes/T7/ASL/kaggle.txt`

contains the API token plus extra shell text.

Important:

- do not blindly `export KAGGLE_API_TOKEN=$(cat kaggle.txt)`
- that breaks auth headers because extra lines are included

If another model needs the token again, it should extract only the actual `KGAT_...` line.

## 21. Recommended Next Steps

If a fresh model is picking up from here, the best next steps are:

1. Live-validate the broader 27-gloss model with real webcam signing.
   - Use `/Volumes/T7/ASL/configs/online_kaggle_expanded30.yaml`.
   - Measure actual live confusions rather than assuming synthetic test quality transfers perfectly.

2. Target the known weak confusions.
   - Especially `RED`, `RIGHT`, `MOTHER`, `APPLE`.
   - Add analysis around handshape/depth/motion differences for those.

3. If broader live performance is good enough, consider promoting the new config.
   - Candidate:
     - `/Volumes/T7/ASL/configs/online_kaggle_expanded30.yaml`
   - But only after real live validation.

4. If live performance still lags, the next likely bottleneck is supervision realism.
   - The current best synthetic result is strong.
   - The next lift probably comes from more realistic continuous supervision, not from random architecture churn.

5. Keep the current blank-control and frame-stride logic unless there is strong evidence against it.
   - Those changes were directly responsible for getting broader-vocab CSLR off the ground.

## 22. Quick Commands for Another Model

### Evaluate the current best broader CSLR checkpoint

```bash
cd /Volumes/T7/ASL
python3 scripts/evaluate.py \
  --checkpoint checkpoints/cslr_kaggle_expanded30_short_motion/best.pt \
  --mode cslr \
  --split test \
  --log-level INFO
```

### Run error analysis

```bash
cd /Volumes/T7/ASL
python3 scripts/analyze_cslr_errors.py \
  --checkpoint checkpoints/cslr_kaggle_expanded30_short_motion/best.pt \
  --split test \
  --top-k 15 \
  --log-level INFO
```

### Start the broader-vocabulary web app

```bash
cd /Volumes/T7/ASL
conda activate asl
python3 scripts/run_web.py \
  --config configs/online_kaggle_expanded30.yaml \
  --mode cslr \
  --port 5050
```

### Retrain the current best broader CSLR setup

```bash
cd /Volumes/T7/ASL
python3 scripts/train.py \
  --config configs/cslr_kaggle_expanded30_short_motion_train.yaml \
  --log-level INFO
```

## 23. Short Bottom Line

If a fresh model only remembers five things, they should be these:

1. The repo is CSLR-first now.
2. The current best broader model is `/Volumes/T7/ASL/checkpoints/cslr_kaggle_expanded30_short_motion/best.pt`.
3. The big recent breakthrough came from temporal subsampling plus explicit blank-row control in CTC training.
4. The model now gets `0.0837` test WER on the rebuilt 27-gloss synthetic continuous test set.
5. The remaining uncertainty is live real-world generalization, not whether the broader CSLR pipeline basically works.
