<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Make it extremely detailed, highly informational, and really long.

Below is a **single, very detailed, end‑to‑end plan** you can hand to an engineer or team and they will know exactly what to do, in what order, and why. It integrates all prior decisions: datasets, skeleton representation, preprocessing, labeling, modeling, training, and online deployment on your **M4 Pro MacBook Pro**.

***

## 0. High‑level architecture and timeline

### 0.1 Overall technical architecture

1. **Data layer (offline)**
    - Acquire selected ASL corpora.
    - For each, **convert videos/keypoints → normalized skeleton sequences** with a unified 52‑joint layout.
    - Store only `(T, D)` skeleton arrays (and optional motion features) plus manifest and vocab.
2. **Model layer**
    - Stage 1: train **isolated sign recognition (ISLR)** models on skeletons (dictionary).
    - Stage 2: train **continuous sign language recognition (CSLR)** models with **CTC** on continuous skeleton sequences.
    - Stage 3: refine with **multi‑scale/multi‑stream** architectures (pose vs motion) as per recent skeleton CSLR work.[^1][^2][^3][^4]
3. **Online layer (runtime)**
    - Stream webcam frames, run **MediaPipe Pose+Hands** on‑device for skeletons.[^5][^6][^7]
    - Apply exactly the same normalization and joint indexing.
    - Use **sliding‑window inference** + ISLR/CSLR to emit gloss sequences in real time.[^8][^9]
4. **Compute layer**
    - Use **PyTorch 2.x + MPS** backend on your M4 Pro GPU as the primary accelerator for both training and inference.[^10][^11][^12][^13]
    - Use **mixed precision (torch.amp)** for speed and reduced memory usage, as recommended for Apple’s MPS backend.[^14][^15][^10]

### 0.2 Rough implementation timeline (assuming one engineer)

- Week 1–2: Environment, dataset download, skeleton preprocessing across all corpora.
- Week 3–4: ISLR training, selection of best backbone.
- Week 5–7: Baseline CSLR training (BiLSTM+CTC) and evaluation.
- Week 8–9: Multi‑stream / multi‑scale CSLR experiments.
- Week 10+: Online CSLR implementation, tuning, and webcam domain adaptation.

***

## 1. Environment and tooling on the M4 Pro

### 1.1 OS and Python setup

- macOS: **Ventura/Sonoma** or later; ensure it supports the latest MPS features.[^11][^12][^10]
- Install **Miniforge** / Conda arm64 environment.

Packages:

- `pytorch`, `torchvision` (Apple Silicon build with MPS enabled).
- `mediapipe` or `mediapipe‑solutions` (Python) for pose+hands keypoints.[^6][^7][^5]
- `opencv-python` for video IO.
- `numpy`, `json`, `h5py` or `numpy` for storage; optional `pandas` for annotation tables.

Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops transparently fall back to CPU.[^12]

### 1.2 Mixed precision on MPS

- Use `torch.amp.autocast(device_type="mps", dtype=torch.float16)` in training forward passes to utilize mixed precision on Apple GPUs.[^15][^10]
- Avoid mixed precision in:
    - Loss reduction,
    - Certain numerically sensitive post‑processing (e.g., some CTC operations), if you see NaNs.[^14]

***

## 2. Data layer: acquisition and high‑level storage

### 2.1 Datasets and what you download

1. **How2Sign** (primary continuous ASL corpus)
    - Download:
        - 2D body–face–hand keypoint clips for frontal view.[^16][^17][^18]
        - Gloss/English sentence annotations and official train/dev/test splits.
2. **ASLLVD (lexicon)** via ASLLRP DAI
    - Download:
        - Multi‑view videos (or just the frontal view to simplify).[^19][^20]
        - Token‑level annotation table with: token IDs, lexical entry IDs, variant IDs, gloss, timing, signer ID, view ID, etc.[^21]
3. **NCSLGR / BU continuous corpora**
    - Download:
        - Utterance‑level videos.
        - ELAN `.eaf` or SignStream XML files with utterance and gloss tiers.[^22][^23][^24]
4. **WLASL (word‑level ASL)**
    - Clone official GitHub / data:
        - Cropped sign clips (mp4).
        - JSON metadata (gloss, signers, difficulty levels, etc.).[^25][^26]

### 2.2 Long‑term storage structure

On external SSD, say `/Volumes/asl-ssd/asl_cslr_data/`:

```text
raw/
  how2sign/          # only if you also store their videos; not required for keypoints
  asllvd/
  asllrp_ncslgr/
  wlasl/

processed/
  keypoints/
    how2sign/
    asllvd/
    asllrp/
    wlasl/
  manifests/
    how2sign.jsonl
    asllvd.jsonl
    asllrp.jsonl
    wlasl.jsonl
    combined_cslr_train.jsonl
    combined_cslr_val.jsonl
    combined_cslr_test.jsonl
    islr_train.jsonl
    islr_val.jsonl
    islr_test.jsonl
    vocab.json
    label_maps/
      wlasl_label_map.json
      asllvd_label_map.json
      bu_corpus_label_map.json
  config/
    preprocessing.json
```

Raw videos can be **deleted or archived** after preprocessing (except How2Sign keypoints, which are used directly).

***

## 3. Canonical skeleton representation (very detailed)

### 3.1 Joint set (52 joints)

Based on **MediaPipe Holistic** Pose+Hands output.[^7][^27][^28][^5][^6]

#### Pose joints (10)

From the 33 pose landmarks, keep:

1. NOSE
2. LEFT_SHOULDER
3. RIGHT_SHOULDER
4. LEFT_ELBOW
5. RIGHT_ELBOW
6. LEFT_WRIST
7. RIGHT_WRIST
8. LEFT_HIP
9. RIGHT_HIP
10. MID_SHOULDERS (synthetic) = 0.5 × (L_SHOULDER + R_SHOULDER)

#### Left hand joints (21)

Using standard MediaPipe Hands ordering (0–20):[^28][^29][^6]

11. LH_WRIST (index 0)
12. LH_THUMB_CMC (1)
13. LH_THUMB_MCP (2)
14. LH_THUMB_IP (3)
15. LH_THUMB_TIP (4)
16. LH_INDEX_MCP (5)
17. LH_INDEX_PIP (6)
18. LH_INDEX_DIP (7)
19. LH_INDEX_TIP (8)
20. LH_MIDDLE_MCP (9)
21. LH_MIDDLE_PIP (10)
22. LH_MIDDLE_DIP (11)
23. LH_MIDDLE_TIP (12)
24. LH_RING_MCP (13)
25. LH_RING_PIP (14)
26. LH_RING_DIP (15)
27. LH_RING_TIP (16)
28. LH_PINKY_MCP (17)
29. LH_PINKY_PIP (18)
30. LH_PINKY_DIP (19)
31. LH_PINKY_TIP (20)

#### Right hand joints (21)

32. RH_WRIST
33. RH_THUMB_CMC
34. RH_THUMB_MCP
35. RH_THUMB_IP
36. RH_THUMB_TIP
37. RH_INDEX_MCP
38. RH_INDEX_PIP
39. RH_INDEX_DIP
40. RH_INDEX_TIP
41. RH_MIDDLE_MCP
42. RH_MIDDLE_PIP
43. RH_MIDDLE_DIP
44. RH_MIDDLE_TIP
45. RH_RING_MCP
46. RH_RING_PIP
47. RH_RING_DIP
48. RH_RING_TIP
49. RH_PINKY_MCP
50. RH_PINKY_PIP
51. RH_PINKY_DIP
52. RH_PINKY_TIP

### 3.2 Flattened feature vector

For frame t, you generate `v_t ∈ ℝ^104`:

$$
v_t = [x_0, y_0, x_1, y_1, \dots, x_{51}, y_{51}]
$$

Where `i=0`..`51` index the joints above in order.

So each sequence is:

- `X = [v_0, v_1, …, v_{T-1}] ∈ ℝ^{T × 104}`.


### 3.3 Coordinate normalization (per frame)

MediaPipe returns coordinates normalized to image width/height: `x_raw, y_raw ∈ [0,1]`.[^5][^6][^28]

For each frame:

1. **Compute reference point:**

$$
ref_x = \frac{x_{L\_SHOULDER} + x_{R\_SHOULDER}}{2},
\quad
ref_y = \frac{y_{L\_SHOULDER} + y_{R\_SHOULDER}}{2}
$$
2. **Compute scale:**

$$
d_{shoulder} = \sqrt{(x_{L\_SHOULDER} - x_{R\_SHOULDER})^2 + (y_{L\_SHOULDER} - y_{R\_SHOULDER})^2}
$$
    - Then `s_ref = max(d_shoulder, ε)` (ε ≈ 1e−3).
3. **Normalize each joint:**

For joint with raw coords `(x_raw, y_raw)`:

$$
x' = \frac{x_{raw} - ref_x}{s_{ref}}, \quad
y' = \frac{y_{raw} - ref_y}{s_{ref}}
$$
    - For MID_SHOULDERS, `(x',y') = (0,0)`.
4. **Missing joint handling:**
    - If a joint is not detected or low confidence, use:
        - Last valid value `(x',y')_{t-1}`, or
        - `(0,0)` (i.e., at ref point) if at t=0.

This gives translation/scale invariance and is standard for skeleton SLR.[^2][^30][^3][^31]

### 3.4 Motion features (optional)

To capture dynamics more explicitly:

- For each frame:

$$
v^{(vel)}_t = v'_t - v'_{t-1}, \quad v^{(vel)}_0 = \mathbf{0}
$$
- Optionally:

$$
v^{(acc)}_t = v^{(vel)}_t - v^{(vel)}_{t-1}
$$

You may store them as `X_vel`, `X_acc` in the same `.npz`. Multi‑cue temporal modeling and dual‑stream skeleton SLR have been shown to benefit from such static+dynamic signals.[^3][^4][^1][^2]

***

## 4. Label normalization and vocab building

### 4.1 Dataset‑specific cleaning

You need mapping tables so all datasets use a common gloss space.

#### WLASL label map

- From WLASL JSON, collect original glosses (`orig_gloss`).[^26][^25]
- Apply a cleaning procedure similar to follow‑up work:[^32][^26]
    - Uppercase all glosses.
    - Remove punctuation and trailing digits when they represent version numbers not true variants (e.g., `EAT1`, `EAT2` → `EAT`).
    - Merge obvious synonyms (e.g., `CANNOT` vs `CAN'T` → `CANNOT`) if consistent with literature.
- Build `wlasl_label_map.json` mapping `orig_gloss` → `canonical_gloss`.


#### ASLLVD label map

- From ASLLVD tokens, identify `(lexical_entry_id, variant_id, base_gloss)`.[^19][^21]
- Decide whether to merge variants:
    - If goal is generic recognition, map all variants of a lexical entry to a single `canonical_gloss` (e.g., `HOUSE`).
- Build `asllvd_label_map.json` mapping token’s (entry,variant) or original gloss string → canonical gloss.


#### BU continuous corpora map (NCSLGR, etc.)

- BU glossing may encode morphological details (e.g., classifiers, aspect markers).[^24][^21][^22]
- For the first pass:
    - Strip morphological suffixes if your goal is base lexeme recognition.
    - Uppercase, standardize multiword signs (`GO_TO` vs `GO-TO`).
- Build `bu_corpus_label_map.json` for NCSLGR/other BU corpora.


#### How2Sign gloss normalization

- How2Sign glosses are already close to standard lexical glosses (uppercase, simple tokens).[^17][^18][^33]
- Still:
    - Uppercase, remove redundant punctuation, standardize hyphens/underscores.


### 4.2 Global vocabulary

1. Collect **all canonical glosses** from:
    - How2Sign, ASLLRP continuous corpora, WLASL, ASLLVD.
2. Count corpus‑level frequencies.
3. Choose a max vocabulary size `V_max` (e.g., 2,500–3,000) to keep CTC head manageable.
4. Build:

```
- `itos = ["<blank>","<pad>","<bos>","<eos>","<unk>",G1,G2,…,G_{V}]`.  
```

    - `stoi` mapping strings to indices.
5. For any canonical gloss not in top‑V after filtering, map to `<unk>`.

All `gloss` lists in manifests are already in canonical form and rely on `stoi` at training time.

***

## 5. Preprocessing pipeline (per dataset)

This uses the canonical skeleton schema and label maps.

### 5.1 How2Sign

**Input:** Provided 2D keypoint sequences and gloss annotations.[^18][^16][^17]

Steps:

1. For each sentence clip:
    - Load How2Sign’s 2D keypoints (body + hands).
    - If they are raw (not normalized as you want), convert to your 52‑joint layout using known indices; else adapt them.
    - Optional: downsample further (e.g., from ~25fps to ~12.5fps) keeping every 2nd frame to align with your global `downsample_factor`.
2. Normalize coordinates per frame using your shoulder‑based scheme.
3. Build `X (T,104)`; optionally `X_vel`.
4. Create `.npz` file with `X`, `frame_times`, and `meta` fields.
5. From How2Sign annotations:
    - For each clip ID, extract original gloss sequence.
    - Normalize each gloss string and apply mapping if necessary.
    - Ensure they exist in `stoi` (else mapped to `<unk>`).
6. Create `how2sign.jsonl` manifest with one entry per clip including `id`, `features_path`, `gloss`, `split`, etc.

**No raw RGB video must be stored**, only skeletons and annotation files.

### 5.2 ASLLVD

**Input:** BU videos + lexicon token table.[^20][^21][^19]

Steps:

1. For each token in the table:
    - Determine video file, view ID (choose frontal), and `[start_time, end_time]`.
    - Use ffmpeg/OpenCV to read frames in that interval.
    - Downsample frames by `r` (global downsample factor).
    - For each kept frame:
        - Run MediaPipe Pose+Hands.
        - Extract required 52 joints.
        - Normalize coordinates.
2. Build `X (T,104)` and optional motion features.
3. Save `.npz`.
4. For label:
    - Use `asllvd_label_map` to map to `canonical_gloss`.
    - `gloss=[canonical_gloss]`.
5. Add lines to `asllvd.jsonl` manifest.
6. After verifying a subset of `.npz` files visually (overlay skeleton on video in a debug tool), delete or move raw videos out of active storage.

### 5.3 NCSLGR / BU continuous corpora

**Input:** Utterance videos + ELAN/SignStream annotations.[^23][^22][^24]

Steps:

1. Parse ELAN/SignStream to obtain:
    - Utterance tier: ID, `[start_time, end_time]`.
    - Gloss tier: list of gloss segments within each utterance.
2. For each utterance:
    - Extract frames over `[start_time, end_time]`; downsample with factor `r`.
    - Run MediaPipe Pose+Hands per selected frame; normalize.
    - Build `X (T,104)`.
3. For the gloss sequence:
    - For each gloss segment in time order, capture its canonical label via BU label map.
4. Save `.npz` and `asllrp.jsonl` entry with full gloss sequence.
5. Optionally drop extremely long utterances or segment them to enforce `T_max_cont`.

### 5.4 WLASL

**Input:** Cropped sign clips + JSON metadata.[^25][^26]

Steps:

1. For each clip:
    - Read all frames; downsample every `r`-th frame.
    - For each kept frame:
        - Run MediaPipe Pose+Hands; normalize; gather 52 joints.
2. Build `X (T,104)`.
3. For label:
    - Extract original gloss; map via `wlasl_label_map` to `canonical_gloss`.
4. Save `.npz` + `wlasl.jsonl` entry.
5. Raw WLASL clip can be deleted after QC.

***

## 6. Dataset composition and sampling strategies

You now have consistent skeleton corpora for:

- `ISLR`: WLASL + ASLLVD (isolated signs).
- `CSLR`: How2Sign + BU continuous (NCSLGR/others).


### 6.1 ISLR dataset

**Manifests:**

- `islr_train.jsonl` = union of WLASL + ASLLVD training splits.
- `islr_val.jsonl`, `islr_test.jsonl` similarly.

Sampling strategy:

- Balanced sampling between WLASL and ASLLVD to avoid either dominating.
- Option: oversample ASLLVD tokens if their labels/variants are important.


### 6.2 CSLR dataset

**Manifests:**

- `combined_cslr_train.jsonl`: How2Sign train + BU ASL train.
- `combined_cslr_val.jsonl`, `combined_cslr_test.jsonl`: dev/test splits similarly.

Sampling:

- Either mix corpora uniformly, or sample proportionally to dataset size.
- You can experiment with upweighting How2Sign initially, since its annotations and conditions are consistent.

***

## 7. Models: architectures tailored to skeleton input

### 7.1 Common base: temporal conv + BiLSTM (Family A)

- Input: `(B, T, 104)` or `(B, T, 208)` if concatenating `X` and `X_vel`.
- Temporal conv encoder:
    - 3 layers of Conv1d with kernel sizes 5 or 7, ReLU, BatchNorm, Dropout 0.1–0.2.[^4][^3]
    - Output: `(B, T, conv_dim)`.
- BiLSTM:
    - `hidden_size`: 256–512, `layers`: 2–3, `bidirectional=True`.
    - Output: `(B, T, 2*hidden_size)`.

This matches many CSLR and trajectory‑based designs, balancing capacity and cost.[^34][^35][^3][^4]

### 7.2 Multi‑scale / dual‑stream (Family B)

Inspired by TCNet and CoSign/CorrNet‑like models for skeleton CSLR:[^35][^1][^2][^3][^4]

- **Multi‑scale temporal conv:**
    - Branch A: Conv1d(k=3).
    - Branch B: Conv1d(k=5).
    - Branch C: Conv1d(k=9).
    - Concatenate channels and project back to conv_dim.
- **Dual stream:**
    - Stream 1: static pose (X).
    - Stream 2: motion (X_vel).
    - Each stream has its own temporal conv + BiLSTM encoder; their features are fused (e.g., concatenation or gating) before CTC head.

This directly leverages findings that combining static and dynamic skeleton signals with adequate receptive field boosts CSLR performance.[^1][^2][^3][^4]

### 7.3 Lightweight Transformer encoder (Family C, optional)

Use the same temporal conv front‑end, but replace BiLSTM with:

- Transformer encoder:
    - hidden_dim=256–384, num_layers=4–6, heads=4–6, dropout=0.1–0.2.
    - Relative/sinusoidal positions.
- Optionally local attention (attention over ±W frames) to manage O(T²) cost when T is large.

This follows the trend of Transformer‑based CSLR (e.g., correlation networks, multi‑cue temporal Transformers).[^30][^36][^3][^4]

***

## 8. Training details and monitoring

### 8.1 ISLR training (Stage 1)

**Objective:** build a strong lexical dictionary model for isolated signs.

- Dataset: `islr_train.jsonl`, `islr_val.jsonl`.
- Input: skeleton sequences `(B, T, 104)`.
- Model: Family A (conv + BiLSTM) with classification head.
- Loss: cross‑entropy over dictionary gloss IDs.

Hyperparameters (example best starting point):

- conv_dim=384, LSTM hidden_size=384, LSTM layers=2.
- downsample_factor=2, no T_max truncation (sequences are short).
- B_islr=64, lr=1e−3, wd=0.01, dropout=0.1 (conv) + 0.2 (FC).
- Epochs: 30–50.

Monitoring:

- Train/val loss and val top‑1 / top‑5 accuracy.
- Macro‑averaged accuracy across glosses to avoid only measuring frequent signs.

Outcome: best ISLR checkpoint becomes your **pretrained backbone** for CSLR.

### 8.2 CSLR training (Stage 2)

**Objective:** continuous gloss recognition from skeleton sequences.

- Dataset: `combined_cslr_train.jsonl`, `combined_cslr_val.jsonl`.
- Input: `(B, T, 104)`, truncated to T_max=256–320.
- Model: Family A or B (conv+BiLSTM), with CTC head.

Hyperparameters:

- From best ISLR: conv_dim, hidden_size.
- layers ∈ {2, 3}.
- r ∈ {2, 3}.
- T_max=256 or 320 (tuned).
- B_cslr=8 (or 6 if memory is tight).
- lr=3e−4, wd=0.01.
- Epochs: 60–80.

Training:

- Initialize from ISLR backbone.
- Fine‑tune end‑to‑end; optionally freeze lowest conv layer for first few epochs if convergence is unstable.

Monitoring:

- Training CTC loss.
- Dev CTC loss.
- Every N epochs (e.g., 5), run **greedy CTC decoding** on dev set and compute gloss‑level WER (and optionally CER).

This follows standard CSLR practice.[^36][^3][^4][^34]

### 8.3 CorrNet/TCNet‑style experiments (Stage 3)

Train multi‑stream/ multi‑scale variants (Family B):

- 3A: multi‑scale conv only.
- 3B: multi‑scale + dual stream (X + X_vel).

Keep other hyperparameters similar to the best CSLR baseline; fine‑tune from the same ISLR or baseline CSLR weights. Compare dev/test WERs.

### 8.4 Lightweight Transformer CSLR (Stage 5, optional)

Train 1–2 Transformer‑based models to see if they outperform the best BiLSTM/CorrNet skeleton CSLR.[^37][^3][^4]

- hidden_dim=256, layers=4, heads=4 (5A).
- hidden_dim=384, layers=4, heads=6 (5B).

May require slightly smaller batch sizes (B_cslr=6) due to attention cost.

***

## 9. Online CSLR design (runtime system)

### 9.1 Live skeleton extraction

At runtime:

1. Open webcam at ~30 fps.
2. For each frame:
    - Downsample in time (e.g., process every 2nd frame).
    - Run MediaPipe Pose+Hands; get raw 2D landmarks.[^6][^7][^5]
    - Compute normalized v_t ∈ ℝ^104 with the exact procedure defined above.
    - Append v_t and timestamp t_sec to a buffer (deque).

### 9.2 Sliding‑window ISLR‑centric online CSLR

Following recent online CSLR work:[^9][^8]

- **Window parameters:**
    - Effective fps `fps_eff ≈ 15`.
    - window_duration ∈ {1.5, 2.0} s → `W = fps_eff * duration` frames.
    - hop_duration ∈ {0.25, 0.5} s → `H = fps_eff * hop`.
    - stability_windows ∈ {2, 3}.

Pipeline:

1. Maintain a rolling buffer of up to ~3–4 seconds (e.g., 60 frames).
2. At each hop:
    - Extract window of last W frames; feed `(1, W, 104)` to ISLR model.
    - Get predicted gloss and confidence.
3. Accumulate window‑level predictions:
    - Keep a recent history of (time_range, gloss, confidence).
    - When the same gloss is **dominant across stability_windows consecutive overlapping windows** and confidence is above threshold, finalize that gloss as a recognized sign.
    - Detect sign boundaries from transitions between stable glosses.

This uses the dictionary model directly for online segmentation and recognition, minimizing latency and memory.

### 9.3 CSLR‑centric streaming variant

Alternatively, use the CSLR model in streaming:

- Maintain buffer of L seconds (2–3 s).
- Periodically (every 0.5–1 s), feed the entire buffer `(1, T_buf, 104)` to CSLR model.
- Greedy‑decode to partial gloss sequence; track changes over calls.
- Use blank probabilities and local stability to decide when a gloss is “complete” (similar to streaming CTC in speech).[^36][^8][^9]

You can combine both (ISLR to segment, CSLR to refine sequences) once baseline pipeline is stable.

***

## 10. Evaluation protocols

### 10.1 Offline

For each trained model:

- **ISLR metrics:**
    - Per‑dataset (WLASL, ASLLVD) top‑1, top‑5 accuracy.
    - Macro‑averaged accuracy across glosses to ensure rare glosses are not ignored.[^26][^32]
- **CSLR metrics:**
    - Gloss‑level WER on:
        - How2Sign dev/test.
        - BU continuous corpora dev/test.
    - Optionally, character‑level error rate over gloss sequences.

Ensure you are evaluating on canonical glosses and consistent splits.

### 10.2 Online

For each online configuration (4A–4D):

1. Offline streaming simulation:
    - For dev/test videos, feed frames sequentially to the webcam pipeline.
    - Compare predicted gloss sequences against reference.
2. Metrics:
    - WER (same as offline).
    - **Latency per gloss:**
        - For each gloss occurrence in ground truth, measure time difference between true end frame and time at which your pipeline emits that gloss.
    - Throughput: average fps processed on M4 (should be >= real‑time).

Plot WER vs latency to choose the optimal configuration for user experience.

***

## 11. M4 Pro‑specific performance tuning

### 11.1 Memory and compute management

- **Skeleton input is compact**; main memory usage comes from:
    - Temporal conv activations `(B,T,conv_dim)`.
    - BiLSTM/Transformer hidden states `(B,T,H)`.

Tuning knobs:

- `T_max` for CSLR (start at 256, increase to 320 if needed).
- `B_cslr`: 4–8, depending on memory.
- `conv_dim`, `hidden_size`: 256–384 are good trade‑offs.


### 11.2 MPS + mixed precision

- Use `torch.amp.autocast(device_type="mps", dtype=torch.float16)` where safe; this is supported in recent macOS and PyTorch and can significantly accelerate training.[^10][^15][^14]
- Monitor for NaNs/Inf in CTC loss; if they appear:
    - Disable mixed precision for the CTC region only, keep it for conv/LSTM layers.
- Exploit the single M4 GPU; no need for multi‑GPU complexities.

***

## 12. Risk management and extensions

### 12.1 Skeleton‑only limitations

- Lack of color/texture cues (skin, clothing, some subtle handshapes) and facial expressions; skeleton‑only may struggle with nuance.[^2][^1]
- Mitigation:
    - Use multi‑cue skeleton features (body + hands + optional face keypoints) to capture more context.[^1]
    - Later, selectively retain RGB for a small subset and build a dual‑stream appearance+skeleton model if needed.


### 12.2 Dataset bias and generalization

- Studio datasets (How2Sign, BU corpora) have controlled conditions, limited signers and backgrounds.[^33][^24][^2]
- For your webcam setting:
    - Fine‑tune CSLR/ISLR on a small **unlabeled or weakly labeled** subset of your own webcam recordings using:
        - Self‑training (pseudo‑labels) or
        - Domain adaptation techniques if desired.


### 12.3 Future directions

- Graph‑based skeleton models (ST‑GCN, multi‑cue GCN/LSTM) using the same joint layout.[^30][^32][^1]
- Multilingual SLR using your schema as a generic interface (CSL‑Daily, PHOENIX14, etc., then cross‑lingual transfer).[^3][^2][^30]
- Full sign language translation (CSLR + gloss→text Transformer) using How2Sign and open‑domain ASL translation corpora.

***

This is now a **complete, deeply specified blueprint** from downloading datasets to having a live, online CSLR demo on your M4 Pro that uses only skeleton‑level data long‑term. If you want, the next step can be to pick a **minimal first milestone path** (e.g., “only How2Sign + WLASL, only BiLSTM family, skip CorrNet/Transformer”), and I’ll trim this into a prioritized to‑do list so you can get a first working prototype as fast as possible without sacrificing the long‑term architecture.
<span style="display:none">[^38][^39][^40][^41][^42]</span>

<div align="center">⁂</div>

[^1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10113557/

[^2]: https://openaccess.thecvf.com/content/ICCV2025W/MSLR/papers/Min_A_Closer_Look_at_Skeleton-based_Continuous_Sign_Language_Recognition_ICCVW_2025_paper.pdf

[^3]: https://arxiv.org/abs/2303.03202

[^4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12920636/

[^5]: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

[^6]: https://mediapipe.readthedocs.io/en/latest/solutions/hands.html

[^7]: https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker

[^8]: https://arxiv.org/html/2401.05336v2

[^9]: https://aclanthology.org/2024.emnlp-main.619/

[^10]: https://developer.apple.com/videos/play/wwdc2023/10050/

[^11]: https://developer.apple.com/metal/pytorch/

[^12]: https://docs.pytorch.org/docs/stable/notes/mps.html

[^13]: https://huggingface.co/docs/accelerate/en/usage_guides/mps

[^14]: https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/

[^15]: https://docs.pytorch.org/docs/stable/amp.html

[^16]: https://how2sign.github.io

[^17]: https://slrtp.com/papers/extended_abstracts/SLRTP.EA.13.014.paper.pdf

[^18]: https://how2sign.github.io/images/poster/How2Sign_Poster.pdf

[^19]: https://www.bu.edu/asllrp/av/dai-asllvd.html

[^20]: https://www.academia.edu/87376066/American_Sign_Language_Lexicon_Video_Dataset_ASLLVD_corpus

[^21]: https://www.semanticscholar.org/paper/Challenges-in-development-of-the-American-Sign-Neidle-Thangali/8bd62710fdaac5e5558bafffecbe4d3107a0c31c

[^22]: https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html

[^23]: http://asl.cs.depaul.edu/corpus/index.html

[^24]: https://www.bu.edu/asllrp/about-datasets.pdf

[^25]: https://github.com/dxli94/WLASL/blob/master/README.md

[^26]: https://www.studocu.vn/vn/document/truong-dai-hoc-bach-khoa-ha-noi/nhap-mon-hoc-may-va-khai-pha-du-lieu/word-level-deep-sign-language-recognition-from-video-a-new-large-scale-dataset-and-methods-comparison/79641511

[^27]: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md

[^28]: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

[^29]: https://www.geeksforgeeks.org/machine-learning/face-and-hand-landmarks-detection-using-python-mediapipe-opencv/

[^30]: https://www.sciencedirect.com/science/article/abs/pii/S0952197624021547

[^31]: https://arxiv.org/html/2509.08661v1

[^32]: https://aclanthology.org/anthology-files/pdf/lrec/2022.lrec-1.797.pdf

[^33]: http://www.iri.upc.edu/files/scidoc/2461-How2Sign:-A-large-scale-multimodal-dataset-for-continuous-American-sign-language.pdf

[^34]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Pu_Iterative_Alignment_Network_for_Continuous_Sign_Language_Recognition_CVPR_2019_paper.pdf

[^35]: https://webspace.science.uu.nl/~salah006/lu24aaai.pdf

[^36]: https://www.nature.com/articles/s41598-024-78319-0

[^37]: https://www.computer.org/csdl/proceedings-article/iccvw/2025/898800e968/2eldqV1Lk9a

[^38]: https://mmla.gse.harvard.edu/tools/holistic/

[^39]: https://www.tencentcloud.com/techpedia/126165

[^40]: https://www.youtube.com/watch?v=EgjwKM3KzGU

[^41]: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html

[^42]: https://github.com/google-ai-edge/mediapipe/issues/5490

