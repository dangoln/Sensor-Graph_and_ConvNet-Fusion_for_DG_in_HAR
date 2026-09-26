# P21 — Pearson-Graph HAR: Architecture & Phase Breakdown

A grounded walkthrough of the P21 project, derived directly from the code in
`gnn/` and `p21/`. P21 is a **LODO domain-generalization** (DG) study for
6-class human-activity recognition (HAR) on the DAGHAR datasets. Its single
contribution over the prior P18b baseline is a **data-driven, per-window
Pearson-correlation graph** in the GNN branch; everything else is frozen.

See `P21_architecture.svg` (same folder) for the visual version of this document.

---

## 0. The one thing to internalize first

**In P21, the ConvNet branch is NOT trained or run.** P21 trains *only* the GNN.
The ConvNet probability files (`convnet_<target>.npz`) are **copied in frozen**
from the earlier P18b project and reused as-is. There is no ConvNet model in the
P21 repository (only `gnn/`); its definition lives in the **P18b** project at
`Project_DA_P18b_FeatAlign/convnet/src/models/deepconvlstm.py` and is documented
in **section 2B** below.

So the familiar "two pipelines run in parallel, then late-fuse" picture is true
of the *final result*, but operationally P21 is:

> retrain the GNN with a new graph construction → late-fuse its probabilities
> against pre-baked, frozen ConvNet probabilities.

**The one and only lever P21 changes versus P18b is how the graph edges are
built** (`edge_mode = pearson` instead of `fixed`). Node features, the CNN
encoder, GraphSAGE, the DANN setup, and the fusion procedure are all
byte-identical to P18b and deliberately held fixed (confirmed in
`p21/gnn_predict.py` and `p21/run_seeds.py`).

---

## 1. Phase-by-phase flow

### Phase 1 — Data
1. **Load DAGHAR** — 6 datasets: KuHar, MotionSense, RealWorld-Thigh,
   RealWorld-Waist, UCI, WISDM. Each sample is a 3-second window of shape
   `[60 timesteps × 6 channels]` (accel x/y/z, gyro x/y/z).
2. **LODO split** — leave one dataset out as the **target-test** set; the other
   5 are **sources**, each with its own `train` / `validation` split. P21 is a
   **pure DG protocol**: *no target data of any kind is loaded* during training
   (`load_target_unlabeled=False`, `normalize_include_target=False`), making it
   numerically comparable to the P18b "base" cells.
3. **Node features = HYBRID (90-dim per node)** — raw time signal (60) ⊕ FFT
   magnitude spectrum (`rfft`, DC bin dropped → 30 bins) = 90 features per node.
   Then **per-channel z-normalization** with statistics fit on **source-train
   only** (reused for val/test → no leakage).

### Phase 2 — Two branches (only the GNN branch is trained)
- **GNN branch** (trained): builds a 6-node sensor graph per window, encodes
  each node with a 1D CNN, runs GraphSAGE message passing, pools to a graph
  embedding, and attaches two heads (activity + domain-adversarial). Produces
  `P_gnn [N, 6]` on the target-test set.
- **ConvNet branch** (frozen): a **DeepConvLSTM** (Ordóñez & Roggen 2016) trained
  in P18b on the **frequency view** (log-magnitude FFT). Its `P_conv [N, 6]` is
  loaded from P18b for the matching target & seed. Full layer spec in **section 2B**.

### Phase 3 — Fusion & classification
- **Late fusion:** `P = w · P_gnn + (1 − w) · P_conv`, with `w` swept over
  `{0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0}` (`p21/run_report.py`).
- **Activity classification:** `argmax(P)` over the 6 classes
  (sit, stand, walk, stair-up, stair-down, run).
- **Reporting:** seed-averaged accuracy with a **KEEP/REVERT gate**: keep if the
  seed-mean fused Δ vs base ≥ +1.0 pp **and** no fold seed-mean drops below −2 pp.

---

## 2. GNN branch — granular layer specification (P21 winning config)

### 2.1 Input → node features
- Window `[60 × 6]` → hybrid features → **6 nodes × 90 features** per graph,
  z-normalized (source-train stats).

### 2.2 Graph construction — the P21 novelty (`gnn/data/pearson.py`)
Computed **per window**, in parallel with the node encoder:
1. Compute `|Pearson ρ|` between the 6 channels on the **raw 60-step signal**
   (not the hybrid 90 — Pearson is invariant to z-scoring, and FFT bins are not a
   meaningful correlation space).
2. 6 channels → **15 candidate undirected pairs**.
3. Score each pair: `score = α·|ρ| + (1 − α)·physical_prior`. The **winning
   config is top-k = 9, α = 1.0** (pure data-driven).
4. Keep the top-k pairs → make **bidirectional** → assign `edge_type ∈
   {intra-accel, intra-gyro, cross-modal}`.
5. **Per-window adjacency from that window's own signal only → zero leakage.**
   GraphSAGE ignores edge *weights*, so Pearson acts purely through *topology
   selection* (which edges exist). Weighted edges are deferred to P22/ChebNet.

### 2.3 Per-node CNN encoder (`CNN1DEncoder`)
Weight sharing = `per_modality`: one encoder for accel nodes 0–2, a separate one
for gyro nodes 3–5.

| # | Layer | Output shape |
|---|-------|--------------|
| 1 | input `[nodes, 90]` → `unsqueeze(1)` | `[nodes, 1, 90]` |
| 2 | `Conv1d(1 → 32, k=5, pad=2)` → `BatchNorm1d(32)` → `ReLU` | `[nodes, 32, 90]` |
| 3 | `Conv1d(32 → 64, k=5, pad=2)` → `BatchNorm1d(64)` → `ReLU` | `[nodes, 64, 90]` |
| 4 | `AdaptiveAvgPool1d(1)` *(temporal_pool = "avg")* | `[nodes, 64]` |
| 5 | `Linear(64 → 64)` projection | `[nodes, 64]` |

> Note: the GNN branch uses **average pooling, not an LSTM**
> (`cnn_temporal_pool="avg"`). The LSTM in your mental model belongs to the
> *external frozen ConvNet branch*, not here.

**Edge-type injection:** `nn.Embedding(3, 64)` lookup on `edge_type`, scaled
×0.1 and `index_add_`'d onto source-node embeddings before message passing.

### 2.4 GraphSAGE message passing (×2 layers)
Per layer (note dropout comes *before* the conv):
```
Dropout(p=0.2) → SAGEConv(64 → 64) → BatchNorm(64) → ReLU
```
Layer 1: 64→64, Layer 2: 64→64.

### 2.5 Readout
`global_mean_pool` over the 6 nodes → graph embedding **h `[batch, 64]`**.

### 2.6 Two heads (DANN wrapper, `gnn/models/da_sensor_graph.py`)
The backbone's own classifier is replaced with `nn.Identity`, and the pooled
embedding `h` feeds two heads:

**Activity head**
```
Linear(64 → 6) → softmax        # CrossEntropy(activity)
```

**Domain head (DANN-over-sources)**
```
GRL(λ) → Linear(64 → 64) → ReLU → Dropout(0.2) → Linear(64 → 5)
                                        # CrossEntropy(domain)
```
The domain discriminator predicts **which of the 5 source datasets** a sample
came from (not source-vs-target — there is no target in P21). The **Gradient
Reversal Layer** is identity on the forward pass and negates/scales the gradient
by −λ on the backward pass, pushing the encoder toward **domain-invariant**
features. `λ` follows the **Ganin schedule** `2/(1+e^(−10p)) − 1`, ramping 0→1
over training.

### 2.7 Output
`softmax(activity_logits)` → `P_gnn [N, 6]`, saved per target in test-CSV order
as `gnn_<target>.npz`.

---

## 2B. ConvNet branch (frozen) — DeepConvLSTM

The ConvNet whose probabilities P21 fuses against is a **faithful PyTorch port of
DeepConvLSTM** (Ordóñez & Roggen 2016, *"Deep Convolutional and LSTM Recurrent
Neural Networks for Multimodal Wearable Activity Recognition"*), defined in
`Project_DA_P18b_FeatAlign/convnet/src/models/deepconvlstm.py`. The architecture
is unchanged from the paper; only the data-forced values differ (channels
113→6, classes 18→6). It was trained in P18b and is **frozen** for P21.

### 2B.1 Input view (the frozen "base" recipe, `da_baseline/convnet_predict.py`)
The winning P18b recipe (`run2_nohann`, ≈74%) feeds the **frequency view**, not
the raw time signal:
- `input_view = "frequency"` — per-window `rfft` magnitude.
- `fft_magnitude = "log"` — **log** magnitude.
- `fft_window = "none"` — no Hann taper.
- `fft_drop_dc = False` — **DC bin kept** → `60//2 + 1 = 31` frequency bins.
- `normalize = "auto"` → `zscore_feature` — per-(bin, channel) z-score, fit on
  source-train only.

Input tensor to the model: **`(B, 1, T, C)`** with `T = 31` bins, `C = 6` channels.

> Note this differs from the GNN branch's frequency features (which use *linear*
> magnitude, drop the DC bin → 30 bins, and only as half of the hybrid vector).
> The two branches deliberately see different views — that diversity is what
> makes the late fusion useful.

### 2B.2 Layer-by-layer specification
Convolutions use kernel `(5, 1)` — they mix **only across time**, kernel width 1
over channels, so each sensor channel keeps its own feature maps. `padding=0`
("valid"), so the time axis shrinks by `(filter_size − 1) = 4` per conv layer.

| # | Layer | Output shape |
|---|-------|--------------|
| 1 | input `(B, 1, T=31, C=6)` | `(B, 1, 31, 6)` |
| 2 | `Conv2d(1 → 64, kernel=(5,1), pad=0)` → `ReLU` | `(B, 64, 27, 6)` |
| 3 | `Conv2d(64 → 64, kernel=(5,1), pad=0)` → `ReLU` | `(B, 64, 23, 6)` |
| 4 | `Conv2d(64 → 64, kernel=(5,1), pad=0)` → `ReLU` | `(B, 64, 19, 6)` |
| 5 | `Conv2d(64 → 64, kernel=(5,1), pad=0)` → `ReLU` | `(B, 64, 15, 6)` |
| 6 | DimShuffle/permute `(0,2,1,3)` | `(B, 15, 64, 6)` |
| 7 | flatten feature grid → per-timestep vector `64·C = 384` | `(B, 15, 384)` |
| 8 | `Dropout(p=0.5)` (input of first recurrent layer) | `(B, 15, 384)` |
| 9 | `LSTM(384 → 128)` (layer 1) | `(B, 15, 128)` |
| 10 | `LSTM(128 → 128)` (layer 2, inter-layer dropout p=0.5) | `(B, 15, 128)` |
| 11 | `SliceLayer(-1)` — keep **last timestep** | `(B, 128)` |
| 12 | `Dropout(p=0.5)` (input of final dense) | `(B, 128)` |
| 13 | `Linear(128 → 6)` → softmax | `(B, 6)` |

(With `T' = 31 − 4·(5−1) = 15` timesteps reaching the LSTM, each a 384-dim
vector = `64 filters × 6 channels`.)

### 2B.3 Regularization & training (P18b, frozen)
- **Dropout p=0.5** on the inputs of *every* dense layer — both LSTMs and the
  final classifier (paper-faithful).
- **Orthogonal weight init** for all Conv2d, Linear, and LSTM weights; biases zero.
- **Optimizer:** RMSProp, lr 1e-3, batch size 100, max 100 epochs.
- **Early stopping:** patience 20; model selection on **val accuracy**.
- **Penultimate embedding** = last-timestep LSTM state (before final dropout+FC) —
  this is the alignment target used by P18b's CORAL/MMD, but P21 doesn't retrain it.

---

## 3. Regularization & training inventory

| Mechanism | Setting / location |
|-----------|--------------------|
| Dropout | p=0.2 before each SAGEConv; p=0.2 inside the domain discriminator |
| BatchNorm | after each `Conv1d` (encoder) and after each `SAGEConv` |
| Weight decay (L2) | 5e-4 (Adam) |
| Gradient clipping | max-norm 1.0 |
| Gradient reversal (adversarial) | GRL with Ganin λ schedule 0→1 |
| LR schedule | Adam lr=1e-3, CosineAnnealingLR → η_min=1e-5 over 100 epochs |
| Early stopping | patience 20 on **source-val** accuracy; best checkpoint restored |
| Loss | `L = CE(activity) + λ · CE(domain)` |
| Batch size / epochs | 64 / 100 (max) |
| Model selection | best source-val accuracy (never target-test) |

---

## 4. How this maps to (and corrects) the common mental model

| Common understanding | Reality in P21 code |
|----------------------|---------------------|
| ConvNet & GNN both run in parallel | Only the **GNN is trained/run**; ConvNet probs are **frozen, copied from P18b** |
| GNN node = raw window | Node features are the **hybrid time+freq (90-dim)** vector, normalized |
| Pearson graph from the window | Yes — computed from the **raw 60-step** signal, in parallel with the encoder ✔ |
| 1D CNN encoder → node embedding | ✔ per-modality (accel/gyro), 2 conv layers + avg-pool + linear |
| GraphSAGE message passing | ✔ **2 layers**, each `dropout→conv→BN→ReLU` |
| Global mean-pool → classifier | ✔ |
| GRL → domain discriminator → domain loss, feeding back to the encoder | ✔ — branches from the **pooled embedding**; classifies **5 source domains** |
| ConvNet sub-steps (freq → CNN → LSTM) | ✔ correct shape — it's a **DeepConvLSTM**: freq view → 4× Conv2d(64,(5,1)) → 2× LSTM(128) → last-step dense. Frozen, defined in P18b (see section 2B) |

---

## 5. Key source files

- `gnn/data/pearson.py` — Pearson adjacency + edge selection (P21 novelty)
- `gnn/data/daghar_loader.py` — DAGHAR loading, LODO split, hybrid features, graph build
- `gnn/models/sensor_graph.py` — `CNN1DEncoder`, `SensorGraphModel`, baselines
- `gnn/models/da_sensor_graph.py` — DANN wrapper (activity + domain heads)
- `gnn/utils/gradient_reversal.py` — GRL + Ganin/linear λ schedules
- `gnn/training/dann_trainer.py` — training loop, losses, early stopping
- `gnn/utils/config.py` — frozen hyperparameters & edge topologies
- `p21/gnn_predict.py` — P21 GNN training/prediction entry point
- `p21/run_seeds.py` — seed sweep; reuses frozen P18b GNN+ConvNet cells
- `p21/run_report.py` — late fusion, seed-averaged report, KEEP/REVERT gate

**Frozen ConvNet branch (in the P18b project):**

- `convnet/src/models/deepconvlstm.py` — DeepConvLSTM model definition
- `convnet/configs/config.py` — model & training hyperparameters
- `da_baseline/convnet_predict.py` — frozen "base" recipe (frequency view) + prob dump

*Reference baseline: P18b base = 76.87 ± 1.07% (seeds 42/1/2); oracle headroom ~85%.*
