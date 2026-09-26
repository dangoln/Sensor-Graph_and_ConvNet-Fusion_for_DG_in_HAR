# P21 (Pearson Graph) — Code-Level Walkthrough

This document walks through exactly what happens, file by file and call by call, when P21 runs — and then pins down precisely how P21's code differs from P6/P9, P15, P18a, P18b, and P23. All file paths below are relative to `PROJECTS/P20_GraphRevival_DG/P21_PearsonGraph/`.

P21 asks one question: **does building the 6-node sensor graph's edges from per-window Pearson correlation (instead of a fixed topology) improve LODO cross-dataset accuracy?** Everything else — features, encoder, GNN backbone, DANN, fusion — is frozen to the P9 recipe. The folder is organized so that freeze is enforced in code, not just in a README.

---

## 0. Folder map and why it's shaped this way

```
P21_PearsonGraph/
├── gnn/                      # the model/data/training library (P21's OWN copy)
│   ├── data/
│   │   ├── daghar_loader.py  # CSV → graphs, LODO fold assembly
│   │   └── pearson.py        # NEW: per-window Pearson adjacency
│   ├── models/
│   │   ├── sensor_graph.py       # SensorGraphModel, CNN1DEncoder, DualViewEncoder
│   │   └── da_sensor_graph.py    # DomainAdversarialWrapper
│   ├── training/dann_trainer.py  # DANNTrainer (incl. dormant CORAL/MMD hook)
│   └── utils/{config.py, seed.py, gradient_reversal.py}
├── p21/
│   ├── gnn_predict.py   # one LODO fold: train + predict, fixed OR pearson edges
│   ├── run_sweep.py     # Stage A — pick best Pearson config on source-val
│   ├── run_seeds.py     # Stage B — seed-average the winner vs Arm A baseline
│   └── run_report.py    # fuse with frozen ConvNet, gate KEEP/REVERT
└── README.md
```

`gnn/` is a **full copy** of the shared library that P6 through P18b all built up, not a symlink or import from another phase's folder. This is deliberate: P21 must remain runnable and reproducible on its own years from now, even if P18b's folder is later reorganized or deleted. The cost is that the copy can drift from the original — which is why, in the comparison section below, I diffed P21's copies of `sensor_graph.py` / `da_sensor_graph.py` against the original `P06_P09_SensorGraph/models/` versions directly, rather than trusting the README's claim that they're "the same architecture."

The `p21/` package is kept separate from `gnn/` for an import-hygiene reason visible directly in code: `gnn_predict.py` line 25 does

```python
GNN_DIR = os.path.join(P21, "gnn")
sys.path.insert(0, GNN_DIR)          # only the GNN codebase is importable here
```

i.e. only the *library* is put on `sys.path`, so `data.daghar_loader`, `models.sensor_graph`, etc. import cleanly without `p21.` prefixes inside `gnn/`'s own modules. `p21/` itself is a proper Python package (it's imported elsewhere as `p21.gnn_predict`), so the orchestration scripts can do `python -m p21.run_sweep` from the P21 root.

---

## 1. Step-by-step: one LODO fold's execution (`p21/gnn_predict.py`)

A "fold" = one of the 6 DAGHAR datasets held out as target; the other 5 are source. `train_and_predict(target, cfg)` is the function that runs one fold end to end. `main()` loops it over all 6 targets.

### Step 1 — Config freeze (`p21_config(args)`, lines 41–72)

```python
cfg = get_config()                      # gnn/utils/config.py — full default dict
cfg["dann_enabled"] = True
cfg["gnn_type"] = "sage"
cfg["feature_mode"] = "hybrid"
cfg["daghar_timesteps"] = 90
...
cfg["edge_mode"] = args.edge_mode        # "fixed" | "pearson"  ← THE ONE LEVER
cfg["pearson_select"] = args.select
cfg["pearson_topk"] = args.topk
cfg["pearson_tau"] = args.tau
cfg["pearson_alpha"] = args.alpha
```

`get_config()` (`gnn/utils/config.py`) returns the full hyperparameter dict (batch_size=64, gnn_hidden=64, gnn_layers=2, gnn_dropout=0.2, cnn_channels=[32,64], patience=20, etc.) with every value documented inline as to which phase introduced it. `p21_config` then overwrites the dozen-or-so fields that constitute "the P9 recipe" to fixed, hard-coded values — these are not CLI-tunable in P21 at all, which is the actual code-level enforcement of "everything but the graph edges is frozen." Three lines explicitly zero out the DA-protocol fields so no target leakage is even possible: `load_target_unlabeled=False`, `normalize_include_target=False`, `align_method="none"`.

### Step 2 — Build the LODO fold and graphs (`build_lodo_sensor_graphs`, in `gnn/data/daghar_loader.py`)

Called once per target inside `train_and_predict`:

```python
lodo = build_lodo_sensor_graphs(cfg["data_root"], target, config=cfg, folder_map=fmap)
```

Internally this:
1. Loads the 5 source datasets' CSVs via `load_daghar_dataset` / `load_daghar_aggregated`, using `_resolve_folder` to case-insensitively match folder names across Colab/local layouts.
2. Splits source into train/val.
3. Loads the target dataset's CSV as the held-out test set, in original row order (preserved deliberately — see Step 6).
4. Fits per-channel z-normalization stats on **source-train only** (never val, never target) — this is the "DG protocol" guarantee, structurally the same code path P17/P18a/P18b use, just with `normalize_include_target` forced off.
5. For each split, computes node features per window via `_compute_node_features` — in `feature_mode="hybrid"`, each of the 6 channel-nodes gets a 90-length feature vector = 60 raw timesteps + 30 FFT-magnitude bins (`daghar_timesteps=90` is literally `60+30`).
6. **Builds the edge set for each split.** This is where P21's lever enters the loader (see Step 3).
7. Wraps each window into a `torch_geometric.data.Data` object (6 nodes, node features, edge_index, edge_type, graph-level label `y`, domain id) via `build_sensor_graphs`.

Returns a dict: `train_graphs`, `val_graphs`, `test_graphs`, `node_feat_dim`, `num_domains`, `n_train`/`n_val`/`n_test`.

### Step 3 — Graph construction: fixed topology vs Pearson adjacency

This is the actual fork point. Inside `build_sensor_graphs`, edges come from one of two places:

**`edge_mode="fixed"` (Arm A control):** a single shared `edge_list` is reused for every window in every split — `EDGES_FULL`, the complete 15-pair K6 graph over the 6 channel-nodes (`config.py`: `(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)`). *(Correction to an earlier verbal description I gave you: `EDGES_FULL` is the complete graph, not a graph that excludes 3 pairs — it's `EDGES_PHYSICAL`, the 9-pair set, that excludes the 6 non-axis-aligned cross-modal pairs, keeping only 3 intra-accel + 3 intra-gyro + 3 axis-matched cross-modal edges.)*

**`edge_mode="pearson"` (Arm B, P21's new code, `gnn/data/pearson.py`):** edges are computed **independently per window**, on the **raw pre-normalization signal** (so no normalization leakage into the topology decision):

```python
R = pearson_abs_matrix(X)        # X: [n_windows, 6, T] → R: [n_windows, 6, 6], vectorized via einsum covariance
edge_sets = select_edges(R, mode=cfg["pearson_select"], topk=cfg["pearson_topk"],
                          tau=cfg["pearson_tau"], alpha=cfg["pearson_alpha"])
```

`select_edges` scores each channel pair `(i,j)` as
```
score_ij = alpha * |rho_ij| + (1 - alpha) * 1[(i,j) in PHYSICAL_PRIOR]
```
where `PHYSICAL_PRIOR = {(0,1),(0,2),(1,2),(3,4),(3,5),(4,5),(0,3),(1,4),(2,5)}` — the same 9 pairs as `EDGES_PHYSICAL`. With `--select topk --topk 9`, the 9 highest-scoring pairs per window become that window's edges; with `--select tau --tau T`, any pair scoring ≥ T is kept (with `--min_edges` as a floor so no window degenerates to an empty graph). `alpha=1.0` is pure data-driven Pearson; `alpha=0.5` blends data-driven and the physical prior 50/50; `alpha=0.0` would degenerate to the fixed physical prior.

`build_sensor_graphs` accepts this optional `edge_sets` array and, if present, indexes into it per-window instead of reusing the shared `edge_list` — so a graph in the Pearson arm can have a genuinely different topology from the graph next to it in the same batch.

One mechanically important fact for interpreting P21's results: the GNN backbone is `SAGEConv` (GraphSAGE), and **GraphSAGE's message passing only uses edge existence, not edge weight** — there is no place in `SAGEConv.forward` that consumes a scalar edge attribute. So Pearson's influence in P21 is purely *topological* (which 9 pairs exist), never a soft attention-like weighting of those pairs. Edge-weighted message passing was deliberately deferred to P22 (ChebNet, which can consume weights) — that ablation is recorded in memory as having failed/reverted.

### Step 4 — Model construction

```python
backbone = SensorGraphModel(cfg)
model = DomainAdversarialWrapper(backbone, num_domains, cfg).to(device)
```

`SensorGraphModel` (`gnn/models/sensor_graph.py`):
- `encoder_mode="single"` in P21 ⇒ uses plain `CNN1DEncoder` (not P12's `DualViewEncoder`) per node: Conv1d stack (channels `[32,64]`, kernel 5) + BatchNorm + ReLU, then `cnn_temporal_pool="avg"` (P11 lever, frozen to average pooling in P21 — `AdaptiveAvgPool1d(1)`), then a Linear projection to `embed_dim=64`.
- `cnn_weight_sharing="per_modality"`: one CNN instance shared across the 3 accel-axis nodes, a separate CNN instance shared across the 3 gyro-axis nodes (not 6 independent CNNs, not 1 CNN for all 6).
- Edge-type embedding: each edge is tagged intra-accel(0) / intra-gyro(1) / cross-modal(2) by `_get_edge_type` in the loader; `SensorGraphModel` embeds this and additively scatters it (× 0.1) into the *source* node's features before message passing — a small structural bias rather than a separate edge-feature channel (because, again, `SAGEConv` has no edge-feature input).
- Two `SAGEConv` layers (`gnn_hidden=64`), each followed by `BatchNorm`, `ReLU`, `Dropout(0.2)`.
- `global_mean_pool` over the 6 node embeddings → one 64-dim graph embedding per window.
- A `classifier` Linear head — which `DomainAdversarialWrapper` immediately strips out and replaces (next step).

`DomainAdversarialWrapper` (`gnn/models/da_sensor_graph.py`):
- In `__init__`, sets `self.backbone.classifier = nn.Identity()` and attaches its own `self.activity_classifier = nn.Linear(gnn_hidden, num_classes)`.
- Also attaches a domain discriminator head behind a `GradientReversalLayer` (`gnn/utils/gradient_reversal.py`) sized to `num_domains` (= number of *source* datasets in this fold, i.e. 5).
- `forward()` returns `(activity_logits, domain_logits)`; `predict()` returns activity logits only (used at test time, no domain head needed); `get_graph_embedding()` exposes the pre-classifier 64-dim embedding (this is the hook P18b's CORAL/MMD loss would read from — see comparison section).

### Step 5 — Training loop (`DANNTrainer`, `gnn/training/dann_trainer.py`)

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
trainer = DANNTrainer(model, optimizer, device, lambda_max=1.0, schedule="ganin",
                      max_grad_norm=1.0, scheduler=scheduler, patience=20)
```

Per epoch, `trainer.train_epoch(train_loader)`:
1. Computes `progress = current_step / total_steps` and the adversarial weight via `ganin_lambda_schedule(progress) = 2/(1+exp(-10*progress)) - 1` (Ganin et al. 2016) — starts near 0 (let the activity classifier stabilize first) and ramps to 1.
2. Forward pass → `activity_logits, domain_logits = model(batch)`.
3. `loss = F.cross_entropy(activity_logits, batch.y) + lambda * F.cross_entropy(domain_logits, batch.domain)`. The domain loss is adversarial only through the `GradientReversalLayer`: forward pass is identity, backward pass negates and scales the gradient by `lambda_val` (`GradientReversalFunction` in `gradient_reversal.py`), so the backbone is pushed to make the graph embedding *less* predictive of which source dataset a window came from — this is the actual domain-invariance mechanism.
4. There is also a dormant `use_align` branch in `train_epoch` that would add a CORAL/MMD alignment loss — `use_align = self.align_method is not None and target_loader is not None and self.align_lambda > 0.0`. P21 never sets `align_method` and never passes a `target_loader`, so this branch is provably dead code in every P21 run, even though the class is shared with P18b. This is the precise code-level proof that P21 is pure DG, not DA.
5. Gradient clipping (`max_grad_norm=1.0`), optimizer step, scheduler step.

`trainer.validate(val_loader)` runs the activity head only on source-val (no domain loss, no GRL). `trainer.update_checkpoint(val_acc, epoch)` saves a checkpoint if `val_acc` improved and tracks `patience` for early stopping. The fold's training loop in `gnn_predict.py` prints a line every `log_every` epochs (default 10) plus every time validation improves, then breaks early if `trainer.stopped_early`.

At the end, `trainer.restore_best()` reloads the best-val-acc checkpoint — *not* the last epoch's weights — before inference.

### Step 6 — Inference and output

```python
test_loader = DataLoader(lodo["test_graphs"], batch_size=bs, shuffle=False)  # ORDER preserved
...
logits = model.predict(batch)
probs.append(F.softmax(logits, dim=1).cpu().numpy())
```

`shuffle=False` matters: the saved `.npz` (`probs`, `labels`) preserves the target CSV's original row order, which is required downstream because `run_report.py` aligns the GNN's probs against the *separately trained* ConvNet's probs purely positionally (`np.array_equal(G["labels"], C["labels"])` is the alignment assertion in `run_report.py` line 38–39 — if either pipeline reordered rows this would fail loudly rather than silently fusing misaligned predictions).

Two files are written per target into `out_dir`: `gnn_<target>.npz` (probs + labels) and a running `val_summary.json` mapping target → best source-val accuracy, used later for protocol-clean config selection (Step 8) and for resumability (`main()` skips a target if both the npz and its `val_summary` entry already exist — every P21 script follows this same resumable pattern).

### Step 7 — Stage A: picking the Pearson config (`p21/run_sweep.py`)

Before seed-averaging anything, `run_sweep.py` runs the GNN once at seed 42 for each of 4 candidate Pearson configs:

```python
GRID = {
    "topk9_a1.0":  ["--select","topk","--topk","9","--alpha","1.0"],
    "topk6_a1.0":  ["--select","topk","--topk","6","--alpha","1.0"],
    "topk12_a1.0": ["--select","topk","--topk","12","--alpha","1.0"],
    "topk9_a0.5":  ["--select","topk","--topk","9","--alpha","0.5"],
}
```

Each config is launched as a subprocess calling `python -m p21.gnn_predict ... --edge_mode pearson --out_dir results/sweep_cfg/<config>`, producing 6 fold npz + a `val_summary.json`. The winner is selected by **mean source-val accuracy across the 6 folds — never target-test accuracy**, even though target-test numbers are also printed (labeled "FYI only"). This is the protocol-cleanliness guarantee: target-test data is never used, even informally, to choose a hyperparameter. The winner is written to `results/p21_sweep_winner.json`.

### Step 8 — Stage B: seed-averaging the winner (`p21/run_seeds.py`)

`run_seeds.py --config <winner> --seeds 1,2` (seed 42's Pearson cell is reused from Stage A if the config matches, via `_copy_npz`). For each seed it builds `results/sweep/<label>_s<seed>/` containing both arms' npz:

- **Arm A (`base_s<seed>`)** is not retrained at all — it's **copied** from `P17_DA_LODO/Project_DA_P18b_FeatAlign/results/sweep/base_s<seed>/` via `_copy_npz`. The justification embedded in the code comment: both P18b's "base" and P21's Arm A are the exact same DG-protocol fixed-topology GraphSAGE run, so retraining would (modulo GPU non-determinism) reproduce the same numbers at real compute cost. If that source directory is missing or incomplete, `run_seeds.py` raises a `SystemExit` with an explicit instructive message rather than silently retraining or silently failing.
- **Arm B (`pearson_<config>_s<seed>`)** retrains the GNN per seed with `--edge_mode pearson`.
- For **both arms**, the ConvNet branch's probabilities are copied in from the corresponding `base_s<seed>` cell's `convnet_*.npz` — the ConvNet is frozen across all of P20's arms (P21 never retrains it), so the same seed's frozen ConvNet output is reused for fusion in both arms. This guarantees that any accuracy delta between Arm A and Arm B is attributable only to the GNN branch's edge construction, not to ConvNet variance.

### Step 9 — Fusion and the KEEP/REVERT gate (`p21/run_report.py`)

`discover()` globs `results/sweep/*_s*`, regex-parsing `<label>_s<seed>` and grouping into `{label: {seed: {target: (gnn_probs, convnet_probs, labels)}}}`.

For each seed, `fuse_best_w(cond)` sweeps a shared fusion weight `w ∈ {0.0,0.3,0.4,0.5,0.6,0.7,1.0}` and computes `accuracy(w*P_gnn + (1-w)*P_convnet)` per fold, picking the single `w` that maximizes that seed's mean fused accuracy across all 6 folds (not a per-fold-tuned weight — one shared weight per seed, to avoid overfitting the fusion weight to the test set fold-by-fold).

`aggregate(seed_map)` then averages `fuse_best_w`'s output across seeds, producing per-condition mean ± std fused accuracy, GNN-only accuracy, and per-fold means.

The gate (`main()`, lines 120–130):
```python
dmean = (a["mean"] - bmean) * 100
worst = min((a["fold_mean"][t] - base["fold_mean"][t]) * 100 for t in TARGETS)
keep = (dmean >= 1.0) and (worst > -2.0)
```
i.e. KEEP only if the seed-averaged mean improvement is ≥ +1.0 percentage point **and** no single fold's seed-mean accuracy *dropped* by more than 2.0pp relative to the fixed-topology baseline. This guards against a config that wins on average by sacrificing one or two folds badly. The full result (per-condition aggregates + gate verdicts) is dumped to `results/p21_seed_report.json`, and the printed verdict explicitly states which downstream phase inherits the decision ("KEEP — carry into P22 (ResChebNet on Pearson edges)" / "REVERT — P22 runs ResChebNet on FIXED edges").

---

## 2. How P21 differs from each other ablation, code-for-code

### vs. P6/P7 (`PROJECTS/P06_P09_SensorGraph/`) — the original fixed-topology + DANN baseline

P6/P7 is the *origin* of `models/sensor_graph.py` and `models/da_sensor_graph.py`; P21's copies in `gnn/models/` are downstream derivatives, not the same file. Diffing them directly:

- `CNN1DEncoder` in P6/P7 has **no `temporal_pool` parameter at all** — it's hard-coded to `nn.AdaptiveAvgPool1d(1)` followed by a Linear projection. P21's copy adds the P11 `temporal_pool ∈ {avg, bigru, attention}` switch (with `bigru` and `attention` code paths present but unused since P21 freezes `cnn_temporal_pool="avg"`). Functionally identical in P21's actual runs, but P21's file is structurally a superset.
- P6/P7 has no `DualViewEncoder` class and no `_make_node_encoder` factory at all — those are P12 additions. P21 has them (inherited as dead code, since `encoder_mode="single"`).
- P6/P7's graph edges are exclusively the fixed `EDGES_FULL`/`EDGES_PHYSICAL`/`EDGES_MINIMAL` topologies from `config.py` — there is no Pearson code path anywhere in P6/P7's loader. P21's `gnn/data/pearson.py` does not exist in P6/P7 at all.
- P6 (`run_lodo.py` in `experiments/`) trains *without* DANN; P7 (`run_dann_lodo.py`) adds the GRL/domain-discriminator. P21 always runs with `dann_enabled=True` — i.e. P21's Arm A control is closer to P7's recipe than P6's, just with the hybrid features and SAGE backbone P9 later settled on.

So: P21 vs P6/P7 is "same family tree, but P21 adds (a) per-window data-driven topology as a new lever and (b) a temporal-pooling/dual-view switch that exists in the file but is inert."

### vs. P9 (`PROJECTS/P06_P09_SensorGraph/`, `results/P09_freqhybrid/`) — the frozen recipe P21 builds on

P9 is not a different architecture from P21's Arm A — it's the recipe P21 freezes verbatim. The `p21_config()` block in `gnn_predict.py` (Step 1 above) is a direct transcription of P9's settings: `gnn_type="sage"`, `feature_mode="hybrid"`, `daghar_timesteps=90`, `dann_lambda_max=1.0`, `dann_schedule="ganin"`. The only code-level difference from a literal P9 run is the addition of the `edge_mode`/`pearson_*` config keys (which default to reproducing P9's fixed-topology behavior when `edge_mode="fixed"`). In other words: **P21 Arm A ≡ P9**, and P21 Arm B ≡ "P9 with one component swapped." This is why P21's README frames Arm A as the control rather than re-deriving a new baseline.

### vs. P15 (`PROJECTS/P11_P16/Project_SensorGraph_P15_TTA/fusion/gnn_predict_tta.py`) — test-time adaptation

P15 trains the *identical* P9-recipe model (`p9_config()` in `gnn_predict_tta.py` is line-for-line the same dict as P21's frozen block) but diverges entirely at **inference time**, not at graph-construction time:

- `predict(model, loader, device)` — the plain DG inference path, identical in spirit to P21's Step 6.
- `adabn(model, loader, device, passes=1)` — deep-copies the trained model, calls `bn.reset_running_stats()` and sets `bn.momentum = None` on every `BatchNorm1d`/`BatchNorm2d` module, switches *only* those BN layers to `.train()` mode (everything else stays in eval), then runs forward passes over the **unlabeled target test set** so BN's running mean/var get recomputed to match the target domain's feature statistics. This is AdaBN.
- `tent(model, loader, device, steps=2, lr=1e-3)` — freezes every parameter except BN affine `weight`/`bias`, then runs a few gradient steps minimizing prediction entropy `-(p*log p).sum(1).mean()` on the unlabeled target batches, via a custom `model_predict_with_grad` that calls `model.backbone.get_graph_embedding(batch)` directly (bypassing the wrapper's no-grad `predict()`).

Crucially, **both AdaBN and TENT consume the target test set's *unlabeled inputs* at inference time** — something P21 never does (`load_target_unlabeled=False` everywhere in P21). The P15 file's own docstring is explicit about this: "AdaBN / TENT use the *unlabeled* target windows at inference. That is test-time domain ADAPTATION, not domain generalization." P21's Pearson edges are computed from the *source*-only training signal pattern (learned once, frozen, applied identically regardless of what target you're evaluating on); P15's adaptation recomputes statistics *per target dataset, at test time*. They're not just different code paths, they're different *evaluation protocols* — P15 is fundamentally not comparable to a pure-DG number without the "with adaptation" caveat the script itself flags.

### vs. P18a (`PROJECTS/P17_DA_LODO/Project_DA_P18a_TransNorm/`) — transductive normalization

P18a uses the exact same `daghar_loader.py` / `build_lodo_sensor_graphs` function P21 uses, with one flag flipped: `normalize_include_target=True`. Concretely, inside the loader, the per-channel z-normalization statistics (mean/std) are fit not just on source-train (P21's path) but on source-train **plus the target's own unlabeled windows**, before any source/target labels are touched. This requires `load_target_unlabeled=True` so those target windows are present at fold-build time. P21 never sets either flag — `p21_config()` hard-codes both to `False`/`"none"` (Step 1) — so P21's normalization statistics are computed from a strictly smaller, target-blind pool. The architectural and training code (model, DANN, trainer) is otherwise identical between P18a and P21; the difference lives entirely in two boolean flags inside the shared loader, which is exactly why this comparison can be made precisely instead of narratively.

### vs. P18b (`PROJECTS/P17_DA_LODO/Project_DA_P18b_FeatAlign/gnn/training/alignment.py`) — CORAL/MMD feature alignment

P18b adds an explicit alignment loss term on top of the shared `DANNTrainer.train_epoch`, computed in `alignment.py`:

- `coral_loss(Fs, Ft)`: squared Frobenius distance between the source-batch and target-batch feature covariance matrices, normalized by `4*d²` (Deep CORAL).
- `mmd_loss(Fs, Ft, bandwidths=(0.5,1.0,2.0,4.0,8.0))`: multi-bandwidth Gaussian RBF kernel MMD, with median-heuristic bandwidth scaling.
- `alignment_loss(method, Fs, Ft)` dispatches between them, returning 0 for `method="none"`.

This plugs into the *exact same* `use_align` branch in `gnn/training/dann_trainer.py`'s `train_epoch` that I noted above is dead code in P21 — P18b is the phase that actually activates it, by setting `align_method ∈ {"coral","mmd"}`, `align_lambda > 0`, and (critically) passing a `target_loader` built from `lodo["target_unlabeled_graphs"]` so `Ft` (target features) exists to align against `Fs` (source features) inside every training step. P21 shares the class but never supplies a `target_loader`, so the branch's `target_loader is not None` guard is always `False`. This is the cleanest code-level statement of "P21 is DG, P18b is DA": same trainer class, same conditional, opposite truth value of one guard.

### vs. P23 (`PROJECTS/P20_GraphRevival_DG/P23_PseudoLabel/p23/`) — pseudo-label self-training

P23 is a teacher/student pipeline, structurally the most different from P21 of all the comparisons:

- `teacher_predict.py`: trains a source-only model (P21-winner recipe) per seed, then dumps its softmax probabilities on the target's **unlabeled** pool (`tu_<target>.npz`) — true labels are loaded alongside *only* for later diagnostic precision logging, explicitly never touched by anything that affects training or selection (the file's docstring says so directly, and `student_predict.py`'s assertion `np.array_equal(y_true, y_graphs)` exists purely to catch alignment bugs, not to use the labels).
- `student_predict.py`: `select_pseudo_labels(P_tu, tau, balance)` filters the teacher's target-unlabeled predictions by confidence threshold `tau` (max softmax prob ≥ tau), optionally `balance="median"` capping each predicted class's count at the median per-class count among classes above threshold (guards against a confidently-wrong class flooding the pseudo-label pool — the docstring explicitly calls out "the WISDM 4-of-6-class trap", referencing that WISDM only has 4 of the 6 activity classes present, per `DAGHAR_DATASET_CLASSES` in `daghar_loader.py`). Selected target windows get `g.y = pseudo_label` and `g.domain = -100` (a sentinel masked out of the DANN domain loss — domain id −100 never matches a valid domain index, so `F.cross_entropy` with `ignore_index=-100` semantics — or equivalent masking — drops these from the adversarial loss), then are concatenated onto the source training set and a **freshly initialized** student model is trained from scratch (not fine-tuned from the teacher, to reduce confirmation bias).

This means P23 is the one phase here that genuinely uses target information during training — albeit only the *model's own* predicted pseudo-labels on unlabeled target windows, never true target labels — which puts it in the transductive-LODO-DA family alongside P18a/P18b, not the DG family P21 belongs to. P21's graph-construction lever (Pearson edges) is in principle orthogonal to P23's teacher/student mechanism: nothing in `pearson.py` or P21's `gnn_predict.py` would need to change to be slotted in as the architecture P23's teacher trains, but as implemented, P23's `common.py` config helper independently freezes "the P21-winner recipe" rather than importing P21's code directly — so the two phases share a recipe by convention, not by code reuse.

---

## 3. One-paragraph summary for the paper's ablation table

P21 isolates graph topology as a single experimental factor inside an otherwise-frozen DG pipeline (P9 recipe: hybrid features, per-modality CNN encoders, 2-layer GraphSAGE, Ganin-scheduled DANN, weighted-softmax fusion with a frozen ConvNet). P6/P7 is the ancestor architecture before the hybrid-feature/DANN-tuning settled; P9 *is* P21's Arm A by construction. P15, P18a, P18b, and P23 each modify a different stage of the *same* shared codebase — P15 at inference (test-time BN/entropy adaptation on unlabeled target), P18a at normalization-statistics-fitting (target-inclusive z-norm), P18b at the training loss (CORAL/MMD feature alignment against unlabeled target batches), P23 at the training *data* (model-generated pseudo-labels on unlabeled target windows) — while P21 modifies graph *construction* (per-window Pearson topology vs fixed K6/physical-prior topology), and is the only one of the five that never reads any target-side signal, labeled or unlabeled, during training or normalization.
