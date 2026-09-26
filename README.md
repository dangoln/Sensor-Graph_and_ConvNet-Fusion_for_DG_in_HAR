# Final_PROJECT — Storage-Optimized GNNs for Cross-Dataset HAR (DAGHAR, LODO)

This folder holds **only the final contributions** of the thesis — the code, the
already-run Colab notebooks, and the result files — so it can be read without the
~30 intermediate project phases (P1–P30) that led here.

> **Every design choice in the final model was reached through a long series of
> ablation phases.** Those phases are not included here; their rationale, numbers and
> negative results are documented in the thesis report:
> `00_Thesis_Report/Thesis_Report.pdf` — Chapter 3 (Methodology) describes the final
> system; Chapter 4 (Empirical Studies, Arcs I–VI) walks through how each component
> was selected.

---

## The two contributions

| # | Contribution | Headline result (LODO, 6 folds, 5 seeds) | Folder |
|---|---|---|---|
| 1 | **Locked generalization model** — GNN generalization branch with an adaptive Pearson-blend graph, late-fused with a frozen ConvNet (DeepConvLSTM) branch | **77.96% ± 0.21%** mean accuracy — matches the DAGHAR ConvNet SOTA (~77%) with no target-domain data | `01_Contribution1_Generalization_Model/` |
| 2 | **Storage-optimization (SO) framework** — five families of SO operators applied to the locked model, and the storage–generalization trade-off | Dropping gyro-z is free (77.97%); tiny backbone 76.97% at ~5.9× smaller; at a matched 9-edge budget, structured edges beat random by +8.25pp; Pareto frontier = {no-gyro-z, tiny} | `02_Contribution2_SO_Framework/` |

Evaluation protocol everywhere: DAGHAR `standardized_view` (6 datasets: KuHar,
MotionSense, RealWorld-Thigh, RealWorld-Waist, UCI, WISDM; 6 channels accel+gyro;
60 timesteps @ 20 Hz; 6 activities), **Leave-One-Dataset-Out**, model selection on
source-validation only, seeds {1, 2, 3, 5, 42}.

---

## Folder map

```
Final_PROJECT/
├── README.md                          ← you are here
├── project_paths.py                   ← the ONE file with paths (edit DEFAULT_DATA_PATH if needed)
├── 00_Check_Setup_Colab.ipynb         ← run first on Colab
├── 00_Thesis_Report/
│   ├── Thesis_Report.pdf              ← full thesis (compiled 2026-09-19)
│   └── framework_figures/             ← overview diagrams used in Ch. 3
│       ├── Proposed_Framework.png         (Contribution 1 pipeline)
│       ├── GNN_branch.png / ConvNet_branch.png
│       └── Proposed_SO_framework.png      (Contribution 2 operators)
│
├── 01_Contribution1_Generalization_Model/
│   ├── P21_Model_Walkthrough.md       ← step-by-step, code-level tour of the locked model (start here)
│   ├── P21_PearsonGraph/              ← GNN generalization branch + late fusion (THE locked model)
│   │   ├── README.md, P21_CODE_WALKTHROUGH.md, OVERVIEW/ (architecture docs, .svg, TikZ)
│   │   ├── gnn/                       ← model / data / training library
│   │   ├── p21/                       ← experiment drivers (predict, sweep, seeds, report/fusion)
│   │   ├── notebooks/Run_P21_PearsonGraph_Colab.ipynb   (already-run)
│   │   └── results/                   ← per-fold probabilities (.npz) + seed reports (.json)
│   └── ConvNet_Branch_Frozen/
│       ├── A_DeepConvLSTM_Reimplementation/   ← the ConvNet branch model itself (PyTorch DeepConvLSTM,
│       │                                         frequency-view recipe "run2_nohann") + its LODO results
│       └── B_Frozen_Predictions_P18b_SourceOnly/  ← the run that produced the FROZEN ConvNet
│                                                  probabilities the fusion actually consumes
│                                                  (source-only "base" arm, seeds 42/1/2)
│
└── 02_Contribution2_SO_Framework/
    └── P24_SO_Compression/
        ├── README.md
        ├── gnn/                       ← the P21 library + node-coarsening / windowing hooks
        ├── p24/                       ← SO operator scripts (see table below)
        ├── notebooks/                 ← 4 Colab runners (+ PDF export of the hardening run)
        └── results/                   ← checkpoints per variant/seed + all SO result JSONs + Pareto figure
```

---

## Contribution 1 — where each component of the locked model lives

All paths relative to `01_Contribution1_Generalization_Model/P21_PearsonGraph/`
unless stated. Thesis section numbers refer to Chapter 3.

| Thesis component (Ch. 3 §3.2) | Code |
|---|---|
| Sensor-as-node graph (6 nodes = accel x/y/z, gyro x/y/z) | `gnn/data/daghar_loader.py` (`build_sensor_graphs`, `build_lodo_sensor_graphs`), edge sets in `gnn/utils/config.py` |
| Hybrid node features (60 raw timesteps + 30 FFT-magnitude bins = 90) | `gnn/data/daghar_loader.py` (`_compute_node_features`, `feature_mode="hybrid"`) |
| CNN-based per-node feature encoder (per-modality weight sharing) | `gnn/models/sensor_graph.py` (`CNN1DEncoder`) |
| GraphSAGE backbone (2 × SAGEConv, hidden 64, mean pooling) | `gnn/models/sensor_graph.py` (`SensorGraphModel`) |
| DANN (gradient reversal, Ganin schedule, domain head) | `gnn/models/da_sensor_graph.py`, `gnn/utils/gradient_reversal.py`, `gnn/training/dann_trainer.py` |
| **Adaptive Pearson-blend graph + top-k selection** (score = 0.5·\|ρ\| + 0.5·physical prior, top-9 per window) | `gnn/data/pearson.py` (`pearson_abs_matrix`, `select_edges`) |
| Frozen recipe / one-fold train + predict | `p21/gnn_predict.py` (`p21_config`, `train_and_predict`) |
| Config selection on source-val (Stage A) & seed confirmation (Stage B) | `p21/run_sweep.py`, `p21/run_seeds.py` |
| **Late fusion** P = w·P_gnn + (1−w)·P_conv, w = 0.6 | `p21/run_report.py` |
| ConvNet (DeepConvLSTM) branch, frequency view (log-FFT, DC kept) | `ConvNet_Branch_Frozen/A_…/src/models/deepconvlstm.py`, `src/data/data_loader.py`; trained per fold by `ConvNet_Branch_Frozen/B_…/da_baseline/convnet_predict.py` |

**Key result files** (`P21_PearsonGraph/results/`):

- `p21_seed_report.json` — canonical 5-seed report: `pearson_topk9_a0.5` (locked model) **77.96 ± 0.21%** fused / 74.41% GNN-only; `base` (fixed-topology control) 76.68 ± 1.36%; `pearson_topk9_a1.0` (pure correlation, 3 seeds) 76.98%.
- `p21_full_metrics.json` — accuracy, macro precision / recall / F1 per fold and per seed.
- `p21_sweep_winner.json` + `sweep_cfg/` — Stage A screening (4 graph configs, seed 42), winner chosen on source-val.
- `sweep/<config>_s<seed>/` — per-fold `gnn_<target>.npz` and `convnet_<target>.npz` (probabilities + labels) used by the fusion report.

**How the frozen ConvNet reaches the fusion.** The ConvNet branch is trained once and
never retrained. Its per-fold probabilities for seeds 42/1/2 were produced in the
P18b source-only baseline run (`ConvNet_Branch_Frozen/B_…/results/sweep/base_s*`) and
copied into `P21_PearsonGraph/results/sweep/base_s*/convnet_*.npz`. For seeds 3 and 5
the seed-42 ConvNet probabilities were reused as a stand-in (the P21 notebook says so
explicitly; files are byte-identical). Earlier fusion phases (P14–P16a) used their own
ConvNet copies; they are not part of the final model.

---

## Contribution 2 — SO framework operators

All paths relative to `02_Contribution2_SO_Framework/P24_SO_Compression/`.
Operators act on the GNN branch only; the ConvNet branch stays frozen.

| Thesis axis (Ch. 3 §3.3) | Operators | Script(s) | Main result file(s) |
|---|---|---|---|
| Architecture capacity reduction | `full` (64/64), `compact` (32/32), `tiny` (16/16), retrained | `p24/train_save.py --variant …` | `results/checkpoints/{full,compact,tiny}*`, `p24_seed_report_5.json` |
| Post-hoc model compression | fp16, dynamic int8, global magnitude pruning 30/50/70 % (no fine-tuning) | `p24/compress_eval.py` | `p24_compression_report.json` |
| Graph-level optimization | Random edge deletion (RED), graph sparsification (GS-intra / GS-cross) vs Pearson top-k at matched budgets k ∈ {9,6,3} (inference-only); edge-density curve | `p24/edge_budget.py`, `p24/edge_budget_multiseed.py`, `p24/edge_curve.py` | `p24_edge_budget_multiseed.json` (5-seed, canonical), `p24_edge_budget.json`, `p24_edge_curve.json` |
| Input-level optimization | Node coarsening (`no_gz`, `accel_only`, `gyro_only`); adaptive windowing (60→40/30) | `p24/train_save.py --nc … / --aw …` | `results/checkpoints/full_nc-*`, `full_aw*`, `p24_seed_report_5.json` |
| Storage–generalization trade-off | Combined axes (no_gz + AW-40 + int8), GenPerByte, uniform metrics, Pareto frontier | `p24/combined_axis.py`, `p24/seed_report_5.py`, `p24/so_metrics_recompute.py`, `p24/pareto_figure.py` | `p24_combined_axis.json`, `p24_so_metrics_*.json`, `p24_pareto.png`, `p24_pareto_points.csv` |

Notebooks: `Run_P24_SO_Compression_Colab.ipynb` (seed-42 pass),
`Run_P24_MultiSeed_Hardening_Colab.ipynb` (5-seed hardening — canonical numbers),
`Run_P24_CombinedAxis_Colab.ipynb`, `Run_P24_SO_Metrics_Recompute_Colab.ipynb`.
`seed_summary.py` / `p24_seed_summary.json` are the older 3-seed summary kept for
provenance; the 5-seed files supersede them.

---

## Running the code on Google Colab

**You don't need to re-run anything to see the thesis results** — every result file is included, and
re-running would reproduce the same numbers up to GPU non-determinism. Run the notebooks only to
verify the pipeline or to extend it.

1. Upload the whole `Final_PROJECT` folder to Google Drive (default location assumed by the notebooks:
   `/content/drive/MyDrive/Thesis/SO_Experiments/Final_PROJECT` — any other location is found
   automatically if it is within a few folder levels of `MyDrive`).
2. **Paths are set in one place: `project_paths.py`** (this folder). The project root is detected
   automatically; the only value you may need to change is `DEFAULT_DATA_PATH`, the folder that holds
   the DAGHAR `standardized_view` datasets (default
   `/content/drive/MyDrive/HAR-Datasets/DAGHAR/standardized_view`). The DAGHAR data itself is not included.
3. Open **`00_Check_Setup_Colab.ipynb`** first (no GPU, ~1 min): it confirms the folder, the data and
   the key result files, and re-prints the headline accuracy from the saved results.
4. Every other notebook starts with the same "Mount Drive & load Final_PROJECT paths" cell, then runs as
   before. Use a GPU runtime (T4 is enough) for training cells.

**Re-running is safe.** The P21 and P24 training scripts are resumable: a fold whose result files already
exist is skipped, so running a notebook in place re-prints/re-aggregates the thesis results instead of
overwriting them. Smoke tests write to separate `_smoke` folders. To reproduce a run *from scratch*, work
on a copy of the folder and delete that component's `results/` sub-folders first. ConvNet re-training
(`Run_Frozen_ConvNet_Colab.ipynb`) always writes to `results/sweep_rerun/`, never over the frozen
predictions.

| Notebook | What it runs | Saved outputs |
|---|---|---|
| `00_Check_Setup_Colab.ipynb` | Setup + results check | new |
| `01_…/P21_PearsonGraph/notebooks/Run_P21_PearsonGraph_Colab.ipynb` | Locked model: Stage A screening, 5-seed training, fusion report, macro metrics | original run |
| `01_…/ConvNet_Branch_Frozen/B_…/notebooks/Run_Frozen_ConvNet_Colab.ipynb` | Inspect / optionally re-train the frozen ConvNet predictions | new |
| `01_…/ConvNet_Branch_Frozen/A_…/notebooks/Run_ConvNet_DAGHAR_Colab.ipynb` | Standalone DeepConvLSTM reimplementation vs DAGHAR's published numbers | original run |
| `02_…/P24_SO_Compression/notebooks/Run_P24_SO_Compression_Colab.ipynb` | SO seed-42 pass (capacity, post-hoc, RED/GS, NC/AW) | original run |
| `02_…/…/Run_P24_MultiSeed_Hardening_Colab.ipynb` | Canonical 5-seed SO numbers | original run |
| `02_…/…/Run_P24_CombinedAxis_Colab.ipynb` | Combined axes + Pareto figure | original run |
| `02_…/…/Run_P24_SO_Metrics_Recompute_Colab.ipynb` | Macro-F1/precision/recall for edge & post-hoc variants | saved without outputs (results present) |
| `01_…/ConvNet_Branch_Frozen/B_…/notebooks/Run_P18b_*.ipynb` | **Historical record only** (full P18b phase; GNN/CORAL code not included) — do not re-run | original run |

Notebooks marked "original run" keep their outputs from the thesis run (when the code lived at its old
Drive location); a note at the top of each says so.

## Provenance (where each folder was copied from)

| Final_PROJECT path | Original location in `PROJECTS/` |
|---|---|
| `01_…/P21_PearsonGraph/` | `P20_GraphRevival_DG/P21_PearsonGraph/` (full copy) |
| `01_…/ConvNet_Branch_Frozen/A_DeepConvLSTM_Reimplementation/` | `P11_P16/Project_ConvNet_DAGHAR/` (code, notebook, report; results limited to the adopted `run2_nohann` recipe) |
| `01_…/ConvNet_Branch_Frozen/B_Frozen_Predictions_P18b_SourceOnly/` | `P17_DA_LODO/Project_DA_P18b_FeatAlign/` (ConvNet code, `convnet_predict.py`, notebooks, `base_s42/s1/s2` only — CORAL arms and the P18b GNN side excluded) |
| `01_…/P21_Model_Walkthrough.md` | `Meetings Planning/P21_Model_Walkthrough.md` |
| `02_…/P24_SO_Compression/` | `P20_GraphRevival_DG/P24_SO_Compression/` (full copy) |
| `00_Thesis_Report/` | `Thesis Prep/latex_doc/main.pdf` and `pics/` |

Cache files (`__pycache__`, `.DS_Store`) and the `.zip` archives were not copied. No
original file in `PROJECTS/` was modified.

**Changes made in Final_PROJECT relative to the originals (2026-09-26):** notebooks' path cells now load
`project_paths.py`; scripts' default paths point to the new layout (P24 → `01_…/P21_PearsonGraph`, P21 →
`ConvNet_Branch_Frozen/B_…`); the GNN/ConvNet configs read the `DATA_PATH` / `PROJECT_PATH` environment
variables; `p24/train_save.py` gained an optional `--out_root` (used only by smoke tests); the P21
seeds-3/5 cell writes its base cells inside P21 instead of the P18b folder; P24 smoke tests no longer
delete real checkpoints. No model, training or evaluation logic was changed.
