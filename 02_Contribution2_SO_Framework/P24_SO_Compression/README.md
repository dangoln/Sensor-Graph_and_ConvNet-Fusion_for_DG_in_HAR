# P24 — Storage Optimization / Compression (the SO in SO-DAGHAR)

**Parent:** `../PROTOCOL_P20.md`. **System under test:** the locked project best —
Pearson-blend graphs (topk9 α0.5) + GraphSAGE + frozen ConvNet fusion,
**77.96 ± 0.21% (DG, canonical 5 seeds; 78.10 ± 0.14% at the earlier 3 seeds)**. P24 is **measurement, not a performance lever** — no gate.

## Why now

Performance work is closed: P21 KEEP (+1.23pp), P22/P23 REVERT, DA rung closed
across alignment and self-training, and parameter-free fusion alternatives all lose
to plain averaging. The remaining thesis question is the one in the title: what does
this accuracy *cost* in storage/compute, and how cheaply can we keep it?

## Five axes — including the original P1/P6 SO suite on the best model

| Axis | How | Compute |
|---|---|---|
| Architecture capacity | retrain `full` (64/64, the winner), `compact` (32/32), `tiny` (16/16) and checkpoint them | ~3 GNN runs |
| Post-hoc compression | fp16, dynamic int8 (Linear layers), global magnitude pruning 30/50/70% — on saved checkpoints, no fine-tuning | inference only |
| **Classic SO — edges (RED, GS)** | `p24/edge_budget.py`: RED (random edges) and GS (type-priority) vs Pearson top-k at matched budgets k ∈ {9,6,3}, re-evaluated on the frozen `full` checkpoints. Answers "does data-driven edge selection beat random/heuristic dropping at equal storage?" | inference only |
| **Classic SO — input (NC, AW)** | `train_save.py --nc accel_only\|gyro_only\|no_gz` (3/3/5-node graphs, Pearson prior remapped, edge-type embedding off) and `--aw 40\|30` (strided 60→N timesteps) | ~5 GNN runs |
| Graph density | accuracy vs edges/window (top-6/9/12), re-read from P21 Stage A dumps | zero |

**Continuity note:** NC/RED/AW/GS are the P1 storage-optimization techniques
(adapted to the sensor graph in P6), now finally evaluated on the best-performing
system — this closes the SO-DAGHAR arc P1 → P24. RED/GS run at inference because
the model accepts arbitrary per-window edge sets (a P21 side benefit); NC/AW change
the input shape and therefore retrain.

Every row reports: params, on-disk size, GNN-only acc, **fused** acc with the frozen
ConvNet (w=0.6, P21 `base_s42` probs), plus CPU ms/window for fp32 vs int8.

## What's in this folder

| Path | Role |
|------|------|
| `gnn/` | P21 GNN codebase, unchanged. |
| `p24/train_save.py` | Trains a capacity variant per fold (seed 42) and **saves checkpoints** + test probs. Resumable. |
| `p24/compress_eval.py` | Loads checkpoints; builds the storage-vs-accuracy table (fp32/fp16/int8/prune30-50-70 × variants) + latency; saves `p24_compression_report.json`. |
| `p24/edge_curve.py` | Report-only edge-density curve from P21 dumps (verified working). |
| `notebooks/Run_P24_SO_Compression_Colab.ipynb` | End-to-end runner. |

## Caveats to carry into the write-up

`compact`/`tiny` are single-seed retrains (note it). Pruning is post-hoc without
fine-tuning — if a pruning row collapses, that is the honest finding, not a bug.
Compression rows are deterministic given a checkpoint. The fused numbers in this
phase are seed-42; the 3-seed reference for the uncompressed system is 78.10 ± 0.14%.
