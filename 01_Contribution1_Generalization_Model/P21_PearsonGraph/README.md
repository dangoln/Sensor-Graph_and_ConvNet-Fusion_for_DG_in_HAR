# P21 — Pearson-Correlation Graph Construction (Arm B vs Arm A)

**Parent:** `../PROTOCOL_P20.md` — ablation design, gates, reference numbers.
**Inspiration:** Yang et al. 2024 (Mathematics 12:556, GDA) — data-driven sensor adjacency.

## The one lever

Replace the fixed full topology (P6-P18) with a **per-window Pearson-correlation
adjacency**: each 3-second window gets its own graph from the |ρ| between its 6
channels. Backbone stays **GraphSAGE** (control); ConvNet branch, fusion harness,
LODO splits, and selection-on-source-val are frozen. Both arms are **DG protocol**
(source-only — no target data of any kind, so numbers are directly comparable to
the P18b `base` cells).

**Technical note:** GraphSAGE ignores edge *weights*, so Pearson acts via
**topology selection** (which pairs become edges). Weighted adjacency is the P22
ChebNet's job (Chebyshev filters consume edge_weight natively via the Laplacian).
Per-window correlation uses only that window's own raw signal — no leakage path.

## What's in this folder

| Path | Role |
|------|------|
| `gnn/` | P18b GNN codebase, unchanged except: `data/pearson.py` (NEW — |ρ| computation + edge selection) and a `edge_mode="pearson"` hook in `data/daghar_loader.py` (per-window `edge_sets`). |
| `p21/gnn_predict.py` | Trains the P9-recipe GNN per fold with fixed or Pearson edges; dumps `gnn_<target>.npz` + `val_summary.json` (best source-val acc, used for config selection). Resumable. |
| `p21/run_sweep.py` | STAGE A: 4 Pearson configs at seed 42, winner picked on **mean source-val acc** (never target-test). |
| `p21/run_seeds.py` | STAGE B: seeds 42/1/2 of the winner. Copies Arm A (`base_s*`) + per-seed ConvNet probs from the P18b sweep — **no baseline or ConvNet retraining**. |
| `p21/run_report.py` | Seed-averaged fused report + KEEP/REVERT gate; also reports GNN-only deltas (informative for P22 even if the fused gate fails). |
| `notebooks/Run_P21_PearsonGraph_Colab.ipynb` | End-to-end Colab runner (smoke test → Stage A → Stage B → report). |

## Sweep grid (Stage A, selection on source-val)

| Config | Edges/window | Note |
|---|---|---|
| `topk9_a1.0` | top-9 pairs, pure Pearson | default density (= physical prior's 9) |
| `topk6_a1.0` | top-6, pure Pearson | sparser |
| `topk12_a1.0` | top-12, pure Pearson | denser |
| `topk9_a0.5` | top-9, score = 0.5·\|ρ\| + 0.5·physical prior | data + prior blend |

## Gate & decision

Seed-mean fused Δ ≥ **+1.0pp** vs base (76.87 ± 1.07%) AND no fold < −2pp.
KEEP → P22 ResChebNet runs on Pearson edges (with true edge weights). REVERT → P22
runs on fixed edges (Arm C), and the negative result is recorded in PROTOCOL_P20.

## Compute budget

Stage A ≈ 4 GNN-only 6-fold runs (~25 min each, T4). Stage B ≈ 2 more. Zero ConvNet
training; zero baseline retraining. Per-epoch logs on every run.
