"""
P24 — CANONICAL 5-SEED RED / GS edge-budget study (hardening pass).

Multi-seed successor to `edge_budget.py`, which is INFERENCE-ONLY on the single
seed-42 `full` checkpoint and fuses against `base_s42`. This script is ADDITIVE
(it imports edge_budget.py's helpers, does not modify it) and repeats the same
policy evaluation for every seed in {1,2,3,5,42}:

    - GNN model  : the `full` checkpoint for that seed
                   (results/checkpoints/full   for seed 42,
                    results/checkpoints/full_s{seed}  otherwise)
    - ConvNet    : P21 base_s{seed} (matching-seed frozen probs)

Policies at matched budgets k in {9,6,3} undirected edges/window:
    pearson-k (P21 mechanism, k=9 reference), RED-k, GS-cross-k, GS-intra-k.

Everything is inference-only given the checkpoints — no retraining here. The
`full_s{seed}` checkpoints are produced by the notebook via
`p24.train_save --variant full --seed {seed}`.

Reports policy×budget seed mean±std (GNN + fused) and a paired Wilcoxon
signed-rank over the 6 folds for pearson-9 vs RED-9 (the "structure ≫ random"
claim). Output: results/p24_edge_budget_multiseed.json.
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P24 = os.path.dirname(HERE)
GNN_DIR = os.path.join(P24, "gnn")
sys.path.insert(0, GNN_DIR)

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from utils.seed import set_seed
from data.daghar_loader import (build_lodo_sensor_graphs, DAGHAR_DATASETS,
                                 load_daghar_dataset)
from data.pearson import pearson_abs_matrix
from p24.train_save import p24_config
from p24.compress_eval import load_model
# reuse the exact policy logic from the single-seed script (no duplication)
from p24.edge_budget import policy_edge_sets, rebuild_edges

TARGETS = list(DAGHAR_DATASETS)
FUSE_W = 0.6
BUDGETS = [9, 6, 3]
SEEDS = [1, 2, 3, 5, 42]
POLICIES = ["pearson", "RED", "GS-cross", "GS-intra"]
DEFAULT_P21 = os.path.normpath(os.path.join(
    P24, "..", "..", "01_Contribution1_Generalization_Model", "P21_PearsonGraph", "results", "sweep"))


def full_ckpt_dir(seed):
    d = os.path.join(P24, "results", "checkpoints",
                     "full" + ("" if seed == 42 else f"_s{seed}"))
    return d if os.path.isfile(os.path.join(d, f"ckpt_{TARGETS[0]}.pt")) else None


def eval_seed(seed, data_path, p21_sweep):
    """Per-(policy,budget) per-fold GNN + fused accuracy for one seed."""
    ck_dir = full_ckpt_dir(seed)
    conv_dir = os.path.join(p21_sweep, f"base_s{seed}")
    has_convnet = all(os.path.isfile(os.path.join(conv_dir, f"convnet_{t}.npz"))
                      for t in TARGETS)
    acc = {(p, k): {} for p in POLICIES for k in BUDGETS}
    fused = {(p, k): {} for p in POLICIES for k in BUDGETS}

    for t in TARGETS:
        ckpt = torch.load(os.path.join(ck_dir, f"ckpt_{t}.pt"),
                          map_location="cpu", weights_only=False)
        cfg = p24_config(data_path, 1, 10, 42, "full")
        cfg["daghar_timesteps"] = ckpt["node_feat_dim"]
        device = torch.device(cfg["device"])
        set_seed(42)   # graph rebuild determinism (independent of the model seed)
        lodo = build_lodo_sensor_graphs(cfg["data_root"], t, config=cfg,
                                        folder_map=cfg["dataset_folder_names"])
        test_graphs = lodo["test_graphs"]
        model = load_model(ckpt, cfg).to(device)

        X_te, _, _ = load_daghar_dataset(cfg["data_root"], t, split="test",
                                         folder_map=cfg["dataset_folder_names"])
        R = pearson_abs_matrix(X_te)

        Pc = yc = None
        if has_convnet:
            C = np.load(os.path.join(conv_dir, f"convnet_{t}.npz"))
            Pc, yc = C["probs"], C["labels"]

        for pol in POLICIES:
            for k in BUDGETS:
                graphs = rebuild_edges(test_graphs, policy_edge_sets(R, pol, k))
                loader = DataLoader(graphs, batch_size=cfg["batch_size"], shuffle=False)
                probs, labels = [], []
                with torch.no_grad():
                    for batch in loader:
                        batch = batch.to(device)
                        probs.append(F.softmax(model.predict(batch), 1).cpu().numpy())
                        labels.append(batch.y.cpu().numpy())
                P = np.concatenate(probs); y = np.concatenate(labels)
                acc[(pol, k)][t] = float((P.argmax(1) == y).mean())
                if Pc is not None:
                    assert np.array_equal(y, yc)
                    fused[(pol, k)][t] = float(
                        ((FUSE_W * P + (1 - FUSE_W) * Pc).argmax(1) == y).mean())
        print(f"    [edge_budget s{seed}:{t}] done", flush=True)
    return acc, fused


def paired_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    eff = float((a - b).mean())
    try:
        from scipy.stats import wilcoxon
        if np.allclose(a - b, 0):
            return {"test": "wilcoxon", "p": 1.0, "mean_diff_pp": eff * 100}
        stat, p = wilcoxon(a, b)
        return {"test": "wilcoxon", "stat": float(stat), "p": float(p),
                "mean_diff_pp": eff * 100}
    except Exception:
        from math import comb
        diff = (a - b); nz = diff[diff != 0]
        n = len(nz); kk = int((nz > 0).sum())
        p = sum(comb(n, i) for i in range(kk, n + 1)) / (2 ** n) if n else 1.0
        return {"test": "sign", "p": float(min(1.0, 2 * p)), "mean_diff_pp": eff * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATA_PATH"))
    ap.add_argument("--p21_sweep", default=DEFAULT_P21)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # per-seed fold means, keyed (policy,k) -> list over seeds of 6-fold-mean
    seed_acc = {(p, k): [] for p in POLICIES for k in BUDGETS}
    seed_fused = {(p, k): [] for p in POLICIES for k in BUDGETS}
    # per-fold seed-means for the Wilcoxon (fused)
    fold_fused = {(p, k): {t: [] for t in TARGETS} for p in POLICIES for k in BUDGETS}
    used = []
    for s in seeds:
        if full_ckpt_dir(s) is None:
            print(f"  [edge_budget] seed {s}: no 'full' checkpoint — "
                  f"run train_save --variant full --seed {s} first. Skipping.", flush=True)
            continue
        print(f"  [edge_budget] seed {s} ...", flush=True)
        acc, fused = eval_seed(s, args.data_path, args.p21_sweep)
        for key in seed_acc:
            seed_acc[key].append(np.mean(list(acc[key].values())))
            if fused[key]:
                seed_fused[key].append(np.mean(list(fused[key].values())))
                for t in TARGETS:
                    fold_fused[key][t].append(fused[key][t])
        used.append(s)

    if not used:
        raise SystemExit("[edge_budget] no seeds had a 'full' checkpoint.")

    print("\n" + "=" * 78)
    print(f"  P24 — 5-SEED EDGE-BUDGET STUDY (RED/GS vs Pearson; frozen 'full' per seed)")
    print(f"  seeds used: {used}   budget = undirected edges/window (full=15; trained on pearson-9)")
    print("=" * 78)
    print(f"\n  {'policy':10s} {'k':>3s} {'GNN mean±std':>15s} {'fused mean±std':>16s}")
    print("  " + "-" * 48)
    out = {}
    for pol in POLICIES:
        for k in BUDGETS:
            ga = np.array(seed_acc[(pol, k)]) * 100
            fa = np.array(seed_fused[(pol, k)]) * 100
            fstr = (f"{fa.mean():7.2f}±{fa.std():4.2f}%" if len(fa) else "     n/a")
            print(f"  {pol:10s} {k:>3d} {ga.mean():6.2f}±{ga.std():4.2f}% {fstr:>16s}")
            out[f"{pol}_k{k}"] = {
                "gnn_mean": float(ga.mean() / 100), "gnn_std": float(ga.std() / 100),
                "fused_mean": float(fa.mean() / 100) if len(fa) else None,
                "fused_std": float(fa.std() / 100) if len(fa) else None,
                "fold_fused_seedmean": {t: float(np.mean(v)) for t, v in
                                        fold_fused[(pol, k)].items()} if len(fa) else None,
            }
        print("  " + "-" * 48)

    # structure ≫ random: pearson-9 vs RED-9 paired over folds (fused seed-means)
    tests = {}
    pk, rk = ("pearson", 9), ("RED", 9)
    if len(seed_fused[pk]) and len(seed_fused[rk]):
        pv = [np.mean(fold_fused[pk][t]) for t in TARGETS]
        rv = [np.mean(fold_fused[rk][t]) for t in TARGETS]
        res = paired_test(pv, rv)
        tests["pearson9_vs_RED9"] = res
        print(f"\n  Paired Wilcoxon (fused, 6 folds): pearson-9 vs RED-9  "
              f"mean Δ {res['mean_diff_pp']:+.2f}pp | {res['test']} p={res['p']:.3f}")

    dest = os.path.join(P24, "results", "p24_edge_budget_multiseed.json")
    json.dump({"seeds": used, "policies": out, "wilcoxon": tests}, open(dest, "w"), indent=2)
    print(f"\n  saved {dest}")
    print("=" * 78)


if __name__ == "__main__":
    main()
