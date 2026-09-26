"""
P24 — classic SO edge techniques (RED / GS) vs Pearson top-k, INFERENCE-ONLY.

The trained `full` checkpoint accepts arbitrary per-window edge sets, so the
P1/P6 edge-side SO suite needs NO retraining: we rebuild the target-test graphs
under each edge policy and re-evaluate the SAME model.

Edge policies, at matched budgets k in {9, 6, 3} undirected edges/window:
    pearson-k : blended-score top-k (the P21 winner mechanism; k=9 = reference)
    RED-k     : Random Edge Deletion — k pairs drawn uniformly per window (P1)
    GS-cross-k: Graph Sparsification — cross-modal pairs first, |rho| tie-break (P1)
    GS-intra-k: intra-modality pairs first, |rho| tie-break (P1)

The thesis question this answers: at a fixed edge-storage budget, does the
data-driven Pearson selection actually beat random/heuristic dropping?

Outputs results/p24_edge_budget.json + printed table (GNN acc + fused w=0.6).
NOTE: the model was TRAINED on pearson-9 graphs; all policies are evaluated on
that same frozen model (train/test edge-policy mismatch is part of the measured
cost — state this in the write-up).
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
from data.daghar_loader import (build_lodo_sensor_graphs, build_edge_index,
                                DAGHAR_DATASETS, load_daghar_dataset)
from data.pearson import pearson_abs_matrix, select_edges, ALL_PAIRS, PHYSICAL_PRIOR
from p24.train_save import p24_config
from p24.compress_eval import load_model

TARGETS = list(DAGHAR_DATASETS)
DEFAULT_CONVNET = os.path.normpath(os.path.join(
    P24, "..", "..", "01_Contribution1_Generalization_Model", "P21_PearsonGraph", "results", "sweep", "base_s42"))
FUSE_W = 0.6
BUDGETS = [9, 6, 3]
ALPHA = 0.5   # the locked P21 blend

CROSS_PAIRS = [p for p in ALL_PAIRS if (p[0] < 3) != (p[1] < 3)]   # 9 cross-modal
INTRA_PAIRS = [p for p in ALL_PAIRS if (p[0] < 3) == (p[1] < 3)]   # 6 intra


def scores_for(R):
    """Blended per-window pair scores, matching the trained graph construction."""
    pv = np.array([1.0 if p in PHYSICAL_PRIOR else 0.0 for p in ALL_PAIRS],
                  dtype=np.float32)
    pi = np.array(ALL_PAIRS)
    return ALPHA * R[:, pi[:, 0], pi[:, 1]] + (1 - ALPHA) * pv[None, :]


def policy_edge_sets(R, policy, k, seed=42):
    """Per-window undirected edge lists for one (policy, budget)."""
    N = R.shape[0]
    if policy == "pearson":
        return select_edges(R, mode="topk", topk=k, alpha=ALPHA)
    if policy == "RED":
        rng = np.random.RandomState(seed)
        return [[ALL_PAIRS[i] for i in rng.choice(len(ALL_PAIRS), size=k, replace=False)]
                for _ in range(N)]
    # GS: type priority first, |rho|-score tie-break inside each priority class
    S = scores_for(R)
    first = CROSS_PAIRS if policy == "GS-cross" else INTRA_PAIRS
    second = INTRA_PAIRS if policy == "GS-cross" else CROSS_PAIRS
    fi = [ALL_PAIRS.index(p) for p in first]
    si = [ALL_PAIRS.index(p) for p in second]
    out = []
    for n in range(N):
        ranked = sorted(fi, key=lambda i: -S[n, i]) + sorted(si, key=lambda i: -S[n, i])
        out.append([ALL_PAIRS[i] for i in ranked[:k]])
    return out


def rebuild_edges(graphs, edge_sets, use_edge_type=True):
    """Clone test graphs with replacement edges (model/features untouched)."""
    out = []
    for g, es in zip(graphs, edge_sets):
        g2 = g.clone()
        ei, et = build_edge_index(es, use_edge_type)
        g2.edge_index = ei
        if et is not None:
            g2.edge_type = et
        out.append(g2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATA_PATH"))
    ap.add_argument("--ckpt_variant", default="full")
    ap.add_argument("--convnet_dir", default=DEFAULT_CONVNET)
    args = ap.parse_args()

    ck_dir = os.path.join(P24, "results", "checkpoints", args.ckpt_variant)
    if not os.path.isfile(os.path.join(ck_dir, f"ckpt_{TARGETS[0]}.pt")):
        raise SystemExit(f"[edge_budget] no checkpoints in {ck_dir} — "
                         f"run p24.train_save --variant {args.ckpt_variant} first.")
    has_convnet = all(os.path.isfile(os.path.join(args.convnet_dir, f"convnet_{t}.npz"))
                      for t in TARGETS)

    policies = ["pearson", "RED", "GS-cross", "GS-intra"]
    acc = {(p, k): {} for p in policies for k in BUDGETS}
    fused = {(p, k): {} for p in policies for k in BUDGETS}

    for t in TARGETS:
        ckpt = torch.load(os.path.join(ck_dir, f"ckpt_{t}.pt"),
                          map_location="cpu", weights_only=False)
        cfg = p24_config(args.data_path, 1, 10, 42, args.ckpt_variant)
        cfg["daghar_timesteps"] = ckpt["node_feat_dim"]
        device = torch.device(cfg["device"])
        set_seed(42)
        lodo = build_lodo_sensor_graphs(cfg["data_root"], t, config=cfg,
                                        folder_map=cfg["dataset_folder_names"])
        test_graphs = lodo["test_graphs"]
        model = load_model(ckpt, cfg).to(device)

        # raw test windows for |rho| (same source as the loader used)
        X_te, _, _ = load_daghar_dataset(cfg["data_root"], t, split="test",
                                         folder_map=cfg["dataset_folder_names"])
        R = pearson_abs_matrix(X_te)

        Pc = yc = None
        if has_convnet:
            C = np.load(os.path.join(args.convnet_dir, f"convnet_{t}.npz"))
            Pc, yc = C["probs"], C["labels"]

        for pol in policies:
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
                        ((FUSE_W*P + (1-FUSE_W)*Pc).argmax(1) == y).mean())
        print(f"  [edge_budget:{t}] done", flush=True)

    print("\n" + "=" * 72)
    print(f"  P24 — EDGE-BUDGET STUDY (classic RED/GS vs Pearson; frozen '{args.ckpt_variant}' model)")
    print("  budget = undirected edges/window (full graph = 15; trained on pearson-9)")
    print("=" * 72)
    print(f"\n  {'policy':10s} {'k':>3s} {'GNN acc':>9s} {'fused':>9s}")
    print("  " + "-" * 36)
    for pol in policies:
        for k in BUDGETS:
            a = np.mean(list(acc[(pol, k)].values())) * 100
            f_ = (np.mean(list(fused[(pol, k)].values())) * 100
                  if fused[(pol, k)] else float("nan"))
            print(f"  {pol:10s} {k:>3d} {a:>8.2f}% {f_:>8.2f}%")
        print("  " + "-" * 36)
    out = os.path.join(P24, "results", "p24_edge_budget.json")
    json.dump({f"{p}_k{k}": {"acc": acc[(p, k)], "fused": fused[(p, k)]}
               for p in policies for k in BUDGETS}, open(out, "w"), indent=2)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
