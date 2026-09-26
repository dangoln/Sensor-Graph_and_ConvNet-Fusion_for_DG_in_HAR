"""
P24 — UNIFORM primary metrics (accuracy, macro-F1, macro-precision, macro-recall)
for the SO techniques whose runs saved accuracy only: RED / GS edge policies and
post-hoc precision (fp16 / int8 / pruning). Inference-only, one seed per call,
resumable; the retrain arms (NC / AW / capacity) already have per-config
probabilities, so they are metric-complete without this pass.

This script is ADDITIVE — it imports helpers from edge_budget.py and
compress_eval.py, edits nothing, and writes only new files:
    results/p24_so_metrics__s{seed}.json     (per-seed, for resumability)
    results/p24_so_metrics_summary.json      (--summary; 5-seed mean/std tables)

Usage (see the notebook):
    python -m p24.so_metrics_recompute --data_path $DATA --seed 42
    python -m p24.so_metrics_recompute --summary
"""
import os, sys, json, glob, argparse
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
from p24.compress_eval import (load_model, predict, state_dict_bytes,
                               quantize_dynamic_int8, global_magnitude_prune)
from p24.edge_budget import policy_edge_sets, rebuild_edges

TARGETS = list(DAGHAR_DATASETS)
FUSE_W = 0.6
BUDGETS = [9, 6, 3]
EDGE_POLICIES = ["pearson", "RED", "GS-cross", "GS-intra"]
POSTHOC_VARIANTS = ["full", "tiny"]
RESULTS = os.path.join(P24, "results")
DEFAULT_SWEEP = os.path.normpath(os.path.join(P24, "..", "..", "01_Contribution1_Generalization_Model", "P21_PearsonGraph",
                                              "results", "sweep"))


# ---------- macro metrics over the classes present in a fold --------------
def macro_prf(y, yp):
    Ps, Rs, Fs = [], [], []
    for c in np.unique(y):
        tp = int(np.sum((yp == c) & (y == c)))
        fp = int(np.sum((yp == c) & (y != c)))
        fn = int(np.sum((yp != c) & (y == c)))
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        Fs.append(2 * P * R / (P + R) if P + R else 0.0); Ps.append(P); Rs.append(R)
    return float(np.mean(Ps)), float(np.mean(Rs)), float(np.mean(Fs))


def fold_metrics(pred, y):
    """(acc, f1, precision, recall) for one fold."""
    acc = float((pred == y).mean())
    p, r, f1 = macro_prf(y, pred)
    return [acc, f1, p, r]


def _ckpt_dir(variant, seed):
    return os.path.join(RESULTS, "checkpoints",
                        variant + ("" if seed == 42 else f"_s{seed}"))


def _convnet(sweep, seed):
    d = os.path.join(sweep, f"base_s{seed}")
    return d if all(os.path.isfile(os.path.join(d, f"convnet_{t}.npz"))
                    for t in TARGETS) else None


# ---------- edge policies (RED / GS / pearson) ----------------------------
def compute_edges(seed, data_path, sweep):
    ck_dir = _ckpt_dir("full", seed)
    conv_dir = _convnet(sweep, seed)
    # accumulate per-fold metric rows, keyed (policy,k) -> {'gnn':[...],'fused':[...]}
    acc = {(p, k): {"gnn": [], "fused": []} for p in EDGE_POLICIES for k in BUDGETS}
    for t in TARGETS:
        ckpt = torch.load(os.path.join(ck_dir, f"ckpt_{t}.pt"),
                          map_location="cpu", weights_only=False)
        cfg = p24_config(data_path, 1, 10, 42, "full")
        cfg["daghar_timesteps"] = ckpt["node_feat_dim"]
        device = torch.device(cfg["device"]); set_seed(42)
        lodo = build_lodo_sensor_graphs(cfg["data_root"], t, config=cfg,
                                        folder_map=cfg["dataset_folder_names"])
        model = load_model(ckpt, cfg).to(device)
        X_te, _, _ = load_daghar_dataset(cfg["data_root"], t, split="test",
                                         folder_map=cfg["dataset_folder_names"])
        R = pearson_abs_matrix(X_te)
        C = np.load(os.path.join(conv_dir, f"convnet_{t}.npz"))
        Pc, yc = C["probs"], C["labels"]
        for pol in EDGE_POLICIES:
            for k in BUDGETS:
                graphs = rebuild_edges(lodo["test_graphs"], policy_edge_sets(R, pol, k))
                loader = DataLoader(graphs, batch_size=cfg["batch_size"], shuffle=False)
                probs, labels = [], []
                with torch.no_grad():
                    for b in loader:
                        b = b.to(device)
                        probs.append(F.softmax(model.predict(b), 1).cpu().numpy())
                        labels.append(b.y.cpu().numpy())
                Pg = np.concatenate(probs); y = np.concatenate(labels)
                assert np.array_equal(y, yc)
                gpred = Pg.argmax(1); fpred = (FUSE_W * Pg + (1 - FUSE_W) * Pc).argmax(1)
                acc[(pol, k)]["gnn"].append(fold_metrics(gpred, y))
                acc[(pol, k)]["fused"].append(fold_metrics(fpred, y))
        print(f"    [edges s{seed}:{t}] done", flush=True)
    # macro-average the 6 folds -> this seed's metric vector
    out = {}
    for (pol, k), d in acc.items():
        out[f"{pol}_k{k}"] = {branch: list(np.array(rows).mean(0))
                              for branch, rows in d.items()}
    return out


# ---------- post-hoc precision (fp16 / int8 / prune) ----------------------
def compute_posthoc(seed, data_path, sweep):
    conv_dir = _convnet(sweep, seed)
    result = {}
    for variant in POSTHOC_VARIANTS:
        ck_dir = _ckpt_dir(variant, seed)
        if not os.path.isfile(os.path.join(ck_dir, f"ckpt_{TARGETS[0]}.pt")):
            continue
        rows = {}   # comp -> {'gnn':[folds], 'fused':[folds], 'size':bytes, 'params':n}
        for t in TARGETS:
            ckpt = torch.load(os.path.join(ck_dir, f"ckpt_{t}.pt"),
                              map_location="cpu", weights_only=False)
            cfg = p24_config(data_path, 1, 10, 42, variant)
            cfg["daghar_timesteps"] = ckpt["node_feat_dim"]
            device = torch.device(cfg["device"]); set_seed(42)
            lodo = build_lodo_sensor_graphs(cfg["data_root"], t, config=cfg,
                                            folder_map=cfg["dataset_folder_names"])
            g = lodo["test_graphs"]; bs = cfg["batch_size"]
            model = load_model(ckpt, cfg)
            C = np.load(os.path.join(conv_dir, f"convnet_{t}.npz"))
            Pc, yc = C["probs"], C["labels"]

            def rec(name, P, y, size_b, params):
                assert np.array_equal(y, yc)
                r = rows.setdefault(name, {"gnn": [], "fused": [], "size": size_b,
                                           "params": params})
                r["gnn"].append(fold_metrics(P.argmax(1), y))
                r["fused"].append(fold_metrics((FUSE_W * P + (1 - FUSE_W) * Pc).argmax(1), y))

            npar = ckpt["n_params"]
            P, y = predict(model.to(device), g, device, bs)
            rec("fp32", P, y, state_dict_bytes(model.cpu()), npar)
            if torch.cuda.is_available():
                import copy
                m16 = copy.deepcopy(model).half().to(device)
                P16, y16 = predict(m16, g, device, bs, half=True)
                rec("fp16", P16, y16, state_dict_bytes(m16.cpu()), npar)
            mq = quantize_dynamic_int8(model)
            Pq, yq = predict(mq, g, torch.device("cpu"), bs)
            rec("int8-dyn", Pq, yq, state_dict_bytes(mq), npar)
            for frac in (0.3, 0.5, 0.7):
                mp, nnz, total = global_magnitude_prune(model, frac)
                Pp, yp = predict(mp.to(device), g, device, bs)
                other = sum(p.numel() for p in model.parameters()) - total
                rec(f"prune{int(frac*100)}", Pp, yp, int(nnz * 8 + other * 4), nnz + other)
            print(f"    [posthoc s{seed}:{variant}:{t}] done", flush=True)
        result[variant] = {c: {"gnn": list(np.array(r["gnn"]).mean(0)),
                               "fused": list(np.array(r["fused"]).mean(0)),
                               "size": r["size"], "params": r["params"]}
                           for c, r in rows.items()}
    return result


# ---------- summary across seeds ------------------------------------------
def summarize():
    files = sorted(glob.glob(os.path.join(RESULTS, "p24_so_metrics__s*.json")))
    if not files:
        raise SystemExit("[summary] no per-seed files — run the recompute first.")
    seeds = [json.load(open(f)) for f in files]
    n = len(seeds)
    names = ["acc", "f1", "precision", "recall"]

    def agg(getter):
        # getter(seedobj) -> list of 4 metrics; returns mean,std across seeds
        arr = np.array([getter(s) for s in seeds]) * 100
        return {names[i]: [float(arr[:, i].mean()), float(arr[:, i].std())] for i in range(4)}

    out = {"n_seeds": n, "edges": {}, "posthoc": {}}
    # edges
    for key in seeds[0]["edges"]:
        out["edges"][key] = {b: agg(lambda s, key=key, b=b: s["edges"][key][b])
                             for b in ("gnn", "fused")}
    # posthoc (use variants/comps present in every seed)
    for variant in seeds[0]["posthoc"]:
        out["posthoc"][variant] = {}
        for comp in seeds[0]["posthoc"][variant]:
            out["posthoc"][variant][comp] = {
                "size_kb": seeds[0]["posthoc"][variant][comp]["size"] / 1024.0,
                "params": seeds[0]["posthoc"][variant][comp]["params"],
                "gnn": agg(lambda s, v=variant, c=comp: s["posthoc"][v][c]["gnn"]),
                "fused": agg(lambda s, v=variant, c=comp: s["posthoc"][v][c]["fused"]),
            }
    dest = os.path.join(RESULTS, "p24_so_metrics_summary.json")
    json.dump(out, open(dest, "w"), indent=2)

    def line(tag, m):
        f = m["fused"]
        print(f"  {tag:22s} acc {f['acc'][0]:5.2f}±{f['acc'][1]:.2f}  "
              f"F1 {f['f1'][0]:5.2f}  P {f['precision'][0]:5.2f}  R {f['recall'][0]:5.2f}")
    print(f"\n=== EDGE POLICIES (5-seed, n={n}) ===")
    for key, m in out["edges"].items():
        line(key, m)
    print(f"\n=== POST-HOC PRECISION (5-seed, n={n}) ===")
    for v in out["posthoc"]:
        for comp, m in out["posthoc"][v].items():
            line(f"{v}/{comp}", m)
    print(f"\n  saved {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATA_PATH"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--p21_sweep", default=DEFAULT_SWEEP)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summarize(); return
    if args.seed is None:
        raise SystemExit("pass --seed S (or --summary)")
    out_path = os.path.join(RESULTS, f"p24_so_metrics__s{args.seed}.json")
    if os.path.isfile(out_path):
        print(f"[so_metrics] seed {args.seed} already done — skipping."); return
    print(f"[so_metrics] seed {args.seed}: edges + post-hoc ...", flush=True)
    payload = {"seed": args.seed,
               "edges": compute_edges(args.seed, args.data_path, args.p21_sweep),
               "posthoc": compute_posthoc(args.seed, args.data_path, args.p21_sweep)}
    json.dump(payload, open(out_path, "w"), indent=2)
    print(f"  saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
