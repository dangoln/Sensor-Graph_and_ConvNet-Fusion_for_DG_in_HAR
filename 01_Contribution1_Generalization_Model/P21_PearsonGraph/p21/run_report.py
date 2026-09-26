"""
P21 — seed-averaged report + KEEP/REVERT gate (Arm B Pearson vs Arm A fixed).

Reads results/sweep/<label>_s<seed>/ cells (from run_seeds.py), fuses GNN+ConvNet
per seed at the best shared weight (same convention as P17/P18b), and reports
seed-averaged accuracy per condition with the standard gate.

Gate: KEEP if seed-mean fused Δ vs base >= +1.0pp AND no fold seed-mean < -2pp.

PROTOCOL NOTE — both arms are DG (source-only, no target data at any point), so
this comparison is internally clean AND comparable to the P18b 'base' numbers.
Pure numpy.
"""
import os, re, json, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P21 = os.path.dirname(HERE)
SWEEP = os.path.join(P21, "results", "sweep")
TARGETS = ["KuHar", "MotionSense", "RealWorld-Thigh", "RealWorld-Waist", "UCI", "WISDM"]
WEIGHTS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
KEEP_MEAN_PP = 1.0
COLLAPSE_PP = -2.0


def metrics_acc(P, y):
    return float((P.argmax(1) == y).mean())


def load_dir(d):
    out = {}
    for t in TARGETS:
        g = os.path.join(d, f"gnn_{t}.npz")
        c = os.path.join(d, f"convnet_{t}.npz")
        if not (os.path.isfile(g) and os.path.isfile(c)):
            continue
        G, C = np.load(g), np.load(c)
        if not np.array_equal(G["labels"], C["labels"]):
            raise SystemExit(f"[report] alignment error in {d} for {t}")
        out[t] = (G["probs"], C["probs"], G["labels"])
    return out


def fuse_best_w(cond):
    """Per-fold fused acc at the single shared w maximizing this seed's mean."""
    sweep = {w: [] for w in WEIGHTS}
    perfold = {w: {} for w in WEIGHTS}
    gnn_only = {}
    for t, (Pg, Pc, y) in cond.items():
        gnn_only[t] = metrics_acc(Pg, y)
        for w in WEIGHTS:
            a = metrics_acc(w * Pg + (1 - w) * Pc, y)
            sweep[w].append(a); perfold[w][t] = a
    best_w = max(WEIGHTS, key=lambda w: np.mean(sweep[w]))
    return best_w, perfold[best_w], float(np.mean(sweep[best_w])), gnn_only


def discover():
    conds = {}
    for d in sorted(glob.glob(os.path.join(SWEEP, "*_s*"))):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(d))
        if not m:
            continue
        label, seed = m.group(1), int(m.group(2))
        cond = load_dir(d)
        if len(cond) == len(TARGETS):
            conds.setdefault(label, {})[seed] = cond
    return conds


def aggregate(seed_map):
    per_fold_vals = {t: [] for t in TARGETS}
    gnn_fold_vals = {t: [] for t in TARGETS}
    means, ws, gnn_means = [], [], []
    for seed, cond in sorted(seed_map.items()):
        bw, fold, m, gnn_only = fuse_best_w(cond)
        for t in TARGETS:
            per_fold_vals[t].append(fold[t])
            gnn_fold_vals[t].append(gnn_only[t])
        means.append(m); ws.append(bw)
        gnn_means.append(float(np.mean(list(gnn_only.values()))))
    return {
        "n_seeds": len(means), "seeds": sorted(seed_map),
        "mean": float(np.mean(means)), "std": float(np.std(means)),
        "per_seed_mean": means, "best_w_per_seed": ws,
        "gnn_only_mean": float(np.mean(gnn_means)),
        "gnn_only_per_seed": gnn_means,
        "fold_mean": {t: float(np.mean(per_fold_vals[t])) for t in TARGETS},
        "fold_std": {t: float(np.std(per_fold_vals[t])) for t in TARGETS},
        "gnn_fold_mean": {t: float(np.mean(gnn_fold_vals[t])) for t in TARGETS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_label", default="base")
    ap.add_argument("--out", default=os.path.join(P21, "results", "p21_seed_report.json"))
    args = ap.parse_args()

    conds = discover()
    if not conds:
        raise SystemExit(f"[report] no complete sweep cells in {SWEEP} (run run_seeds.py).")
    agg = {label: aggregate(sm) for label, sm in conds.items()}

    base = agg.get(args.base_label)
    bmean = base["mean"] if base else None

    def pct(x): return f"{x*100:5.2f}"
    print("\n" + "=" * 78)
    print("  P21 — SEED-AVERAGED: Pearson graph construction vs fixed topology")
    print("  (both arms DG protocol — source-only, frozen ConvNet branch & fusion)")
    print("=" * 78)
    print(f"\n  {'condition':22s} {'seeds':>5s} {'fused mean±std':>15s} "
          f"{'GNN-only':>9s} {'Δ vs base':>11s}  gate")
    print("  " + "-" * 72)
    order = ([args.base_label] if base else []) + \
            sorted([l for l in agg if l != args.base_label],
                   key=lambda l: -agg[l]["mean"])
    gate_rows = {}
    for label in order:
        a = agg[label]
        ms = f"{pct(a['mean'])}±{a['std']*100:4.2f}%"
        gs = f"{pct(a['gnn_only_mean'])}%"
        if base and label != args.base_label:
            dmean = (a["mean"] - bmean) * 100
            worst = min((a["fold_mean"][t] - base["fold_mean"][t]) * 100 for t in TARGETS)
            keep = (dmean >= KEEP_MEAN_PP) and (worst > COLLAPSE_PP)
            gate_rows[label] = {"delta_pp": dmean, "worst_fold_delta_pp": worst, "keep": keep}
            print(f"  {label:22s} {a['n_seeds']:5d} {ms:>15s} {gs:>9s} "
                  f"{dmean:+9.2f}pp  {'KEEP' if keep else 'revert'}")
        else:
            print(f"  {label:22s} {a['n_seeds']:5d} {ms:>15s} {gs:>9s} {'(baseline)':>11s}")

    if base and gate_rows:
        best = max(gate_rows, key=lambda l: agg[l]["mean"])
        print(f"\n  Per-fold seed-means — base vs {best} (fused | GNN-only):")
        print(f"  {'fold':16s} {'base':>8s} {best:>10s} {'Δpp':>7s}   "
              f"{'base-GNN':>9s} {'pears-GNN':>10s}")
        print("  " + "-" * 68)
        for t in TARGETS:
            b, d = base["fold_mean"][t], agg[best]["fold_mean"][t]
            bg, dg = base["gnn_fold_mean"][t], agg[best]["gnn_fold_mean"][t]
            print(f"  {t:16s} {pct(b):>7s}% {pct(d):>9s}% {(d-b)*100:+6.2f}   "
                  f"{pct(bg):>8s}% {pct(dg):>9s}%")
        g = gate_rows[best]
        print(f"\n  VERDICT: {best}  {g['delta_pp']:+.2f}pp (worst fold {g['worst_fold_delta_pp']:+.2f}pp)"
              f"  -> {'KEEP — carry into P22 (ResChebNet on Pearson edges)' if g['keep'] else 'REVERT — P22 runs ResChebNet on FIXED edges (Arm C)'}")
    print("\n  Gate: keep if seed-mean fused Δ >= +1.0pp AND no fold seed-mean < -2pp.")
    print("  Reference: P18b base = 76.87 ± 1.07% (seeds 42/1/2). Oracle headroom ~85%.")
    print("=" * 78)

    json.dump({"phase": "P21_pearson_graph", "base_label": args.base_label,
               "aggregate": agg, "gate": gate_rows}, open(args.out, "w"), indent=2)
    print(f"\n  saved {args.out}")


if __name__ == "__main__":
    main()
