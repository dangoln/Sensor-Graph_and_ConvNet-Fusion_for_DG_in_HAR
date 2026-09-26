"""
P24 — seed-averaged summary for the headline SO arms.

Compares, over seeds 42/1/2:
    full          : the locked P21 winner cells (reused — NOT retrained here)
    full_nc-no_gz : 5-channel system ("dropping gyro-z is free?")
    tiny          : quarter-width model ("97% of fused accuracy at ~1/3 size?")

Each arm's GNN probs (seed s) are fused with the SAME SEED's frozen ConvNet probs
from the P21 base cells, at the standard w=0.6. Reports seed mean ± std (fused and
GNN-only) and the per-fold means, next to the 78.10 ± 0.14% reference.

Arm dirs expected (from train_save.py):
    results/checkpoints/full_nc-no_gz[, _s1, _s2]
    results/checkpoints/tiny[, _s1, _s2]
Pure numpy.
"""
import os, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P24 = os.path.dirname(HERE)
TARGETS = ["KuHar", "MotionSense", "RealWorld-Thigh", "RealWorld-Waist", "UCI", "WISDM"]
SEEDS = [42, 1, 2]
FUSE_W = 0.6
DEFAULT_P21 = os.path.normpath(os.path.join(
    P24, "..", "..", "01_Contribution1_Generalization_Model", "P21_PearsonGraph", "results", "sweep"))

ARMS = ["full_nc-no_gz", "tiny"]


def arm_dir(arm, seed):
    d = os.path.join(P24, "results", "checkpoints",
                     arm + (f"_s{seed}" if seed != 42 else ""))
    return d if all(os.path.isfile(os.path.join(d, f"gnn_{t}.npz"))
                    for t in TARGETS) else None


def p21_base_dir(p21_sweep, seed):
    d = os.path.join(p21_sweep, f"base_s{seed}")
    return d if os.path.isdir(d) else None


def evaluate(gnn_dir, conv_dir):
    g_acc, f_acc = {}, {}
    for t in TARGETS:
        G = np.load(os.path.join(gnn_dir, f"gnn_{t}.npz"))
        C = np.load(os.path.join(conv_dir, f"convnet_{t}.npz"))
        Pg, y = G["probs"], G["labels"]
        Pc, yc = C["probs"], C["labels"]
        if not np.array_equal(y, yc):
            raise SystemExit(f"[seed_summary] alignment error: {gnn_dir} vs {conv_dir} ({t})")
        g_acc[t] = float((Pg.argmax(1) == y).mean())
        f_acc[t] = float(((FUSE_W*Pg + (1-FUSE_W)*Pc).argmax(1) == y).mean())
    return g_acc, f_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p21_sweep", default=DEFAULT_P21,
                    help="P21 results/sweep (base_s* convnet probs + full-arm reuse)")
    args = ap.parse_args()

    # 'full' = the P21 winner cells themselves (same recipe, already 3-seeded)
    arms = {"full (P21 winner)": {
        s: (os.path.join(args.p21_sweep, f"pearson_topk9_a0.5_s{s}"),
            os.path.join(args.p21_sweep, f"base_s{s}")) for s in SEEDS}}
    for arm in ARMS:
        cells = {}
        for s in SEEDS:
            g = arm_dir(arm, s)
            c = p21_base_dir(args.p21_sweep, s)
            if g and c:
                cells[s] = (g, c)
        arms[arm] = cells

    print("\n" + "=" * 76)
    print("  P24 — SEED-AVERAGED headline SO arms (fused w=0.6, frozen ConvNet/seed)")
    print("  Reference: full system 78.10 ± 0.14% (P21, seeds 42/1/2)")
    print("=" * 76)
    print(f"\n  {'arm':20s} {'seeds':>5s} {'fused mean±std':>15s} {'GNN-only':>9s} "
          f"{'Δ vs full':>10s}")
    print("  " + "-" * 64)
    agg = {}
    for arm, cells in arms.items():
        if not cells:
            print(f"  {arm:20s}  (no complete cells — run train_save with --seed 1/2)")
            continue
        fused_means, gnn_means = [], []
        fold_vals = {t: [] for t in TARGETS}
        for s, (g, c) in sorted(cells.items()):
            ga, fa = evaluate(g, c)
            gnn_means.append(np.mean(list(ga.values())))
            fused_means.append(np.mean(list(fa.values())))
            for t in TARGETS:
                fold_vals[t].append(fa[t])
        agg[arm] = {"n_seeds": len(cells), "seeds": sorted(cells),
                    "fused_mean": float(np.mean(fused_means)),
                    "fused_std": float(np.std(fused_means)),
                    "gnn_mean": float(np.mean(gnn_means)),
                    "fold_mean": {t: float(np.mean(v)) for t, v in fold_vals.items()}}
    ref = agg.get("full (P21 winner)", {}).get("fused_mean")
    for arm, a in agg.items():
        d = (f"{(a['fused_mean']-ref)*100:+.2f}pp" if ref and arm != "full (P21 winner)"
             else "(reference)")
        print(f"  {arm:20s} {a['n_seeds']:5d} "
              f"{a['fused_mean']*100:8.2f}±{a['fused_std']*100:4.2f}% "
              f"{a['gnn_mean']*100:8.2f}% {d:>10s}")
    if len(agg) > 1:
        print(f"\n  Per-fold fused seed-means:")
        print("  " + f"{'fold':16s}" + "".join(f"{a[:14]:>16s}" for a in agg))
        for t in TARGETS:
            print("  " + f"{t:16s}" +
                  "".join(f"{agg[a]['fold_mean'][t]*100:15.2f}%" for a in agg))
    print("\n  Single-seed arms (compact, accel_only, aw40/30, ...) remain in the")
    print("  step-9 table; only the headline claims carry seed error bars.")
    print("=" * 76)
    out = os.path.join(P24, "results", "p24_seed_summary.json")
    json.dump(agg, open(out, "w"), indent=2)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
