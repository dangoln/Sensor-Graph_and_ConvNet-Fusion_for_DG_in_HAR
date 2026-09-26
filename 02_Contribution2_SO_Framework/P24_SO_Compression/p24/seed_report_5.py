"""
P24 — CANONICAL 5-SEED report for the thin SO arms (hardening pass).

This is the multi-seed successor to `seed_summary.py` (which is frozen at the
old 3-seed {42,1,2} protocol and only covered no_gz/tiny). This script is
ADDITIVE — it does not modify or replace seed_summary.py; it reads the same
checkpoint layout and aggregates across the canonical 5 seeds {1,2,3,5,42}.

For every arm it fuses each seed's saved GNN probs with the SAME SEED's frozen
ConvNet probs from P21 (`base_s{seed}`), at the locked w=0.6, and reports:
    - fused mean ± std and GNN-only mean ± std across the available seeds
    - per-fold fused seed-means (6 LODO targets)
    - GenPerByte (fused-acc per KB of GNN state-dict) with a seed-derived CI
    - paired Wilcoxon signed-rank tests across the 6 folds for the key contrasts
      (default: no_gz vs full).  n=6 folds → report effect size alongside p.

Arms (checkpoint dir base name under results/checkpoints/):
    full         -> P21 winner cells (pearson_topk9_a0.5_s{seed} GNN + base_s{seed})
    accel_only   -> full_nc-accel_only
    gyro_only    -> full_nc-gyro_only
    aw40         -> full_aw40
    aw30         -> full_aw30
    no_gz        -> full_nc-no_gz          (headline arm)
    tiny         -> tiny                    (headline arm)
seed 42 dirs carry no suffix; seeds 1/2/3/5 use the `_s{seed}` suffix
(exactly as train_save.py writes them).

Output: results/p24_seed_report_5.json + printed tables. Pure numpy (+ scipy for
Wilcoxon; degrades gracefully to a sign test if scipy is absent).
"""
import os, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P24 = os.path.dirname(HERE)
TARGETS = ["KuHar", "MotionSense", "RealWorld-Thigh", "RealWorld-Waist", "UCI", "WISDM"]
SEEDS = [1, 2, 3, 5, 42]            # canonical set
FUSE_W = 0.6
DEFAULT_P21 = os.path.normpath(os.path.join(
    P24, "..", "..", "01_Contribution1_Generalization_Model", "P21_PearsonGraph", "results", "sweep"))

# arm label -> checkpoint dir base name (None => special P21-winner handling)
ARMS = {
    "full (P21 winner)": None,
    "accel_only":        "full_nc-accel_only",
    "gyro_only":         "full_nc-gyro_only",
    "aw40":              "full_aw40",
    "aw30":              "full_aw30",
    "no_gz":             "full_nc-no_gz",
    "tiny":              "tiny",
}
# contrasts run as paired Wilcoxon across the 6 folds (arm_a vs arm_b, on
# fused seed-mean per fold)
WILCOXON_CONTRASTS = [("no_gz", "full (P21 winner)")]


def _suffix(seed):
    return "" if seed == 42 else f"_s{seed}"


def gnn_dir_for(arm_base, seed, p21_sweep):
    """Return the dir holding gnn_{t}.npz for this arm+seed, or None if incomplete."""
    if arm_base is None:                       # full = P21 winner pearson cell
        d = os.path.join(p21_sweep, f"pearson_topk9_a0.5_s{seed}")
    else:
        d = os.path.join(P24, "results", "checkpoints", arm_base + _suffix(seed))
    ok = all(os.path.isfile(os.path.join(d, f"gnn_{t}.npz")) for t in TARGETS)
    return d if ok else None


def convnet_dir_for(seed, p21_sweep):
    d = os.path.join(p21_sweep, f"base_s{seed}")
    ok = all(os.path.isfile(os.path.join(d, f"convnet_{t}.npz")) for t in TARGETS)
    return d if ok else None


def gnn_state_dict_kb(arm_base, seed):
    """GNN model size in KB from a saved checkpoint (deterministic across seeds).
    Returns None when checkpoints aren't present (e.g. the P21-winner cells store
    only probs, not a .pt).  Returns None if torch is unavailable (GenPerByte is
    then simply omitted; the accuracy tables are unaffected)."""
    import io
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if arm_base is None:
        arm_base, seed = "full", 42            # proxy: full-capacity P21 GNN
    d = os.path.join(P24, "results", "checkpoints", arm_base + _suffix(seed))
    ck_path = os.path.join(d, f"ckpt_{TARGETS[0]}.pt")
    if not os.path.isfile(ck_path):
        # fall back to the seed-42 dir for the same arm
        d0 = os.path.join(P24, "results", "checkpoints", arm_base)
        ck_path = os.path.join(d0, f"ckpt_{TARGETS[0]}.pt")
        if not os.path.isfile(ck_path):
            return None
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    buf = io.BytesIO(); torch.save(ck["state_dict"], buf)
    return buf.getbuffer().nbytes / 1024.0


def eval_seed(gnn_dir, conv_dir):
    """Per-fold GNN-only and fused accuracy for one seed."""
    g_acc, f_acc = {}, {}
    for t in TARGETS:
        G = np.load(os.path.join(gnn_dir, f"gnn_{t}.npz"))
        C = np.load(os.path.join(conv_dir, f"convnet_{t}.npz"))
        Pg, y = G["probs"], G["labels"]
        Pc, yc = C["probs"], C["labels"]
        if not np.array_equal(y, yc):
            raise SystemExit(f"[seed_report_5] label mismatch {gnn_dir} vs {conv_dir} ({t})")
        g_acc[t] = float((Pg.argmax(1) == y).mean())
        f_acc[t] = float(((FUSE_W * Pg + (1 - FUSE_W) * Pc).argmax(1) == y).mean())
    return g_acc, f_acc


def paired_test(a, b):
    """Paired Wilcoxon signed-rank over folds (a,b are per-fold arrays). Falls
    back to an exact sign test if scipy missing or n too small."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    eff = float(diff.mean())
    try:
        from scipy.stats import wilcoxon
        # zero_method='wilcox' drops zero-diffs; guard the all-equal case
        if np.allclose(diff, 0):
            return {"test": "wilcoxon", "stat": 0.0, "p": 1.0, "mean_diff_pp": eff * 100}
        stat, p = wilcoxon(a, b)
        return {"test": "wilcoxon", "stat": float(stat), "p": float(p),
                "mean_diff_pp": eff * 100}
    except Exception:
        from math import comb
        nz = diff[diff != 0]
        n = len(nz); k = int((nz > 0).sum())
        p = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n) if n else 1.0
        return {"test": "sign", "n_nonzero": n, "n_pos": k, "p": float(min(1.0, 2 * p)),
                "mean_diff_pp": eff * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p21_sweep", default=DEFAULT_P21,
                    help="P21 results/sweep (base_s* ConvNet + pearson_topk9_a0.5_s* GNN)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                    help="comma list; default canonical 1,2,3,5,42")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    agg = {}
    fold_seedmean = {}    # arm -> {target: fused seed-mean}  (for Wilcoxon)
    for label, base in ARMS.items():
        fused_means, gnn_means = [], []
        used = []
        fold_vals = {t: [] for t in TARGETS}
        for s in seeds:
            g = gnn_dir_for(base, s, args.p21_sweep)
            c = convnet_dir_for(s, args.p21_sweep)
            if not (g and c):
                continue
            ga, fa = eval_seed(g, c)
            gnn_means.append(np.mean(list(ga.values())))
            fused_means.append(np.mean(list(fa.values())))
            for t in TARGETS:
                fold_vals[t].append(fa[t])
            used.append(s)
        if not used:
            agg[label] = {"n_seeds": 0}
            continue
        kb = gnn_state_dict_kb(base, used[0])
        fmean = float(np.mean(fused_means))
        fstd = float(np.std(fused_means))
        gpb = (fmean / kb) if kb else None                    # GenPerByte (acc/KB)
        gpb_ci = (fstd / kb) if kb else None
        agg[label] = {
            "n_seeds": len(used), "seeds": used,
            "fused_mean": fmean, "fused_std": fstd,
            "gnn_mean": float(np.mean(gnn_means)), "gnn_std": float(np.std(gnn_means)),
            "gnn_kb": kb,
            "genperbyte_acc_per_kb": gpb, "genperbyte_ci": gpb_ci,
            "fold_seedmean": {t: float(np.mean(v)) for t, v in fold_vals.items()},
        }
        fold_seedmean[label] = agg[label]["fold_seedmean"]

    # ── accuracy table ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  P24 — CANONICAL 5-SEED report (fused w=0.6, frozen ConvNet per seed)")
    print(f"  seeds requested: {seeds}   |   canonical headline uses {{1,2,3,5,42}}")
    print("  Reference: P21 winner fused 77.96 ± 0.21% (5-seed)")
    print("=" * 90)
    ref = agg.get("full (P21 winner)", {}).get("fused_mean")
    print(f"\n  {'arm':20s} {'n':>2s} {'fused mean±std':>16s} {'GNN-only':>10s} "
          f"{'Δ vs full':>11s} {'GenPerByte':>12s}")
    print("  " + "-" * 76)
    for label, a in agg.items():
        if a.get("n_seeds", 0) == 0:
            print(f"  {label:20s}   (no complete cells yet — train the missing seeds)")
            continue
        d = ("(reference)" if label == "full (P21 winner)" or not ref
             else f"{(a['fused_mean']-ref)*100:+.2f}pp")
        gpb = (f"{a['genperbyte_acc_per_kb']:.4f}" if a["genperbyte_acc_per_kb"] else "  n/a")
        print(f"  {label:20s} {a['n_seeds']:>2d} "
              f"{a['fused_mean']*100:7.2f}±{a['fused_std']*100:4.2f}% "
              f"{a['gnn_mean']*100:8.2f}% {d:>11s} {gpb:>12s}")

    # ── per-fold table ────────────────────────────────────────────────────────
    complete = [l for l, a in agg.items() if a.get("n_seeds", 0) > 0]
    if len(complete) > 1:
        print("\n  Per-fold fused seed-means:")
        print("  " + f"{'fold':16s}" + "".join(f"{l[:13]:>15s}" for l in complete))
        for t in TARGETS:
            print("  " + f"{t:16s}" +
                  "".join(f"{agg[l]['fold_seedmean'][t]*100:14.2f}%" for l in complete))

    # ── paired Wilcoxon contrasts ────────────────────────────────────────────
    tests = {}
    print("\n  Paired Wilcoxon signed-rank across the 6 LODO folds (fused seed-means):")
    for a_lbl, b_lbl in WILCOXON_CONTRASTS:
        if a_lbl in fold_seedmean and b_lbl in fold_seedmean:
            av = [fold_seedmean[a_lbl][t] for t in TARGETS]
            bv = [fold_seedmean[b_lbl][t] for t in TARGETS]
            res = paired_test(av, bv)
            tests[f"{a_lbl} vs {b_lbl}"] = res
            print(f"    {a_lbl} vs {b_lbl}: mean Δ {res['mean_diff_pp']:+.2f}pp "
                  f"| {res['test']} p={res['p']:.3f}  (n=6 folds — read effect size first)")
        else:
            print(f"    {a_lbl} vs {b_lbl}: skipped (an arm has no complete seeds yet)")

    out = os.path.join(P24, "results", "p24_seed_report_5.json")
    json.dump({"arms": agg, "wilcoxon": tests, "seeds": seeds}, open(out, "w"), indent=2)
    print(f"\n  saved {out}")
    print("=" * 90)


if __name__ == "__main__":
    main()
