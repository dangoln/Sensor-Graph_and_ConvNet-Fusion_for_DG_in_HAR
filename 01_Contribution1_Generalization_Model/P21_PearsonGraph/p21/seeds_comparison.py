"""
P21 — seeds_comparison.py

Builds paper-ready tables and figures from the ALREADY-SAVED results files:
    results/p21_seed_report.json   (from run_report.py)
    results/p21_full_metrics.json  (from the metrics cell — precision/recall/F1)
    results/sweep_cfg/<cfg>/val_summary.json  (Stage A source-val selection)

Does NOT retrain or touch any .npz — pure post-hoc aggregation/plotting from
JSON already on disk. Safe to re-run any time after run_report.py.

Run from anywhere (path-independent, same convention as run_report.py):
    python -m p21.seeds_comparison
    python -m p21.seeds_comparison --winner pearson_topk9_a0.5 --base_label base

Outputs (written to results/):
    p21_paper_tables.md         7 markdown tables
    p21_fig1_perfold_bars.png
    p21_fig2_perseed_robustness.png
    p21_fig3_stageA_selection.png
"""
import os, json, glob, re, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
P21 = os.path.dirname(HERE)
RESULTS = os.path.join(P21, "results")
TARGETS = ["KuHar", "MotionSense", "RealWorld-Thigh", "RealWorld-Waist", "UCI", "WISDM"]
SHORT = {"KuHar": "KuHar", "MotionSense": "MotnSns", "RealWorld-Thigh": "RW-Thigh",
         "RealWorld-Waist": "RW-Waist", "UCI": "UCI", "WISDM": "WISDM"}


def load_json(path, hint):
    if not os.path.isfile(path):
        raise SystemExit(f"[seeds_comparison] missing {path}\n  -> run {hint} first.")
    return json.load(open(path))


def discover_stage_a():
    """Mean source-val accuracy per Stage A config, from results/sweep_cfg/*/val_summary.json."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RESULTS, "sweep_cfg", "*"))):
        f = os.path.join(d, "val_summary.json")
        if os.path.isfile(f):
            vals = json.load(open(f))
            row = {t: vals[t] * 100 for t in TARGETS if t in vals}
            if len(row) == len(TARGETS):
                out[os.path.basename(d)] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner", default="pearson_topk9_a0.5",
                     help="condition label in p21_seed_report.json to treat as Arm B")
    ap.add_argument("--base_label", default="base")
    ap.add_argument("--seed_report", default=os.path.join(RESULTS, "p21_seed_report.json"))
    ap.add_argument("--full_metrics", default=os.path.join(RESULTS, "p21_full_metrics.json"))
    ap.add_argument("--out_dir", default=RESULTS)
    args = ap.parse_args()

    seed_report = load_json(args.seed_report, "run_report.py")
    full_metrics = load_json(args.full_metrics, "the metrics cell (precision/recall/F1)")

    agg = seed_report["aggregate"]
    if args.base_label not in agg:
        raise SystemExit(f"[seeds_comparison] '{args.base_label}' not in {args.seed_report}")
    if args.winner not in agg:
        raise SystemExit(f"[seeds_comparison] '{args.winner}' not in {args.seed_report}. "
                          f"Available: {list(agg)}")
    base = agg[args.base_label]
    pearson = agg[args.winner]
    md = []

    # ============ TABLE 1: Main ablation — base vs winner, per fold ============
    md.append(f"## Table 1 — Main Result: {args.base_label} vs {args.winner} (seed-averaged)\n")
    md.append(f"| Fold | {args.base_label} % | {args.winner} % | Δ (pp) |")
    md.append("|---|---|---|---|")
    for t in TARGETS:
        b_m, b_s = base["fold_mean"][t] * 100, base["fold_std"][t] * 100
        p_m, p_s = pearson["fold_mean"][t] * 100, pearson["fold_std"][t] * 100
        md.append(f"| {t} | {b_m:.2f} ± {b_s:.2f} | {p_m:.2f} ± {p_s:.2f} | {p_m - b_m:+.2f} |")
    b_mean, b_std = base["mean"] * 100, base["std"] * 100
    p_mean, p_std = pearson["mean"] * 100, pearson["std"] * 100
    md.append(f"| **Mean** | **{b_mean:.2f} ± {b_std:.2f}** | **{p_mean:.2f} ± {p_std:.2f}** | "
              f"**{p_mean - b_mean:+.2f}** |")
    md.append("")

    # ============ TABLE 2: Per-seed breakdown ============
    md.append("## Table 2 — Per-Seed Breakdown (robustness against seed luck)\n")
    md.append(f"| Seed | {args.base_label} fused % | {args.winner} fused % | Δ (pp) |")
    md.append("|---|---|---|---|")
    b_map = dict(zip(base["seeds"], base["per_seed_mean"]))
    p_map = dict(zip(pearson["seeds"], pearson["per_seed_mean"]))
    common_seeds = sorted(set(b_map) & set(p_map))
    if set(b_map) != set(p_map):
        print(f"[seeds_comparison] WARNING: seed sets differ — base={sorted(b_map)} "
              f"winner={sorted(p_map)}; only common seeds shown.")
    for s in common_seeds:
        bb, pp = b_map[s] * 100, p_map[s] * 100
        md.append(f"| {s} | {bb:.2f} | {pp:.2f} | {pp - bb:+.2f} |")
    md.append(f"| **mean±std** | **{b_mean:.2f}±{b_std:.2f}** | **{p_mean:.2f}±{p_std:.2f}** | "
              f"**{p_mean - b_mean:+.2f}** |")
    md.append("")

    # ============ TABLE 3: Precision / Recall / F1 (macro), per fold ============
    md.append(f"## Table 3 — Precision / Recall / F1 (macro, present-classes-only), "
              f"{full_metrics.get('seeds', [])} seeds\n")
    md.append("| Fold | Accuracy % | Precision % | Recall % | F1 % |")
    md.append("|---|---|---|---|---|")
    s = full_metrics["seed_avg_summary"]
    for t in TARGETS + ["OVERALL"]:
        if t not in s:
            continue
        row = s[t]
        label = "**OVERALL (pooled)**" if t == "OVERALL" else t
        md.append(f"| {label} | {row['acc_mean']:.2f} ± {row['acc_std']:.2f} | "
                  f"{row['prec_mean']:.2f} | {row['rec_mean']:.2f} | "
                  f"{row['f1_mean']:.2f} ± {row['f1_std']:.2f} |")
    md.append("")

    # ============ TABLE 4: GNN-only vs Fused ============
    md.append(f"## Table 4 — GNN-Only vs Fused (isolates the graph-construction effect)\n")
    md.append("| Arm | GNN-only % | Fused % | Fusion gain (pp) |")
    md.append("|---|---|---|---|")
    bg = base["gnn_only_mean"] * 100
    bgs = float(np.std(base["gnn_only_per_seed"])) * 100
    pg = pearson["gnn_only_mean"] * 100
    pgs = float(np.std(pearson["gnn_only_per_seed"])) * 100
    md.append(f"| {args.base_label} | {bg:.2f} ± {bgs:.2f} | {b_mean:.2f} ± {b_std:.2f} | {b_mean - bg:+.2f} |")
    md.append(f"| {args.winner} | {pg:.2f} ± {pgs:.2f} | {p_mean:.2f} ± {p_std:.2f} | {p_mean - pg:+.2f} |")
    md.append(f"| **GNN-only gain ({args.winner} − {args.base_label})** | **{pg - bg:+.2f} pp** | | |")
    md.append("")

    # ============ TABLE 5: Stage A config comparison (auto-discovered) ============
    stage_a = discover_stage_a()
    if stage_a:
        winner_cfg = args.winner.replace("pearson_", "")
        md.append("## Table 5 — Stage A Config Selection (source-val accuracy, seed 42 — "
                  "selection signal, never target-test)\n")
        cols = " | ".join(SHORT[t] for t in TARGETS)
        md.append(f"| Config | {cols} | Mean |")
        md.append("|---|" + "---|" * (len(TARGETS) + 1))
        for cfg, vals in stage_a.items():
            row = [vals[t] for t in TARGETS]
            mean = float(np.mean(row))
            marker = " **← selected**" if cfg == winner_cfg else ""
            md.append(f"| {cfg}{marker} | " + " | ".join(f"{v:.2f}" for v in row) + f" | {mean:.2f} |")
        md.append("")
    else:
        print("[seeds_comparison] no results/sweep_cfg/*/val_summary.json found — skipping Table 5.")

    # ============ TABLE 6: Gate summary ============
    gate = seed_report.get("gate", {}).get(args.winner)
    if gate:
        md.append("## Table 6 — Keep/Revert Gate\n")
        md.append("| Criterion | Threshold | Observed | Pass? |")
        md.append("|---|---|---|---|")
        md.append(f"| Mean Δ vs base | ≥ +1.0 pp | {gate['delta_pp']:+.2f} pp | "
                  f"{'✓' if gate['delta_pp'] >= 1.0 else '✗'} |")
        md.append(f"| Worst-fold Δ | > −2.0 pp | {gate['worst_fold_delta_pp']:+.2f} pp | "
                  f"{'✓' if gate['worst_fold_delta_pp'] > -2.0 else '✗'} |")
        md.append(f"| **Verdict** | | | **{'KEEP' if gate['keep'] else 'REVERT'}** |")
        md.append("")

    # ============ TABLE 7: Overall Accuracy/Precision/Recall/F1, per seed ============
    md.append("## Table 7 — Overall (pooled-fold) Accuracy / Precision / Recall / F1, per seed\n")
    md.append("| Seed | Accuracy % | Precision % | Recall % | F1 % |")
    md.append("|---|---|---|---|---|")
    per_seed = full_metrics.get("per_seed", {})
    accs, precs, recs, f1s = [], [], [], []
    for seed_key in sorted(per_seed, key=lambda x: int(x)):
        ov = per_seed[seed_key].get("OVERALL")
        if not ov:
            continue
        accs.append(ov["accuracy"]); precs.append(ov["precision"])
        recs.append(ov["recall"]); f1s.append(ov["f1"])
        md.append(f"| {seed_key} | {ov['accuracy']:.2f} | {ov['precision']:.2f} | "
                  f"{ov['recall']:.2f} | {ov['f1']:.2f} |")
    if accs:
        md.append(f"| **mean ± std** | **{np.mean(accs):.2f} ± {np.std(accs):.2f}** | "
                  f"**{np.mean(precs):.2f} ± {np.std(precs):.2f}** | "
                  f"**{np.mean(recs):.2f} ± {np.std(recs):.2f}** | "
                  f"**{np.mean(f1s):.2f} ± {np.std(f1s):.2f}** |")
    md.append("")

    md_path = os.path.join(args.out_dir, "p21_paper_tables.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\n[seeds_comparison] saved {md_path}")

    # ================= FIGURES =================
    # Fig 1 — per-fold bars with error bars
    b_m = [base["fold_mean"][t] * 100 for t in TARGETS]
    b_s = [base["fold_std"][t] * 100 for t in TARGETS]
    p_m = [pearson["fold_mean"][t] * 100 for t in TARGETS]
    p_s = [pearson["fold_std"][t] * 100 for t in TARGETS]
    x = np.arange(len(TARGETS)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, b_m, w, yerr=b_s, capsize=4, label=args.base_label, color="#888888")
    ax.bar(x + w / 2, p_m, w, yerr=p_s, capsize=4, label=args.winner, color="#2b6cb0")
    ax.axhline(b_mean, color="#888888", ls="--", lw=1, alpha=0.6)
    ax.axhline(p_mean, color="#2b6cb0", ls="--", lw=1, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in TARGETS])
    ax.set_ylabel("LODO accuracy (%)")
    ax.set_title(f"P21: Per-fold accuracy, {args.base_label} vs {args.winner} (seed mean ± std)")
    ax.legend()
    ylo = max(0, min(min(b_m), min(p_m)) - 15)
    ax.set_ylim(ylo, 95)
    for i, (bm, pm) in enumerate(zip(b_m, p_m)):
        ax.annotate(f"{pm - bm:+.1f}", (x[i], max(bm, pm) + max(b_s[i], p_s[i]) + 1.5),
                    ha="center", fontsize=8, color="#1a4971")
    plt.tight_layout()
    fig1_path = os.path.join(args.out_dir, "p21_fig1_perfold_bars.png")
    plt.savefig(fig1_path, dpi=200); plt.close()

    # Fig 2 — per-seed dot plot
    bx = [b_map[s] * 100 for s in common_seeds]
    px = [p_map[s] * 100 for s in common_seeds]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([args.base_label] * len(bx), bx, color="#888888", s=70, zorder=3)
    ax.scatter([args.winner] * len(px), px, color="#2b6cb0", s=70, zorder=3)
    for s, b, p in zip(common_seeds, bx, px):
        ax.plot([args.base_label, args.winner], [b, p], color="#cccccc", lw=1, zorder=1)
        ax.annotate(f"s{s}", (0.02, b), fontsize=8, color="#555555")
    ax.scatter([args.base_label], [np.mean(bx)], color="black", marker="_", s=400, zorder=4)
    ax.scatter([args.winner], [np.mean(px)], color="black", marker="_", s=400, zorder=4)
    ax.set_ylabel("Fused LODO accuracy (%)")
    ax.set_title(f"P21: Per-seed fused accuracy — {args.base_label} vs {args.winner}\n"
                f"(seeds: {', '.join(str(s) for s in common_seeds)})")
    plt.tight_layout()
    fig2_path = os.path.join(args.out_dir, "p21_fig2_perseed_robustness.png")
    plt.savefig(fig2_path, dpi=200); plt.close()

    # Fig 3 — Stage A config comparison
    fig3_path = None
    if stage_a:
        names = list(stage_a.keys())
        vals = [float(np.mean(list(stage_a[n].values()))) for n in names]
        winner_cfg = args.winner.replace("pearson_", "")
        colors = ["#2b6cb0" if n == winner_cfg else "#aaaaaa" for n in names]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, vals, color=colors)
        ax.set_ylabel("Mean source-val accuracy (%)")
        ax.set_title("Stage A: config selection (source-val only)")
        lo = min(vals) - 3
        ax.set_ylim(max(0, lo), max(vals) + 3)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (i, v + 0.3), ha="center", fontsize=9)
        plt.tight_layout()
        fig3_path = os.path.join(args.out_dir, "p21_fig3_stageA_selection.png")
        plt.savefig(fig3_path, dpi=200); plt.close()

    print(f"[seeds_comparison] saved {fig1_path}")
    print(f"[seeds_comparison] saved {fig2_path}")
    if fig3_path:
        print(f"[seeds_comparison] saved {fig3_path}")


if __name__ == "__main__":
    main()
