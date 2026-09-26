"""
P21 — seeds_comparison.py

Builds paper-ready tables and figures from the ALREADY-SAVED results files:
    results/p21_seed_report.json   (from run_report.py)
    results/p21_full_metrics.json  (from the metrics cell — precision/recall/F1)
    results/sweep_cfg/<cfg>/val_summary.json  (Stage A source-val selection)

Does NOT retrain or touch any .npz — pure post-hoc aggregation/plotting from
JSON already on disk. Safe to re-run any time after run_report.py.

IMPORTANT — how to invoke for rich tables in Colab:
    Rich HTML-styled tables only render if this runs INSIDE the notebook
    kernel. `!python -m p21.seeds_comparison` runs as a subprocess and will
    only ever print plain text, even with this version of the script.
    Use an in-kernel call instead:

        import sys
        sys.path.insert(0, PROJECT_PATH)   # PROJECT_PATH already set + chdir'd
        from p21 import seeds_comparison
        seeds_comparison.main([])

    Command-line use still works the same (e.g. for local/non-notebook runs):
        python -m p21.seeds_comparison
    — it just falls back to plain printed tables when no notebook display
    backend is available.

Outputs (written to results/):
    p21_paper_tables.md         7 markdown tables (unchanged from before)
    p21_fig1_perfold_bars.png
    p21_fig2_perseed_robustness.png
    p21_fig3_stageA_selection.png
"""
import os, json, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from IPython.display import display, Markdown
    _IPY = True
except ImportError:
    _IPY = False

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


# ----------------------------------------------------------------------------
# Rich-table rendering helpers
# ----------------------------------------------------------------------------
def _color_signed(v):
    """Green for positive deltas, red for negative — used on Δ/gain columns."""
    try:
        s = str(v).replace("pp", "").replace("%", "").strip()
        num = float(s)
    except (ValueError, TypeError):
        return ""
    if num > 0:
        return "color: #1a7f37; font-weight: 600"
    if num < 0:
        return "color: #c4262e; font-weight: 600"
    return ""


def show_table(df, title, delta_cols=None, bold_last_row=True, highlight_rows=None):
    """
    Render a styled HTML table in a notebook (display), or a clean plain-text
    table otherwise. `highlight_rows` is an optional boolean list (len == len(df))
    marking rows to tint (e.g. the selected Stage A config).
    """
    delta_cols = delta_cols or []
    if _IPY:
        try:
            display(Markdown(f"#### {title}"))
            sty = (df.style
                     .hide(axis="index")
                     .set_table_styles([
                         {"selector": "th", "props": [("background-color", "#f0f2f5"),
                                                       ("text-align", "center"),
                                                       ("padding", "6px 10px"),
                                                       ("border", "1px solid #ddd")]},
                         {"selector": "td", "props": [("text-align", "center"),
                                                       ("padding", "5px 10px"),
                                                       ("border", "1px solid #eee")]},
                     ]))
            for c in delta_cols:
                if c in df.columns:
                    sty = sty.map(_color_signed, subset=[c])
            if bold_last_row and len(df) > 0:
                last_idx = df.index[-1]
                sty = sty.apply(
                    lambda row: ["font-weight: 700; background-color: #f7f8fa"
                                 if row.name == last_idx else "" for _ in row],
                    axis=1)
            if highlight_rows:
                def _hl(row):
                    return ["background-color: #dbe9f9" if highlight_rows[row.name] else ""
                            for _ in row]
                sty = sty.apply(_hl, axis=1)
            display(sty)
            return
        except Exception as e:
            print(f"[seeds_comparison] rich display failed ({e}); falling back to plain text.")
    # plain-text fallback (no notebook display backend, or display failed)
    print(f"\n{title}")
    print(df.to_string(index=False))


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner", default="pearson_topk9_a0.5",
                     help="condition label in p21_seed_report.json to treat as Arm B")
    ap.add_argument("--base_label", default="base")
    ap.add_argument("--seed_report", default=os.path.join(RESULTS, "p21_seed_report.json"))
    ap.add_argument("--full_metrics", default=os.path.join(RESULTS, "p21_full_metrics.json"))
    ap.add_argument("--out_dir", default=RESULTS)
    args = ap.parse_args(argv)

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
    md = []  # markdown file content — unchanged behavior from before

    b_mean, b_std = base["mean"] * 100, base["std"] * 100
    p_mean, p_std = pearson["mean"] * 100, pearson["std"] * 100

    # ============ TABLE 1 ============
    rows = []
    for t in TARGETS:
        bm, bs = base["fold_mean"][t] * 100, base["fold_std"][t] * 100
        pm, ps = pearson["fold_mean"][t] * 100, pearson["fold_std"][t] * 100
        rows.append([t, f"{bm:.2f} ± {bs:.2f}", f"{pm:.2f} ± {ps:.2f}", f"{pm - bm:+.2f}"])
    rows.append(["Mean", f"{b_mean:.2f} ± {b_std:.2f}", f"{p_mean:.2f} ± {p_std:.2f}",
                 f"{p_mean - b_mean:+.2f}"])
    df1 = pd.DataFrame(rows, columns=["Fold", f"{args.base_label} %", f"{args.winner} %", "Δ (pp)"])
    show_table(df1, f"Table 1 — Main Result: {args.base_label} vs {args.winner} (seed-averaged)",
               delta_cols=["Δ (pp)"])
    md.append(f"## Table 1 — Main Result: {args.base_label} vs {args.winner} (seed-averaged)\n")
    md.append(f"| Fold | {args.base_label} % | {args.winner} % | Δ (pp) |")
    md.append("|---|---|---|---|")
    for r in rows[:-1]:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    md.append(f"| **Mean** | **{rows[-1][1]}** | **{rows[-1][2]}** | **{rows[-1][3]}** |")
    md.append("")

    # ============ TABLE 2 ============
    b_map = dict(zip(base["seeds"], base["per_seed_mean"]))
    p_map = dict(zip(pearson["seeds"], pearson["per_seed_mean"]))
    common_seeds = sorted(set(b_map) & set(p_map))
    if set(b_map) != set(p_map):
        print(f"[seeds_comparison] WARNING: seed sets differ — base={sorted(b_map)} "
              f"winner={sorted(p_map)}; only common seeds shown.")
    rows2 = []
    for s in common_seeds:
        bb, pp = b_map[s] * 100, p_map[s] * 100
        rows2.append([str(s), f"{bb:.2f}", f"{pp:.2f}", f"{pp - bb:+.2f}"])
    rows2.append(["mean ± std", f"{b_mean:.2f}±{b_std:.2f}", f"{p_mean:.2f}±{p_std:.2f}",
                  f"{p_mean - b_mean:+.2f}"])
    df2 = pd.DataFrame(rows2, columns=["Seed", f"{args.base_label} fused %",
                                        f"{args.winner} fused %", "Δ (pp)"])
    show_table(df2, "Table 2 — Per-Seed Breakdown (robustness against seed luck)",
               delta_cols=["Δ (pp)"])
    md.append("## Table 2 — Per-Seed Breakdown (robustness against seed luck)\n")
    md.append(f"| Seed | {args.base_label} fused % | {args.winner} fused % | Δ (pp) |")
    md.append("|---|---|---|---|")
    for r in rows2[:-1]:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    md.append(f"| **mean±std** | **{rows2[-1][1]}** | **{rows2[-1][2]}** | **{rows2[-1][3]}** |")
    md.append("")

    # ============ TABLE 3 ============
    s = full_metrics["seed_avg_summary"]
    rows3 = []
    for t in TARGETS + ["OVERALL"]:
        if t not in s:
            continue
        row = s[t]
        label = "OVERALL (pooled)" if t == "OVERALL" else t
        rows3.append([label, f"{row['acc_mean']:.2f} ± {row['acc_std']:.2f}",
                      f"{row['prec_mean']:.2f}", f"{row['rec_mean']:.2f}",
                      f"{row['f1_mean']:.2f} ± {row['f1_std']:.2f}"])
    df3 = pd.DataFrame(rows3, columns=["Fold", "Accuracy %", "Precision %", "Recall %", "F1 %"])
    show_table(df3, f"Table 3 — Precision / Recall / F1 (macro, present-classes-only), "
                     f"{full_metrics.get('seeds', [])} seeds")
    md.append(f"## Table 3 — Precision / Recall / F1 (macro, present-classes-only), "
              f"{full_metrics.get('seeds', [])} seeds\n")
    md.append("| Fold | Accuracy % | Precision % | Recall % | F1 % |")
    md.append("|---|---|---|---|---|")
    for r in rows3[:-1]:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    last = rows3[-1]
    md.append(f"| **{last[0]}** | {last[1]} | {last[2]} | {last[3]} | {last[4]} |")
    md.append("")

    # ============ TABLE 4 ============
    bg = base["gnn_only_mean"] * 100
    bgs = float(np.std(base["gnn_only_per_seed"])) * 100
    pg = pearson["gnn_only_mean"] * 100
    pgs = float(np.std(pearson["gnn_only_per_seed"])) * 100
    rows4 = [
        [args.base_label, f"{bg:.2f} ± {bgs:.2f}", f"{b_mean:.2f} ± {b_std:.2f}", f"{b_mean - bg:+.2f}"],
        [args.winner, f"{pg:.2f} ± {pgs:.2f}", f"{p_mean:.2f} ± {p_std:.2f}", f"{p_mean - pg:+.2f}"],
        [f"GNN-only gain ({args.winner} − {args.base_label})", "", "", f"{pg - bg:+.2f}"],
    ]
    df4 = pd.DataFrame(rows4, columns=["Arm", "GNN-only %", "Fused %", "Δ / gain (pp)"])
    show_table(df4, "Table 4 — GNN-Only vs Fused (isolates the graph-construction effect)",
               delta_cols=["Δ / gain (pp)"])
    md.append("## Table 4 — GNN-Only vs Fused (isolates the graph-construction effect)\n")
    md.append("| Arm | GNN-only % | Fused % | Fusion gain (pp) |")
    md.append("|---|---|---|---|")
    md.append(f"| {rows4[0][0]} | {rows4[0][1]} | {rows4[0][2]} | {rows4[0][3]} |")
    md.append(f"| {rows4[1][0]} | {rows4[1][1]} | {rows4[1][2]} | {rows4[1][3]} |")
    md.append(f"| **{rows4[2][0]}** | **{rows4[2][3]} pp** | | |")
    md.append("")

    # ============ TABLE 5 ============
    stage_a = discover_stage_a()
    if stage_a:
        winner_cfg = args.winner.replace("pearson_", "")
        rows5, hl = [], []
        for cfg, vals in stage_a.items():
            row = [vals[t] for t in TARGETS]
            mean = float(np.mean(row))
            is_winner = (cfg == winner_cfg)
            label = f"{cfg}  ← selected" if is_winner else cfg
            rows5.append([label] + [f"{v:.2f}" for v in row] + [f"{mean:.2f}"])
            hl.append(is_winner)
        cols5 = ["Config"] + [SHORT[t] for t in TARGETS] + ["Mean"]
        df5 = pd.DataFrame(rows5, columns=cols5)
        show_table(df5, "Table 5 — Stage A Config Selection (source-val accuracy, seed 42 — "
                         "selection signal, never target-test)", bold_last_row=False,
                   highlight_rows=hl)
        md.append("## Table 5 — Stage A Config Selection (source-val accuracy, seed 42 — "
                  "selection signal, never target-test)\n")
        md.append(f"| {' | '.join(cols5)} |")
        md.append("|" + "---|" * len(cols5))
        for r in rows5:
            label = r[0] if "← selected" not in r[0] else r[0].replace("← selected", "**← selected**")
            md.append(f"| {label} | " + " | ".join(r[1:]) + " |")
        md.append("")
    else:
        print("[seeds_comparison] no results/sweep_cfg/*/val_summary.json found — skipping Table 5.")

    # ============ TABLE 6 ============
    gate = seed_report.get("gate", {}).get(args.winner)
    if gate:
        rows6 = [
            ["Mean Δ vs base", "≥ +1.0 pp", f"{gate['delta_pp']:+.2f} pp",
             "✓" if gate["delta_pp"] >= 1.0 else "✗"],
            ["Worst-fold Δ", "> −2.0 pp", f"{gate['worst_fold_delta_pp']:+.2f} pp",
             "✓" if gate["worst_fold_delta_pp"] > -2.0 else "✗"],
            ["Verdict", "", "", "KEEP" if gate["keep"] else "REVERT"],
        ]
        df6 = pd.DataFrame(rows6, columns=["Criterion", "Threshold", "Observed", "Pass?"])
        show_table(df6, "Table 6 — Keep/Revert Gate")
        md.append("## Table 6 — Keep/Revert Gate\n")
        md.append("| Criterion | Threshold | Observed | Pass? |")
        md.append("|---|---|---|---|")
        md.append(f"| {rows6[0][0]} | {rows6[0][1]} | {rows6[0][2]} | {rows6[0][3]} |")
        md.append(f"| {rows6[1][0]} | {rows6[1][1]} | {rows6[1][2]} | {rows6[1][3]} |")
        md.append(f"| **Verdict** | | | **{rows6[2][3]}** |")
        md.append("")

    # ============ TABLE 7 ============
    per_seed = full_metrics.get("per_seed", {})
    rows7, accs, precs, recs, f1s = [], [], [], [], []
    for seed_key in sorted(per_seed, key=lambda x: int(x)):
        ov = per_seed[seed_key].get("OVERALL")
        if not ov:
            continue
        accs.append(ov["accuracy"]); precs.append(ov["precision"])
        recs.append(ov["recall"]); f1s.append(ov["f1"])
        rows7.append([seed_key, f"{ov['accuracy']:.2f}", f"{ov['precision']:.2f}",
                      f"{ov['recall']:.2f}", f"{ov['f1']:.2f}"])
    if accs:
        rows7.append(["mean ± std", f"{np.mean(accs):.2f} ± {np.std(accs):.2f}",
                      f"{np.mean(precs):.2f} ± {np.std(precs):.2f}",
                      f"{np.mean(recs):.2f} ± {np.std(recs):.2f}",
                      f"{np.mean(f1s):.2f} ± {np.std(f1s):.2f}"])
    df7 = pd.DataFrame(rows7, columns=["Seed", "Accuracy %", "Precision %", "Recall %", "F1 %"])
    show_table(df7, "Table 7 — Overall (pooled-fold) Accuracy / Precision / Recall / F1, per seed")
    md.append("## Table 7 — Overall (pooled-fold) Accuracy / Precision / Recall / F1, per seed\n")
    md.append("| Seed | Accuracy % | Precision % | Recall % | F1 % |")
    md.append("|---|---|---|---|---|")
    for r in rows7[:-1] if accs else rows7:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    if accs:
        last = rows7[-1]
        md.append(f"| **{last[0]}** | **{last[1]}** | **{last[2]}** | **{last[3]}** | **{last[4]}** |")
    md.append("")

    md_path = os.path.join(args.out_dir, "p21_paper_tables.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"\n[seeds_comparison] saved {md_path}")

    # ================= FIGURES (unchanged) =================
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
    plt.savefig(fig1_path, dpi=200)
    if _IPY:
        display(fig)
    plt.close()

    bx = [b_map[s] * 100 for s in common_seeds]
    px = [p_map[s] * 100 for s in common_seeds]
    fig2, ax = plt.subplots(figsize=(7, 5))
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
    plt.savefig(fig2_path, dpi=200)
    if _IPY:
        display(fig2)
    plt.close()

    fig3_path = None
    if stage_a:
        names = list(stage_a.keys())
        vals = [float(np.mean(list(stage_a[n].values()))) for n in names]
        winner_cfg = args.winner.replace("pearson_", "")
        colors = ["#2b6cb0" if n == winner_cfg else "#aaaaaa" for n in names]
        fig3, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, vals, color=colors)
        ax.set_ylabel("Mean source-val accuracy (%)")
        ax.set_title("Stage A: config selection (source-val only)")
        lo = min(vals) - 3
        ax.set_ylim(max(0, lo), max(vals) + 3)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (i, v + 0.3), ha="center", fontsize=9)
        plt.tight_layout()
        fig3_path = os.path.join(args.out_dir, "p21_fig3_stageA_selection.png")
        plt.savefig(fig3_path, dpi=200)
        if _IPY:
            display(fig3)
        plt.close()

    print(f"[seeds_comparison] saved {fig1_path}")
    print(f"[seeds_comparison] saved {fig2_path}")
    if fig3_path:
        print(f"[seeds_comparison] saved {fig3_path}")


if __name__ == "__main__":
    main()
