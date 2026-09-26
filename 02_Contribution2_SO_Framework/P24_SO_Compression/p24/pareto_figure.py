"""
P24 — Pareto-front figure for the compression-generalization trade-off (Ch. 7).

Reads the result JSONs already produced by the hardening pass and draws the
single figure that carries the storage chapter: fused LODO accuracy (y) vs GNN
model size (x, log scale), every operating point overlaid, with the efficient
(Pareto-optimal) frontier highlighted. The ConvNet branch is frozen and shared
across all points, so GNN state-dict size is the honest storage axis for the
compression story.

Sources (whichever exist under results/):
    p24_seed_report_5.json      -> the 5-seed arms (full, no_gz, tiny, accel_only,
                                   gyro_only, aw40, aw30) with fused_mean + gnn_kb
    p24_combined_axis.json      -> the stacked no_gz+AW-40(+int8) operating points

Inference-only / no GPU. Outputs:
    results/p24_pareto.png          the figure
    results/p24_pareto_points.csv   the underlying table (accuracy, size, on-frontier)
"""
import os, json, csv, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
P24 = os.path.dirname(HERE)
RES = os.path.join(P24, "results")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_points():
    """Return list of dicts: {label, acc(0-1), kb, group}."""
    pts = []
    sr = os.path.join(RES, "p24_seed_report_5.json")
    if os.path.isfile(sr):
        d = json.load(open(sr))
        for arm, a in d.get("arms", {}).items():
            if a.get("n_seeds", 0) and a.get("fused_mean") and a.get("gnn_kb"):
                label = arm.replace(" (P21 winner)", "")
                pts.append({"label": label, "acc": a["fused_mean"],
                            "kb": a["gnn_kb"], "group": "single-axis / arm"})
    ca = os.path.join(RES, "p24_combined_axis.json")
    if os.path.isfile(ca):
        d = json.load(open(ca))
        for label, r in d.get("points", {}).items():
            if r.get("fused_mean") and r.get("gnn_kb"):
                if label.startswith("full"):
                    continue                       # already have 'full' from seed report
                pts.append({"label": label.split("  ")[0], "acc": r["fused_mean"],
                            "kb": r["gnn_kb"], "group": "combined-axis"})
    # de-dup by label, keep smaller size
    seen = {}
    for p in pts:
        if p["label"] not in seen or p["kb"] < seen[p["label"]]["kb"]:
            seen[p["label"]] = p
    return list(seen.values())


def frontier(pts):
    """Mark Pareto-optimal points (want high acc, low kb)."""
    for p in sorted(pts, key=lambda x: x["kb"]):
        p["frontier"] = not any(
            (q["kb"] <= p["kb"] and q["acc"] > p["acc"]) or
            (q["kb"] < p["kb"] and q["acc"] >= p["acc"])
            for q in pts if q is not p)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RES, "p24_pareto.png"))
    args = ap.parse_args()

    pts = frontier(load_points())
    if not pts:
        raise SystemExit("[pareto] no result JSONs found — run seed_report_5 "
                         "(and combined_axis) first.")

    # number the points by accuracy (desc) so labels never overlap — the point
    # carries only an index; a side panel maps index -> name/acc/size.
    pts = sorted(pts, key=lambda x: -x["acc"])
    for i, p in enumerate(pts, 1):
        p["idx"] = i

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(11.4, 5.8))
    gs = GridSpec(1, 2, width_ratios=[1.9, 1.0], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    leg = fig.add_subplot(gs[1]); leg.axis("off")

    # frontier line (efficient points, sorted by size)
    front = sorted([p for p in pts if p["frontier"]], key=lambda x: x["kb"])
    if len(front) > 1:
        ax.plot([p["kb"] for p in front], [p["acc"] * 100 for p in front],
                "--", color="#999", lw=1.3, zorder=1)

    colors = {"single-axis / arm": "#2a8f4e", "combined-axis": "#d23c3c"}
    for p in pts:
        on = p["frontier"]
        ax.scatter(p["kb"], p["acc"] * 100, s=210 if on else 150,
                   c=colors.get(p["group"], "#666"),
                   edgecolors="black" if on else "#333",
                   linewidths=1.6 if on else 0.6, zorder=3,
                   marker="*" if p["group"] == "combined-axis" else "o")
        ax.annotate(str(p["idx"]), (p["kb"], p["acc"] * 100), zorder=4,
                    ha="center", va="center", fontsize=7.5,
                    color="white" if p["group"] == "single-axis / arm" else "white",
                    fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("GNN model size  (KB, state-dict; log scale)  —  smaller is cheaper")
    ax.set_ylabel("Fused LODO accuracy  (%)")
    ax.set_title("P24 — Compression–Generalization Trade-off (DAGHAR LODO)", fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.margins(x=0.12, y=0.12)

    # side legend: index -> label, accuracy, size, frontier marker
    lines = [r"$\bf{Operating\ points}$  (● arm  ★ combined;  ✦ = Pareto-efficient)", ""]
    for p in pts:
        star = "  ✦" if p["frontier"] else "   "
        grp = "★" if p["group"] == "combined-axis" else "●"
        lines.append(f"{p['idx']:>2}{star}  {grp} {p['label']:<16s} "
                     f"{p['acc']*100:5.1f}%   {p['kb']:6.1f} KB")
    leg.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", family="monospace",
             fontsize=8.6, transform=leg.transAxes)
    leg.text(0.0, 0.02,
             "GNN size = state-dict bytes (ConvNet frozen/shared, excluded).\n"
             "Combined point = seed 42; arms = 5-seed mean.",
             va="bottom", ha="left", fontsize=7, color="#555", transform=leg.transAxes)

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"  saved {args.out}")

    csv_out = os.path.join(RES, "p24_pareto_points.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "group", "fused_acc_pct", "gnn_kb", "on_frontier"])
        for p in sorted(pts, key=lambda x: -x["acc"]):
            w.writerow([p["label"], p["group"], round(p["acc"] * 100, 2),
                        round(p["kb"], 1), p["frontier"]])
    print(f"  saved {csv_out}")
    print("\n  Pareto-efficient points:",
          ", ".join(p["label"] for p in front))


if __name__ == "__main__":
    main()
