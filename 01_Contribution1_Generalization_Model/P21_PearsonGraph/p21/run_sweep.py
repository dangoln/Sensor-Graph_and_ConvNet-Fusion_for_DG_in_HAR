"""
P21 — STAGE A: Pearson-config sweep at one seed, SELECTED ON SOURCE-VAL.

Runs each Pearson graph-construction config (seed 42 only) and reports the mean
best source-val accuracy across the 6 folds. The winning config is chosen on
source-val — never on target-test — keeping config selection protocol-clean.
(Target-test numbers are printed for the record but marked as non-selective.)

Grid (4 configs, each ≈ one 6-fold GNN run):
    topk9_a1.0   : top-9 pairs,  pure Pearson
    topk6_a1.0   : top-6 pairs,  pure Pearson (sparser)
    topk12_a1.0  : top-12 pairs, pure Pearson (denser)
    topk9_a0.5   : top-9 pairs,  blended 50/50 with the physical prior

Each cell is resumable (per-fold npz + val_summary.json are skipped if present).
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P21 = os.path.dirname(HERE)
SWEEP_CFG = os.path.join(P21, "results", "sweep_cfg")
TARGETS = ["KuHar", "MotionSense", "RealWorld-Thigh", "RealWorld-Waist", "UCI", "WISDM"]

GRID = {
    "topk9_a1.0":  ["--select", "topk", "--topk", "9",  "--alpha", "1.0"],
    "topk6_a1.0":  ["--select", "topk", "--topk", "6",  "--alpha", "1.0"],
    "topk12_a1.0": ["--select", "topk", "--topk", "12", "--alpha", "1.0"],
    "topk9_a0.5":  ["--select", "topk", "--topk", "9",  "--alpha", "0.5"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATA_PATH"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--configs", default="all",
                    help="comma-separated subset of: " + ",".join(GRID))
    args = ap.parse_args()

    configs = list(GRID) if args.configs == "all" else args.configs.split(",")
    for c in configs:
        if c not in GRID:
            raise SystemExit(f"unknown config '{c}' (choose from {list(GRID)})")

    for c in configs:
        out_dir = os.path.join(SWEEP_CFG, c)
        cmd = [sys.executable, "-m", "p21.gnn_predict",
               "--data_path", args.data_path, "--targets", "all",
               "--epochs", str(args.epochs), "--log_every", str(args.log_every),
               "--seed", str(args.seed), "--edge_mode", "pearson",
               "--out_dir", out_dir] + GRID[c]
        print(f"\n>>> [{c}] {' '.join(cmd[2:])}", flush=True)
        subprocess.run(cmd, cwd=P21, check=True)

    # ── Report: select on SOURCE-VAL ─────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  P21 STAGE A — Pearson config sweep (seed 42)")
    print("  SELECTION METRIC = mean best source-val acc (target-test shown FYI only)")
    print("=" * 74)
    print(f"\n  {'config':14s} {'src-val mean':>13s} {'tgt-test mean (FYI)':>20s}")
    print("  " + "-" * 50)
    rows = {}
    for c in sorted(GRID):
        d = os.path.join(SWEEP_CFG, c)
        vp = os.path.join(d, "val_summary.json")
        if not os.path.isfile(vp):
            continue
        vals = json.load(open(vp))
        if len(vals) < len(TARGETS):
            print(f"  {c:14s}  (incomplete: {len(vals)}/{len(TARGETS)} folds)")
            continue
        sv = float(np.mean([vals[t] for t in TARGETS]))
        accs = []
        for t in TARGETS:
            z = np.load(os.path.join(d, f"gnn_{t}.npz"))
            accs.append(float((z["probs"].argmax(1) == z["labels"]).mean()))
        tt = float(np.mean(accs))
        rows[c] = sv
        print(f"  {c:14s} {sv*100:12.2f}% {tt*100:19.2f}%")
    if rows:
        best = max(rows, key=rows.get)
        print(f"\n  WINNER (by source-val): {best}")
        print(f"  -> Stage B: python -m p21.run_seeds --config {best} --seeds 1,2")
        json.dump({"winner": best, "src_val": rows},
                  open(os.path.join(P21, "results", "p21_sweep_winner.json"), "w"), indent=2)
    print("=" * 74)


if __name__ == "__main__":
    main()
