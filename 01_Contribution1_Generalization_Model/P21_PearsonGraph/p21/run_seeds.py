"""
P21 — STAGE B: seed-average the winning Pearson config vs the Arm A baseline.

Builds the sweep tree  results/sweep/<label>_s<seed>/  that run_report.py reads:

  base_s42 / base_s1 / base_s2
      COPIED from the P18b seed sweep (Arm A: fixed full topology + GraphSAGE,
      source-only, seeds 42/1/2 — already trained, 76.87 ± 1.07%). The ConvNet
      branch is FROZEN across P20 arms, so its per-seed probs are reused too.

  pearson_<cfg>_s42 / _s1 / _s2
      GNN retrained with Pearson edges (seed 42 is copied from Stage A if the
      config matches). The matching seed's ConvNet npz are copied in from the
      base cell — fusing pearson-GNN(seed s) with convnet(seed s).

Resumable: complete cells are skipped.
"""
import os, sys, glob, shutil, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
P21 = os.path.dirname(HERE)
SWEEP = os.path.join(P21, "results", "sweep")
SWEEP_CFG = os.path.join(P21, "results", "sweep_cfg")
TARGETS = ["KuHar", "MotionSense", "RealWorld-Thigh", "RealWorld-Waist", "UCI", "WISDM"]

# Default location of the P18b seed-sweep cells (Arm A + per-seed ConvNet probs),
# relative to the PROJECTS tree. Override with --p18b_sweep if your Drive layout differs.
DEFAULT_P18B = os.path.normpath(os.path.join(
    P21, "..", "ConvNet_Branch_Frozen", "B_Frozen_Predictions_P18b_SourceOnly", "results", "sweep"))

GRID = {
    "topk9_a1.0":  ["--select", "topk", "--topk", "9",  "--alpha", "1.0"],
    "topk6_a1.0":  ["--select", "topk", "--topk", "6",  "--alpha", "1.0"],
    "topk12_a1.0": ["--select", "topk", "--topk", "12", "--alpha", "1.0"],
    "topk9_a0.5":  ["--select", "topk", "--topk", "9",  "--alpha", "0.5"],
}


def _has(d, prefix):
    return len(glob.glob(os.path.join(d, f"{prefix}_*.npz"))) >= len(TARGETS)


def _copy_npz(src, dst, prefix):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in glob.glob(os.path.join(src, f"{prefix}_*.npz")):
        shutil.copy2(f, dst); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATA_PATH"))
    ap.add_argument("--config", required=True, choices=list(GRID))
    ap.add_argument("--seeds", default="42,1,2", help="comma-separated seeds")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--p18b_sweep", default=DEFAULT_P18B,
                    help="path to P18b results/sweep (base_s* cells)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    label = f"pearson_{args.config}"

    # 1) Reuse Arm A baseline cells (gnn + convnet) from P18b — no retraining.
    for s in seeds:
        src = os.path.join(args.p18b_sweep, f"base_s{s}")
        dst = os.path.join(SWEEP, f"base_s{s}")
        if _has(dst, "gnn") and _has(dst, "convnet"):
            print(f"[reuse] base_s{s}: already present", flush=True)
            continue
        if not (_has(src, "gnn") and _has(src, "convnet")):
            raise SystemExit(
                f"[reuse] P18b base cell incomplete or missing: {src}\n"
                f"        Point --p18b_sweep at the P18b results/sweep folder "
                f"(it must contain base_s{s} with 6 gnn_*.npz + 6 convnet_*.npz).")
        n = _copy_npz(src, dst, "gnn") + _copy_npz(src, dst, "convnet")
        print(f"[reuse] base_s{s}: copied {n} npz from P18b", flush=True)

    # 2) Pearson arm per seed.
    for s in seeds:
        cell = os.path.join(SWEEP, f"{label}_s{s}")
        # Seed 42 GNN may already exist from Stage A — reuse it.
        stage_a = os.path.join(SWEEP_CFG, args.config)
        if s == 42 and not _has(cell, "gnn") and _has(stage_a, "gnn"):
            n = _copy_npz(stage_a, cell, "gnn")
            print(f"[reuse] {label}_s42: copied {n} gnn npz from Stage A", flush=True)
        if not _has(cell, "gnn"):
            cmd = [sys.executable, "-m", "p21.gnn_predict",
                   "--data_path", args.data_path, "--targets", "all",
                   "--epochs", str(args.epochs), "--log_every", str(args.log_every),
                   "--seed", str(s), "--edge_mode", "pearson",
                   "--out_dir", cell] + GRID[args.config]
            print(f"\n>>> [{label} s{s}] {' '.join(cmd[2:])}", flush=True)
            subprocess.run(cmd, cwd=P21, check=True)
        else:
            print(f"[skip] {label}_s{s}: gnn complete", flush=True)
        # ConvNet branch is frozen: fuse with the SAME SEED's baseline ConvNet probs.
        if not _has(cell, "convnet"):
            n = _copy_npz(os.path.join(SWEEP, f"base_s{s}"), cell, "convnet")
            print(f"[reuse] {label}_s{s}: copied {n} convnet npz (frozen branch, seed {s})",
                  flush=True)

    print("\nDone. Now run:  python -m p21.run_report", flush=True)


if __name__ == "__main__":
    main()
