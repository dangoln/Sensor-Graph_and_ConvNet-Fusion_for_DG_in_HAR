"""
P24 — edge-sparsity curve, REPORT-ONLY (zero retraining).

The P21 Stage A sweep already trained the winner recipe at top-6 / top-9 / top-12
edges per window (alpha=1.0) plus the alpha=0.5 winner at top-9. This script
re-reads those existing dumps and reports accuracy vs graph density — the
storage-relevant question "how few edges does the sensor graph need?".

Source: ../../01_Contribution1_Generalization_Model/P21_PearsonGraph/results/sweep_cfg/<config>/
"""
import os, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P24 = os.path.dirname(HERE)
DEFAULT_P21_CFG = os.path.normpath(os.path.join(
    P24, "..", "..", "01_Contribution1_Generalization_Model", "P21_PearsonGraph", "results", "sweep_cfg"))
TARGETS = ["KuHar", "MotionSense", "RealWorld-Thigh", "RealWorld-Waist", "UCI", "WISDM"]

CURVE = [  # (label, undirected edges/window, config dir)
    ("top-6  (a=1.0)", 6,  "topk6_a1.0"),
    ("top-9  (a=1.0)", 9,  "topk9_a1.0"),
    ("top-12 (a=1.0)", 12, "topk12_a1.0"),
    ("top-9  (a=0.5)*", 9, "topk9_a0.5"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p21_cfg", default=DEFAULT_P21_CFG)
    args = ap.parse_args()

    print("\n" + "=" * 66)
    print("  P24 — accuracy vs graph density (P21 Stage A dumps, seed 42)")
    print("  full graph = 15 undirected edges/window; * = the locked winner")
    print("=" * 66)
    print(f"\n  {'config':18s} {'edges':>6s} {'GNN tgt-test':>13s} {'src-val':>9s}")
    print("  " + "-" * 50)
    out = {}
    for label, k, cfg in CURVE:
        d = os.path.join(args.p21_cfg, cfg)
        vp = os.path.join(d, "val_summary.json")
        if not os.path.isfile(vp):
            print(f"  {label:18s}  (missing: {d})")
            continue
        sv = float(np.mean(list(json.load(open(vp)).values())))
        accs = []
        for t in TARGETS:
            z = np.load(os.path.join(d, f"gnn_{t}.npz"))
            accs.append(float((z["probs"].argmax(1) == z["labels"]).mean()))
        tt = float(np.mean(accs))
        out[cfg] = {"edges": k, "tgt_test": tt, "src_val": sv}
        print(f"  {label:18s} {k:>6d} {tt*100:>12.2f}% {sv*100:>8.2f}%")
    print("\n  Reading: edges/window is the per-graph storage knob "
          "(vs 15 for a full graph).")
    print("=" * 66)
    json.dump(out, open(os.path.join(P24, "results", "p24_edge_curve.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
