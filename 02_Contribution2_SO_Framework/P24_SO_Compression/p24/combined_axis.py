"""
P24 — COMBINED-AXIS operating point (Thesis_Final_Structure §2.3).

Demonstrates that the compression axes COMPOSE into one deployable model rather
than only being reported in isolation. Stacks three axes on the locked P21 recipe:

    structural : no_gz  (drop gyro-z  -> 5 sensor nodes)
              +  AW-40  (60 -> 40 timesteps/window)
    precision  : int8 dynamic quantization (post-hoc, CPU; nn.Linear layers)

The combined arm is trained ONCE by the existing train_save.py with
`--variant full --nc no_gz --aw 40` (arm dir: results/checkpoints/full_nc-no_gz_aw40)
— no new training code. This script is INFERENCE-ONLY: it evaluates that
checkpoint at fp32 and int8, and for reference the uncompressed `full` arm at
fp32, all fused with the frozen ConvNet (base_s42, w=0.6). It reports GNN-only +
fused accuracy, params, deploy size (GNN state-dict bytes), the compression
factor, and Δ vs the full reference — so "the axes compose for ~X pp at ~N× smaller"
is a single measured configuration instead of an extrapolation from isolated cuts.

Seed 42 (composition demonstration; every constituent single-axis arm already
carries 5-seed error bars in p24_seed_report_5.json).

Output: results/p24_combined_axis.json + printed table.
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P24 = os.path.dirname(HERE)
GNN_DIR = os.path.join(P24, "gnn")
sys.path.insert(0, GNN_DIR)

import torch
from data.daghar_loader import build_lodo_sensor_graphs, DAGHAR_DATASETS
from utils.seed import set_seed
from p24.train_save import p24_config
from p24.compress_eval import (load_model, predict, state_dict_bytes,
                               quantize_dynamic_int8)

TARGETS = list(DAGHAR_DATASETS)
FUSE_W = 0.6
DEFAULT_CONVNET = os.path.normpath(os.path.join(
    P24, "..", "..", "01_Contribution1_Generalization_Model", "P21_PearsonGraph", "results", "sweep", "base_s42"))

# operating points to evaluate:  label -> (arm_dir, nc, aw, quantize?)
COMBINED_ARM = "full_nc-no_gz_aw40"
POINTS = [
    ("full  (reference, fp32)",         "full",        None,    0,  False),
    ("no_gz+AW40  (fp32)",              COMBINED_ARM,  "no_gz", 40, False),
    ("no_gz+AW40+int8  (deployable)",   COMBINED_ARM,  "no_gz", 40, True),
]


def eval_point(arm_dir, nc, aw, quantize, data_path, convnet_dir):
    """Per-fold GNN-only + fused accuracy for one operating point. Also returns
    the GNN deploy size in bytes (constant across folds; taken from fold 0)."""
    ck_dir = os.path.join(P24, "results", "checkpoints", arm_dir)
    if not os.path.isfile(os.path.join(ck_dir, f"ckpt_{TARGETS[0]}.pt")):
        raise SystemExit(f"[combined_axis] missing checkpoints in {ck_dir} — "
                         f"train it first: p24.train_save --variant full "
                         f"{'--nc '+nc if nc else ''} {'--aw '+str(aw) if aw else ''}")
    has_conv = all(os.path.isfile(os.path.join(convnet_dir, f"convnet_{t}.npz"))
                   for t in TARGETS)
    g_acc, f_acc = {}, {}
    size_bytes = params = None
    for t in TARGETS:
        ckpt = torch.load(os.path.join(ck_dir, f"ckpt_{t}.pt"),
                          map_location="cpu", weights_only=False)
        cfg = p24_config(data_path, 1, 10, 42, "full", nc=nc, aw=aw)
        cfg["daghar_timesteps"] = ckpt["node_feat_dim"]
        set_seed(42)
        lodo = build_lodo_sensor_graphs(cfg["data_root"], t, config=cfg,
                                        folder_map=cfg["dataset_folder_names"])
        model = load_model(ckpt, cfg)

        if quantize:
            m = quantize_dynamic_int8(model)          # CPU int8
            dev = torch.device("cpu")
        else:
            dev = torch.device(cfg["device"])
            m = model.to(dev)
        if size_bytes is None:
            size_bytes = state_dict_bytes(m if quantize else m.cpu())
            params = ckpt["n_params"]
            if not quantize:
                m = m.to(dev)

        P, y = predict(m, lodo["test_graphs"], dev, cfg["batch_size"])
        g_acc[t] = float((P.argmax(1) == y).mean())
        if has_conv:
            C = np.load(os.path.join(convnet_dir, f"convnet_{t}.npz"))
            Pc, yc = C["probs"], C["labels"]
            assert np.array_equal(y, yc), f"alignment {arm_dir}/{t}"
            f_acc[t] = float(((FUSE_W * P + (1 - FUSE_W) * Pc).argmax(1) == y).mean())
        print(f"    [{arm_dir}{'/int8' if quantize else ''}:{t}] "
              f"gnn {g_acc[t]*100:.2f}%", flush=True)
    return g_acc, f_acc, size_bytes, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATA_PATH"))
    ap.add_argument("--convnet_dir", default=DEFAULT_CONVNET)
    args = ap.parse_args()

    rows = {}
    for label, arm, nc, aw, quant in POINTS:
        print(f"  [combined_axis] {label} ...", flush=True)
        ga, fa, size_b, params = eval_point(arm, nc, aw, quant,
                                            args.data_path, args.convnet_dir)
        rows[label] = {
            "arm": arm, "nc": nc, "aw": aw, "int8": quant,
            "gnn_mean": float(np.mean(list(ga.values()))),
            "fused_mean": float(np.mean(list(fa.values()))) if fa else None,
            "gnn_kb": size_b / 1024.0, "params": int(params),
            "per_fold_fused": fa,
        }

    ref = rows["full  (reference, fp32)"]
    ref_fused = ref["fused_mean"]; ref_kb = ref["gnn_kb"]
    for label, r in rows.items():
        r["delta_fused_pp"] = ((r["fused_mean"] - ref_fused) * 100
                               if (r["fused_mean"] and ref_fused) else None)
        r["compression_x"] = (ref_kb / r["gnn_kb"]) if r["gnn_kb"] else None

    print("\n" + "=" * 84)
    print("  P24 — COMBINED-AXIS operating point (seed 42; fused w=0.6, frozen ConvNet)")
    print("  Do the compression axes COMPOSE? no_gz + AW-40 + int8 stacked vs full.")
    print("=" * 84)
    print(f"\n  {'operating point':32s} {'GNN':>7s} {'fused':>7s} "
          f"{'Δ vs full':>10s} {'GNN size':>10s} {'shrink':>7s}")
    print("  " + "-" * 78)
    for label, r in rows.items():
        f = f"{r['fused_mean']*100:.2f}%" if r["fused_mean"] else "  n/a"
        d = f"{r['delta_fused_pp']:+.2f}pp" if r["delta_fused_pp"] is not None else " ref"
        shrink = f"{r['compression_x']:.1f}x" if r["compression_x"] else "  -"
        print(f"  {label:32s} {r['gnn_mean']*100:6.2f}% {f:>7s} {d:>10s} "
              f"{r['gnn_kb']:>7.1f}KB {shrink:>7s}")
    print("=" * 84)

    out = os.path.join(P24, "results", "p24_combined_axis.json")
    json.dump({"fuse_w": FUSE_W, "seed": 42, "reference": "full (fp32)",
               "points": rows}, open(out, "w"), indent=2)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
