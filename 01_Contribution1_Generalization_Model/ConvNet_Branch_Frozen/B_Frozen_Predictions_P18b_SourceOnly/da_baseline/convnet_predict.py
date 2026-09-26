"""
P17 — DA source-only BASELINE, ConvNet side.

Trains the DeepConvLSTM replication (run2_nohann frequency recipe, the ~74% best)
per LODO fold and dumps per-sample target-test probabilities + labels for fusion.
Source-only: the target's unlabeled train split is LOADED (P17 plumbing for P18) but
NOT used for training or model selection — this run reproduces the DG number.

Differences vs the P16a fusion dump:
  * calls build_lodo_fold(..., with_target_unlabeled=True) and reports the pool size;
  * full per-epoch training logs every `--log_every` epochs (train/val loss + acc).

Runs in its OWN subprocess with only convnet/ on sys.path.
Output: <out_dir>/convnet_<target>.npz  with arrays  probs [N,6] float32, labels [N] int64.
"""
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P17 = os.path.dirname(HERE)
CN_DIR = os.path.join(P17, "convnet")
sys.path.insert(0, CN_DIR)           # only the ConvNet codebase is importable here

import torch
from configs.config import get_config
from src.data.data_loader import build_lodo_fold
from src.models.deepconvlstm import build_model
from src.training.trainer import train_model
from src.utils.common import set_seed, get_device

# ConvNet internal dataset key  ->  canonical (GNN-style) name used for npz filenames
CANON = {
    "KuHar": "KuHar", "MotionSense": "MotionSense",
    "RealWorld_thigh": "RealWorld-Thigh", "RealWorld_waist": "RealWorld-Waist",
    "UCI": "UCI", "WISDM": "WISDM",
}


def best_config(data_path, epochs, seed=42):
    cfg = get_config()
    if data_path:
        cfg.paths.data_path = data_path
    # run2_nohann (best, ~74%): frequency + log magnitude + no taper + val-acc selection
    cfg.data.input_view = "frequency"
    cfg.data.fft_magnitude = "log"
    cfg.data.fft_window = "none"
    cfg.data.fft_drop_dc = False
    cfg.data.normalize = "auto"          # -> zscore_feature for the frequency view
    cfg.train.model_selection_metric = "val_acc"
    cfg.train.max_epochs = epochs
    cfg.train.early_stopping_patience = 20
    cfg.train.seed = seed
    return cfg.sync()


def train_and_predict(target_key, cfg, device, log_every, transductive_norm=False,
                      align_method="none", align_lambda=0.0):
    set_seed(cfg.train.seed)
    use_align = align_method not in (None, "none", "") and align_lambda > 0.0
    # Load the target-unlabeled pool (needed for alignment / plumbing).
    train_loader, val_loader, test_loader, target_unlabeled_loader, meta = build_lodo_fold(
        cfg, target_key, with_target_unlabeled=True, transductive_norm=transductive_norm
    )
    n_tu = meta.get("n_target_unlabeled", 0)
    tag = (f"  ALIGN[{align_method} λ={align_lambda:g}]" if use_align
           else ("  [transductive-norm]" if transductive_norm else ""))
    print(f"  [cnn:{CANON[target_key]}] source-train={meta['n_train']}  "
          f"source-val={meta['n_val']}  target-test={meta['n_test']}  "
          f"target-unlabeled={n_tu}{tag}", flush=True)

    model = build_model(cfg.model).to(device)
    train_model(model, train_loader, val_loader, device, cfg.train,
                cfg.data.num_classes, log_prefix=f"    [cnn:{CANON[target_key]}] ",
                verbose=True, log_every=log_every,
                target_loader=(target_unlabeled_loader if use_align else None),
                align_method=(align_method if use_align else None),
                align_lambda=(align_lambda if use_align else 0.0))

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:                       # shuffle=False -> test.csv order
            x = x.to(device)
            p = model.predict_proba(x).cpu().numpy()    # softmax [B, 6]
            probs.append(p)
            labels.append(y.numpy())
    P = np.concatenate(probs, axis=0).astype(np.float32)
    yv = np.concatenate(labels, axis=0).astype(np.int64)
    acc = float((P.argmax(1) == yv).mean())
    print(f"  [cnn:{CANON[target_key]}] TARGET-TEST acc = {acc*100:.2f}%  (N={len(yv)})", flush=True)
    return P, yv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATA_PATH"))
    ap.add_argument("--out_dir", default=None, help="defaults to results/probs_dg or probs_align")
    ap.add_argument("--targets", default="all", help="comma-separated canonical names, or 'all'")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=5, help="training-log cadence (epochs)")
    ap.add_argument("--transductive_norm", action="store_true",
                    help="P18a (reverted): fit Standardizer on source-train ∪ target-unlabeled")
    ap.add_argument("--align", choices=["none", "coral", "mmd"], default="none",
                    help="P18b: source→target feature alignment method")
    ap.add_argument("--align_lambda", type=float, default=1.0,
                    help="weight on the alignment loss (only used if --align != none)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (for seed-averaging)")
    args = ap.parse_args()

    use_align = args.align != "none" and args.align_lambda > 0.0
    mode = (f"align={args.align} λ={args.align_lambda:g}" if use_align
            else ("transductive-norm (P18a)" if args.transductive_norm
                  else "source-only baseline (DG-norm)"))
    default_dir = "probs_align" if use_align else "probs_dg"
    out_dir = args.out_dir or os.path.join(P17, "results", default_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    cfg0 = best_config(args.data_path, args.epochs, args.seed)
    keys = list(CANON.keys())
    if args.targets != "all":
        want = set(args.targets.split(","))
        keys = [k for k in keys if CANON[k] in want or k in want]
    print(f"[ConvNet] P18b | mode={mode} | seed={args.seed} | run2_nohann freq recipe | "
          f"data={cfg0.paths.data_path} | device={device} | "
          f"seq_len={cfg0.model.window_size} | targets={[CANON[k] for k in keys]} | "
          f"out={out_dir}", flush=True)
    if use_align:
        print(f"[ConvNet] NOTE: {args.align.upper()} aligns source↔target-unlabeled on the "
              "last-timestep features. No target labels; selection on source-val.", flush=True)

    for k in keys:
        cfg = best_config(args.data_path, args.epochs, args.seed)   # fresh per fold
        P, y = train_and_predict(k, cfg, device, args.log_every, args.transductive_norm,
                                 args.align, args.align_lambda)
        out = os.path.join(out_dir, f"convnet_{CANON[k]}.npz")
        np.savez(out, probs=P, labels=y)
        print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
