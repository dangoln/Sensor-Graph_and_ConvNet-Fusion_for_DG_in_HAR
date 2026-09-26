"""
P24 — post-hoc compression evaluation on the saved checkpoints.

For each trained variant checkpoint, evaluates the storage-accuracy trade-off:

  fp32        : the checkpoint as trained (reference)
  fp16        : half-precision weights (2x smaller; CUDA eval, CPU fallback skips)
  int8-dyn    : dynamic int8 quantization of all nn.Linear layers (CPU eval;
                covers the classifier, projections, and SAGEConv internals)
  prune30/50/70 : global magnitude pruning (smallest |w| zeroed across Linear+Conv);
                accuracy as-is (no fine-tuning — honest post-hoc numbers) with
                theoretical sparse storage (nnz * 4 bytes + indices)

Reports, per variant x compression: params, on-disk size, GNN-only acc (6-fold
mean), and FUSED acc with the frozen ConvNet probs (P21 base_s42 cell, w=0.6).
Latency: mean CPU ms/window on a 512-window sample.

Output: results/p24_compression_report.json + printed table.
"""
import os, io, sys, json, glob, time, argparse, copy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P24 = os.path.dirname(HERE)
GNN_DIR = os.path.join(P24, "gnn")
sys.path.insert(0, GNN_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from utils.config import get_config
from utils.seed import set_seed
from data.daghar_loader import build_lodo_sensor_graphs, DAGHAR_DATASETS
from models.sensor_graph import SensorGraphModel
from models.da_sensor_graph import DomainAdversarialWrapper
from p24.train_save import p24_config, VARIANTS

TARGETS = list(DAGHAR_DATASETS)
DEFAULT_CONVNET = os.path.normpath(os.path.join(
    P24, "..", "..", "01_Contribution1_Generalization_Model", "P21_PearsonGraph", "results", "sweep", "base_s42"))
FUSE_W = 0.6


def state_dict_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def load_model(ckpt, cfg):
    backbone = SensorGraphModel(cfg)
    model = DomainAdversarialWrapper(backbone, ckpt["num_domains"], cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict(model, graphs, device, batch_size, half=False):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if half:
                batch.x = batch.x.half()
            logits = model.predict(batch)
            probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
            labels.append(batch.y.cpu().numpy())
    return np.concatenate(probs).astype(np.float32), np.concatenate(labels).astype(np.int64)


def global_magnitude_prune(model, fraction):
    """Zero the smallest |w| globally across Linear/Conv1d weights. Returns nnz info."""
    m = copy.deepcopy(model)
    weights = [p for mod in m.modules()
               if isinstance(mod, (nn.Linear, nn.Conv1d))
               for n, p in mod.named_parameters(recurse=False) if n == "weight"]
    if not weights:
        return m, 0, 0
    allw = torch.cat([w.detach().abs().flatten() for w in weights])
    k = int(len(allw) * fraction)
    if k > 0:
        thresh = allw.kthvalue(k).values.item()
        with torch.no_grad():
            for w in weights:
                w.mul_((w.abs() > thresh).float())
    total = sum(w.numel() for w in weights)
    nnz = sum(int((w != 0).sum()) for w in weights)
    return m, nnz, total


def quantize_dynamic_int8(model):
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).cpu(), {nn.Linear}, dtype=torch.qint8)


def measure_latency_cpu(model, graphs, batch_size=1, n=256):
    loader = DataLoader(graphs[:n], batch_size=batch_size, shuffle=False)
    model = model.cpu().eval()
    t0 = time.perf_counter()
    cnt = 0
    with torch.no_grad():
        for batch in loader:
            model.predict(batch)
            cnt += batch.num_graphs
    return (time.perf_counter() - t0) * 1000.0 / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATA_PATH"))
    ap.add_argument("--variants", default="all", help="comma list of trained variants")
    ap.add_argument("--convnet_dir", default=DEFAULT_CONVNET,
                    help="frozen ConvNet probs (P21 base_s42) for fused numbers")
    ap.add_argument("--skip_latency", action="store_true")
    args = ap.parse_args()

    variants = [v for v in VARIANTS
                if os.path.isfile(os.path.join(P24, "results", "checkpoints", v,
                                               f"ckpt_{TARGETS[0]}.pt"))] \
        if args.variants == "all" else args.variants.split(",")
    if not variants:
        raise SystemExit("[P24] no checkpoints found — run p24.train_save first.")

    has_convnet = all(os.path.isfile(os.path.join(args.convnet_dir, f"convnet_{t}.npz"))
                      for t in TARGETS)
    if not has_convnet:
        print(f"[P24] WARNING: ConvNet probs not found at {args.convnet_dir} — "
              f"fused numbers skipped.", flush=True)

    report = {}
    for variant in variants:
        ck_dir = os.path.join(P24, "results", "checkpoints", variant)
        rows = {}   # compression -> {acc per fold, fused per fold, size, params}
        lat = {}
        for t in TARGETS:
            ckpt = torch.load(os.path.join(ck_dir, f"ckpt_{t}.pt"),
                              map_location="cpu", weights_only=False)
            cfg = p24_config(args.data_path, 1, 10, 42, variant)
            cfg["daghar_timesteps"] = ckpt["node_feat_dim"]
            device = torch.device(cfg["device"])
            set_seed(42)
            lodo = build_lodo_sensor_graphs(cfg["data_root"], t, config=cfg,
                                            folder_map=cfg["dataset_folder_names"])
            test_graphs = lodo["test_graphs"]
            bs = cfg["batch_size"]
            model = load_model(ckpt, cfg)

            Pc = yc = None
            if has_convnet:
                C = np.load(os.path.join(args.convnet_dir, f"convnet_{t}.npz"))
                Pc, yc = C["probs"], C["labels"]

            def record(name, P, y, size_bytes, params):
                if not np.array_equal(y, yc if yc is not None else y):
                    raise SystemExit(f"[P24] alignment error {variant}/{name}/{t}")
                r = rows.setdefault(name, {"acc": {}, "fused": {}, "size": size_bytes,
                                           "params": params})
                r["acc"][t] = float((P.argmax(1) == y).mean())
                if Pc is not None:
                    r["fused"][t] = float(((FUSE_W*P + (1-FUSE_W)*Pc).argmax(1) == y).mean())

            n_params = ckpt["n_params"]

            # fp32
            P, y = predict(model.to(device), test_graphs, device, bs)
            record("fp32", P, y, state_dict_bytes(model), n_params)

            # fp16 (CUDA only)
            if torch.cuda.is_available():
                m16 = copy.deepcopy(model).half().to(device)
                P16, y16 = predict(m16, test_graphs, device, bs, half=True)
                record("fp16", P16, y16, state_dict_bytes(m16.cpu()), n_params)

            # int8 dynamic (CPU)
            mq = quantize_dynamic_int8(model)
            Pq, yq = predict(mq, test_graphs, torch.device("cpu"), bs)
            record("int8-dyn", Pq, yq, state_dict_bytes(mq), n_params)

            # magnitude pruning
            for frac in (0.3, 0.5, 0.7):
                mp, nnz, total = global_magnitude_prune(model, frac)
                Pp, yp = predict(mp.to(device), test_graphs, device, bs)
                other = sum(p.numel() for p in model.parameters()) - total
                sparse_bytes = int(nnz * 8 + other * 4)   # value+index per nnz, rest fp32
                record(f"prune{int(frac*100)}", Pp, yp, sparse_bytes, nnz + other)

            if not args.skip_latency and t == TARGETS[0]:
                lat["fp32"] = measure_latency_cpu(copy.deepcopy(model), test_graphs)
                lat["int8-dyn"] = measure_latency_cpu(mq, test_graphs)
            print(f"  [{variant}:{t}] done", flush=True)

        report[variant] = {"rows": rows, "latency_ms_per_window_cpu": lat}

    # ── table ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 86)
    print("  P24 — STORAGE vs ACCURACY (GNN branch; fused with frozen ConvNet at w=0.6)")
    print("  Reference system: P21 winner fused 78.10 ± 0.14% (3 seeds); this table seed 42")
    print("=" * 86)
    for variant, rep in report.items():
        print(f"\n  VARIANT {variant}  {VARIANTS[variant]}")
        print(f"  {'compression':12s} {'params':>10s} {'size':>10s} "
              f"{'GNN acc':>8s} {'fused':>8s} {'Δfused vs fp32':>15s}")
        print("  " + "-" * 70)
        ref = None
        for name, r in rep["rows"].items():
            acc = np.mean(list(r["acc"].values())) * 100
            fused = np.mean(list(r["fused"].values())) * 100 if r["fused"] else float("nan")
            if name == "fp32":
                ref = fused
            d = f"{fused-ref:+.2f}pp" if ref == ref and fused == fused else "—"
            size_mb = r["size"] / 1e6
            print(f"  {name:12s} {r['params']:>10,} {size_mb:>8.2f}MB "
                  f"{acc:>7.2f}% {fused:>7.2f}% {d:>15s}")
        if rep["latency_ms_per_window_cpu"]:
            print(f"  CPU latency (ms/window): " +
                  "  ".join(f"{k}={v:.2f}" for k, v in
                            rep["latency_ms_per_window_cpu"].items()))
    print("=" * 86)
    out = os.path.join(P24, "results", "p24_compression_report.json")
    json.dump(report, open(out, "w"), indent=2, default=float)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
