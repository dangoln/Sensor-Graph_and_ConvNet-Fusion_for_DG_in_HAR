"""
P17 setup: install PyTorch Geometric (needed by the GNN side) and verify the
shared DAGHAR data path for both the GNN and ConvNet pipelines.
"""
import os, sys, subprocess


def install_deps():
    print("Installing PyTorch Geometric ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch-geometric"])
    print("Done.")


def resolve_folder(data_root, folder):
    if os.path.isdir(os.path.join(data_root, folder)):
        return folder
    if os.path.isdir(data_root):
        tl = folder.lower()
        for n in os.listdir(data_root):
            if n.lower() == tl and os.path.isdir(os.path.join(data_root, n)):
                return n
    return folder


def verify():
    print("\nVerifying imports ...")
    try:
        import torch
        print(f"  OK torch {torch.__version__}  CUDA={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     GPU {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print("  FAIL torch:", e); return False
    try:
        import torch_geometric
        print(f"  OK torch_geometric {torch_geometric.__version__}")
    except Exception:
        print("  FAIL torch_geometric — run install_deps()"); return False

    data_root = os.environ.get("DATA_PATH",
        "/content/drive/MyDrive/HAR-Datasets/DAGHAR/standardized_view")
    print(f"\nChecking data at: {data_root}")
    ok = True
    for ds in ["KuHar", "MotionSense", "RealWorld-Thigh", "RealWorld-Waist", "UCI", "WISDM"]:
        real = resolve_folder(data_root, ds)
        path = os.path.join(data_root, real)
        if os.path.isdir(path):
            n = len([f for f in os.listdir(path) if f.endswith(".csv")])
            note = "" if real == ds else f"  (matched '{real}')"
            print(f"  OK {ds:18s} {n} CSVs{note}")
        else:
            print(f"  MISSING {ds:18s} ({path})"); ok = False
    return ok


if __name__ == "__main__":
    install_deps()
    ok = verify()
    if ok:
        print("\nSetup complete. Next: run the GNN and ConvNet probability dumps, then fuse.")
    else:
        print("\nSome checks failed — fix the data path above before running.")
