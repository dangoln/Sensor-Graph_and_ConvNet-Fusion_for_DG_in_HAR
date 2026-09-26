"""
Final_PROJECT — the ONE place where file paths are defined.

Every notebook in this folder loads this file first, and every script receives its
paths from here (via the notebook) or falls back to paths relative to its own
location inside Final_PROJECT. So:

  * The Final_PROJECT folder location is detected automatically from where this
    file lives — moving/renaming the folder needs no edits here.
  * The only thing you may need to edit is DATA_PATH below (where the DAGHAR
    `standardized_view` folder is). You can also leave this file alone and set an
    environment variable instead:  os.environ["DATA_PATH"] = "/your/path"
    before importing this module.

Usage inside a notebook (already done in each notebook's first code cell):

    import sys; sys.path.insert(0, FINAL_PROJECT_ROOT)
    import project_paths as PP
    PP.activate("p21")          # "p21" | "p24" | "convnet_a" | "convnet_b"
"""
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# EDIT HERE if your DAGHAR data lives elsewhere.
# Folder that contains KuHar/, MotionSense/, RealWorld-Thigh/ (or Realworld-thigh/),
# RealWorld-Waist/, UCI/, WISDM/ — each with train.csv / validation.csv / test.csv.
DEFAULT_DATA_PATH = "/content/drive/MyDrive/HAR-Datasets/DAGHAR/standardized_view"
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)

# Everything below is derived automatically — no need to edit.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

C1_DIR = os.path.join(PROJECT_ROOT, "01_Contribution1_Generalization_Model")
C2_DIR = os.path.join(PROJECT_ROOT, "02_Contribution2_SO_Framework")

# Contribution 1 — locked generalization model
P21_DIR        = os.path.join(C1_DIR, "P21_PearsonGraph")
P21_SWEEP      = os.path.join(P21_DIR, "results", "sweep")       # per-seed GNN + ConvNet probs
P21_SWEEP_CFG  = os.path.join(P21_DIR, "results", "sweep_cfg")   # Stage-A graph-config screening
P21_BASE_S42   = os.path.join(P21_SWEEP, "base_s42")             # frozen ConvNet probs, seed 42

CONVNET_DIR    = os.path.join(C1_DIR, "ConvNet_Branch_Frozen")
CONVNET_A_DIR  = os.path.join(CONVNET_DIR, "A_DeepConvLSTM_Reimplementation")
CONVNET_B_DIR  = os.path.join(CONVNET_DIR, "B_Frozen_Predictions_P18b_SourceOnly")
CONVNET_FROZEN_SWEEP = os.path.join(CONVNET_B_DIR, "results", "sweep")        # ORIGINAL frozen probs (never overwritten)
CONVNET_RERUN_SWEEP  = os.path.join(CONVNET_B_DIR, "results", "sweep_rerun")  # where any ConvNet re-run writes

# Contribution 2 — storage-optimization framework
P24_DIR        = os.path.join(C2_DIR, "P24_SO_Compression")

COMPONENTS = {
    "p21":       P21_DIR,
    "p24":       P24_DIR,
    "convnet_a": CONVNET_A_DIR,
    "convnet_b": CONVNET_B_DIR,
}

DATASETS = ["KuHar", "MotionSense", "RealWorld-Thigh", "RealWorld-Waist", "UCI", "WISDM"]


def _check_data(path):
    """True if all six DAGHAR dataset folders exist (case-insensitive match)."""
    if not os.path.isdir(path):
        return False, [f"folder not found: {path}"]
    present = {n.lower() for n in os.listdir(path)}
    missing = [d for d in DATASETS if d.lower() not in present]
    return (not missing), missing


def activate(component, check_data=True, chdir=True):
    """Set up a notebook/session for one component of Final_PROJECT.

    - exports DATA_PATH (and PROJECT_PATH for the ConvNet code) as env vars, so
      `!python -m ...` shell commands and every script see the same paths;
    - checks the data folder and the component folder exist;
    - cd's into the component folder and puts it on sys.path.
    Returns the component's folder path.
    """
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {list(COMPONENTS)}")
    workdir = COMPONENTS[component]
    assert os.path.isdir(workdir), f"component folder missing: {workdir}"

    os.environ["DATA_PATH"] = DATA_PATH
    os.environ["PROJECT_PATH"] = workdir          # read by the ConvNet configs
    os.environ["FINAL_PROJECT_ROOT"] = PROJECT_ROOT

    if check_data:
        ok, missing = _check_data(DATA_PATH)
        assert ok, (f"DAGHAR data not found / incomplete at DATA_PATH={DATA_PATH} "
                    f"(missing: {missing}). Edit DEFAULT_DATA_PATH in project_paths.py "
                    f"or set os.environ['DATA_PATH'] before importing it.")

    if chdir:
        os.chdir(workdir)
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    summary(component)
    return workdir


def summary(component=None):
    print("Final_PROJECT paths")
    print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"  DATA_PATH    : {DATA_PATH}")
    if component:
        print(f"  component    : {component} -> {COMPONENTS[component]}")
        print(f"  cwd          : {os.getcwd()}")


if __name__ == "__main__":
    summary()
    ok, missing = _check_data(DATA_PATH)
    print("  data check   :", "OK" if ok else f"MISSING {missing}")
    for k, v in COMPONENTS.items():
        print(f"  {k:10s}: {'OK ' if os.path.isdir(v) else 'MISSING '}{v}")
