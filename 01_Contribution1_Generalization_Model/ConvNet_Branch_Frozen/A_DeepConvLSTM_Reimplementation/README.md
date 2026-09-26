# ConvNet-in-DAGHAR

Re-implementation of the **DeepConvLSTM** model (Ordóñez & Roggen, 2016,
*"Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable
Activity Recognition"*) applied to the **DAGHAR** benchmark with
**Leave-One-Dataset-Out (LODO)** cross-dataset validation.

The original reference code is Theano/Lasagne (Python 2) and ships only an
inference script with weights for OPPORTUNITY (113 channels / 18 classes). That
is unrunnable on modern Colab and useless for DAGHAR (6 channels / 6 classes), so
the **architecture is faithfully re-implemented in PyTorch and trained from
scratch**. The network graph is unchanged; only the values the data forces are
adapted (input channels, window length, number of classes).

## What stays identical to the paper

| Component        | Paper (DeepConvLSTM)        | Here                         |
|------------------|-----------------------------|------------------------------|
| Conv layers      | 4 × Conv2D, 64 filters, 5×1 | identical                    |
| Conv activation  | ReLU, valid padding         | identical                    |
| Recurrent core   | 2 × LSTM, 128 units         | identical                    |
| Dropout          | p = 0.5 on dense inputs     | identical                    |
| Classifier       | Dense + softmax, last step  | identical (logits + CE loss) |
| Weight init      | random orthogonal           | identical                    |
| Optimizer        | RMSProp, lr 1e-3, ρ = 0.9   | identical                    |
| Loss             | cross-entropy               | identical                    |

## What changes (forced by DAGHAR)

| Item            | Paper (OPPORTUNITY) | DAGHAR (here)                          |
|-----------------|---------------------|----------------------------------------|
| Input channels  | 113                 | **6** (accel x/y/z, gyro x/y/z)        |
| Window length   | 24 (sliding window) | **60** (DAGHAR windows are pre-segmented) |
| Sliding window  | applied             | **not needed** (rows already windowed) |
| Classes         | 18 gestures         | **6** standard activities              |
| LSTM input dim  | 64 × 113 = 7232     | 64 × 6 = 384                           |
| Normalisation   | min-max to [0,1]    | none (standardized_view already z-scored) |

DAGHAR standard activity codes: `0 sit, 1 stand, 2 walk, 3 stair up,
4 stair down, 5 run`.

## Input views (time vs. frequency)

DAGHAR evaluates every model on **two views of the same standardized windows**,
and reports both. The view is a *preprocessing step in the data pipeline* — the
DeepConvLSTM architecture is byte-for-byte identical in both cases. Toggle it
with `config.data.input_view` (or `--input_view`):

| `input_view` | What the model sees | Seq. length | DAGHAR ConvNet LODO acc. |
|--------------|---------------------|-------------|--------------------------|
| `"time"`      | raw standardized signal (native DeepConvLSTM input) | 60 | 60.3% (their Table 9) |
| `"frequency"` | per-window FFT magnitude (`rfft`)                   | 31 | 77.1% (their Table 9) |

The original DeepConvLSTM code contains **no** FFT; the frequency view is
DAGHAR's own benchmark choice, applied uniformly to all models. For the frequency
view, FFT magnitudes span very different scales per bin, so normalisation
defaults to per-feature z-score fitted on the source-train set (`normalize:
"auto"`).

### Frequency-view recipe (run2)

The model architecture is **identical** across runs; these are preprocessing /
training-recipe knobs only:

| Knob | Config field | run1 | run2 (default) |
|------|--------------|------|----------------|
| FFT magnitude | `data.fft_magnitude` | `amplitude` | `log` (`log1p(\|FFT\|)`) |
| FFT taper | `data.fft_window` | `none` | `hann` |
| Drop DC bin | `data.fft_drop_dc` | `False` | `False` |
| Model selection | `train.model_selection_metric` | `val_loss` | `val_acc` |
| Max epochs / patience | `train.max_epochs` / `early_stopping_patience` | 100 / 15 | 100 / 20 |

`log` magnitude compresses the heavy-tailed FFT amplitudes into a far more
learnable feature space; the Hann window reduces spectral leakage; and selecting
the checkpoint by **validation accuracy** avoids restoring the under-trained
early-epoch weights that `val_loss` picks under domain shift.

### Run management

Results are written to `results/<run_name>/<input_view>/` so successive runs
never overwrite each other. There is a **single codebase** (no forked copies) —
old behaviour is fully reproducible from the same code by passing run1 flags, e.g.
`--run_name run1_repro --fft_magnitude amplitude --fft_window none
--model_selection_metric val_loss`.

> Note: your original run1 results live at `results/time/` and
> `results/frequency/` (created before `run_name` existed); new runs default to
> `results/run2/...`.

## LODO protocol

For each of the 6 datasets (KuHar, MotionSense, RealWorld-thigh,
RealWorld-waist, UCI, WISDM):

- **train** on the concatenated `train.csv` of the other 5 datasets,
- **model-select / early-stop** on their concatenated `validation.csv`,
- **evaluate** on the held-out dataset's `test.csv`.

Reported per fold and aggregated (mean ± std): accuracy, weighted F1, macro F1,
per-class F1, confusion matrix.

## Project layout

```
Project_ConvNet_DAGHAR/
├── configs/
│   └── config.py            # ALL paths + hyper-parameters (single source of truth)
├── src/
│   ├── data/
│   │   ├── dataset.py       # CSV -> (N,1,60,6) tensors, channel selection, z-score
│   │   └── data_loader.py   # LODO fold construction + DataLoaders
│   ├── models/
│   │   └── deepconvlstm.py  # faithful PyTorch DeepConvLSTM (reusable)
│   ├── training/
│   │   ├── trainer.py       # generic train/val loop, RMSProp, early stopping
│   │   └── metrics.py       # accuracy / F1 / confusion matrix / fold summary
│   └── utils/
│       └── common.py        # seeding, device, JSON IO
├── experiments/
│   ├── run_lodo.py          # main LODO driver (all 6 folds)
│   └── smoke_test.py        # tiny end-to-end check on a few CSVs
├── notebooks/
│   └── Run_ConvNet_DAGHAR_Colab.ipynb
├── requirements.txt
└── README.md
```

The modules are intentionally model-agnostic: to try a different model later,
add it under `src/models/` and point `build_model` / config at it — the data,
training, metrics and LODO driver are reused unchanged.

## Running on Colab

1. Upload this `Project_ConvNet_DAGHAR/` folder to
   `/content/drive/MyDrive/Thesis/SO_Experiments/P11-test/`.
2. Make sure the DAGHAR data is at
   `/content/drive/MyDrive/HAR-Datasets/DAGHAR/standardized_view`
   with the 6 dataset sub-folders, each containing `train.csv`,
   `validation.csv`, `test.csv`.
3. Open `notebooks/Run_ConvNet_DAGHAR_Colab.ipynb` and run all cells,
   **or** in a cell:

```python
import os, sys
from google.colab import drive
drive.mount('/content/drive')

PROJECT_PATH = '/content/drive/MyDrive/Thesis/SO_Experiments/P11-test/Project_ConvNet_DAGHAR'
DATA_PATH    = '/content/drive/MyDrive/HAR-Datasets/DAGHAR/standardized_view'
sys.path.insert(0, PROJECT_PATH)
os.environ['PROJECT_PATH'] = PROJECT_PATH
os.environ['DATA_PATH']    = DATA_PATH

!cd "$PROJECT_PATH" && python -m experiments.run_lodo
```

Results (per-fold JSON + checkpoints + `lodo_summary.json`) are written to
`<PROJECT_PATH>/results/`.

## Running locally / CLI

```bash
cd Project_ConvNet_DAGHAR
pip install -r requirements.txt

# run2 frequency recipe (targets DAGHAR's ~77% ConvNet column):
python -m experiments.run_lodo --data_path /path/to/standardized_view \
    --run_name run2 --input_view frequency \
    --fft_magnitude log --fft_window hann \
    --model_selection_metric val_acc --max_epochs 100 --patience 20

# time-domain (matches DAGHAR's ~60% ConvNet column):
python -m experiments.run_lodo --data_path /path/to/standardized_view \
    --run_name run2 --input_view time --model_selection_metric val_acc

# quick subset:
python -m experiments.run_lodo --targets UCI,WISDM --max_epochs 5
```

## Notes & assumptions

- Normalisation defaults to `config.data.normalize = "auto"`: **none** for the
  time view (the `standardized_view` is already z-normalised) and **per-feature
  z-score** for the frequency view (FFT magnitudes span very different scales per
  bin). All stats are fit on **source-train only** (no target leakage). Override
  with `"none"`, `"zscore_channel"`, or `"zscore_feature"`.
- MotionSense carries extra `attitude.*` / `gravity.*` columns; only the 6
  channels common to every dataset are used, so all folds share an identical
  input space — required for cross-dataset evaluation.
- Not every dataset contains all 6 activities (e.g. WISDM has no stair
  up/down; UCI has no run). The model always outputs 6 classes; absent classes
  simply do not appear in that fold's test labels. Metrics use `zero_division=0`.
