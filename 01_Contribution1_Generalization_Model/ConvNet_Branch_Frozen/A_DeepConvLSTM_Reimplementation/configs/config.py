"""
Central configuration for the ConvNet-in-DAGHAR project.

Everything that a user might want to tweak (paths, model hyper-parameters,
training settings, LODO protocol) lives here so the rest of the codebase stays
generic and re-usable for future models / pipelines.

The defaults reproduce the original DeepConvLSTM paper
(Ordonez & Roggen, 2016) as faithfully as the DAGHAR data allows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
@dataclass
class PathConfig:
    # Root of this project (where run_lodo.py lives). Set via the PROJECT_PATH env
    # var (Final_PROJECT/project_paths.py does this); defaults to this code folder.
    project_path: str = os.environ.get(
        "PROJECT_PATH",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    # Folder that contains the 6 DAGHAR dataset sub-folders, each with
    # train.csv / validation.csv / test.csv
    data_path: str = os.environ.get(
        "DATA_PATH",
        "/content/drive/MyDrive/HAR-Datasets/DAGHAR/standardized_view",
    )
    # Where results (metrics, checkpoints, logs) are written.
    results_dir: str = os.environ.get("RESULTS_DIR", "results")
    # Sub-folder grouping a run's outputs so successive runs never overwrite
    # each other. Final layout: <results_dir>/<run_name>/<input_view>/...
    # ("run1" = the original time/frequency results, stored at results/time and
    #  results/frequency directly, before run_name was introduced.)
    run_name: str = os.environ.get("RUN_NAME", "run2")


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    # The 6 DAGHAR datasets in the standardized_view folder.
    datasets: List[str] = field(
        default_factory=lambda: [
            "KuHar",
            "MotionSense",
            "RealWorld_thigh",   # folder name on disk may use a hyphen; see name_map
            "RealWorld_waist",
            "UCI",
            "WISDM",
        ]
    )
    # Maps the canonical dataset key above to the actual folder name on disk.
    # (Disk folders use hyphens: 'Realworld-thigh', 'Realworld-waist'.)
    folder_name_map: dict = field(
        default_factory=lambda: {
            "KuHar": "KuHar",
            "MotionSense": "MotionSense",
            "RealWorld_thigh": "RealWorld-Thigh",
            "RealWorld_waist": "RealWorld-Waist",
            "UCI": "UCI",
            "WISDM": "WISDM",
        }
    )

    # The 6 sensor channels common to every DAGHAR dataset (standardized view).
    # Each channel is a 60-sample time series -> columns "<channel>-0" .. "<channel>-59".
    channels: List[str] = field(
        default_factory=lambda: [
            "accel-x", "accel-y", "accel-z",
            "gyro-x", "gyro-y", "gyro-z",
        ]
    )
    window_size: int = 60          # samples per window (DAGHAR standardized view)
    label_column: str = "standard activity code"

    # Input representation fed to the model:
    #   "time"      -> raw standardized windows (DeepConvLSTM's native input)
    #   "frequency" -> per-window FFT magnitude (rfft), DAGHAR's frequency view
    # The MODEL ARCHITECTURE is identical for both; only the input changes.
    input_view: str = "time"       # one of {"time", "frequency"}
    # --- Frequency-view (FFT) options. These are PREPROCESSING choices applied
    #     before the (unchanged) model; they do not alter the architecture. ---
    # Drop the DC (bin 0) component.
    fft_drop_dc: bool = False
    # Magnitude scaling of the spectrum:
    #   "amplitude" -> |FFT|              (run1 default)
    #   "power"     -> |FFT|^2
    #   "log"       -> log1p(|FFT|)       (recommended: compresses heavy tails)
    fft_magnitude: str = "log"
    # Tapering window applied before the FFT to reduce spectral leakage:
    #   "none" (run1 default) or "hann".
    fft_window: str = "hann"

    # DAGHAR standard activity codes (for reporting only).
    class_names: List[str] = field(
        default_factory=lambda: [
            "sit",        # 0
            "stand",      # 1
            "walk",       # 2
            "stair up",   # 3
            "stair down", # 4
            "run",        # 5
        ]
    )
    num_classes: int = 6

    # Normalisation applied AFTER the input-view transform, fit on the source
    # TRAIN set only (no target leakage):
    #   "auto"           -> "none" for the time view (data already z-scored),
    #                       "zscore_feature" for the frequency view (FFT
    #                       magnitudes span very different scales per bin).
    #   "none"           -> no extra normalisation
    #   "zscore_channel" -> per-channel z-score (stats pooled over time/bins)
    #   "zscore_feature" -> per-(timestep-or-bin, channel) z-score
    normalize: str = "auto"

    def resolved_normalize(self) -> str:
        if self.normalize != "auto":
            return self.normalize
        return "zscore_feature" if self.input_view == "frequency" else "none"


# --------------------------------------------------------------------------- #
#  Model  (DeepConvLSTM - paper defaults)
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    name: str = "DeepConvLSTM"
    in_channels: int = 6           # DAGHAR sensor channels (paper: 113)
    num_classes: int = 6           # DAGHAR activities    (paper: 18)
    window_size: int = 60          # input sequence length (paper: 24)

    num_conv_layers: int = 4       # paper: 4 convolutional layers
    num_filters: int = 64          # paper: 64 feature maps
    filter_size: int = 5           # paper: 5x1 kernels (time x 1)

    num_lstm_layers: int = 2       # paper: 2 recurrent layers
    lstm_hidden: int = 128         # paper: 128 hidden units

    dropout: float = 0.5           # paper: p=0.5 on inputs of every dense layer


# --------------------------------------------------------------------------- #
#  Training  (paper defaults)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    optimizer: str = "rmsprop"     # paper: RMSProp
    learning_rate: float = 1e-3    # paper: 10e-3 -> 1e-3
    rho: float = 0.9               # paper: decay factor rho = 0.9
    batch_size: int = 100          # paper: 100
    max_epochs: int = 100
    early_stopping_patience: int = 20   # epochs without improvement in monitor
    # Which validation metric selects the best checkpoint / drives early stopping:
    #   "val_loss"    -> lowest source-val cross-entropy (run1 default)
    #   "val_acc"     -> highest source-val accuracy  (recommended under shift)
    #   "val_f1_macro"-> highest source-val macro F1
    model_selection_metric: str = "val_acc"
    weight_init: str = "orthogonal"     # paper: random orthogonal init
    num_workers: int = 2
    seed: int = 42
    device: str = "cuda"           # falls back to cpu automatically if unavailable


# --------------------------------------------------------------------------- #
#  LODO protocol
# --------------------------------------------------------------------------- #
@dataclass
class LODOConfig:
    # For each fold one dataset is the held-out target; the remaining 5 are sources.
    # Train on the *train* split of the 5 sources, model-select on their *validation*
    # split, and evaluate on the target dataset's *test* split.
    source_train_split: str = "train"
    source_val_split: str = "validation"
    target_eval_split: str = "test"
    # If you would rather evaluate on the whole held-out dataset, set this to
    # ["train", "validation", "test"].
    target_eval_splits: List[str] = field(default_factory=lambda: ["test"])


# --------------------------------------------------------------------------- #
#  Top-level experiment config
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lodo: LODOConfig = field(default_factory=LODOConfig)

    def effective_seq_len(self) -> int:
        """Sequence length the model actually sees, given the input view."""
        if self.data.input_view == "frequency":
            # rfft of a length-N real window yields N//2 + 1 magnitude bins.
            n_bins = self.data.window_size // 2 + 1
            if self.data.fft_drop_dc:
                n_bins -= 1
            return n_bins
        return self.data.window_size

    def sync(self) -> "ExperimentConfig":
        """Keep cross-cutting fields consistent (channels, classes, seq length)."""
        self.model.in_channels = len(self.data.channels)
        self.model.num_classes = self.data.num_classes
        self.model.window_size = self.effective_seq_len()
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def get_config() -> ExperimentConfig:
    """Return the default experiment configuration (already synced)."""
    return ExperimentConfig().sync()
