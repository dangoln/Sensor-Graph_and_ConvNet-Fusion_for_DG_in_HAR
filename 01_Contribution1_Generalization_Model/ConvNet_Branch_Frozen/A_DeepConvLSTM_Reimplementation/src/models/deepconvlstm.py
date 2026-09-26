"""
DeepConvLSTM - faithful PyTorch port of the original Lasagne/Theano model from
Ordonez & Roggen (2016), "Deep Convolutional and LSTM Recurrent Neural Networks
for Multimodal Wearable Activity Recognition".

Original Lasagne graph (from the paper's reference implementation):

    input  : (B, 1, T, C)
    conv1  : Conv2D(64, (5,1))          # ReLU
    conv2  : Conv2D(64, (5,1))          # ReLU
    conv3  : Conv2D(64, (5,1))          # ReLU
    conv4  : Conv2D(64, (5,1))          # ReLU      -> (B, 64, T', C)
    shuff  : DimShuffle (0, 2, 1, 3)               -> (B, T', 64, C)
    lstm1  : LSTM(128)                              # features per step = 64*C
    lstm2  : LSTM(128)
    dense  : Dense(num_classes, softmax) applied per time step
    output : SliceLayer(-1)  -> last time step

The architecture itself is UNCHANGED. The only values adapted for DAGHAR are the
ones the data forces: input channels C (113 -> 6), window length T (24 -> 60) and
number of classes (18 -> 6). All of these are passed in via config, so the model
class is reusable as-is for any HAR dataset.

Faithful details preserved:
  * 4 convolutional layers, 64 filters, 5x1 kernels, ReLU, valid (no) padding
    -> the time dimension shrinks by (filter_size-1) per layer, exactly as in the
       paper (T' = T - 4*(filter_size-1)).
  * convolution mixes only across time (kernel width 1 over channels), so each
    sensor channel keeps its own feature maps, giving 64*C features per timestep
    fed to the LSTM (paper: 64*113 = 7232; DAGHAR: 64*6 = 384).
  * 2 stacked LSTM layers, 128 hidden units.
  * dropout p=0.5 on the inputs of every dense layer (the two LSTMs and the final
    classifier), matching the paper's regularisation.
  * orthogonal weight initialisation.
  * classification uses the last time step (SliceLayer(-1)).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 6,
        window_size: int = 60,
        num_conv_layers: int = 4,
        num_filters: int = 64,
        filter_size: int = 5,
        num_lstm_layers: int = 2,
        lstm_hidden: int = 128,
        dropout: float = 0.5,
        weight_init: str = "orthogonal",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.window_size = window_size
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.lstm_hidden = lstm_hidden

        # ---- Convolutional front-end: 4 x Conv2d(64, (5,1)) + ReLU ----------
        convs = []
        c_in = 1
        for _ in range(num_conv_layers):
            convs.append(
                nn.Conv2d(
                    in_channels=c_in,
                    out_channels=num_filters,
                    kernel_size=(filter_size, 1),  # (time, channel) -> conv over time only
                    stride=1,
                    padding=0,                     # "valid": shrinks time dim like the paper
                )
            )
            convs.append(nn.ReLU(inplace=True))
            c_in = num_filters
        self.conv = nn.Sequential(*convs)

        # Sequence length after the conv stack (valid convolutions).
        self.seq_len_out = window_size - num_conv_layers * (filter_size - 1)
        if self.seq_len_out <= 0:
            raise ValueError(
                f"window_size={window_size} too small for {num_conv_layers} conv "
                f"layers of size {filter_size} (got T'={self.seq_len_out})."
            )

        # Features fed to the LSTM at each timestep: num_filters * in_channels.
        self.lstm_input_size = num_filters * in_channels

        # ---- Recurrent core: 2 x LSTM(128) ---------------------------------
        # Dropout p=0.5 on the *input* of every dense (recurrent) layer.
        # nn.LSTM applies dropout between stacked layers, but NOT on the first
        # layer's input, so we add an explicit dropout module for that.
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # ---- Classifier: Dense(softmax) on the last timestep ---------------
        self.classifier_dropout = nn.Dropout(dropout)  # dropout on input of dense
        self.fc = nn.Linear(lstm_hidden, num_classes)
        # NOTE: softmax is folded into nn.CrossEntropyLoss during training; raw
        # logits are returned here. Use `predict_proba` for probabilities.

        if weight_init == "orthogonal":
            self._orthogonal_init()

    # ------------------------------------------------------------------ #
    def _orthogonal_init(self) -> None:
        """Random orthogonal initialisation (paper: 'randomly orthogonally initialized')."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, T, C)  ->  logits : (B, num_classes)
        """
        b = x.shape[0]

        # Conv front-end -> (B, 64, T', C)
        x = self.conv(x)

        # DimShuffle (0, 2, 1, 3): (B, 64, T', C) -> (B, T', 64, C)
        x = x.permute(0, 2, 1, 3).contiguous()

        # Flatten the (64, C) feature grid per timestep -> (B, T', 64*C)
        x = x.view(b, self.seq_len_out, self.lstm_input_size)

        # Dropout on the input of the first recurrent (dense) layer.
        x = self.input_dropout(x)

        # 2-layer LSTM -> (B, T', hidden)
        out, _ = self.lstm(x)

        # SliceLayer(-1): keep only the last timestep -> (B, hidden)
        last = out[:, -1, :]

        # Dropout on the input of the final dense layer, then classify.
        last = self.classifier_dropout(last)
        logits = self.fc(last)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)


def build_model(model_cfg) -> DeepConvLSTM:
    """Factory that builds a DeepConvLSTM from a ModelConfig dataclass."""
    return DeepConvLSTM(
        in_channels=model_cfg.in_channels,
        num_classes=model_cfg.num_classes,
        window_size=model_cfg.window_size,
        num_conv_layers=model_cfg.num_conv_layers,
        num_filters=model_cfg.num_filters,
        filter_size=model_cfg.filter_size,
        num_lstm_layers=model_cfg.num_lstm_layers,
        lstm_hidden=model_cfg.lstm_hidden,
        dropout=model_cfg.dropout,
    )
