
"""CardioFed: Baseline and Multi-Scale Residual ECG models."""

import torch
import torch.nn as nn


# --------------------------------------------------
# BASELINE MODEL
# --------------------------------------------------

def conv_block(c_in, c_out, k):
    return nn.Sequential(
        nn.Conv1d(
            c_in,
            c_out,
            kernel_size=k,
            padding=k // 2
        ),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2)
    )


class ECGNet(nn.Module):
    """Original baseline 1D CNN."""

    def __init__(
        self,
        n_leads=12,
        n_classes=5,
        use_lstm=False,
        dropout=0.3
    ):
        super().__init__()

        self.features = nn.Sequential(
            conv_block(n_leads, 64, 7),
            conv_block(64, 128, 5),
            conv_block(128, 256, 3)
        )

        self.use_lstm = use_lstm

        if use_lstm:
            self.lstm = nn.LSTM(
                256,
                128,
                batch_first=True,
                bidirectional=True
            )
            feat = 256
        else:
            feat = 256

        self.head = nn.Sequential(
            nn.Linear(feat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):

        h = self.features(x)

        if self.use_lstm:
            h, _ = self.lstm(h.permute(0, 2, 1))
            h = h.mean(dim=1)
        else:
            h = h.mean(dim=2)

        return self.head(h)


# --------------------------------------------------
# MULTI-SCALE RESIDUAL BLOCK
# --------------------------------------------------

class MultiScaleResidualBlock(nn.Module):
    """
    Three parallel convolution branches.

    Kernel sizes: 3, 7, 15.

    Branch features are concatenated, projected,
    and combined with a residual shortcut.
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        branch_channels = out_channels // 3

        self.branch3 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch7 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=7,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch15 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=15,
                padding=7,
                bias=False
            ),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

        combined_channels = branch_channels * 3

        self.projection = nn.Sequential(
            nn.Conv1d(
                combined_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm1d(out_channels)
        )

        # Adjust residual dimensions when necessary.
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):

        # Parallel feature extraction.
        f3 = self.branch3(x)
        f7 = self.branch7(x)
        f15 = self.branch15(x)

        # Feature fusion.
        combined = torch.cat(
            [f3, f7, f15],
            dim=1
        )

        combined = self.projection(combined)

        # Residual addition.
        residual = self.shortcut(x)

        return self.activation(combined + residual)


# --------------------------------------------------
# MULTI-SCALE RESIDUAL ECG MODEL
# --------------------------------------------------

class MultiScaleResidualECGNet(nn.Module):
    """
    Proposed CardioFed architecture.

    Input:  [B, 12, 1000]
    Output: [B, 5] logits
    """

    def __init__(
        self,
        n_leads=12,
        n_classes=5,
        dropout=0.35
    ):

        super().__init__()

        # Initial feature extraction.
        self.stem = nn.Sequential(
            nn.Conv1d(
                n_leads,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )

        # Multi-scale feature extraction stages.
        self.stage1 = nn.Sequential(
            MultiScaleResidualBlock(64, 96),
            nn.MaxPool1d(2)
        )

        self.stage2 = nn.Sequential(
            MultiScaleResidualBlock(96, 192),
            nn.MaxPool1d(2)
        )

        self.stage3 = nn.Sequential(
            MultiScaleResidualBlock(192, 288),
            nn.MaxPool1d(2)
        )

        # Global average pooling.
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Five-class multi-label prediction.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(288, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.global_pool(x)

        return self.classifier(x)
