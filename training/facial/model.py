"""CNN-LSTM architecture for DAiSEE facial affect detection.

Input:  (B, T, 3, 96, 96)   — batch of T-frame clips
Output: (B, num_classes)    — class logits
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → BN → ReLU → MaxPool (halves spatial dims)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class FrameCNN(nn.Module):
    """Per-frame spatial encoder: (B,3,96,96) → (B,512).

    Four conv blocks halve the spatial dims each time:
      96 → 48 → 24 → 12 → 6  then global average pool → linear projection.
    """

    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):           # (B, 3, 96, 96)
        x = self.blocks(x)          # (B, 256, 6, 6)
        x = self.pool(x).flatten(1) # (B, 256)
        return self.proj(x)         # (B, out_dim)


class CNNLSTMModel(nn.Module):
    """CNN feature extractor + LSTM temporal aggregator.

    Works on sequences of frames (clips), producing a single affect prediction
    per clip using the final hidden state of the LSTM.
    """

    def __init__(
        self,
        num_classes: int = 4,
        cnn_out_dim: int = 512,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.cnn     = FrameCNN(out_dim=cnn_out_dim)
        self.lstm    = nn.LSTM(cnn_out_dim, hidden_size,
                               num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, num_classes)

    def forward(self, x):                      # x: (B, T, 3, 96, 96)
        B, T, C, H, W = x.shape
        feats = self.cnn(x.view(B * T, C, H, W))  # (B*T, 512)
        feats = feats.view(B, T, -1)               # (B, T, 512)
        _, (h_n, _) = self.lstm(feats)             # h_n: (num_layers, B, 256)
        out = self.dropout(h_n[-1])                # (B, 256)
        return self.head(out)                      # (B, num_classes)


def build_model(cfg: dict) -> CNNLSTMModel:
    """Instantiate CNNLSTMModel from a config dict (model section of config.yaml)."""
    return CNNLSTMModel(
        num_classes=cfg.get("num_classes", 4),
        cnn_out_dim=cfg.get("cnn_out_dim", 512),
        hidden_size=cfg.get("hidden_size", 256),
        num_layers=cfg.get("num_layers", 1),
        dropout=cfg.get("dropout", 0.5),
    )
