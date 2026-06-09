"""Behavioral affect Bi-LSTM (architecture.md line 928).

Tiny sequence classifier: consumes a 30-step (1s-bin) feature sequence and
predicts one of 4 affect classes. ~50k-150k params — runs on CPU at serve time.
"""

import torch
import torch.nn as nn


class BehavioralBiLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, n_layers: int = 1,
                 n_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):                 # x: (B, T, n_features)
        out, _ = self.lstm(x)             # (B, T, 2*hidden)
        pooled = out.mean(dim=1)          # mean-pool over time
        return self.fc(self.drop(pooled))


def build_model(cfg: dict, n_features: int) -> "BehavioralBiLSTM":
    return BehavioralBiLSTM(
        n_features=n_features,
        hidden_size=cfg.get("hidden_size", 64),
        n_layers=cfg.get("n_layers", 1),
        n_classes=cfg.get("n_classes", 4),
        dropout=cfg.get("dropout", 0.3),
    )
