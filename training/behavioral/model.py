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


def build_model(cfg: dict, n_features: int, n_classes: int | None = None) -> "BehavioralBiLSTM":
    """Build the Bi-LSTM. `n_classes` overrides cfg (used for the 2-class pretrain proxy head)."""
    return BehavioralBiLSTM(
        n_features=n_features,
        hidden_size=cfg.get("hidden_size", 64),
        n_layers=cfg.get("n_layers", 1),
        n_classes=n_classes if n_classes is not None else cfg.get("n_classes", 4),
        dropout=cfg.get("dropout", 0.3),
    )


def freeze_lstm(model: "BehavioralBiLSTM", frozen: bool = True) -> None:
    """Freeze/unfreeze the shared LSTM encoder (the FC head always stays trainable).

    Used for transfer learning: freeze the pretrained encoder for the first few fine-tune
    epochs so the fresh head adapts without disturbing the transferred weights, then unfreeze.
    """
    for p in model.lstm.parameters():
        p.requires_grad = not frozen


def load_pretrained_lstm(model: "BehavioralBiLSTM", checkpoint_path: str) -> bool:
    """Load ONLY the LSTM encoder weights from a pretrained checkpoint into `model`.

    The FC head is left fresh (the pretrain proxy head has a different class count than the
    4-class fine-tune head — see plan). Returns True on success, False if the file is absent
    (so the caller can fall back to from-scratch). Checkpoint shape matches what
    `pretrain_bilstm.py` saves: {"model_state": state_dict, ...}.
    """
    import os

    import torch

    if not os.path.exists(checkpoint_path):
        return False
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    lstm_only = {k[len("lstm."):]: v for k, v in state.items() if k.startswith("lstm.")}
    if not lstm_only:
        return False
    model.lstm.load_state_dict(lstm_only)   # strict — the encoder must match exactly
    return True
