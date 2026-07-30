"""Stage 0 — pretrain the behavioral Bi-LSTM ENCODER on public human data.

Transfer learning (ml-training-guide-behavioral.md §3): pretrain the shared Bi-LSTM on real
human keyboard/mouse interaction data (EmoSurv + DUX) on a BINARY PROXY task
(high-arousal/emotion vs low-arousal/neutral), then fine-tune the real 4-class head on Phase A
data in `train_bilstm.py`. Public emotion labels do NOT map to the 4 learning-affect states, so
we deliberately train a proxy — the goal is for the LSTM to learn the temporal structure of
human input, not to predict the public labels.

Saves ONLY a reusable checkpoint (`pretrain.checkpoint`) whose LSTM weights `train_bilstm.py`
loads via `model.load_pretrained_lstm`. The reported thesis result is the fine-tuned model, not
anything here. See DATASET_CARD.md for the honesty rules + dataset coverage gaps (no scroll;
EmoSurv has no mouse).

Usage:  python pretrain_bilstm.py [--config config.yaml]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import WindowDataset, windows_to_arrays, split_by_participant, fit_zscore, apply_zscore
from external_datasets import load_external_windows, PROXY_LABELS
from feature_engineering import N_FEATURES
from model import build_model


def _evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return logits.argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pcfg = cfg["pretrain"]
    seed = cfg["data"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- public data -> proxy-labeled windows (reuses the SHARED extract_features) ----
    windows = load_external_windows(cfg)
    if not windows:
        raise SystemExit(
            "No external pretraining windows. Download DUX (Zenodo) and/or EmoSurv (IEEE "
            "DataPort) into the `external:` paths in config.yaml — see DATASET_CARD.md.")
    print(f"=== PRETRAIN (proxy={pcfg.get('proxy','arousal')}): {len(windows)} windows, "
          f"{len({w['participant'] for w in windows})} participants ===")

    # windows carry `events`; windows_to_arrays runs extract_features -> (N, 30, 13).
    for w in windows:                         # windows_to_arrays keys on label_idx
        w["label_idx"] = int(w["proxy_label"])
    X, y, pid = windows_to_arrays(windows)
    print("Feature tensor:", X.shape, "| proxy class counts:", np.bincount(y).tolist())

    tr, va, te, groups = split_by_participant(pid, seed)
    mean, std = fit_zscore(X[tr])
    Xtr, Xva = apply_zscore(X[tr], mean, std), apply_zscore(X[va], mean, std)
    ytr, yva = y[tr], y[va]

    counts = np.bincount(ytr, minlength=len(PROXY_LABELS)).astype(np.float64)
    w_ = counts.sum() / np.maximum(counts, 1)
    class_w = torch.tensor((w_ / w_.sum() * len(PROXY_LABELS)), dtype=torch.float32, device=device)

    model = build_model(cfg["model"], N_FEATURES, n_classes=len(PROXY_LABELS)).to(device)
    print("Encoder params:", f"{sum(p.numel() for p in model.lstm.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    crit = nn.CrossEntropyLoss(weight=class_w)
    loader = DataLoader(WindowDataset(Xtr, ytr), batch_size=cfg["training"]["batch_size"], shuffle=True)

    best_f1, best_state = -1.0, None
    for epoch in range(1, int(pcfg.get("epochs", 40)) + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        vf1 = f1_score(yva, _evaluate(model, Xva, yva, device), average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state = vf1, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d}  loss={loss.item():.3f}  proxy val_macroF1={vf1:.3f}")

    model.load_state_dict(best_state)
    out = Path(__file__).parent / pcfg["checkpoint"]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "proxy": pcfg.get("proxy", "arousal"),
                "n_features": N_FEATURES, "note": "encoder pretrained on EmoSurv+DUX proxy task"},
               out)
    print(f"saved pretrained encoder -> {out.resolve()}  (best proxy val macroF1={best_f1:.3f})")
    print("PRETRAIN DONE")


if __name__ == "__main__":
    main()
