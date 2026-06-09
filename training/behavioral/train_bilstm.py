"""Train the behavioral affect Bi-LSTM.

Currently runs on SYNTHETIC data (config data.use_synthetic: true) to validate the
pipeline and produce a demo model. When Phase A data exists, set use_synthetic: false
and point at phase_a/{train,val,test}.pkl (load_phase_a stub below).

Usage:  python train_bilstm.py [--config config.yaml]
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import FEATURE_NAMES, N_FEATURES
from synthetic_data import generate_dataset, LABELS
from model import build_model
from dataset import (WindowDataset, windows_to_arrays, split_by_participant,
                     fit_zscore, apply_zscore)
import data_quality


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return logits.argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = cfg["data"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = cfg["labels"]

    # ---- data ----
    if cfg["data"]["use_synthetic"]:
        print("=== SYNTHETIC data (pipeline validation / demo) ===")
        windows = generate_dataset(cfg["data"]["synthetic_participants"], seed)
    else:
        raise SystemExit("Phase A data path not implemented yet — set use_synthetic: true.")

    X, y, pid = windows_to_arrays(windows)
    print("Feature tensor:", X.shape, "(windows, timesteps, features)")
    print("Data quality:")
    data_quality.report(y, pid, labels)

    tr, va, te, groups = split_by_participant(pid, seed)
    print(f"Split by participant -> train {groups[0]} | val {groups[1]} | test {groups[2]}")

    mean, std = fit_zscore(X[tr])
    Xtr, Xva, Xte = (apply_zscore(X[m], mean, std) for m in (tr, va, te))
    ytr, yva, yte = y[tr], y[va], y[te]

    # ---- class weights (engaged dominates) ----
    counts = np.bincount(ytr, minlength=len(labels)).astype(np.float64)
    w = (counts.sum() / np.maximum(counts, 1)) if cfg["training"]["class_weighting"] else np.ones(len(labels))
    w = w / w.sum() * len(labels)
    class_w = torch.tensor(w, dtype=torch.float32, device=device)

    model = build_model(cfg["model"], N_FEATURES).to(device)
    print("Model params:", f"{sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    crit = nn.CrossEntropyLoss(weight=class_w)
    loader = DataLoader(WindowDataset(Xtr, ytr), batch_size=cfg["training"]["batch_size"], shuffle=True)

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        val_pred = evaluate(model, Xva, yva, device)
        vf1 = f1_score(yva, val_pred, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, patience = vf1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d}  loss={loss.item():.3f}  val_macroF1={vf1:.3f}")
        if patience >= cfg["training"]["early_stopping_patience"]:
            print(f"  early stop @ epoch {epoch}"); break

    model.load_state_dict(best_state)

    # ---- test eval (held-out participants) ----
    pred = evaluate(model, Xte, yte, device)
    print("\n=== TEST (held-out participants) ===")
    print(f"  accuracy={ (pred==yte).mean():.3f}  weighted F1={f1_score(yte,pred,average='weighted',zero_division=0):.3f}"
          f"  macro F1={f1_score(yte,pred,average='macro',zero_division=0):.3f}")
    print(classification_report(yte, pred, labels=list(range(len(labels))),
                                target_names=labels, zero_division=0))
    print("confusion (rows=true):\n", confusion_matrix(yte, pred, labels=list(range(len(labels)))))
    pid_te = pid[te]
    print("per-participant accuracy:",
          {p: round(float((pred[pid_te == p] == yte[pid_te == p]).mean()), 2) for p in sorted(set(pid_te))})

    # ---- save artifacts ----
    out = Path(__file__).parent / cfg["paths"]["models"]
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg,
                "feature_names": FEATURE_NAMES, "labels": labels}, out / "behavioral_bilstm_best.pt")
    json.dump({"feature_names": FEATURE_NAMES, "mean": mean.tolist(), "std": std.tolist()},
              open(out / "behavioral_feature_stats.json", "w"), indent=2)
    dummy = torch.zeros(1, X.shape[1], N_FEATURES, device=device)
    torch.onnx.export(model, dummy, str(out / "behavioral_bilstm.onnx"), dynamo=False,
                      input_names=["window"], output_names=["logits"],
                      dynamic_axes={"window": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    print("\nsaved: behavioral_bilstm_best.pt / .onnx / behavioral_feature_stats.json ->", out.resolve())
    print("BEHAVIORAL DONE")


if __name__ == "__main__":
    main()
