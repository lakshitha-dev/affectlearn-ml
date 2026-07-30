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
from model import build_model, load_pretrained_lstm, freeze_lstm
from dataset import (WindowDataset, windows_to_arrays, split_by_participant,
                     fit_zscore, apply_zscore)
import data_quality


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        return logits.argmax(1).cpu().numpy()


def train_eval(cfg, Xtr, ytr, Xva, yva, Xte, yte, pid_te, labels, device, class_w,
               *, pretrained_ckpt=None, freeze_epochs=0, tag="main"):
    """Build → (optionally load pretrained encoder + freeze) → train → test-eval.

    Returns (model, test_macro_f1). Run once normally, or twice (scratch vs pretrained) for
    the transfer-learning ablation. Prints per-arm metrics + per-participant accuracy.
    """
    model = build_model(cfg["model"], N_FEATURES).to(device)
    src = "scratch"
    if pretrained_ckpt and load_pretrained_lstm(model, str(Path(__file__).parent / pretrained_ckpt)):
        src = "pretrained"
        if freeze_epochs > 0:
            freeze_lstm(model, True)
    print(f"\n[{tag}] init={src}  params={sum(p.numel() for p in model.parameters()):,}"
          + (f"  (encoder frozen for first {freeze_epochs} epochs)"
             if src == "pretrained" and freeze_epochs else ""))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    crit = nn.CrossEntropyLoss(weight=class_w)
    loader = DataLoader(WindowDataset(Xtr, ytr), batch_size=cfg["training"]["batch_size"], shuffle=True)

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        if src == "pretrained" and freeze_epochs and epoch == freeze_epochs + 1:
            freeze_lstm(model, False); print(f"[{tag}] unfroze encoder @ epoch {epoch}")
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        vf1 = f1_score(yva, evaluate(model, Xva, yva, device), average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, patience = vf1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{tag}]  epoch {epoch:2d}  loss={loss.item():.3f}  val_macroF1={vf1:.3f}")
        if patience >= cfg["training"]["early_stopping_patience"]:
            print(f"[{tag}]  early stop @ epoch {epoch}"); break

    model.load_state_dict(best_state)
    pred = evaluate(model, Xte, yte, device)
    tf1 = f1_score(yte, pred, average="macro", zero_division=0)
    print(f"[{tag}] TEST (held-out participants): accuracy={(pred==yte).mean():.3f}  "
          f"weighted F1={f1_score(yte,pred,average='weighted',zero_division=0):.3f}  macro F1={tf1:.3f}")
    print(classification_report(yte, pred, labels=list(range(len(labels))),
                                target_names=labels, zero_division=0))
    print("confusion (rows=true):\n", confusion_matrix(yte, pred, labels=list(range(len(labels)))))
    print("per-participant accuracy:",
          {p: round(float((pred[pid_te == p] == yte[pid_te == p]).mean()), 2) for p in sorted(set(pid_te))})
    return model, tf1


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
        windows = generate_dataset(cfg["data"]["synthetic_participants"], seed,
                                   cfg.get("synthetic", {}))
        noisy = sum(w["label"] != w["true_label"] for w in windows)
        print(f"  label noise: {noisy}/{len(windows)} windows have a wrong self-report")
    else:
        from phase_a import load_phase_a_windows
        export_path = cfg["data"]["phase_a_export"]
        print(f"=== PHASE A real data ({export_path}) ===")
        raw = json.load(open(export_path))
        export_events = raw["items"] if isinstance(raw, dict) and "items" in raw else raw
        windows = load_phase_a_windows(
            export_events, propagate_ms=int(cfg["data"].get("label_propagate_ms", 0)))
        if not windows:
            raise SystemExit(
                f"No labeled windows in {export_path}: need behavioral_affect_detected events "
                "with a `features` payload joined to self_report labels. Is Phase A data present?")
        print(f"  {len(windows)} labeled windows across "
              f"{len({w['participant'] for w in windows})} participants")

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

    # ---- train (transfer-learning aware): pretrain-then-fine-tune + optional ablation ----
    pcfg = cfg.get("pretrain", {})
    ckpt = pcfg.get("checkpoint")
    freeze = int(pcfg.get("freeze_lstm_epochs", 0))
    have_ckpt = bool(ckpt) and (Path(__file__).parent / ckpt).exists()
    use_pre = (not cfg["data"]["use_synthetic"]) and bool(pcfg.get("enabled")) and have_ckpt
    if (not cfg["data"]["use_synthetic"]) and pcfg.get("enabled") and not have_ckpt:
        print(f"  (pretrain enabled but checkpoint '{ckpt}' not found → training from scratch; "
              "run pretrain_bilstm.py first)")
    args_te = (Xtr, ytr, Xva, yva, Xte, yte, pid[te], labels, device, class_w)

    if use_pre and pcfg.get("ablation"):
        print("\n=== ABLATION: from-scratch vs pretrained-then-fine-tuned (same held-out participants) ===")
        _, f1_scratch = train_eval(cfg, *args_te, pretrained_ckpt=None, tag="scratch")
        model, f1_pre = train_eval(cfg, *args_te, pretrained_ckpt=ckpt, freeze_epochs=freeze, tag="pretrained")
        print(f"\n>>> ABLATION: scratch macroF1={f1_scratch:.3f} | pretrained macroF1={f1_pre:.3f} "
              f"| delta={f1_pre - f1_scratch:+.3f}  (report both; deploy pretrained only if delta > 0)")
    else:
        model, _ = train_eval(cfg, *args_te,
                              pretrained_ckpt=(ckpt if use_pre else None),
                              freeze_epochs=(freeze if use_pre else 0),
                              tag=("pretrained" if use_pre else "scratch"))

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
