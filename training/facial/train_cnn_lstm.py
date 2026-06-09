"""CNN-LSTM training on DAiSEE (FR13, FR23).

Usage (from the training/facial/ directory):
    python train_cnn_lstm.py --config config.yaml

The script:
  1. Loads pre-processed .npy clips produced by preprocess.py
  2. Trains a CNN-LSTM with focal loss, AdamW, cosine LR, and mixed precision
  3. Saves the best checkpoint (by val weighted-F1) to checkpoints/
  4. Exports the best model to ONNX after training
"""

import sys
import os
import argparse
import logging
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score

# allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))
from model   import build_model
from losses  import FocalLoss
from dataset import DAiSEEDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    """Return (loss, weighted_f1) on loader."""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            total_loss += criterion(logits, labels).item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    n = len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / n, f1


def export_onnx(model, cfg, out_path: str, device):
    """Export the model to ONNX for backend handoff."""
    model.eval()
    T = cfg["model"]["frames_per_clip"]
    H = cfg["model"]["input_size"]
    dummy = torch.randn(1, T, 3, H, H, device=device)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["clip"],
        output_names=["logits"],
        dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    log.info("ONNX exported → %s", out_path)


# ── main training loop ────────────────────────────────────────────────────────

def train(cfg: dict):
    paths   = cfg["paths"]
    mcfg    = cfg["model"]
    tcfg    = cfg["training"]
    target  = cfg.get("target_affect", "Engagement")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── datasets ──────────────────────────────────────────────────────────────
    labels_dir = Path(paths["labels_dir"])
    pre_dir    = paths["daisee_preprocessed"]

    train_ds = DAiSEEDataset(pre_dir, labels_dir / "TrainLabels.csv",
                             split="Train", target=target, augment=True)
    val_ds   = DAiSEEDataset(pre_dir, labels_dir / "ValidationLabels.csv",
                             split="Validation", target=target, augment=False)

    log.info("Train clips: %d  |  Val clips: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"],
                              shuffle=True,  num_workers=tcfg["num_workers"],
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=tcfg["batch_size"],
                              shuffle=False, num_workers=tcfg["num_workers"],
                              pin_memory=True)

    # ── model + loss ──────────────────────────────────────────────────────────
    model     = build_model(mcfg).to(device)
    alpha     = train_ds.class_weights().to(device)
    criterion = FocalLoss(gamma=tcfg["focal_gamma"], alpha=alpha)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{total_params:,}")

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=tcfg["epochs"]
    )
    scaler = GradScaler(enabled=tcfg["mixed_precision"] and device.type == "cuda")

    # ── checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = Path(paths["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "cnn_lstm_best.pt"

    best_f1      = 0.0
    patience_cnt = 0
    patience     = tcfg["early_stopping_patience"]

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips, labels = clips.to(device), labels.to(device)

            optimiser.zero_grad()
            with autocast(enabled=tcfg["mixed_precision"] and device.type == "cuda"):
                logits = model(clips)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            # unscale BEFORE grad clipping (mixed-precision requirement)
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_f1 = evaluate(model, val_loader, device)

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_f1=%.4f  lr=%.2e",
            epoch, tcfg["epochs"], avg_train_loss, val_loss, val_f1,
            optimiser.param_groups[0]["lr"],
        )

        # ── checkpoint ────────────────────────────────────────────────────────
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "val_f1":      val_f1,
                "cfg":         cfg,
            }, best_ckpt)
            log.info("  ✓ New best saved (val_f1=%.4f)", best_f1)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)",
                         epoch, patience)
                break

    log.info("Training complete. Best val F1: %.4f", best_f1)

    # ── ONNX export ───────────────────────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    onnx_path = str(ckpt_dir / "cnn_lstm_best.onnx")
    export_onnx(model, cfg, onnx_path, device)

    return best_f1


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-LSTM on DAiSEE")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    train(cfg)
