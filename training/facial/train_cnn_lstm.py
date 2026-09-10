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
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

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

def evaluate(model, loader, device, return_metrics: bool = False):
    """Return (loss, weighted_f1), or (loss, weighted_f1, metrics) when asked.

    Weighted F1 alone is why no engagement run in this project ever had an AUC or a kappa:
    this function computed only CE loss and weighted F1, and neither `crop_gap_eval.py` nor
    `mediapipe_retrain_pipeline.py` filled the gap. On a corpus where one class holds 95% of
    the clips, weighted F1 is dominated by the majority class and a model that never predicts
    the minority still scores well -- so the metrics that decide whether a detector is usable
    were never available at training time. They are now, alongside the majority baseline that
    every accuracy must be read against.

    `probs` is the positive-class probability for a 2-class head, else the full softmax, so a
    caller can dump per-clip probabilities and recompute anything later. Nothing in this repo
    previously produced such a dump for a facial run.
    """
    model.eval()
    all_preds, all_labels, all_probs, total_loss = [], [], [], 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            total_loss += criterion(logits, labels).item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
    n = len(all_labels)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    if not return_metrics:
        return total_loss / n, f1

    y = np.asarray(all_labels)
    pred = np.asarray(all_preds)
    probs = np.asarray(all_probs)
    n_classes = probs.shape[1]
    counts = np.bincount(y, minlength=n_classes)

    m = {
        "n": int(n),
        "n_classes": int(n_classes),
        "class_counts": counts.tolist(),
        "accuracy": float((pred == y).mean()),
        # The number every accuracy must be quoted beside.
        "majority_baseline": float(counts.max() / n),
        "weighted_f1": float(f1),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
    }
    if n_classes == 2:
        m["positive_rate"] = float(counts[1] / n)
        m["positive_recall"] = float((pred[y == 1] == 1).mean()) if counts[1] else None
        m["positive_precision"] = float((y[pred == 1] == 1).mean()) if (pred == 1).any() else None
        # AUC is undefined on a single-class split, and an AUC over a handful of negatives is
        # noise rather than skill -- the reason the engagement ANY-cut figure of 0.7942, taken
        # over four negatives, is disowned in reports/facial_confusion/FINDINGS.md.
        m["auc"] = float(roc_auc_score(y, probs[:, 1])) if len(np.unique(y)) == 2 else None
        m["min_class_n"] = int(counts.min())
    else:
        m["per_class_f1"] = f1_score(y, pred, average=None, zero_division=0).tolist()
    return total_loss / n, f1, m


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

    binary_cut = cfg.get("binary_cut") or None

    train_ds = DAiSEEDataset(pre_dir, labels_dir / "TrainLabels.csv",
                             split="Train", target=target, binary_cut=binary_cut,
                             augment=True)
    val_ds   = DAiSEEDataset(pre_dir, labels_dir / "ValidationLabels.csv",
                             split="Validation", target=target, binary_cut=binary_cut,
                             augment=False)

    # The head must match the label space, and the config's `num_classes` predates binary
    # support -- deriving it here means the two cannot disagree.
    mcfg = {**mcfg, "num_classes": train_ds.num_classes}

    log.info("Train clips: %d  |  Val clips: %d  |  target %s  |  cut %s  |  %d classes",
             len(train_ds), len(val_ds), target, binary_cut or "none (4-level)",
             train_ds.num_classes)
    # The class distribution the run actually trained on. Never recorded for engagement
    # before: check_daisee_labels.py can produce it but was only ever invoked for Confusion.
    import collections as _c
    log.info("Train label counts: %s | Val label counts: %s",
             dict(sorted(_c.Counter(train_ds.labels).items())),
             dict(sorted(_c.Counter(val_ds.labels).items())))

    if tcfg.get("balanced_sampling", False):
        labels_arr  = np.array(train_ds.labels)
        class_count = np.bincount(labels_arr, minlength=train_ds.num_classes).astype(np.float64)
        per_class_w = 1.0 / np.sqrt(np.maximum(class_count, 1.0))   # sqrt-balanced
        sample_w    = per_class_w[labels_arr]
        sampler     = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(sample_w, dtype=torch.double), len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"],
                                  sampler=sampler, num_workers=tcfg["num_workers"],
                                  pin_memory=True)
        log.info("Class-balanced sampling ON (sqrt). Class counts: %s", class_count.astype(int).tolist())
    else:
        train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"],
                                  shuffle=True,  num_workers=tcfg["num_workers"],
                                  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=tcfg["batch_size"],
                              shuffle=False, num_workers=tcfg["num_workers"],
                              pin_memory=True)

    # ── model + loss ──────────────────────────────────────────────────────────
    model     = build_model(mcfg).to(device)

    # anti-overfit: freeze pretrained backbone (fully, or all-but-last-block)
    if mcfg.get("backbone") == "resnet18" and tcfg.get("freeze_backbone_full", False):
        for p in model.cnn.net.parameters():
            p.requires_grad_(False)
        log.info("Froze ENTIRE backbone (train LSTM + head only)")
    elif mcfg.get("backbone") == "resnet18" and tcfg.get("freeze_backbone", False):
        frozen = 0
        for name, p in model.cnn.net.named_parameters():
            if not name.startswith("layer4"):
                p.requires_grad_(False); frozen += 1
        log.info("Froze %d backbone tensors (layer4 + LSTM + head trainable)", frozen)

    # with balanced sampling the loss must NOT also re-weight by class (double-correction)
    ls = tcfg.get("label_smoothing", 0.0)
    if tcfg.get("balanced_sampling", False):
        criterion = FocalLoss(gamma=tcfg["focal_gamma"], alpha=None, label_smoothing=ls)
    else:
        criterion = FocalLoss(gamma=tcfg["focal_gamma"], alpha=train_ds.class_weights().to(device), label_smoothing=ls)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{total_params:,}")

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=tcfg["epochs"]
    )
    scaler = GradScaler(enabled=tcfg["mixed_precision"] and device.type == "cuda")

    # ── checkpoint directory ──────────────────────────────────────────────────
    # Checkpoint names are keyed by TARGET AFFECT. They used to be the fixed
    # "cnn_lstm_best.pt"/".onnx", which meant training any non-Engagement target silently
    # OVERWROTE the deployed Engagement model on Drive — including the 47 MB ONNX the backend
    # serves. Engagement keeps the historic filenames so existing artifacts and
    # `behavioral_inference`/`predict.py` paths still resolve; every other target gets its own
    # name, matching the convention `predict_camera.py` already expects
    # (cnn_lstm_confusion.pt, cnn_lstm_boredom.pt, cnn_lstm_frustration.pt).
    ckpt_dir = Path(paths["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # The binary cut is part of the artifact's identity, for the same reason the target is.
    # `Engagement` keeps the historic `cnn_lstm_best` name ONLY for the 4-level model the
    # backend serves; a binary engagement run is a different model answering a different
    # question, and writing it to that name would silently replace the deployed artifact --
    # the exact failure this block was written to prevent, one level deeper. The deployed
    # confusion model already follows this convention: `cnn_lstm_confusion_anycut.onnx`.
    stem = "cnn_lstm_best" if target == "Engagement" else f"cnn_lstm_{target.lower()}"
    if binary_cut:
        stem = f"cnn_lstm_{target.lower()}_{binary_cut}cut"
    best_ckpt = ckpt_dir / f"{stem}.pt"
    if best_ckpt.exists():
        log.warning("checkpoint %s already exists and WILL be overwritten if this run beats "
                    "val_f1=0; move it aside first if you need to keep it", best_ckpt)
    log.info("target_affect=%s -> checkpoint %s", target, best_ckpt.name)

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
                # Stamped so a loaded checkpoint can never be mistaken for a different target.
                # `val_f1` is the score the checkpoint was SELECTED on and is optimistically
                # biased; the reportable number comes from the held-out Test split.
                "target_affect": target,
                "provenance": {"selected_on": "Validation weighted-F1", "reportable": False},
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
    onnx_path = str(ckpt_dir / f"{stem}.onnx")
    export_onnx(model, cfg, onnx_path, device)

    return best_f1


# ── entry point ───────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Make a run repeatable, which until now it was not.

    Every checkpoint this script has produced came from an unseeded run: no `manual_seed`, no numpy
    seed, and a `WeightedRandomSampler` feeding multiple loader workers on top. That is why the
    thesis has to state that the served facial models are not bit-reproducible, and it is worth
    fixing before a new corpus inherits the same defect rather than after.

    `cudnn.benchmark` is turned off alongside: left on, it picks convolution algorithms by timing
    them, so two runs of identical code can take different numerical paths.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-LSTM on DAiSEE")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed everything for a repeatable run. Overrides training.seed in the "
                             "config. Omit both and the run is unseeded, as all prior runs were.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed")
    if seed is not None:
        set_seed(int(seed))
        logging.info("seeded run: seed=%d, cudnn deterministic", int(seed))
    else:
        logging.warning("UNSEEDED run - this checkpoint will not be bit-reproducible. "
                        "Pass --seed or set training.seed in the config.")
    train(cfg)
