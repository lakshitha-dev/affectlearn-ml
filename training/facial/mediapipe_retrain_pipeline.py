"""Consolidated Colab pipeline: re-preprocess DAiSEE with the MediaPipe cropper
(serve-aligned) and retrain the engagement CNN-LSTM in two variants:

  * focal   — current recipe (FocalLoss, 4-way softmax head). The serve-ALIGNED
              baseline: shows how much accuracy plain alignment recovers.
  * corn    — CORN ordinal regression (Shi/Cao/Raschka 2021, 3 rank logits).
              Targets the High<->Very-High boundary that the crop-gap measurement
              showed is the model's dominant error.

Self-contained (inlines model + losses + dataset) so it does not depend on the
version of training/facial/*.py present on the runtime. Writes the preprocessed
.npy cache to LOCAL /content (fast I/O) and durable checkpoints + results to
Drive. Resume-friendly preprocessing (skips existing .npy).

Usage:
    python mediapipe_retrain_pipeline.py            # full run
    python mediapipe_retrain_pipeline.py --smoke 30 # 30 clips/split, 1 epoch/variant
"""

import os
import csv
import json
import time
import argparse
import urllib.request
import tempfile
import zipfile
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# ── paths ─────────────────────────────────────────────────────────────────────
DRIVE   = "/content/drive/MyDrive/affectlearn-ml"
ZIP     = f"{DRIVE}/data/DAiSEE.zip"
LABELS  = f"{DRIVE}/data/DAiSEE/Labels"
TFLITE  = f"{DRIVE}/models/blaze_face_short_range.tflite"
CKPT_DIR = f"{DRIVE}/models"
OUT_JSON = f"{DRIVE}/eval_crop_gap/mediapipe_retrain_result.json"
PRE     = "/content/daisee_mp"            # local fast cache (lost on runtime reset)

SPLITS = {"Train": "TrainLabels.csv",
          "Validation": "ValidationLabels.csv",
          "Test": "TestLabels.csv"}
TARGET = "Engagement"
NUM_CLASSES = 4

FRAMES_PER_CLIP = 16
CANDIDATE_FRAMES = 32
INPUT_SIZE = 96
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_CONF = 0.5
FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# training hyperparams (mirror config.yaml run #3 — the 0.509 recipe)
BATCH = 64
EPOCHS = 40
LR = 2.0e-4
WD = 1.0e-4
FOCAL_GAMMA = 2.0
PATIENCE = 12


# ── MediaPipe crop (serve-aligned, == preprocess.ts / AC4) ────────────────────
def _ensure_tflite():
    p = Path(TFLITE)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_DETECTOR_URL, str(p))
    return str(p)


_DET = None


def _detector():
    global _DET
    if _DET is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        _DET = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_ensure_tflite()),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=MIN_CONF))
    return _DET


def _mp_crop(frame_rgb, size=INPUT_SIZE):
    import mediapipe as mp
    h, w = frame_rgb.shape[:2]
    res = _detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                       data=np.ascontiguousarray(frame_rgb)))
    if not res.detections:
        return None
    det = res.detections[0]
    if det.categories and det.categories[0].score < MIN_CONF:
        return None
    b = det.bounding_box
    x1, y1 = max(0, int(b.origin_x)), max(0, int(b.origin_y))
    x2, y2 = min(w, int(b.origin_x + b.width)), min(h, int(b.origin_y + b.height))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(frame_rgb[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_LINEAR)


def _normalize(frame_uint8):
    x = frame_uint8.astype(np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def _read_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None
    idxs = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out or None


def preprocess_clip(video_path):
    frames = _read_frames(video_path, CANDIDATE_FRAMES)
    if frames is None:
        return None
    crops = [c for c in (_mp_crop(f) for f in frames) if c is not None]
    if len(crops) < FRAMES_PER_CLIP:
        return None
    sel = np.linspace(0, len(crops) - 1, FRAMES_PER_CLIP, dtype=int)
    return np.stack([_normalize(crops[i]) for i in sel])


# ── parallel preprocessing from the zip ───────────────────────────────────────
_ZF = None


def _worker_init():
    global _ZF
    _ZF = zipfile.ZipFile(ZIP, "r")


def _worker_proc(entry, split):
    """Extract one .avi from the zip, MediaPipe-preprocess, save .npy. Returns status."""
    cid = Path(entry).name
    out_path = Path(PRE) / split / (Path(cid).stem + ".npy")
    if out_path.exists():
        return "cached"
    try:
        data = _ZF.read(entry)
    except Exception:
        return "unreadable"
    tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    tmp.write(data)
    tmp.close()
    try:
        arr = preprocess_clip(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if arr is None:
        return "noface"
    np.save(str(out_path), arr)
    return "ok"


def _label_ids(split):
    ids = set()
    with open(f"{LABELS}/{SPLITS[split]}", newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}
            ids.add(Path(row["ClipID"].strip()).stem)
    return ids


def preprocess_split(split, smoke=0):
    (Path(PRE) / split).mkdir(parents=True, exist_ok=True)
    prefix = f"DAiSEE/DataSet/{split}/"
    with zipfile.ZipFile(ZIP, "r") as zf:
        entries = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".avi")]
    if smoke:
        # keep only labelled clips so the smoke set can actually build a dataset
        ids = _label_ids(split)
        entries = [e for e in entries if Path(e).stem in ids][:smoke]
    print(f"[preprocess] {split}: {len(entries)} clips, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    counts = {}
    with Pool(N_WORKERS, initializer=_worker_init) as pool:
        for i, status in enumerate(pool.imap_unordered(partial(_worker_proc, split=split), entries, chunksize=4), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 500 == 0:
                print(f"  {split} {i}/{len(entries)} ({time.time()-t0:.0f}s) {counts}", flush=True)
    print(f"[preprocess] {split} done in {time.time()-t0:.0f}s -> {counts}", flush=True)


# ── dataset ───────────────────────────────────────────────────────────────────
class DAiSEEDataset(Dataset):
    def __init__(self, split, augment=False):
        self.augment = augment
        label_map = {}
        with open(f"{LABELS}/{SPLITS[split]}", newline="") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items()}
                label_map[row["ClipID"].strip()] = int(row[TARGET])
        self.clips, self.labels = [], []
        for npy in sorted((Path(PRE) / split).glob("*.npy")):
            key = npy.stem + ".avi"
            if key in label_map:
                self.clips.append(npy)
                self.labels.append(label_map[key])
        if not self.clips:
            raise RuntimeError(f"no clips for {split} in {PRE}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = np.load(self.clips[idx])
        if self.augment:
            if np.random.rand() < 0.5:
                clip = clip[:, :, :, ::-1].copy()
            clip = np.clip(clip + np.random.uniform(-0.3, 0.3), -3.0, 3.0)
            clip = np.clip(clip * np.random.uniform(0.8, 1.2), -3.0, 3.0)
        return torch.from_numpy(clip.copy()), self.labels[idx]


# ── model (resnet18 + LSTM; head size depends on loss type) ───────────────────
class FrameCNN(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feat = net.fc.in_features
        net.fc = nn.Identity()
        self.net = net
        self.proj = nn.Identity() if out_dim == feat else nn.Linear(feat, out_dim)

    def forward(self, x):
        return self.proj(self.net(x))


class CNNLSTMModel(nn.Module):
    def __init__(self, n_out, cnn_out_dim=512, hidden=256, dropout=0.5):
        super().__init__()
        self.cnn = FrameCNN(cnn_out_dim)
        self.lstm = nn.LSTM(cnn_out_dim, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = self.cnn(x.view(B * T, C, H, W)).view(B, T, -1)
        _, (h_n, _) = self.lstm(feats)
        return self.head(self.dropout(h_n[-1]))


def freeze_to_layer4(model):
    frozen = 0
    for name, p in model.cnn.net.named_parameters():
        if not name.startswith("layer4"):
            p.requires_grad_(False)
            frozen += 1
    return frozen


# ── losses ────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def corn_loss(logits, y, num_classes):
    """CORN ordinal loss (Shi, Cao & Raschka, 2021). logits: (N, num_classes-1)."""
    losses = 0.0
    num_examples = 0
    for task_i in range(num_classes - 1):
        mask = y > (task_i - 1)            # samples still 'in play' for rank task_i
        if mask.sum() < 1:
            continue
        lab = (y[mask] > task_i).float()   # 1 if label exceeds task_i
        pred = logits[mask, task_i]
        num_examples += int(mask.sum())
        losses += -torch.sum(F.logsigmoid(pred) * lab
                             + (F.logsigmoid(pred) - pred) * (1.0 - lab))
    return losses / max(num_examples, 1)


def corn_label_from_logits(logits):
    """Rank-monotonic prediction: count how many cumulative-prob thresholds pass 0.5."""
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    return torch.sum(probas > 0.5, dim=1)


# ── train / eval one variant ──────────────────────────────────────────────────
def evaluate(model, loader, device, loss_type):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for clips, labels in loader:
            logits = model(clips.to(device))
            if loss_type == "corn":
                p = corn_label_from_logits(logits)
            else:
                p = logits.argmax(1)
            preds.extend(p.cpu().numpy())
            gts.extend(labels.numpy())
    return gts, preds


def metrics(gts, preds):
    return {
        "n": len(gts),
        "weighted_f1": round(float(f1_score(gts, preds, average="weighted", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(gts, preds, average="macro", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(gts, preds)), 4),
        "per_class_f1": [round(float(x), 4) for x in
                         f1_score(gts, preds, average=None, labels=[0, 1, 2, 3], zero_division=0)],
        "confusion": confusion_matrix(gts, preds, labels=[0, 1, 2, 3]).tolist(),
    }


def train_variant(loss_type, train_ds, val_ds, test_ds, device, epochs):
    n_out = NUM_CLASSES - 1 if loss_type == "corn" else NUM_CLASSES
    model = CNNLSTMModel(n_out).to(device)
    freeze_to_layer4(model)

    # balanced (sqrt-inverse-freq) sampling — same as the 0.509 recipe
    labels_arr = np.array(train_ds.labels)
    cc = np.bincount(labels_arr, minlength=NUM_CLASSES).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(cc, 1.0))
    sampler = WeightedRandomSampler(torch.as_tensor(w[labels_arr], dtype=torch.double),
                                    len(labels_arr), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    crit = (FocalLoss(FOCAL_GAMMA) if loss_type == "focal"
            else (lambda lo, ta: corn_loss(lo, ta, NUM_CLASSES)))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"[train:{loss_type}] cc={cc.astype(int).tolist()} "
          f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    ckpt = Path(CKPT_DIR) / f"cnn_lstm_mp_{loss_type}_best.pt"
    best_f1, wait, best_test = 0.0, 0, None
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = crit(model(clips), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
        sched.step()
        vg, vp = evaluate(model, val_loader, device, loss_type)
        vf1 = f1_score(vg, vp, average="weighted", zero_division=0)
        print(f"[train:{loss_type}] ep{ep:02d}/{epochs} loss={tl/len(train_loader):.4f} val_wF1={vf1:.4f}", flush=True)
        if vf1 > best_f1:
            best_f1, wait = vf1, 0
            tg, tp = evaluate(model, test_loader, device, loss_type)
            best_test = metrics(tg, tp)
            torch.save({"model_state": model.state_dict(), "val_f1": vf1,
                        "loss_type": loss_type, "test": best_test}, ckpt)
            print(f"  ✓ best val_wF1={vf1:.4f}  test_wF1={best_test['weighted_f1']}", flush=True)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[train:{loss_type}] early stop @ep{ep}", flush=True)
                break
    return {"best_val_wf1": round(best_f1, 4), "test": best_test, "ckpt": str(ckpt)}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    epochs = 1 if args.smoke else EPOCHS

    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    print("workers:", N_WORKERS, "| smoke:", args.smoke, flush=True)

    # Preprocess BEFORE touching CUDA (multiprocessing fork + initialized CUDA = trouble).
    for split in SPLITS:
        preprocess_split(split, smoke=args.smoke)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    train_ds = DAiSEEDataset("Train", augment=True)
    val_ds = DAiSEEDataset("Validation", augment=False)
    test_ds = DAiSEEDataset("Test", augment=False)
    print(f"clips: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    results = {"refs": {"haar_train_haar_serve": 0.509, "haar_train_mp_serve": 0.480}}
    results["mp_focal"] = train_variant("focal", train_ds, val_ds, test_ds, device, epochs)
    results["mp_corn"] = train_variant("corn", train_ds, val_ds, test_ds, device, epochs)

    Path(OUT_JSON).write_text(json.dumps(results, indent=2))
    print("\n================ RETRAIN RESULT ================", flush=True)
    print("REF Haar-train/Haar-serve wF1 : 0.509", flush=True)
    print("REF Haar-train/MP-serve   wF1 : 0.480  (today's production)", flush=True)
    for k in ("mp_focal", "mp_corn"):
        t = results[k]["test"]
        print(f"{k:9s} test wF1={t['weighted_f1']} macroF1={t['macro_f1']} "
              f"acc={t['accuracy']} perclass={t['per_class_f1']}", flush=True)
    print(f"JSON -> {OUT_JSON}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
