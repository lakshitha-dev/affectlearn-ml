"""Measure the train/serve preprocessing gap for the facial engagement model.

Evaluates the canonical CNN-LSTM (`cnn_lstm_best.pt`) on the DAiSEE Test set
preprocessed TWO ways and reports the weighted-F1 difference:

  * HAAR : OpenCV Haar cascade + 10% padding + centre-crop fallback
           == what the model was TRAINED on.
  * MP   : MediaPipe blaze_face_short_range, raw bbox (no pad), drop faceless
           == what the live browser SENDS at serve time (preprocess.ts / AC4).

The HAAR number reproduces the reported 0.509 (matched train/test conditions);
the MP number is the real production accuracy. The gap is what we lose today.

Fully self-contained: inlines the model arch + both croppers so it does not
depend on the version of training/facial/*.py present on the runtime.

Run (after Drive is mounted):
    python crop_gap_eval.py
Outputs JSON + a human summary to  <DRIVE>/eval_crop_gap/ .
"""

import os
import csv
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# ── paths (Google Drive) ──────────────────────────────────────────────────────
DRIVE   = "/content/drive/MyDrive/affectlearn-ml"
RAW_DIR = f"{DRIVE}/data/DAiSEE"                       # has DataSet/Test/...
ZIP     = f"{DRIVE}/data/DAiSEE.zip"                   # fallback if not extracted
LABELS  = f"{DRIVE}/data/DAiSEE/Labels/TestLabels.csv"
CKPT    = f"{DRIVE}/models/cnn_lstm_best.pt"
TFLITE  = f"{DRIVE}/models/blaze_face_short_range.tflite"
OUT_DIR = f"{DRIVE}/eval_crop_gap"
TARGET  = "Engagement"

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

LABEL_COLS = ("Boredom", "Engagement", "Confusion", "Frustration")


# ── model (inlined from training/facial/model.py) ─────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.MaxPool2d(2))

    def forward(self, x):
        return self.net(x)


class FrameCNN(nn.Module):
    def __init__(self, out_dim=512, backbone="scratch"):
        super().__init__()
        if backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
            self.net = net
            self.proj = nn.Identity() if out_dim == feat_dim else nn.Linear(feat_dim, out_dim)
        elif backbone == "scratch":
            self.net = nn.Sequential(
                ConvBlock(3, 32), ConvBlock(32, 64), ConvBlock(64, 128),
                ConvBlock(128, 256), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
            self.proj = nn.Linear(256, out_dim)
        else:
            raise ValueError(backbone)

    def forward(self, x):
        return self.proj(self.net(x))


class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=4, cnn_out_dim=512, hidden_size=256,
                 num_layers=1, dropout=0.5, backbone="scratch"):
        super().__init__()
        self.cnn = FrameCNN(out_dim=cnn_out_dim, backbone=backbone)
        self.lstm = nn.LSTM(cnn_out_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = self.cnn(x.view(B * T, C, H, W)).view(B, T, -1)
        _, (h_n, _) = self.lstm(feats)
        return self.head(self.dropout(h_n[-1]))


def build_model(mcfg):
    return CNNLSTMModel(
        num_classes=mcfg.get("num_classes", 4),
        cnn_out_dim=mcfg.get("cnn_out_dim", 512),
        hidden_size=mcfg.get("hidden_size", 256),
        num_layers=mcfg.get("num_layers", 1),
        dropout=mcfg.get("dropout", 0.5),
        backbone=mcfg.get("backbone", "scratch"))


# ── shared frame helpers ──────────────────────────────────────────────────────
def _read_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None
    idxs = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames or None


def _normalize(frame_uint8):
    x = frame_uint8.astype(np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


# ── HAAR cropper (the OLD / training pipeline) ────────────────────────────────
_cascade = None


def _haar():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _cascade


def _haar_crop(frame_rgb, size=INPUT_SIZE):
    h, w = frame_rgb.shape[:2]
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = _haar().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        px, py = int(fw * 0.10), int(fh * 0.10)
        x1, y1 = max(0, x - px), max(0, y - py)
        x2, y2 = min(w, x + fw + px), min(h, y + fh + py)
        if x2 > x1 and y2 > y1:
            return cv2.resize(frame_rgb[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_LINEAR)
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return cv2.resize(frame_rgb[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_LINEAR)


def haar_clip(video_path):
    frames = _read_frames(video_path, FRAMES_PER_CLIP)
    if frames is None or len(frames) < FRAMES_PER_CLIP:
        return None
    return np.stack([_normalize(_haar_crop(f)) for f in frames[:FRAMES_PER_CLIP]])


# ── MEDIAPIPE cropper (the NEW / serve pipeline) ──────────────────────────────
_detector = None


def _ensure_tflite():
    p = Path(TFLITE)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_DETECTOR_URL, str(p))
    return str(p)


def _mp_detector():
    global _detector
    if _detector is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_ensure_tflite()),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=MIN_CONF)
        _detector = vision.FaceDetector.create_from_options(opts)
    return _detector


def _mp_crop(frame_rgb, size=INPUT_SIZE):
    import mediapipe as mp
    h, w = frame_rgb.shape[:2]
    res = _mp_detector().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb)))
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


def mp_clip(video_path):
    frames = _read_frames(video_path, CANDIDATE_FRAMES)
    if frames is None:
        return None
    crops = [c for c in (_mp_crop(f) for f in frames) if c is not None]
    if len(crops) < FRAMES_PER_CLIP:
        return None
    sel = np.linspace(0, len(crops) - 1, FRAMES_PER_CLIP, dtype=int)
    return np.stack([_normalize(crops[i]) for i in sel])


# ── data enumeration ──────────────────────────────────────────────────────────
def load_labels():
    m = {}
    with open(LABELS, newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}
            m[row["ClipID"].strip()] = int(row[TARGET])
    return m


def iter_test_clips():
    """Yield (clip_id, video_path) for the Test split.

    Prefers an extracted DataSet/Test/ tree; falls back to streaming each .avi
    out of DAiSEE.zip to a temp file (cleaned up after the consumer is done).
    """
    split = Path(RAW_DIR) / "DataSet" / "Test"
    if split.is_dir():
        for subj in sorted(split.iterdir()):
            if not subj.is_dir():
                continue
            for clip in sorted(subj.iterdir()):
                if not clip.is_dir():
                    continue
                v = clip / (clip.name + ".avi")
                if v.exists():
                    yield v.name, str(v)
        return

    import tempfile
    import zipfile
    prefix = "DAiSEE/DataSet/Test/"
    with zipfile.ZipFile(ZIP, "r") as zf:
        entries = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".avi")]
        print(f"zip Test entries: {len(entries)}", flush=True)
        for entry in entries:
            cid = Path(entry).name
            try:
                data = zf.read(entry)
            except Exception as e:
                print("skip unreadable zip entry", entry, e, flush=True)
                continue
            with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
                tmp.write(data)
                tp = tmp.name
            try:
                yield cid, tp
            finally:
                try:
                    os.unlink(tp)
                except OSError:
                    pass


# ── eval ──────────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    return {
        "n": len(y_true),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "per_class_f1": [round(float(x), 4) for x in
                         f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)],
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
    }


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    ckpt = torch.load(CKPT, map_location=device)
    mcfg = ckpt.get("cfg", {}).get("model", {"backbone": "resnet18"})
    model = build_model(mcfg).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print("model loaded; reported val_f1 =", ckpt.get("val_f1"), "| backbone =", mcfg.get("backbone"), flush=True)

    labels = load_labels()
    print(f"labels loaded: {len(labels)}", flush=True)

    rows = []          # (label, haar_pred|None, mp_pred|None)
    mp_dropped = 0
    haar_dropped = 0
    seen = 0
    t0 = time.time()

    @torch.no_grad()
    def predict(arr):
        x = torch.from_numpy(arr.copy()).unsqueeze(0).to(device)  # (1,T,3,96,96)
        return int(model(x).argmax(1).item())

    for cid, path in iter_test_clips():
        if cid not in labels:
            continue
        seen += 1
        y = labels[cid]
        h = haar_clip(path)
        m = mp_clip(path)
        hp = predict(h) if h is not None else None
        mp_ = predict(m) if m is not None else None
        if h is None:
            haar_dropped += 1
        if m is None:
            mp_dropped += 1
        rows.append((y, hp, mp_))
        if seen % 100 == 0:
            print(f"  {seen} clips  ({time.time()-t0:.0f}s)", flush=True)

    # full-set per method
    haar_full = [(y, p) for y, p, _ in rows if p is not None]
    mp_full   = [(y, p) for y, _, p in rows if p is not None]
    # common set: both methods produced a prediction
    common    = [(y, hp, mp_) for y, hp, mp_ in rows if hp is not None and mp_ is not None]

    result = {
        "ckpt": CKPT,
        "reported_val_f1": ckpt.get("val_f1"),
        "target": TARGET,
        "total_labelled_clips": len(rows),
        "haar_dropped": haar_dropped,
        "mp_dropped_no_face": mp_dropped,
        "haar_full_set": metrics([y for y, _ in haar_full], [p for _, p in haar_full]),
        "mp_full_set":   metrics([y for y, _ in mp_full],   [p for _, p in mp_full]),
        "common_set_haar": metrics([y for y, _, _ in common], [hp for _, hp, _ in common]),
        "common_set_mp":   metrics([y for y, _, _ in common], [mp_ for _, _, mp_ in common]),
    }
    h_f1 = result["common_set_haar"]["weighted_f1"]
    m_f1 = result["common_set_mp"]["weighted_f1"]
    result["gap_weighted_f1_haar_minus_mp"] = round(h_f1 - m_f1, 4)

    out_json = Path(OUT_DIR) / "crop_gap_result.json"
    out_json.write_text(json.dumps(result, indent=2))

    print("\n================ CROP GAP RESULT ================", flush=True)
    print(f"Common clips (both croppers OK): {len(common)}", flush=True)
    print(f"MediaPipe dropped (no face)    : {mp_dropped}", flush=True)
    print(f"HAAR  (train cond.) wF1 : {h_f1}", flush=True)
    print(f"MP    (serve cond.) wF1 : {m_f1}", flush=True)
    print(f"GAP (Haar - MP)         : {result['gap_weighted_f1_haar_minus_mp']}", flush=True)
    print(f"JSON -> {out_json}", flush=True)
    print("=================================================", flush=True)


if __name__ == "__main__":
    main()
