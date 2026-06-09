"""Shared preprocessing pipeline.

CRITICAL: This file MUST stay byte-identical to
  affectlearn/backend/app/services/model_inference.py
Any change here must ship with a matching change in the backend.

Pipeline: BGR video -> RGB -> OpenCV face crop -> 96x96 -> ImageNet normalize -> (T,3,H,W) float32
"""

import os
import zipfile
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FRAMES_PER_CLIP = 16
INPUT_SIZE = 96
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_cascade = None

def _get_cascade():
    global _cascade
    if _cascade is not None:
        return _cascade
    xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _cascade = cv2.CascadeClassifier(xml)
    return _cascade


# -- per-frame helpers ---------------------------------------------------------

def extract_frames(video_path: str, num_frames: int = FRAMES_PER_CLIP):
    """Sample num_frames evenly spaced from the clip. Returns (T,H,W,3) uint8 RGB or None."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None

    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < num_frames:
        return None
    return np.stack(frames[:num_frames])


def crop_face(frame_rgb: np.ndarray, detector=None, size: int = INPUT_SIZE) -> np.ndarray:
    """Detect the largest face and crop + resize to size x size.
    Falls back to centre-square crop when no face is detected.
    """
    h, w = frame_rgb.shape[:2]
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24)
    )

    if len(faces) > 0:
        # pick the largest face by area
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(fw * 0.10)
        pad_y = int(fh * 0.10)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)
        if x2 > x1 and y2 > y1:
            return cv2.resize(frame_rgb[y1:y2, x1:x2], (size, size),
                              interpolation=cv2.INTER_LINEAR)

    # fallback: centre-square crop
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return cv2.resize(frame_rgb[y0:y0+side, x0:x0+side], (size, size),
                      interpolation=cv2.INTER_LINEAR)


def normalize(frame_uint8: np.ndarray) -> np.ndarray:
    """uint8 HWC [0,255] -> float32 CHW, ImageNet-normalised."""
    x = frame_uint8.astype(np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


# -- full clip pipeline --------------------------------------------------------

def preprocess_clip(video_path: str):
    """Run the full pipeline on one video clip.

    Returns float32 ndarray of shape (T, 3, H, W), or None on failure.
    """
    frames = extract_frames(video_path)
    if frames is None:
        return None

    processed = [normalize(crop_face(frame)) for frame in frames]
    return np.stack(processed)


# -- batch preprocessing -------------------------------------------------------

def _clips_from_zip(zip_path: str, split: str):
    """Stream (clip_id, tmp_path) pairs by extracting one .avi at a time."""
    import tempfile

    prefix = f"DAiSEE/DataSet/{split}/"
    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".avi")]
        for entry in entries:
            clip_id = Path(entry).name
            try:
                data = zf.read(entry)
            except Exception as e:
                log.warning("Skipping unreadable zip entry %s: %s", entry, e)
                continue
            with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            yield clip_id, tmp_path
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _clips_from_dir(raw_dir: str, split: str):
    """Yield (clip_id, video_path) from an already-extracted DAiSEE directory."""
    split_path = Path(raw_dir) / "DataSet" / split
    for subject_dir in sorted(split_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        for clip_dir in sorted(subject_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            video = clip_dir / (clip_dir.name + ".avi")
            if video.exists():
                yield video.name, str(video)


def run_preprocessing(
    out_dir: str,
    zip_path: str = None,
    raw_dir: str = None,
    splits=("Train", "Validation", "Test"),
    smoke_n: int = 0,
):
    """Preprocess all clips and save each as a .npy file."""
    out_root = Path(out_dir)
    failed = []

    for split in splits:
        split_out = out_root / split
        split_out.mkdir(parents=True, exist_ok=True)

        source = _clips_from_zip(zip_path, split) if zip_path \
            else _clips_from_dir(raw_dir, split)

        count = 0
        for clip_id, video_path in source:
            npy_path = split_out / (Path(clip_id).stem + ".npy")
            if npy_path.exists():
                count += 1
                continue

            arr = preprocess_clip(video_path)
            if arr is None:
                log.warning("Failed: %s", clip_id)
                failed.append(clip_id)
            else:
                np.save(str(npy_path), arr)

            count += 1
            if count % 200 == 0:
                log.info("%s: %d clips done", split, count)
            if smoke_n and count >= smoke_n:
                log.info("Smoke-test done for %s (%d clips)", split, smoke_n)
                break

        log.info("%s finished -- %d clips, %d failed", split, count, len(failed))

    if failed:
        fail_log = out_root / "failed_clips.txt"
        fail_log.write_text("\n".join(failed))
        log.warning("%d clips failed -> %s", len(failed), fail_log)


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, yaml

    p = argparse.ArgumentParser(description="Preprocess DAiSEE clips to .npy")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--smoke", type=int, default=0)
    args = p.parse_args()

    cfg   = yaml.safe_load(open(args.config))
    paths = cfg["paths"]

    zip_p = paths.get("daisee_zip", "")
    raw_p = paths.get("daisee_raw", "")

    run_preprocessing(
        out_dir=paths["daisee_preprocessed"],
        zip_path=zip_p if Path(zip_p).exists() else None,
        raw_dir=raw_p  if Path(raw_p).is_dir() else None,
        smoke_n=args.smoke,
    )
