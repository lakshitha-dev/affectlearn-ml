"""Shared preprocessing pipeline.

CRITICAL — TRAIN/SERVE PREPROCESSING CONTRACT (three files, one behaviour):
  1. THIS file                         affectlearn-ml/training/facial/preprocess.py   (training)
  2. affectlearn/frontend/src/lib/preprocess.ts                                        (browser serve)
  3. affectlearn/backend/app/services/model_inference.py                               (server serve, story 4.4)

All three MUST produce the same 96x96 RGB ImageNet-normalised CHW float32 crop
from the same face, or the model silently misclassifies on inputs it never saw.

Face crops use the SAME detector the browser uses: MediaPipe Face Detection
`blaze_face_short_range`, confidence >= 0.5, the raw detection bounding box with
NO padding, resized to 96x96. Frames with no face (or score < 0.5) are DROPPED,
never centre-cropped — this mirrors `preprocess.ts` (AC4) where low-confidence
frames are skipped rather than zero-padded.

Pipeline: BGR video -> RGB -> MediaPipe face crop (raw bbox, no pad) -> 96x96
          -> ImageNet normalize -> (T,3,H,W) float32
"""

import os
import zipfile
import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FRAMES_PER_CLIP = 16
INPUT_SIZE = 96
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# MediaPipe face detector — identical model + threshold to the browser
# (frontend/src/lib/mediapipe-config.ts / mediapipe-loader.ts).
MIN_DETECTION_CONFIDENCE = 0.5
# Oversample candidate frames so we can keep FRAMES_PER_CLIP *with a detected
# face*. DAiSEE is near-frontal webcam footage so faces are present in almost
# every frame; oversampling only matters for the rare blink/turn/occlusion.
CANDIDATE_FRAMES = 32
# blaze_face_short_range model asset — same URL the browser loads at runtime.
FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
# Where to cache the .tflite locally; overridable via config (paths.face_detector_model).
_DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "blaze_face_short_range.tflite")

_detector = None


def _ensure_model(model_path: str) -> str:
    """Download blaze_face_short_range.tflite if it isn't cached yet."""
    p = Path(model_path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading MediaPipe face detector -> %s", p)
        urllib.request.urlretrieve(FACE_DETECTOR_URL, str(p))
    return str(p)


def _get_detector(model_path: str = None):
    """Lazily create a singleton MediaPipe FaceDetector (IMAGE running mode)."""
    global _detector
    if _detector is not None:
        return _detector

    # Imported lazily so importing this module (e.g. for constants) doesn't
    # require mediapipe to be installed.
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = _ensure_model(model_path or _DEFAULT_MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )
    _detector = vision.FaceDetector.create_from_options(options)
    return _detector


# -- per-frame helpers ---------------------------------------------------------

def extract_frames(video_path: str, num_frames: int = CANDIDATE_FRAMES):
    """Sample num_frames evenly spaced from the clip. Returns (T,H,W,3) uint8 RGB or None.

    Oversamples (CANDIDATE_FRAMES) so the caller can keep FRAMES_PER_CLIP frames
    that actually contain a detected face.
    """
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

    if not frames:
        return None
    return np.stack(frames)


def crop_face(frame_rgb: np.ndarray, size: int = INPUT_SIZE):
    """Detect a face with MediaPipe and crop + resize to size x size.

    Returns the size x size uint8 RGB crop, or None when no face is detected
    with confidence >= MIN_DETECTION_CONFIDENCE (frame is dropped — matching
    the browser, which never centre-crops a faceless frame).

    The raw detection bounding box is used with NO padding, identical to
    `preprocess.ts::cropAndNormalize` (drawImage of the bbox straight to 96x96).
    """
    import mediapipe as mp

    h, w = frame_rgb.shape[:2]
    detector = _get_detector()

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(frame_rgb))
    result = detector.detect(mp_image)
    if not result.detections:
        return None

    det = result.detections[0]  # browser uses detections[0]
    if det.categories and det.categories[0].score < MIN_DETECTION_CONFIDENCE:
        return None

    bbox = det.bounding_box  # origin_x, origin_y, width, height (pixels)
    x1 = max(0, int(bbox.origin_x))
    y1 = max(0, int(bbox.origin_y))
    x2 = min(w, int(bbox.origin_x + bbox.width))
    y2 = min(h, int(bbox.origin_y + bbox.height))
    if x2 <= x1 or y2 <= y1:
        return None

    return cv2.resize(frame_rgb[y1:y2, x1:x2], (size, size),
                      interpolation=cv2.INTER_LINEAR)


def normalize(frame_uint8: np.ndarray) -> np.ndarray:
    """uint8 HWC [0,255] -> float32 CHW, ImageNet-normalised."""
    x = frame_uint8.astype(np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


# -- full clip pipeline --------------------------------------------------------

def preprocess_clip(video_path: str, frames_per_clip: int = FRAMES_PER_CLIP):
    """Run the full pipeline on one video clip.

    Oversamples candidate frames, keeps only those with a detected face, then
    selects `frames_per_clip` of them evenly across the retained set. Returns a
    float32 ndarray of shape (T, 3, H, W), or None when fewer than
    `frames_per_clip` frames contain a detectable face (clip dropped).
    """
    frames = extract_frames(video_path)
    if frames is None:
        return None

    cropped = []
    for frame in frames:
        face = crop_face(frame)
        if face is not None:
            cropped.append(face)

    if len(cropped) < frames_per_clip:
        return None  # too few real-face frames — drop (matches serve semantics)

    # Pick frames_per_clip evenly across the retained (in-order) face crops.
    sel = np.linspace(0, len(cropped) - 1, frames_per_clip, dtype=int)
    processed = [normalize(cropped[i]) for i in sel]
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
    no_face = []

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
                # Distinguish "couldn't read" from "no detectable face" for diagnostics.
                if extract_frames(video_path) is None:
                    log.warning("Failed (unreadable): %s", clip_id)
                    failed.append(clip_id)
                else:
                    log.warning("Dropped (no face): %s", clip_id)
                    no_face.append(clip_id)
            else:
                np.save(str(npy_path), arr)

            count += 1
            if count % 200 == 0:
                log.info("%s: %d clips done", split, count)
            if smoke_n and count >= smoke_n:
                log.info("Smoke-test done for %s (%d clips)", split, smoke_n)
                break

        log.info("%s finished -- %d clips, %d unreadable, %d no-face",
                 split, count, len(failed), len(no_face))

    if failed:
        (out_root / "failed_clips.txt").write_text("\n".join(failed))
        log.warning("%d clips unreadable -> failed_clips.txt", len(failed))
    if no_face:
        (out_root / "no_face_clips.txt").write_text("\n".join(no_face))
        log.warning("%d clips dropped (no detectable face) -> no_face_clips.txt", len(no_face))


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

    # Optional override for the cached detector model path.
    if paths.get("face_detector_model"):
        _DEFAULT_MODEL_PATH = paths["face_detector_model"]

    run_preprocessing(
        out_dir=paths["daisee_preprocessed"],
        zip_path=zip_p if Path(zip_p).exists() else None,
        raw_dir=raw_p  if Path(raw_p).is_dir() else None,
        smoke_n=args.smoke,
    )
