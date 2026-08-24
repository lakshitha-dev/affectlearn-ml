"""MediaPipe BlazeFace face-crop generator — the geometry the DEPLOYED facial model was trained on.

## Why this file exists

`cnn_lstm_confusion_anycut.onnx` (AUC 0.6414, DAiSEE ANY-cut) was produced in Colab by a script of
this name that was **never committed**. Its only trace was a line in
`reports/facial_confusion/FINDINGS.md`:

    Replaced with MediaPipe BlazeFace (`crops_mp.py`) at identical geometry -- 10% padding,
    largest detection, centre-crop fallback.

Meanwhile the committed `training/facial/preprocess.py` documents RAW-BBOX cropping with no padding
and frames dropped when no face is found, and claims byte-for-byte parity with the browser. Both
statements cannot be true of the same artifact, and the model card
(`backend/models/cnn_lstm_confusion_anycut.json`) sides with this file:

    "preprocessing": "MediaPipe BlazeFace crop, 10% pad, largest face, centre-crop fallback;
                      ImageNet-normalised CHW float32"

The consequence of the gap was measurable: served with raw boxes, a face filled ~100% of the 96x96
input where training had it fill ~69%. Because the ResNet18 backbone is FULLY FROZEN, the LSTM+head
had no adapted response to the shifted framing and live P(confused) collapsed into a ~0.05 band
around 0.5 -- while the same weights spread properly on their own test set (413/1638 predicted
positive). Nothing in CI could catch it, because the geometry lived only in an uncommitted file.

So this module is committed for one reason: to be the versioned, authoritative statement of the
crop geometry, mirrored by `frontend/src/lib/preprocess.ts` and pinned by
`frontend/src/lib/preprocess.parity.test.ts`.

The geometry below is transcribed from `evaluation/crop_gap_eval.py::_haar_crop`, which is the
implementation `crops_mp.py` replicated. Only the DETECTOR differs (BlazeFace vs Haar); the
padding, selection rule and fallback are identical.

## Contract

    crop:      largest detection, expanded 10% of w/h on EACH side, clamped to frame
    fallback:  largest centred square when no face is detected (frame is KEPT, never dropped)
    resize:    cv2.INTER_LINEAR to 96x96
    normalise: /255, then ImageNet mean/std, then HWC -> CHW float32

Any change here must be mirrored in `preprocess.ts` and the parity fixture, or the two drift again.
"""

from __future__ import annotations

import numpy as np

INPUT_SIZE = 96
BOX_PAD_FRACTION = 0.10  # per side; 1.44x area
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_DETECTION_CONFIDENCE = 0.5


def pad_box(x: float, y: float, w: float, h: float, frame_w: int, frame_h: int):
    """Expand a detection box 10% per side, clamped to the frame.

    Mirrors `_haar_crop`'s padding exactly, including the clamp-at-zero and clamp-at-extent.
    Returns `(x1, y1, x2, y2)` as ints.
    """
    px, py = int(w * BOX_PAD_FRACTION), int(h * BOX_PAD_FRACTION)
    x1, y1 = max(0, int(x) - px), max(0, int(y) - py)
    x2 = min(frame_w, int(x) + int(w) + px)
    y2 = min(frame_h, int(y) + int(h) + py)
    return x1, y1, x2, y2


def center_crop_box(frame_w: int, frame_h: int):
    """Largest centred square. The fallback for a frame with no detected face.

    Training KEPT these frames rather than dropping them, so clips contained occasional non-face
    frames and the LSTM was fitted over that distribution.
    """
    side = min(frame_h, frame_w)
    y0, x0 = (frame_h - side) // 2, (frame_w - side) // 2
    return x0, y0, x0 + side, y0 + side


def largest_detection(boxes):
    """The largest box by area. Training used `max(faces, key=w*h)`.

    Taking the first detection instead means a bystander can be picked over the learner.
    `boxes` is a sequence of `(x, y, w, h)`.
    """
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def crop_face(frame_rgb: np.ndarray, detector=None) -> np.ndarray:
    """Crop one RGB frame to 96x96 using the deployed model's training geometry.

    `detector` is a MediaPipe `FaceDetector` in IMAGE running mode. When it is None, or finds
    nothing above `MIN_DETECTION_CONFIDENCE`, the centre-crop fallback is used -- the frame is
    never dropped.
    """
    import cv2

    h, w = frame_rgb.shape[:2]
    boxes = []

    if detector is not None:
        import mediapipe as mp

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_img)
        for det in result.detections or []:
            score = det.categories[0].score if det.categories else 0.0
            if score < MIN_DETECTION_CONFIDENCE:
                continue
            bb = det.bounding_box
            boxes.append((bb.origin_x, bb.origin_y, bb.width, bb.height))

    best = largest_detection(boxes)
    if best is not None:
        x1, y1, x2, y2 = pad_box(*best, frame_w=w, frame_h=h)
        if x2 > x1 and y2 > y1:
            return cv2.resize(
                frame_rgb[y1:y2, x1:x2], (INPUT_SIZE, INPUT_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )

    x1, y1, x2, y2 = center_crop_box(w, h)
    return cv2.resize(
        frame_rgb[y1:y2, x1:x2], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR
    )


def normalize(frame_uint8: np.ndarray) -> np.ndarray:
    """uint8 HWC [0,255] -> float32 CHW, ImageNet-normalised.

    Identical to `preprocess.py::normalize` and to `normalizeRgbaToChwFloat32` in the browser --
    normalisation was never the problem; geometry was.
    """
    x = frame_uint8.astype(np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
