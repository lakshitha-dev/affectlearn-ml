"""Live webcam multi-affect prediction.

Loads every available DAiSEE affect model (Engagement / Boredom / Confusion /
Frustration) from a models directory and overlays each affect's predicted level
on the live webcam feed, with a face bounding box.

Usage:
    python predict_camera.py --models-dir "G:\\My Drive\\affectlearn-ml\\models"
    python predict_camera.py --models-dir ./models --camera 0

Press 'q' in the video window to quit.
Requires: torch, torchvision (for the resnet18 backbone), opencv-python, numpy.

Note: affect is a slow-changing, clip-level signal — the models see a rolling
window of recent frames, so labels update every ~half-second, not per-frame.
"""

import sys
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model      import build_model                                         # noqa: E402
from preprocess import crop_face, normalize, FRAMES_PER_CLIP, _get_cascade  # noqa: E402

CLASS_NAMES = ["Very Low", "Low", "High", "Very High"]
COLORS = [(0, 0, 255), (0, 165, 255), (0, 200, 0), (0, 255, 0)]  # BGR per level
# affect -> candidate checkpoint filenames (first found wins)
AFFECTS = {
    "Engagement":  ["cnn_lstm_engagement.pt", "cnn_lstm_best.pt"],
    "Boredom":     ["cnn_lstm_boredom.pt"],
    "Confusion":   ["cnn_lstm_confusion.pt"],
    "Frustration": ["cnn_lstm_frustration.pt"],
}
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "models"


def load_models(models_dir: str, device):
    models = {}
    for affect, files in AFFECTS.items():
        for fn in files:
            p = Path(models_dir) / fn
            if p.exists():
                ck = torch.load(p, map_location=device)
                m = build_model(ck["cfg"]["model"]).to(device)
                m.load_state_dict(ck["model_state"]); m.eval()
                models[affect] = m
                break
    return models


def build_clip(frames_rgb, device):
    idx = np.linspace(0, len(frames_rgb) - 1, FRAMES_PER_CLIP, dtype=int)
    clip = np.stack([normalize(crop_face(frames_rgb[i])) for i in idx])
    return torch.from_numpy(clip).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser(description="Live webcam multi-affect prediction")
    ap.add_argument("--models-dir", default=str(DEFAULT_DIR),
                    help="dir containing cnn_lstm_<affect>.pt files")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--window", type=int, default=48, help="rolling buffer size in frames")
    ap.add_argument("--every", type=int, default=12, help="re-predict every N frames")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args.models_dir, device)
    if not models:
        sys.exit(f"No affect models found in {args.models_dir}\n"
                 "Expected cnn_lstm_engagement.pt / _boredom / _confusion / _frustration.")
    print(f"Loaded affects: {', '.join(models)} on {device}. Press 'q' to quit.")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)   # CAP_DSHOW = fast open on Windows
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera} (try --camera 1).")

    buf = deque(maxlen=args.window)
    preds = {a: ("warming up", 0.0) for a in models}
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            n += 1

            if len(buf) >= FRAMES_PER_CLIP and n % args.every == 0:
                x = build_clip(list(buf), device)
                with torch.no_grad():
                    for a, m in models.items():
                        p = F.softmax(m(x), 1)[0].cpu().numpy()
                        preds[a] = (CLASS_NAMES[int(p.argmax())], float(p.max()))

            # face bounding box
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.1,
                                                    minNeighbors=4, minSize=(48, 48))
            if len(faces):
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

            # affect panel (one line per loaded affect)
            cv2.rectangle(frame, (0, 0), (340, 18 + 26 * len(models)), (0, 0, 0), -1)
            for i, (a, (lbl, cf)) in enumerate(preds.items()):
                col = COLORS[CLASS_NAMES.index(lbl)] if lbl in CLASS_NAMES else (220, 220, 220)
                cv2.putText(frame, f"{a:11}: {lbl}  {cf*100:3.0f}%", (8, 24 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            cv2.imshow("AffectLearn — live affects (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
