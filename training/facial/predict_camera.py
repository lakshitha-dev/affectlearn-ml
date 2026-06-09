"""Live webcam engagement prediction.

Opens the default camera, runs the trained CNN-LSTM on a rolling window of frames,
and overlays the predicted engagement level (with per-class bars) in real time.

Usage:
    python predict_camera.py --checkpoint "G:\\My Drive\\affectlearn-ml\\models\\cnn_lstm_best.pt"
    python predict_camera.py --camera 0

Press 'q' in the video window to quit.
Requires: torch, torchvision (for the resnet18 backbone), opencv-python, numpy.

Note: engagement is a slow-changing, clip-level signal — the model sees a rolling
window of recent frames, so the label updates every ~half-second, not per-frame.
"""

import sys
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))
from model      import build_model                                         # noqa: E402
from preprocess import crop_face, normalize, FRAMES_PER_CLIP, _get_cascade  # noqa: E402

CLASS_NAMES = ["Very Low", "Low", "High", "Very High"]
COLORS = [(0, 0, 255), (0, 165, 255), (0, 200, 0), (0, 255, 0)]  # BGR per class
DEFAULT_CKPT = Path(__file__).resolve().parents[2] / "models" / "cnn_lstm_best.pt"


def load_model(checkpoint: str, device):
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model(ckpt["cfg"]["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("cfg", {}).get("target_affect", "Engagement")


def build_clip(frames_rgb):
    """frames_rgb: list of HWC RGB uint8 frames -> tensor (1, T, 3, 96, 96)."""
    idx = np.linspace(0, len(frames_rgb) - 1, FRAMES_PER_CLIP, dtype=int)
    clip = np.stack([normalize(crop_face(frames_rgb[i])) for i in idx])
    return torch.from_numpy(clip).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser(description="Live webcam engagement prediction")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--window", type=int, default=48, help="rolling buffer size in frames")
    ap.add_argument("--every", type=int, default=8, help="re-predict every N frames")
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}\n"
                 "Pass --checkpoint /path/to/cnn_lstm_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, target = load_model(args.checkpoint, device)
    print(f"Loaded {target} model on {device}. Opening camera {args.camera} — press 'q' to quit.")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)   # CAP_DSHOW = fast open on Windows
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera} (try --camera 1).")

    buf = deque(maxlen=args.window)
    label, conf, probs, n = "warming up…", 0.0, None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            n += 1

            if len(buf) >= FRAMES_PER_CLIP and n % args.every == 0:
                with torch.no_grad():
                    probs = F.softmax(model(build_clip(list(buf)).to(device)), 1)[0].cpu().numpy()
                pred = int(probs.argmax())
                label, conf = CLASS_NAMES[pred], float(probs[pred])

            # ---- overlay ----
            color = COLORS[CLASS_NAMES.index(label)] if label in CLASS_NAMES else (200, 200, 200)

            # face bounding box (same detector the model crops with)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.1,
                                                    minNeighbors=4, minSize=(48, 48))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {conf*100:.0f}%", (x, max(22, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{target}: {label}  {conf*100:4.1f}%", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if probs is not None:
                for i, name in enumerate(CLASS_NAMES):
                    y = 64 + i * 24
                    cv2.rectangle(frame, (10, y - 14), (10 + int(probs[i] * 200), y), COLORS[i], -1)
                    cv2.putText(frame, f"{name} {probs[i]*100:4.1f}%", (220, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("AffectLearn — live engagement (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
