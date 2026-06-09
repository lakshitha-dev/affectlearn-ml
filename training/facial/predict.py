"""Standalone inference — predict the affect level for a single video.

Runs the SAME preprocessing as training (largest-face crop → 16 evenly-spaced
frames → 96×96 → ImageNet normalisation), loads the trained CNN-LSTM, and prints
the predicted class plus per-class probabilities.

Usage:
    python predict.py --video path/to/clip.avi
    python predict.py --video clip.mp4 --checkpoint /path/to/cnn_lstm_best.pt

Notes:
  * Works with any video format OpenCV can read (.avi, .mp4, .mov, ...).
  * The model architecture is read from the checkpoint itself, so this works for
    either the `scratch` or `resnet18` backbone without extra flags.
  * Requires: torch, opencv-python(-headless), numpy  (+ torchvision if the
    checkpoint used the resnet18 backbone).
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

# allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))
from model      import build_model        # noqa: E402
from preprocess import preprocess_clip     # noqa: E402

CLASS_NAMES = ["Very Low", "Low", "High", "Very High"]
DEFAULT_CKPT = Path(__file__).resolve().parents[2] / "models" / "cnn_lstm_best.pt"


def load_model(checkpoint: str, device, model_cfg: dict | None = None):
    """Load a trained CNN-LSTM. Architecture comes from the checkpoint's own
    embedded config unless `model_cfg` is provided."""
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = model_cfg or ckpt.get("cfg", {}).get("model")
    if cfg is None:
        raise ValueError("Checkpoint has no embedded model config — pass --config.")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    target = ckpt.get("cfg", {}).get("target_affect", "Engagement")
    return model, target


def predict(video_path: str, model, device):
    """Return (pred_index, prob_vector) for one video, or (None, None) on failure."""
    clip = preprocess_clip(video_path)            # (T, 3, 96, 96) float32 or None
    if clip is None:
        return None, None
    x = torch.from_numpy(clip).unsqueeze(0).to(device)   # (1, T, 3, 96, 96)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs


def main():
    ap = argparse.ArgumentParser(description="Predict affect level for one video clip")
    ap.add_argument("--video", required=True, help="Path to a video file")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                    help="Path to cnn_lstm_best.pt (default: <repo>/models/cnn_lstm_best.pt)")
    ap.add_argument("--config", default=None,
                    help="Optional config.yaml (only if the checkpoint lacks an embedded cfg)")
    args = ap.parse_args()

    if not Path(args.video).exists():
        sys.exit(f"Video not found: {args.video}")
    if not Path(args.checkpoint).exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}\n"
                 "Pass --checkpoint /path/to/cnn_lstm_best.pt (it is gitignored; "
                 "grab it from the models/ folder on Drive).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = None
    if args.config:
        import yaml
        model_cfg = yaml.safe_load(open(args.config))["model"]

    model, target = load_model(args.checkpoint, device, model_cfg)
    pred, probs = predict(args.video, model, device)

    if pred is None:
        sys.exit("Could not extract a clip (video unreadable or fewer than 16 frames).")

    print(f"\n{target} prediction for '{Path(args.video).name}':")
    print(f"  → {CLASS_NAMES[pred]}   ({probs[pred] * 100:.1f}% confidence)\n")
    print("  Per-class probability:")
    for i, name in enumerate(CLASS_NAMES):
        bar = "#" * int(round(probs[i] * 30))
        print(f"    {name:10} {probs[i] * 100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
