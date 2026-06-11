"""Export the trained facial CNN-LSTM to ONNX for backend serving.

Produces an ONNX graph matching the backend's serve-side contract
(`affectlearn/backend/app/services/model_inference.py`):
    input  name = "clip"    shape (B, T, 3, 96, 96)  float32
    output name = "logits"  shape (B, 4)
batch and time (T) are dynamic so the server's 16-frame window works regardless of the
training clip length.

Usage:
    python export_onnx.py --checkpoint models/cnn_lstm_best.pt --out cnn_lstm_best.onnx
    # then copy cnn_lstm_best.onnx into affectlearn/backend/models/ and set AFFECT_MODEL_KIND

The checkpoint may be a raw state_dict or a dict containing "model_state"/"state_dict"/"model".
Architecture is read from config.yaml (model section), so it stays in lockstep with training.
"""

import argparse
from pathlib import Path

import torch
import yaml

from model import build_model

_HERE = Path(__file__).parent


def _load_state_dict(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt  # assume it's already a state_dict


def main() -> None:
    ap = argparse.ArgumentParser(description="Export facial CNN-LSTM to ONNX")
    ap.add_argument("--checkpoint", required=True, help="path to the trained .pt checkpoint")
    ap.add_argument("--out", default="cnn_lstm_best.onnx", help="output ONNX path")
    ap.add_argument("--config", default=str(_HERE / "config.yaml"))
    ap.add_argument("--frames", type=int, default=16, help="sample clip length T for tracing")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model = build_model(cfg.get("model", cfg))
    model.load_state_dict(_load_state_dict(args.checkpoint))
    model.eval()

    dummy = torch.randn(1, args.frames, 3, 96, 96)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["clip"], output_names=["logits"],
        dynamic_axes={"clip": {0: "batch", 1: "frames"}, "logits": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )

    # sanity: re-load and run a forward pass through onnxruntime if available
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"clip": dummy.numpy().astype("float32")})[0]
        assert out.shape == (1, cfg.get("model", {}).get("num_classes", 4))
        print(f"[export_onnx] OK -> {args.out}  (verified logits shape {tuple(out.shape)})")
        print("[export_onnx] np", np.asarray(out).round(3).tolist())
    except ImportError:
        print(f"[export_onnx] exported -> {args.out} (install onnxruntime to verify)")


if __name__ == "__main__":
    main()
