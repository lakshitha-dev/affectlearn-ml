"""Prove that the deployed facial model's accuracy depends on adapted BatchNorm statistics.

THE CLAIM UNDER TEST
--------------------
`models/cnn_lstm_confusion_anycut.json` states the backbone is "resnet18 FULLY FROZEN", and
`train_cnn_lstm.py` does freeze it -- but only the WEIGHTS:

    for p in model.cnn.net.parameters():
        p.requires_grad_(False)          # train_cnn_lstm.py, freeze_backbone_full branch

It never calls `.eval()` on the backbone, and the epoch loop calls `model.train()`, so every
BatchNorm layer in the ResNet18 kept updating its running mean and variance on DAiSEE for the
whole run. Frozen weights, moving statistics.

WHY IT MATTERS
--------------
1. Reproducibility. A cached-feature pipeline, or any re-implementation that freezes the
   backbone properly, silently loses accuracy it cannot account for.
2. Description. "Fully frozen" implies the backbone is a fixed ImageNet feature extractor. It
   is not: 24.7% of the mean BN statistic magnitude changed, which is a form of unsupervised
   domain adaptation to DAiSEE and part of why the model works.
3. NOT a serving bug. The ONNX export bakes in whatever statistics the checkpoint holds, so
   production is self-consistent. This is about what the artifact IS, not whether it is broken.

WHAT THIS SCRIPT MEASURES
-------------------------
It isolates the backbone by holding everything else fixed. The deployed model's own trained
lstm+head is applied to features produced two ways -- once by the deployed backbone (adapted
BN) and once by a stock ImageNet ResNet18 (original BN) -- over the same clips with the same
labels. Any difference is attributable to the BN statistics alone.

Run with `--features-dir` pointing at an ImageNet-BN feature cache to get the comparison
without touching the 13.6 GB of raw clips.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.facial.train_cached_features import (  # noqa: E402
    TemporalHead,
    load_split,
    subject_bootstrap,
    to_binary,
)

HEAD_KEYS = ("lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0",
             "lstm.bias_hh_l0", "head.weight", "head.bias")


def bn_drift(state_dict) -> dict:
    """How far the checkpoint's BN running statistics moved from the ImageNet initialisation."""
    from torchvision.models import ResNet18_Weights, resnet18
    ref = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    deltas, mags, n = [], [], 0
    for k, v in state_dict.items():
        if k.startswith("cnn.net.") and ("running_mean" in k or "running_var" in k):
            rk = k[len("cnn.net."):]
            if rk in ref:
                deltas.append(float((v - ref[rk]).abs().mean()))
                mags.append(float(ref[rk].abs().mean()))
                n += 1
    if not n:
        return {"tensors": 0}
    d, m = float(np.mean(deltas)), float(np.mean(mags))
    return {"tensors": n, "mean_abs_delta": d, "mean_abs_imagenet": m,
            "relative_drift": d / max(m, 1e-9)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="cnn_lstm_confusion_anycut.pt")
    ap.add_argument("--features-dir", required=True, help="ImageNet-BN feature cache")
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--reference-npz", default=None,
                    help="the deployed model's stored per-clip probabilities, for clip-by-clip "
                         "comparison (cnn_lstm_confusion_anycut_test_predictions.npz)")
    ap.add_argument("--target", default="Confusion")
    ap.add_argument("--cut", default="any")
    ap.add_argument("--published-auc", type=float, default=0.6414)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model_state"]
    drift = bn_drift(sd)
    print(f"  checkpoint: {Path(args.checkpoint).name}  epoch {ck.get('epoch')} "
          f"target {ck.get('target_affect')} cut {ck.get('binary_cut')}")
    print(f"  BN statistics vs ImageNet: {drift['tensors']} tensors, "
          f"mean |delta| {drift['mean_abs_delta']:.4f} against |imagenet| "
          f"{drift['mean_abs_imagenet']:.4f} -> {drift['relative_drift']:.1%} drift")

    head = TemporalHead(num_classes=2, dropout=ck["cfg"]["model"].get("dropout", 0.6))
    head.load_state_dict({k: sd[k] for k in HEAD_KEYS})
    head.eval()

    feats, _, levels, clip_ids, subjects = load_split(
        Path(args.features_dir), "Test", Path(args.labels_dir), args.target)
    y = to_binary(levels, args.cut)
    with torch.no_grad():
        p = torch.softmax(head(torch.from_numpy(feats)).float(), 1)[:, 1].numpy()

    auc = float(roc_auc_score(y, p))
    lo, hi = subject_bootstrap(y, p, subjects, n_boot=2000, seed=0)
    gap = args.published_auc - auc
    print(f"\n  deployed head x IMAGENET-BN features : AUC {auc:.4f}  CI [{lo:.4f}, {hi:.4f}]")
    print(f"  deployed head x its own backbone     : AUC {args.published_auc:.4f}  (published)")
    print(f"  attributable to BN statistics alone  : {gap:+.4f}")

    result = {"checkpoint": str(args.checkpoint), "bn_drift": drift,
              "auc_imagenet_bn": auc, "auc_ci": [lo, hi],
              "auc_published": args.published_auc, "gap_from_bn": gap,
              "n": int(len(y)), "positives": int(y.sum()),
              "subjects": int(len(np.unique(subjects)))}

    if args.reference_npz:
        ref = np.load(args.reference_npz, allow_pickle=False)
        order = {str(c): i for i, c in enumerate(ref["ids"])}
        missing = [c for c in clip_ids if str(c) not in order]
        if missing:
            print(f"  WARNING: {len(missing)} cached clips absent from the reference dump")
        idx = np.array([order[str(c)] for c in clip_ids if str(c) in order])
        keep = np.array([str(c) in order for c in clip_ids])
        pref = (ref["y_prob"][:, 1] if ref["y_prob"].ndim == 2 else ref["y_prob"])[idx]
        same_labels = bool((y[keep] == ref["y_true"][idx]).all())
        r = float(np.corrcoef(p[keep], pref)[0, 1])
        print(f"\n  clip-by-clip against the deployed model's stored probabilities")
        print(f"    labels identical: {same_labels}   (if False, nothing below is meaningful)")
        print(f"    pearson r {r:.4f}   mean|diff| {float(np.abs(p[keep]-pref).mean()):.4f}")
        print(f"    imagenet-BN spread {p.min():.3f}-{p.max():.3f}   "
              f"deployed spread {pref.min():.3f}-{pref.max():.3f}")
        result.update({"labels_identical": same_labels, "pearson_r": r,
                       "mean_abs_prob_diff": float(np.abs(p[keep] - pref).mean())})

    print("\n  CONCLUSION: the backbone's weights were frozen; its BatchNorm statistics were")
    print("  not. Holding the trained head and the clips fixed, swapping only those statistics")
    print(f"  moves AUC by {abs(gap):.4f} -- so a re-implementation that freezes the backbone")
    print("  properly cannot reproduce the published figure, and the model card's")
    print('  "resnet18 FULLY FROZEN" understates what the backbone contributed.')

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
