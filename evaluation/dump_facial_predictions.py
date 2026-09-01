"""Run a facial checkpoint over a DAiSEE split and persist the per-clip predictions.

WHY THIS EXISTS
---------------
`reports/facial_confusion/FINDINGS.md:108` credits a script named `eval_all_checkpoints.py`
with producing the AUC/kappa table for every facial checkpoint. **That script is not in the
repository, in any branch.** Nor are the per-clip probabilities it would have produced:
`reports/facial_confusion/` contains `FINDINGS.md` and nothing else, while every DUX and
EmoSurv result set ships its `*_predictions.npz`. The consequence is that every DAiSEE number
in Chapter 4 is prose that cannot be re-derived from anything committed, and re-asking a
question of a finished run (a different binary cut, a threshold sweep, a bootstrap interval)
requires re-running inference against a 17 GB Drive-only corpus.

The same file records the cost of that gap directly: "There was previously no per-clip
probability dumper anywhere in the repo, which is precisely why the binary recompute had been
blocked on Colab for so long."

This script closes it. It writes a `Predictions` record in the format
`evaluation/predictions.py` defines, so `confusion_matrix.py --collapse` and every other
evaluation script works offline afterwards, on a laptop, without the corpus.

TWO THINGS IT DELIBERATELY DOES
-------------------------------
1. Prints the majority baseline beside the accuracy. On DAiSEE engagement, a model scoring
   0.9353 accuracy sits 0.0128 BELOW the 0.9481 baseline, and only the comparison reveals it.
2. Reports the minority-class count, because a metric's precision is bounded by it. The
   engagement ANY cut leaves four negatives in the whole test split, which is why the AUC of
   0.7942 computed over them is disowned in FINDINGS.md. The worst-case 95% interval on a
   recall estimated from n examples is ~1.96*sqrt(0.25/n): 4 examples gives +/-49 points, 85
   gives +/-10.6, 503 gives +/-4.4.

USAGE
-----
    python evaluation/dump_facial_predictions.py \
        --checkpoint /content/drive/MyDrive/affectlearn-ml/checkpoints/cnn_lstm_best.pt \
        --config training/facial/config.yaml \
        --split Test --target Engagement --binary-cut high \
        --out reports/facial_engagement/engagement_high_test.npz

Needs the preprocessed corpus, so in practice it runs on Colab. Everything downstream of the
`.npz` does not.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.predictions import Predictions  # noqa: E402
from training.facial.dataset import DAiSEEDataset  # noqa: E402
from training.facial.model import build_model  # noqa: E402

# Label names per cut, so the stored record is self-describing.
LABELS_4 = ("very_low", "low", "high", "very_high")
BINARY_LABELS = {
    "any": ("absent", "present"),
    "high": ("low_or_absent", "high"),
}
# The engagement HIGH cut is the one worth naming plainly: the positive class is "engaged".
ENGAGEMENT_HIGH_LABELS = ("disengaged", "engaged")


def recall_ci_half_width(n: int) -> float:
    """Worst-case 95% half-width for a recall estimated on `n` minority examples."""
    return 1.96 * math.sqrt(0.25 / n) if n else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="training/facial/config.yaml")
    ap.add_argument("--split", default="Test", choices=("Train", "Validation", "Test"))
    ap.add_argument("--target", default=None, help="defaults to the checkpoint's stamp")
    ap.add_argument("--binary-cut", default=None, choices=("any", "high"))
    ap.add_argument("--out", required=True, help="destination .npz")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    stamped = ckpt.get("target_affect")
    target = args.target or stamped or cfg.get("target_affect", "Engagement")
    if stamped and args.target and stamped != args.target:
        # The mismatch that produces confident nonsense rather than an error.
        print(f"  REFUSING: checkpoint is stamped {stamped!r}, you asked for {args.target!r}")
        return 2

    ds = DAiSEEDataset(
        cfg["paths"]["daisee_preprocessed"],
        Path(cfg["paths"]["labels_dir"]) / f"{args.split}Labels.csv",
        split=args.split,
        target=target,
        binary_cut=args.binary_cut,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    mcfg = {**cfg["model"], "num_classes": ds.num_classes}
    model = build_model(mcfg).to(device)
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # A silently partial load is how a head-shape change becomes a mystery result.
        print(f"  state_dict mismatch — missing={list(missing)} unexpected={list(unexpected)}")
        if any("head" in k for k in list(missing) + list(unexpected)):
            print("  the classifier head does not match; is --binary-cut right for this checkpoint?")
            return 2
    model.eval()

    y_true, y_pred, y_prob, ids = [], [], [], []
    with torch.no_grad():
        for i, (clips, labels) in enumerate(loader):
            logits = model(clips.to(device))
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            y_prob.extend(probs)
            y_pred.extend(probs.argmax(1))
            y_true.extend(labels.numpy())
            lo = i * args.batch_size
            ids.extend(p.stem for p in ds.clips[lo:lo + len(labels)])

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if args.binary_cut:
        labels = (ENGAGEMENT_HIGH_LABELS if (target == "Engagement" and args.binary_cut == "high")
                  else BINARY_LABELS[args.binary_cut])
    else:
        labels = LABELS_4

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Predictions(
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        ids=np.asarray(ids, dtype=str), labels=list(labels),
    ).save(str(out))

    counts = np.bincount(y_true, minlength=len(labels))
    minority = int(counts.min())
    acc = float((y_pred == y_true).mean())
    baseline = float(counts.max() / len(y_true))

    print(f"  wrote {out}  n={len(y_true)}  classes={list(labels)}")
    print(f"  class counts      : {counts.tolist()}")
    print(f"  accuracy          : {acc:.4f}")
    print(f"  majority baseline : {baseline:.4f}   <- read the accuracy against this")
    if acc < baseline:
        print(f"  ** BELOW baseline by {baseline - acc:.4f} — a constant predictor scores higher **")
    print(f"  minority class n  : {minority}  -> recall measurable to "
          f"+/-{recall_ci_half_width(minority)*100:.1f} pts (worst case, 95%)")
    if minority < 25:
        print("  ** too few minority examples to estimate a rate at any useful precision **")

    # The probability SPREAD, which decides whether a fixed threshold can work at all. AUC is
    # rank-based and so says nothing about it: a model can hold its AUC exactly while emitting
    # every score inside a narrow band, and a gate set at a round number then fires on roughly
    # half of all cycles regardless of the signal carried.
    if y_prob.shape[1] == 2:
        p1 = y_prob[:, 1]
        print(f"  P(positive) spread: {p1.min():.3f}-{p1.max():.3f} "
              f"(mean {p1.mean():.3f}, sd {p1.std():.3f})")
        print(f"  share above 0.50  : {(p1 >= 0.5).mean()*100:.1f}%")
        for q in (0.80, 0.90, 0.95):
            print(f"  threshold at p{int(q*100)}  : {np.quantile(p1, q):.4f}")

    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "target": target,
        "binary_cut": args.binary_cut, "split": args.split,
        "n": int(len(y_true)), "class_counts": counts.tolist(),
        "labels": list(labels), "accuracy": acc, "majority_baseline": baseline,
        "minority_class_n": minority,
        "recall_ci_half_width": recall_ci_half_width(minority),
    }, indent=2))
    print(f"  wrote {meta}")
    print("  now offline-reusable: "
          f"python evaluation/confusion_matrix.py --predictions {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
