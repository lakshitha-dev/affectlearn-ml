"""Confusion matrix + per-class precision/recall/F1, from a saved prediction record.

Produces the two artefacts Chapter 4's ceiling argument actually rests on and which the repo
never contained: the confusion matrix figure, and the per-class table. The claim "the dominant
error is the confusion of adjacent High and Very High clips" is currently asserted with no
displayed evidence; this is what supplies it.

METRICS. Weighted-F1 is the headline for the imbalanced 4-class task, but accuracy is reported
beside it and per-class recall is reported for every class, because aggregate scores hide exactly
the failure that matters here — a model can post a respectable weighted-F1 while never once
identifying the lowest engagement level. There is direct evidence in this project that accuracy
misleads on this distribution: moving to the production cropper RAISED accuracy 0.520 -> 0.540
while weighted-F1 FELL 0.509 -> 0.480. Cohen's kappa is included because it, unlike accuracy, is
chance-corrected; one published detector reached A' = 0.99 on frustration at kappa = 0.23.

FIGURE. A confusion matrix encodes count MAGNITUDE, so the colour job is sequential: a single
hue, light -> dark. Not a rainbow and not a diverging map — neither has a meaningful midpoint
here, and both imply structure the data does not have. A single hue also survives greyscale
printing, which matters for a thesis. Cells are directly labelled (a 4x4 grid is small enough
that every cell should carry its number), label ink flips to white on dark cells for contrast,
and the axes are recessive so the data dominates.

Run:
    python confusion_matrix.py --pred preds.npz
    python confusion_matrix.py --pred preds.npz --normalize true --out-dir figs/
    # collapse 4 engagement levels to binary (deletes the H<->VH boundary):
    python confusion_matrix.py --pred preds.npz \
        --collapse "disengaged=Very Low,Low" --collapse "engaged=High,Very High"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from predictions import Predictions  # noqa: E402

# Single-hue sequential ramp, light -> dark. Matplotlib's "Blues" is perceptually ordered and
# greyscale-safe; see the module docstring for why sequential rather than rainbow/diverging.
SEQUENTIAL_CMAP = "Blues"


def compute(preds: Predictions) -> dict:
    """All metrics, as plain JSON-able types."""
    y_true, y_pred, labels = preds.y_true, preds.y_pred, preds.labels
    n_classes = preds.n_classes
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    out = {
        "n": int(len(preds)),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {
            labels[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(n_classes)
        },
    }

    # Classes the model NEVER gets right are the headline limitation, not a footnote.
    out["zero_recall_classes"] = [labels[i] for i in range(n_classes) if rec[i] == 0.0]
    absent = [labels[i] for i in range(n_classes) if support[i] == 0]
    if absent:
        out["classes_absent_from_this_split"] = absent

    # Macro-F1 is mechanically capped when a class is missing from the split — flag it so the
    # number is never read as if the model had a fair shot at every class.
    if absent:
        k = n_classes
        out["macro_f1_ceiling"] = (k - len(absent)) / k

    if preds.y_prob is not None:
        try:
            if n_classes == 2:
                out["auc"] = float(roc_auc_score(y_true, preds.y_prob[:, 1]))
            else:
                out["auc_ovr_macro"] = float(
                    roc_auc_score(y_true, preds.y_prob, multi_class="ovr", average="macro")
                )
        except ValueError as e:
            out["auc_error"] = str(e)  # e.g. a class absent from y_true

    # The single most informative error for an ordinal task: adjacent-class confusion.
    if n_classes > 2:
        off = cm.sum() - np.trace(cm)
        adj = sum(cm[i, i + 1] + cm[i + 1, i] for i in range(n_classes - 1))
        out["errors_total"] = int(off)
        out["errors_between_adjacent_classes"] = int(adj)
        out["adjacent_error_share"] = float(adj / off) if off else 0.0
        pairs = {
            f"{labels[i]}<->{labels[j]}": int(cm[i, j] + cm[j, i])
            for i in range(n_classes) for j in range(i + 1, n_classes)
        }
        out["worst_confused_pair"] = max(pairs, key=pairs.get) if pairs else None
        out["confused_pairs"] = dict(sorted(pairs.items(), key=lambda kv: -kv[1]))

    return out


def plot(preds: Predictions, out_path: Path, normalize: str | None = None, title: str = "") -> Path:
    """Render the matrix. `normalize`: None | 'true' (row) | 'pred' (column)."""
    import matplotlib
    matplotlib.use("Agg")           # headless: no display needed on Colab or CI
    import matplotlib.pyplot as plt

    n = preds.n_classes
    cm = confusion_matrix(preds.y_true, preds.y_pred, labels=list(range(n)))
    shown = cm.astype(float)
    if normalize == "true":
        denom = shown.sum(axis=1, keepdims=True)
        shown = np.divide(shown, denom, out=np.zeros_like(shown), where=denom != 0)
    elif normalize == "pred":
        denom = shown.sum(axis=0, keepdims=True)
        shown = np.divide(shown, denom, out=np.zeros_like(shown), where=denom != 0)

    size = max(3.4, 1.05 * n + 1.4)
    fig, ax = plt.subplots(figsize=(size, size * 0.92), dpi=300)
    im = ax.imshow(shown, cmap=SEQUENTIAL_CMAP, vmin=0.0,
                   vmax=shown.max() if shown.max() > 0 else 1.0)

    ax.set_xticks(range(n), preds.labels, rotation=30, ha="right")
    ax.set_yticks(range(n), preds.labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title, pad=10)

    # Recessive frame: keep the data dominant.
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)

    # Direct cell labels. Ink flips to white on dark cells so contrast holds at both ends of
    # the ramp; text stays neutral (never the ramp colour).
    hi = shown.max() if shown.max() > 0 else 1.0
    for i in range(n):
        for j in range(n):
            text = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if shown[i, j] > 0.6 * hi else "#1a1a1a")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("proportion of true class" if normalize else "clips", fontsize=8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    # PDF alongside PNG: LaTeX wants vector, Word wants raster.
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def summary(m: dict) -> str:
    lines = [
        f"n = {m['n']}   classes: {', '.join(m['labels'])}",
        "",
        f"  accuracy     {m['accuracy']:.4f}",
        f"  weighted F1  {m['weighted_f1']:.4f}   <- headline for this imbalance",
        f"  macro F1     {m['macro_f1']:.4f}",
        f"  Cohen kappa  {m['cohen_kappa']:.4f}   <- chance-corrected",
    ]
    for k in ("auc", "auc_ovr_macro"):
        if k in m:
            lines.append(f"  {k:12s} {m[k]:.4f}")
    if "auc_error" in m:
        lines.append(f"  AUC unavailable: {m['auc_error']}")

    lines += ["", f"  {'class':16s} {'prec':>6s} {'recall':>7s} {'F1':>6s} {'n':>6s}"]
    for lab, s in m["per_class"].items():
        flag = "   <-- NEVER CORRECT" if s["recall"] == 0.0 else ""
        lines.append(
            f"  {lab:16s} {s['precision']:6.3f} {s['recall']:7.3f} {s['f1']:6.3f} "
            f"{s['support']:6d}{flag}"
        )

    lines += ["", "  confusion (rows = true):"]
    width = max(len(str(v)) for row in m["confusion_matrix"] for v in row)
    for lab, row in zip(m["labels"], m["confusion_matrix"]):
        lines.append(f"    {lab:16s} " + " ".join(f"{v:>{width}d}" for v in row))

    if m.get("zero_recall_classes"):
        lines += ["", "  *** classes with ZERO recall: "
                  + ", ".join(m["zero_recall_classes"])
                  + " — the aggregate scores above are blind to this ***"]
    if m.get("classes_absent_from_this_split"):
        lines += [f"  *** classes ABSENT from this split: "
                  f"{', '.join(m['classes_absent_from_this_split'])} — macro-F1 is capped at "
                  f"{m['macro_f1_ceiling']:.2f} and is not comparable ***"]
    if "adjacent_error_share" in m:
        lines += [
            "",
            f"  adjacent-class errors: {m['errors_between_adjacent_classes']}/{m['errors_total']} "
            f"({m['adjacent_error_share']:.1%} of all errors)",
            f"  worst confused pair:   {m['worst_confused_pair']}",
        ]
    return "\n".join(lines)


def _parse_collapse(specs: list[str]) -> dict[str, list[str]]:
    """`--collapse "engaged=High,Very High"` -> {"engaged": ["High", "Very High"]}."""
    out: dict[str, list[str]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"--collapse needs NAME=label,label — got {spec!r}")
        name, members = spec.split("=", 1)
        out[name.strip()] = [m.strip() for m in members.split(",") if m.strip()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="prediction record (.npz or .json)")
    ap.add_argument("--out-dir", default=None, help="write metrics.json + figure here")
    ap.add_argument("--normalize", choices=["true", "pred"], default=None,
                    help="row- or column-normalise the FIGURE (metrics stay on raw counts)")
    ap.add_argument("--collapse", action="append", default=[],
                    help='merge classes, e.g. --collapse "engaged=High,Very High" (repeatable)')
    ap.add_argument("--title", default="")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()

    preds = Predictions.load(a.pred)
    if a.collapse:
        preds = preds.collapse(_parse_collapse(a.collapse))
        print(f"collapsed to {preds.n_classes} classes: {', '.join(preds.labels)}\n")

    metrics = compute(preds)
    print(summary(metrics))

    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = "confusion_binary" if a.collapse else "confusion"
        (out / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (out / f"{stem}_summary.txt").write_text(summary(metrics), encoding="utf-8")
        print(f"\nwrote {out / (stem + '_metrics.json')}")
        if not a.no_figure:
            p = plot(preds, out / f"{stem}_matrix.png", normalize=a.normalize, title=a.title)
            print(f"wrote {p} (+ .pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
