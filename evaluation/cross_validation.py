"""Grouped cross-validation with mean +- sd — the honest alternative to a single split.

WHY GROUPED, AND WHY CV AT ALL
------------------------------
A single 70/15/15 split is not reportable at this scale. Two reasons, both measured in this
project:

  * With uneven group counts per class, a random group split leaves a whole class out of a fold
    on most seeds (17/20 for the real section distribution). Folds must be built per class.
  * A single split gives one point estimate with no spread, so two models differing by 0.002 look
    ordered when they are not. Several claims in the thesis rest on "within run-to-run noise"
    without ever establishing a noise floor. CV over folds is that floor.

GROUPS ARE MANDATORY, not optional. Windows from one participant (or one section, under codebook
labelling) are not independent: consecutive 30-second slices of the same person reading the same
passage share almost everything. Splitting at window level leaks and inflates every metric. The
group column is whatever `participant` holds — learner id for multi-subject data, section title
for the single-subject codebook scheme.

Note this makes the folds *group*-independent, which is weaker than subject-independent when the
groups are sections from one learner. Report which it is; do not call a section-independent split
subject-independent.

Run (as a library — you supply fit/predict):
    from cross_validation import grouped_cv
    res = grouped_cv(X, y, groups, fit_predict=my_fn, n_splits=5, labels=LABELS)
    print(format_report(res))
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score


def stratified_group_folds(groups, y, n_splits: int = 5, seed: int = 42) -> list[np.ndarray]:
    """Assign each GROUP to a fold, balancing classes across folds.

    Greedy: process classes from rarest to most common and deal their groups round-robin into the
    folds that currently hold fewest of that class. This keeps the rare classes — the ones that
    actually break a split — represented everywhere.
    """
    groups = np.asarray(groups)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    # One label per group (mode), as in splits.group_labels.
    per_group: dict = {}
    for g, lab in zip(groups.tolist(), y.tolist()):
        per_group.setdefault(g, []).append(lab)
    group_label = {g: Counter(v).most_common(1)[0][0] for g, v in per_group.items()}

    fold_of: dict = {}
    counts = np.zeros((n_splits, int(y.max()) + 1), dtype=int)
    # Rarest class first: its groups get the best pick of folds.
    by_rarity = sorted(Counter(group_label.values()).items(), key=lambda kv: kv[1])
    for label, _ in by_rarity:
        gs = sorted([g for g, lab in group_label.items() if lab == label])
        rng.shuffle(gs)
        for g in gs:
            f = int(np.argmin(counts[:, label]))
            fold_of[g] = f
            counts[f, label] += 1

    return [np.array([fold_of[g] == f for g in groups], dtype=bool) for f in range(n_splits)]


def grouped_cv(X, y, groups, fit_predict, n_splits: int = 5, seed: int = 42,
               labels: list[str] | None = None) -> dict:
    """Run grouped CV. `fit_predict(X_tr, y_tr, X_te) -> y_pred` (or `(y_pred, y_prob)`).

    Returns per-fold metrics plus mean +- sd. Folds that lose a class are FLAGGED rather than
    silently averaged, because macro-F1 from such a fold is mechanically capped.
    """
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    n_classes = len(labels) if labels else int(y.max()) + 1
    test_masks = stratified_group_folds(groups, y, n_splits, seed)

    folds = []
    for i, te in enumerate(test_masks):
        tr = ~te
        if not tr.any() or not te.any():
            folds.append({"fold": i, "skipped": "empty train or test fold"})
            continue

        out = fit_predict(X[tr], y[tr], X[te])
        y_pred = np.asarray(out[0] if isinstance(out, tuple) else out)

        present = sorted(set(y[te].tolist()))
        missing = [labels[c] if labels else c for c in range(n_classes) if c not in present]
        rec = recall_score(y[te], y_pred, labels=list(range(n_classes)),
                           average=None, zero_division=0)
        folds.append({
            "fold": i,
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_groups_test": int(len(set(groups[te].tolist()))),
            "accuracy": float(accuracy_score(y[te], y_pred)),
            "weighted_f1": float(f1_score(y[te], y_pred, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y[te], y_pred, average="macro", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(y[te], y_pred)),
            "per_class_recall": {(labels[c] if labels else str(c)): float(rec[c])
                                 for c in range(n_classes)},
            "classes_missing_from_fold": missing,
        })

    scored = [f for f in folds if "skipped" not in f]
    agg = {}
    for key in ("accuracy", "weighted_f1", "macro_f1", "cohen_kappa"):
        vals = np.array([f[key] for f in scored], dtype=float)
        agg[key] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(vals.min()), "max": float(vals.max())}
    if labels:
        agg["per_class_recall"] = {
            lab: {
                "mean": float(np.mean([f["per_class_recall"][lab] for f in scored])),
                "sd": float(np.std([f["per_class_recall"][lab] for f in scored], ddof=1))
                if len(scored) > 1 else 0.0,
            }
            for lab in labels
        }

    return {
        "n_splits": n_splits, "seed": seed, "n": int(len(y)),
        "n_groups": int(len(set(groups.tolist()))),
        "labels": labels, "folds": folds, "aggregate": agg,
        "folds_missing_a_class": [f["fold"] for f in scored if f["classes_missing_from_fold"]],
    }


def format_report(res: dict) -> str:
    lines = [
        f"Grouped {res['n_splits']}-fold CV — {res['n']} windows across {res['n_groups']} groups "
        f"(seed {res['seed']})",
        "",
        f"  {'fold':>4s} {'n_te':>5s} {'grp':>4s} {'acc':>7s} {'wF1':>7s} {'macF1':>7s} {'kappa':>7s}",
    ]
    for f in res["folds"]:
        if "skipped" in f:
            lines.append(f"  {f['fold']:>4d}  SKIPPED ({f['skipped']})")
            continue
        flag = "  <-- missing " + ",".join(map(str, f["classes_missing_from_fold"])) \
            if f["classes_missing_from_fold"] else ""
        lines.append(
            f"  {f['fold']:>4d} {f['n_test']:>5d} {f['n_groups_test']:>4d} {f['accuracy']:>7.4f} "
            f"{f['weighted_f1']:>7.4f} {f['macro_f1']:>7.4f} {f['cohen_kappa']:>7.4f}{flag}"
        )

    a = res["aggregate"]
    lines += ["", "  mean +- sd over folds:"]
    for key in ("accuracy", "weighted_f1", "macro_f1", "cohen_kappa"):
        s = a[key]
        lines.append(f"    {key:12s} {s['mean']:.4f} +- {s['sd']:.4f}   "
                     f"[{s['min']:.4f}, {s['max']:.4f}]")

    if "per_class_recall" in a:
        lines += ["", "  per-class recall (mean +- sd):"]
        for lab, s in a["per_class_recall"].items():
            flag = "   <-- never detected" if s["mean"] == 0.0 else ""
            lines.append(f"    {lab:16s} {s['mean']:.4f} +- {s['sd']:.4f}{flag}")

    if res["folds_missing_a_class"]:
        lines += ["", f"  *** folds {res['folds_missing_a_class']} lost a class — their macro-F1 is "
                  "capped and the mean above is optimistic ***"]

    sd = a["weighted_f1"]["sd"]
    lines += ["", f"  NOISE FLOOR: weighted-F1 sd across folds is {sd:.4f}. Two models differing "
              f"by less than ~{2 * sd:.4f} are not distinguishable by this evaluation."]
    return "\n".join(lines)


def save(res: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return path
