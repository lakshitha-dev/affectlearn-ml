"""EmoSurv: a real-human, subject-independent affect result from keystroke dynamics.

WHAT THIS IS FOR
----------------
The platform's own four-state pilot data does not exist yet, and collecting it needs an ethics
determination that has not arrived. EmoSurv is the one corpus already on disk with REAL human
affect labels from a meaningful number of participants (83), and `load_emosurv` was discarding
those labels to build a one-bit arousal proxy. This script recovers them.

It gives the behavioural branch a reportable number with a proper subject-independent split —
which n=1 self-collection can never provide, however carefully it is run.

WHAT IT IS NOT, AND THIS MUST BE STATED WHEREVER IT IS REPORTED
--------------------------------------------------------------
1. WRONG TAXONOMY. Neutral / Happy / Sad / Angry / Calm are basic emotions from a typing task.
   They are NOT bored / confused / engaged / frustrated during learning. A result here supports
   the PREMISE that keystroke dynamics carry affective signal; it does not answer RQ1.
2. KEYBOARD ONLY. EmoSurv has no mouse and no scroll, so most of the 16-feature vector is
   structurally dead — only keystroke_count, typing_rhythm_std, backspace_pct, pause_count and the
   idle clock vary. This validates the KEYBOARD portion of the model, not the whole thing.
3. NEUTRAL DOMINATES (55.6%). Always predicting Neutral scores 55.6% accuracy while detecting
   nothing, so accuracy alone is meaningless here. The majority baseline is printed for exactly
   that reason, and Cohen's kappa is the headline.
4. EMOTION IS ENTANGLED WITH PARTICIPANT. Most participants contributed only one or two emotions
   (13 have one, 42 have two), so within-participant the label is nearly constant. Keystroke style
   is famously identifying, so a window-level or passage-level split would let the model recognise
   the TYPIST instead of the emotion. Splitting by PARTICIPANT is therefore not a nicety here — it
   is the only control that makes the number mean anything.

Run:
    python train_emosurv.py                 # 5-class, 5-fold subject-independent
    python train_emosurv.py --non-neutral   # 4-class, excluding the dominant Neutral class
    python train_emosurv.py --out-dir ../../reports/emosurv
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2] / "evaluation"))
from compare_models import TorchSequenceClassifier, _WeightedGBDT  # noqa: E402
from aggregate_features import aggregate  # noqa: E402
from confusion_matrix import compute, plot, summary  # noqa: E402
from cross_validation import format_report, grouped_cv  # noqa: E402
from external_datasets import EMOSURV_LABELS, load_emosurv  # noqa: E402
from feature_engineering import FEATURE_NAMES, extract_features  # noqa: E402
from predictions import Predictions  # noqa: E402


def load(emosurv_dir: str, non_neutral: bool):
    windows = load_emosurv(emosurv_dir)
    if not windows:
        raise SystemExit(
            f"No EmoSurv windows under {emosurv_dir}. The corpus is a free IEEE DataPort download "
            "(DOI 10.21227/eae6-pk42), non-commercial research use, and is gitignored — place the "
            "typing CSVs there first."
        )
    if non_neutral:
        windows = [w for w in windows if w["emotion_index"] != 0]
        labels = EMOSURV_LABELS[1:]
        remap = {old: new for new, old in enumerate(range(1, 5))}
    else:
        labels = list(EMOSURV_LABELS)
        remap = {i: i for i in range(5)}

    X = np.stack([extract_features(w["events"], 0) for w in windows])
    y = np.array([remap[w["emotion_index"]] for w in windows], dtype=np.int64)
    groups = np.array([w["participant"] for w in windows])       # SUBJECT-independent
    return X, y, groups, labels, windows


def describe(X, y, groups, labels, windows) -> dict:
    counts = collections.Counter(y.tolist())
    majority = max(counts.values()) / len(y)
    per_p = collections.defaultdict(set)
    for w in windows:
        per_p[w["participant"]].add(w["emotion_label"])
    coverage = collections.Counter(len(v) for v in per_p.values())

    # Which of the 16 channels actually carry information on a keyboard-only corpus?
    flat = X.reshape(-1, X.shape[-1])
    varying = [FEATURE_NAMES[i] for i in range(X.shape[-1]) if flat[:, i].std() > 1e-9]

    print(f"windows {len(y)}  participants {len(set(groups.tolist()))}  classes {len(labels)}")
    print("  class distribution:")
    for i, lab in enumerate(labels):
        n = counts.get(i, 0)
        print(f"    {lab:9s} {n:5d}  {100 * n / len(y):5.1f}%")
    print(f"  MAJORITY-CLASS BASELINE: {majority:.4f} accuracy — beat this or nothing was learned")
    print(f"  emotions per participant: {dict(sorted(coverage.items()))}  "
          "(most cover 1-2, so emotion is entangled with typist -> split by participant)")
    print(f"  features carrying signal: {len(varying)}/{X.shape[-1]}  ({', '.join(varying)})")
    return {"majority_baseline": float(majority),
            "class_counts": {labels[i]: int(counts.get(i, 0)) for i in range(len(labels))},
            "emotions_per_participant": {str(k): v for k, v in sorted(coverage.items())},
            "varying_features": varying}


def _fit_predict_bilstm(n_features, seed):
    def fn(Xtr, ytr, Xte):
        n_classes = int(max(ytr.max(), Xte.shape[0] and 0) + 1)
        est = TorchSequenceClassifier(int(ytr.max()) + 1, n_features, seed=seed).fit(Xtr, ytr)
        return est.predict(Xte)
    return fn


def _fit_predict_gbdt(seed):
    def fn(Xtr, ytr, Xte):
        return _WeightedGBDT(seed).fit(aggregate(Xtr), ytr).predict(aggregate(Xte))
    return fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emosurv", default="../../data/external/emosurv")
    ap.add_argument("--non-neutral", action="store_true",
                    help="drop the dominant Neutral class and classify the four emotions only")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    X, y, groups, labels, windows = load(a.emosurv, a.non_neutral)
    meta = describe(X, y, groups, labels, windows)

    results = {"labels": labels, "n": int(len(y)),
               "n_participants": int(len(set(groups.tolist()))),
               "folds": a.folds, "seed": a.seed, **meta, "arms": {}}

    arms = {
        "bilstm": _fit_predict_bilstm(X.shape[-1], a.seed),
        "gbdt": _fit_predict_gbdt(a.seed),
    }
    for name, fn in arms.items():
        print(f"\n=== {name} — {a.folds}-fold SUBJECT-independent CV ===")
        res = grouped_cv(X, y, groups, fn, n_splits=a.folds, seed=a.seed, labels=labels)
        print(format_report(res))
        results["arms"][name] = res
        acc = res["aggregate"]["accuracy"]["mean"]
        verdict = ("ABOVE" if acc > meta["majority_baseline"] else "AT OR BELOW")
        print(f"\n  vs majority baseline ({meta['majority_baseline']:.4f}): {acc:.4f} -> {verdict}")

    best = max(results["arms"], key=lambda k: results["arms"][k]["aggregate"]["macro_f1"]["mean"])
    results["best_arm_by_macro_f1"] = best
    print(f"\nbest arm by macro-F1: {best}")

    print("\n" + "=" * 78)
    print("REPORTING CONSTRAINTS — carry these wherever this number appears")
    print("=" * 78)
    print("  * Basic emotions from a TYPING task, not the four learning states. Supports the")
    print("    premise that keystroke dynamics carry affect; does NOT answer RQ1.")
    print(f"  * Keyboard-only: {len(meta['varying_features'])}/{X.shape[-1]} features vary.")
    print(f"  * Majority baseline {meta['majority_baseline']:.4f} — quote kappa, not accuracy.")
    print("  * Subject-independent split, which is mandatory here: most participants contributed")
    print("    only 1-2 emotions, so a looser split would measure typist identity.")

    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = "emosurv_4class" if a.non_neutral else "emosurv_5class"
        (out / f"{tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

        # Pooled out-of-fold predictions -> one confusion matrix over every participant.
        fn = arms[best]
        from cross_validation import stratified_group_folds
        pooled = np.full(len(y), -1, dtype=np.int64)
        for te in stratified_group_folds(groups, y, a.folds, a.seed):
            tr = ~te
            if tr.any() and te.any():
                pooled[te] = fn(X[tr], y[tr], X[te])
        keep = pooled >= 0
        preds = Predictions(y[keep], pooled[keep], labels, groups=groups[keep])
        preds.save(out / f"{tag}_predictions.npz")
        m = compute(preds)
        (out / f"{tag}_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        print("\n=== pooled out-of-fold confusion (best arm) ===")
        print(summary(m))
        plot(preds, out / f"{tag}_matrix.png", normalize="true",
             title=f"EmoSurv {len(labels)}-class — {best}, subject-independent")
        print(f"\nwrote {out / (tag + '.json')} + predictions + matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
