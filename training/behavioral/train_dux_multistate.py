"""DUX v1: multi-state affect detection from interaction behaviour, human-annotated.

WHY THIS EXISTS ON TOP OF `train_dux_confusion.py`
--------------------------------------------------
v0 could only support a binary confused-vs-not task: it had 4,202 confusion annotations and almost
nothing else. v1 (the release with emotional triggers ENABLED) has 38,183 confusion annotations
plus 11,676 Joy, 7,099 Anger and 3,709 Surprise, over 36 sessions. That is enough to ask a harder
and much more relevant question: can interaction behaviour separate SEVERAL affective states, not
just detect one?

This matters because the platform's taxonomy has four states, and a binary confusion detector
answers only a quarter of RQ1.

HOW THE LABELS MAP TO THE PLATFORM'S STATES - AND WHERE THE MAPPING STOPS
------------------------------------------------------------------------
  Confusion -> CONFUSED       direct, same construct, no reinterpretation needed
  Anger     -> FRUSTRATED     a PROXY. Frustration is commonly modelled as low-intensity,
                              goal-blockage anger, and AFFDEX has no frustration channel, so Anger
                              is the closest available. It is not the same construct and must be
                              named as a proxy every time it is reported.
  Joy       -> (nothing)      Joy is NOT engagement. A learner can be engaged and unsmiling, and
                              amused while off-task. It is kept as its own class because dropping
                              an annotated state would silently move its windows into the negative
                              class, but it must NEVER be relabelled "engaged".
  BOREDOM   -> unavailable    absent from AFFDEX entirely, and therefore from the human annotation
                              scheme built on it. Boredom cannot be studied in DUX at all.
  ENGAGEMENT-> unavailable    `emotion_manual_Engagement` covers 489 of 590,738 rows in v1 and is
                              identically zero in v0. Too sparse to learn or to evaluate.

So this experiment reaches TWO of the platform's four states (confused, frustrated-as-anger-proxy).
The remaining two have to come from the platform's own data or, for engagement, from DAiSEE.

THE NEGATIVE CLASS IS AN ABSENCE, NOT A JUDGEMENT
-------------------------------------------------
`emotion_manual_Neutral` is never annotated in either file. "none" therefore means "no annotator
marked any emotion here", which conflates genuine calm with unlabelled time. That inflates apparent
performance on the none class and is a real limitation, not a technicality.

Run:
    python train_dux_multistate.py                    # v0 + v1, whatever is on disk
    python train_dux_multistate.py --v1-only          # 36 trigger-enabled sessions only
    python train_dux_multistate.py --out-dir ../../reports/dux_multistate
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2] / "evaluation"))
from aggregate_features import aggregate  # noqa: E402
from confusion_matrix import compute, plot, summary  # noqa: E402
from cross_validation import stratified_group_folds  # noqa: E402
from dux_sensitivity import per_participant_z  # noqa: E402
from external_datasets import (DUX_AFFECTIVA, _DUX_TYPE, _MIN_EVENTS, _SCREEN_H,  # noqa: E402
                               _SCREEN_W, WINDOW_MS, _is_backspace)
from feature_engineering import extract_features  # noqa: E402
from predictions import Predictions  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

# Order matters: index 0 is the negative/unannotated class.
STATES = ["none", "Confusion", "Anger", "Joy"]
PLATFORM_MAP = {"Confusion": "confused", "Anger": "frustrated (PROXY)", "Joy": "-- no mapping --",
                "none": "not annotated"}


def load_multistate(dux_dir: str, files_wanted: list[str], window_ms: int) -> list[dict]:
    """Windows labelled by whichever annotated state has the highest intensity in the window.

    Ties go to the earlier entry in STATES, which biases toward Confusion - the state with by far
    the most annotation, so the tie-break cannot manufacture a rare class.
    """
    d = Path(dux_dir)
    files = [d / n for n in files_wanted if (d / n).exists()]
    if not files:
        return []
    aff_cols = [f"emotion_affectiva_{c}" for c in DUX_AFFECTIVA]
    man_cols = [f"emotion_manual_{s}" for s in STATES[1:]]
    cols = ["session", "timestamp", "type", "key", "x", "y", "yPosition"] + man_cols + aff_cols

    out: list[dict] = []
    for f in files:
        raw = pd.read_csv(f, sep="\t", usecols=cols, low_memory=False)
        for c in man_cols + aff_cols:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw[man_cols] = raw[man_cols].fillna(0.0)
        raw["ts"] = pd.to_numeric(raw["timestamp"], errors="coerce")
        raw = raw.dropna(subset=["ts"])

        for sess, g_all in raw.groupby("session"):
            g_all = g_all.sort_values("ts")
            t0 = int(g_all["ts"].min())
            wi_all = ((g_all["ts"] - t0) // window_ms).astype(int)
            lab, facial = {}, {}
            for w, sub in g_all.groupby(wi_all):
                peak = sub[man_cols].max()                     # per-channel max within the window
                lab[int(w)] = 0 if peak.max() < 1.0 else int(np.argmax(peak.to_numpy())) + 1
                facial[int(w)] = sub[aff_cols].mean().to_numpy(dtype=np.float64)

            g = g_all[g_all["type"].isin(_DUX_TYPE)].copy()
            if g.empty:
                continue
            is_bs = _is_backspace(g["key"])
            g["type"] = g["type"].map(_DUX_TYPE)
            g["x"] = np.clip(pd.to_numeric(g["x"], errors="coerce").fillna(0.0) / _SCREEN_W, 0, 1)
            g["y"] = np.clip(pd.to_numeric(g["y"], errors="coerce").fillna(0.0) / _SCREEN_H, 0, 1)
            g["ypos"] = pd.to_numeric(g["yPosition"], errors="coerce")
            g["key"] = np.where(is_bs, "Backspace", np.where(g["type"] == "key", "a", ""))
            g = g.reset_index(drop=True)
            g["dy"] = 0.0
            srows = g.index[g["type"] == "scroll"]
            if len(srows):
                g.loc[srows, "dy"] = g.loc[srows, "ypos"].diff().fillna(0.0).to_numpy()

            for w, sub in g.groupby(((g["ts"] - t0) // window_ms).astype(int)):
                if len(sub) < _MIN_EVENTS:
                    continue
                ev = sub[["ts", "type", "x", "y", "key", "dy"]].copy()
                ev["ts"] = (ev["ts"] - t0) - int(w) * window_ms
                out.append({"participant": f"dux_{f.stem}_{sess}", "label": lab.get(int(w), 0),
                            "affectiva": facial.get(int(w)), "events": ev.reset_index(drop=True)})
    return out


def _gbdt(seed: int, n_classes: int):
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=150, learning_rate=0.06, min_samples_leaf=10,
        l2_regularization=1.0, class_weight="balanced", random_state=seed)


def grouped_oof(X, y, groups, seed: int, n_splits: int):
    """Pooled out-of-fold predictions over STRATIFIED grouped folds.

    Leave-one-participant-out is not usable here: with four classes and rare states, a single held-
    out participant often contains only one class, so per-fold metrics are undefined. Stratified
    grouped folds keep every class present in every fold while never splitting a participant across
    train and test.
    """
    prob = np.full((len(y), len(STATES)), np.nan)
    for te in stratified_group_folds(groups, y, n_splits, seed):
        tr = ~te
        if not (tr.any() and te.any()) or len(np.unique(y[tr])) < 2:
            continue
        est = _gbdt(seed, len(STATES)).fit(X[tr], y[tr])
        p = est.predict_proba(X[te])
        # A fold missing a class yields fewer probability columns; scatter them back by class id so
        # the pooled matrix stays aligned to STATES rather than to the fold's local ordering.
        for j, c in enumerate(est.classes_):
            prob[te, int(c)] = p[:, j]
        prob[te] = np.nan_to_num(prob[te], nan=0.0)
    return prob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dux", default="../../data/external/dux")
    ap.add_argument("--v1-only", action="store_true",
                    help="use only v1 (triggers ENABLED); v0's triggers were disabled")
    ap.add_argument("--window-s", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw", action="store_true", help="skip per-learner z-scoring")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    wanted = ["v1.csv"] if a.v1_only else ["v0.csv", "v1.csv"]
    win = load_multistate(a.dux, wanted, a.window_s * 1000)
    if not win:
        raise SystemExit(f"No DUX windows under {a.dux} for {wanted}")

    Xb = np.stack([extract_features(w["events"], 0, window_length_ms=a.window_s * 1000,
                                   bin_length_ms=1_000) for w in win])
    Xf = np.stack([w["affectiva"] for w in win])
    y = np.array([w["label"] for w in win], dtype=np.int64)
    groups = np.array([w["participant"] for w in win])
    ok = ~np.isnan(Xf).any(axis=1)
    Xb, Xf, y, groups = Xb[ok], Xf[ok], y[ok], groups[ok]

    counts = collections.Counter(y.tolist())
    majority = max(counts.values()) / len(y)
    print(f"files {wanted}  windows {len(y)}  participants {len(set(groups.tolist()))}  "
          f"window {a.window_s}s")
    print(f"  {'state':12s} {'n':>6s} {'share':>7s}   platform mapping")
    for i, s in enumerate(STATES):
        n = counts.get(i, 0)
        print(f"  {s:12s} {n:6d} {100 * n / len(y):6.1f}%   {PLATFORM_MAP[s]}")
    print(f"  MAJORITY BASELINE {majority:.4f} accuracy. Quote macro-F1 and kappa, never accuracy.")
    absent = [STATES[i] for i in range(len(STATES)) if counts.get(i, 0) < a.folds]
    if absent:
        print(f"  WARNING: {absent} have fewer windows than folds - metrics for them are unstable")

    Ab = aggregate(Xb)
    if not a.raw:
        Ab, Xf = per_participant_z(Ab, groups), per_participant_z(Xf, groups)
        print("  per-learner z-scoring applied to both channels")

    results = {"files": wanted, "window_s": a.window_s, "n": int(len(y)),
               "n_participants": int(len(set(groups.tolist()))), "folds": a.folds,
               "states": STATES, "platform_mapping": PLATFORM_MAP,
               "class_counts": {STATES[i]: int(counts.get(i, 0)) for i in range(len(STATES))},
               "majority_baseline": float(majority), "arms": {}}

    out_dir = Path(a.out_dir) if a.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name, X in (("behavioural", Ab), ("facial", Xf), ("fused", np.hstack([Ab, Xf]))):
        prob = grouped_oof(X, y, groups, a.seed, a.folds)
        keep = ~np.isnan(prob).all(axis=1)
        recs = Predictions(y[keep], prob[keep].argmax(1), STATES,
                           y_prob=prob[keep], groups=groups[keep])
        m = compute(recs)
        print(f"\n=== {name} ({X.shape[1]} features) ===")
        print(summary(m))
        results["arms"][name] = {k: v for k, v in m.items() if k != "matrix"}
        if out_dir:
            recs.save(out_dir / f"{name}_predictions.npz")
            plot(recs, out_dir / f"{name}_matrix.png", normalize="true",
                 title=f"DUX human annotation, {len(STATES)}-class - {name}")

    print("\n" + "=" * 78)
    print(f"  {'arm':14s} {'macroF1':>8s} {'kappa':>7s} {'acc':>7s}   per-state recall")
    for name, m in results["arms"].items():
        rec = "  ".join(f"{s[:4]}={m['per_class'][s]['recall']:.2f}" for s in STATES)
        print(f"  {name:14s} {m['macro_f1']:8.3f} {m['cohen_kappa']:7.3f} "
              f"{m['accuracy']:7.3f}   {rec}")
    # Deliberately NOT "below baseline means it learned nothing". Every arm here is fitted with
    # class_weight="balanced", which trades majority-class accuracy for minority-class recall on
    # purpose, so accuracy below the majority baseline is the EXPECTED cost of that choice rather
    # than evidence of failure. kappa is the metric that separates the two cases: kappa > 0 with
    # non-trivial minority recall means real learning at a deliberately shifted operating point,
    # whereas kappa ~ 0 means nothing was learned however the accuracy reads.
    print(f"\n  majority baseline accuracy {majority:.3f} (always predict '{STATES[0]}')")
    print("  Arms are class-weighted, so accuracy BELOW this baseline is expected and is not by")
    print("  itself a failure. Judge by kappa and per-state recall: kappa ~ 0 means nothing learned;")
    print("  kappa > 0 with real minority recall means learning at a shifted operating point.")
    print("  A state whose recall is ~0.00 was not learned AT ALL, whatever the aggregate says.")
    print("  Confusion -> confused; Anger -> frustrated is a PROXY; Joy maps to NOTHING and must")
    print("  never be reported as engagement. Boredom is absent from AFFDEX and so from DUX.")

    if out_dir:
        (out_dir / "multistate.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {out_dir / 'multistate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
