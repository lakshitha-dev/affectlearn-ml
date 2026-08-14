"""DUX: behavioural vs facial vs fused detection of HUMAN-ANNOTATED confusion.

WHY THIS EXPERIMENT EXISTS
--------------------------
Every other route to a reportable Model B number was blocked. No public corpus carries
mouse+keyboard+scroll with learning-affect labels (confirmed three times, and stated in print by
the DUX authors themselves). EmoSurv has real labels but is keyboard-only and gave kappa 0.08-0.12
subject-independent, i.e. nothing. Training on the Affectiva channels would be distillation - a
behavioural model imitating a commercial facial classifier - not affect detection.

`emotion_manual_Confusion` escapes all of that. It is a HUMAN judgement, recorded on the same
sessions as both the interaction events and the facial channels, so:

  * the behavioural arm predicts an independent label, not another model's output;
  * the facial arm predicts the SAME label on the SAME windows, so the two are directly comparable;
  * a fused arm can be evaluated against both, which is a real ablation rather than two unrelated
    test sets glued together.

This is the only genuine unimodal-vs-multimodal comparison available without new data collection.

WHAT IT CANNOT SUPPORT - CARRY THESE WHEREVER THE NUMBER APPEARS
---------------------------------------------------------------
1. ONE STATE, NOT FOUR. This is binary confused-vs-not. Bored, frustrated and engaged are absent:
   `emotion_manual_Engagement` is identically zero across all 146,119 rows, so DUX cannot supply a
   human engagement label at all.
2. NOT A LEARNING TASK. DUX participants used business software (travel-expense forms), not
   learning material. Confusion during form-filling is not confusion during study.
3. TINY. 273 windows, 46 positive, 10 participants. Leave-one-participant-out is the only defensible
   protocol, and even so the confidence intervals are wide. This is evidence, not proof.
4. THE FACIAL ARM IS NOT MODEL A. DUX ships AFFDEX channel outputs, not video, so the CNN-LSTM
   cannot be run on it. The facial arm here is 12 commercial-classifier scores used as features.

Run:
    python train_dux_confusion.py
    python train_dux_confusion.py --out-dir ../../reports/dux_confusion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2] / "evaluation"))
from aggregate_features import aggregate  # noqa: E402
from confusion_matrix import compute, plot, summary  # noqa: E402
from external_datasets import DUX_AFFECTIVA, load_dux_confusion  # noqa: E402
from feature_engineering import extract_features  # noqa: E402
from predictions import Predictions  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

LABELS = ["not_confused", "confused"]


def load(dux_dir: str, threshold: float):
    windows = load_dux_confusion(dux_dir, threshold=threshold)
    if not windows:
        raise SystemExit(
            f"No DUX windows under {dux_dir}. v0.csv is a free Zenodo download "
            "(10.5281/zenodo.7778612, CC BY) and is gitignored - place it there first."
        )
    # Behavioural: the SHARED extractor, so these are the same 16 features the platform computes
    # at serve time. Facial: the 12 AFFDEX channels bin-averaged over the same window.
    Xb = np.stack([extract_features(w["events"], 0) for w in windows])
    Xf = np.stack([w["affectiva"] for w in windows])
    y = np.array([w["label"] for w in windows], dtype=np.int64)
    groups = np.array([w["participant"] for w in windows])
    ok = ~np.isnan(Xf).any(axis=1)
    if not ok.all():
        print(f"  dropped {int((~ok).sum())} windows with missing facial channels")
    return Xb[ok], Xf[ok], y[ok], groups[ok]


def _gbdt(seed: int) -> HistGradientBoostingClassifier:
    # Depth-capped and heavily regularised on purpose: with 46 positives an unconstrained booster
    # memorises participants. class_weight balances the 17/83 split so the positive class is not
    # simply ignored.
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=120, learning_rate=0.06, min_samples_leaf=8,
        l2_regularization=1.0, class_weight="balanced", random_state=seed,
    )


def leave_one_participant_out(X, y, groups, seed: int):
    """Pooled out-of-fold probabilities under LOPO.

    Pooling rather than averaging per-fold metrics is deliberate: several participants contribute
    only 2 positive windows, and an AUC computed on 2 positives is meaningless. One AUC over the
    pooled predictions - every window scored by a model that never saw that participant - is the
    honest summary.
    """
    prob = np.full(len(y), np.nan)
    for p in np.unique(groups):
        te = groups == p
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        prob[te] = _gbdt(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return prob


def _permutation_p(y, prob, groups, seed: int, n: int = 2000) -> float:
    """P(AUC this high | labels carry no information), permuting WITHIN participant.

    Within-participant permutation preserves each participant's positive rate, so the null cannot
    be beaten by learning who the participant is - which, with 10 subjects and famously identifying
    motor behaviour, is the failure mode that matters here.
    """
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(y, prob)
    hits = 0
    for _ in range(n):
        yp = y.copy()
        for p in np.unique(groups):
            m = groups == p
            yp[m] = rng.permutation(y[m])
        if len(np.unique(yp)) > 1 and roc_auc_score(yp, prob) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def _bootstrap_auc(y, prob, seed: int, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    out = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) > 1:
            out.append(roc_auc_score(y[b], prob[b]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan,) * 2


def run_arm(name: str, X, y, groups, seed: int, out_dir: Path | None) -> dict:
    prob = leave_one_participant_out(X, y, groups, seed)
    keep = ~np.isnan(prob)
    yk, pk, gk = y[keep], prob[keep], groups[keep]
    pred = (pk >= 0.5).astype(np.int64)

    auc = roc_auc_score(yk, pk)
    lo, hi = _bootstrap_auc(yk, pk, seed)
    perm = _permutation_p(yk, pk, gk, seed)

    two_col = np.column_stack([1.0 - pk, pk])
    records = Predictions(yk, pred, LABELS, y_prob=two_col, groups=gk)
    m = compute(records)

    print(f"\n=== arm: {name} ({X.shape[1]} features) ===")
    print(f"  AUC            {auc:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  permutation p  {perm:.4f}   (within-participant null)")
    print(summary(m))
    if out_dir:
        records.save(out_dir / f"{name}_predictions.npz")
        plot(records, out_dir / f"{name}_matrix.png", normalize="true",
             title=f"DUX human-annotated confusion - {name}, LOPO")
    return {"n_features": int(X.shape[1]), "auc": float(auc), "auc_ci95": [lo, hi],
            "permutation_p": float(perm), **{k: v for k, v in m.items() if k != "matrix"},
            "_prob": pk.tolist(), "_y": yk.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dux", default="../../data/external/dux")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="min MAX annotation intensity in a window to call it confused")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--per-learner-z", action="store_true",
                    help="z-score each feature within the participant's own windows before "
                         "classifying (see dux_sensitivity.py: worth +0.079 AUC at 30 s)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir) if a.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    Xb, Xf, y, groups = load(a.dux, a.threshold)
    pos = int(y.sum())
    majority = 1.0 - pos / len(y)
    print(f"windows {len(y)}  participants {len(set(groups.tolist()))}  "
          f"confused {pos} ({100 * pos / len(y):.1f}%)")
    print(f"  MAJORITY BASELINE: {majority:.4f} accuracy by always saying 'not confused' while")
    print("  detecting nothing. Quote AUC and confused-class recall, never bare accuracy.")
    print(f"  positives per participant: "
          f"{{{', '.join(f'{p.split(chr(95))[-1]}:{int(y[groups == p].sum())}' for p in np.unique(groups))}}}")

    # Fusion is early/feature-level: one classifier over both channels. With 46 positives a late
    # fusion layer would have more parameters than data to fit them.
    Ab = aggregate(Xb)
    if a.per_learner_z:
        # Both channels are normalised, not just the behavioural one: a fusion comparison in which
        # one arm gets the better representation and the other does not is not an ablation.
        from dux_sensitivity import per_participant_z
        Ab, Xf = per_participant_z(Ab, groups), per_participant_z(Xf, groups)
        print("  per-learner z-scoring APPLIED to both channels")
    arms = {"behavioural": Ab, "facial": Xf, "fused": np.hstack([Ab, Xf])}
    results = {"n": int(len(y)), "n_participants": int(len(set(groups.tolist()))),
               "n_confused": pos, "majority_baseline": float(majority),
               "threshold": a.threshold, "seed": a.seed,
               "affectiva_channels": DUX_AFFECTIVA, "arms": {}}
    for name, X in arms.items():
        results["arms"][name] = run_arm(name, X, y, groups, a.seed, out_dir)

    print("\n" + "=" * 78)
    print("ARM COMPARISON (pooled LOPO)")
    print("=" * 78)
    print(f"  {'arm':14s} {'AUC':>6s}  {'95% CI':>16s}  {'perm p':>7s}  {'confused recall':>15s}")
    for name, r in results["arms"].items():
        rec = r["per_class"]["confused"]["recall"]
        print(f"  {name:14s} {r['auc']:6.3f}  [{r['auc_ci95'][0]:.3f}, {r['auc_ci95'][1]:.3f}]  "
              f"{r['permutation_p']:7.4f}  {rec:15.3f}")

    best_uni = max(("behavioural", "facial"), key=lambda k: results["arms"][k]["auc"])
    gain = results["arms"]["fused"]["auc"] - results["arms"][best_uni]["auc"]
    print(f"\n  fusion vs best unimodal ({best_uni}): AUC {gain:+.3f}")
    print("  D'Mello & Kory (2015) prior: 9.83% mean / 6.60% MEDIAN gain, but only 4.59% on")
    print("  NATURAL data (12.7% on acted). A gain far")
    print("  above that on 46 positives is more likely noise than a finding.")
    lo_f, hi_f = results["arms"]["fused"]["auc_ci95"]
    lo_u, hi_u = results["arms"][best_uni]["auc_ci95"]
    if not (lo_f > hi_u or lo_u > hi_f):
        print("  NOTE: the two confidence intervals OVERLAP - this comparison cannot establish that")
        print("  fusion helps. Report it as inconclusive, not as a gain.")

    if out_dir:
        (out_dir / "dux_confusion.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {out_dir / 'dux_confusion.json'} + predictions + matrices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
