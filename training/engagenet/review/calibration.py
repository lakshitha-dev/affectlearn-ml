"""Calibration of the deployed geometry model's Test scores, and a gate sweep on calibrated scores.

    python training/engagenet/review/calibration.py --work C:/engagenet

The adaptation gate treats the model's P(disengaged) as a confidence and compares it with a fixed
floor, so how well that probability is calibrated matters. The model is fitted with balanced class
weights, which shifts scores upward relative to the base rate. This reports:

* expected calibration error (10 equal-width bins, weighted by bin count), Brier score and the
  reliability table for the committed Test predictions;
* a NESTED Platt recalibration: a logistic map from logit(score) to label, fitted only on the
  Train subset's out-of-fold predictions (`results/Train_4_lean_preds.npz`, participant-grouped
  5-fold) and then applied unchanged to Test. Test labels are never used to fit it;
* the same calibration metrics after recalibration, and the gate sweep on recalibrated scores.

AUC is unchanged by a monotone recalibration, so only calibration and gate behaviour move.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import _common as C

N_BINS = 10
EPS = 1e-6


def reliability(y, p, n_bins=N_BINS):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            rows.append({"bin": [float(edges[b]), float(edges[b + 1])], "n": 0,
                         "mean_score": None, "fraction_positive": None})
            continue
        ms, fp = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(ms - fp)
        rows.append({"bin": [float(edges[b]), float(edges[b + 1])], "n": int(m.sum()),
                     "mean_score": ms, "fraction_positive": fp})
    return {"ece": float(ece), "brier": float(brier_score_loss(y, p)),
            "mean_score": float(p.mean()), "base_rate": float(y.mean()), "table": rows}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(C.WORK))
    a = ap.parse_args()
    work = pathlib.Path(a.work)

    y, p, g, ids = C.committed_test_predictions(work)
    tr = np.load(work / "results" / "Train_4_lean_preds.npz", allow_pickle=False)
    ytr, ptr = tr["y_true"].astype(int), tr["y_prob"].astype(float)
    if np.isnan(ptr).any():
        raise SystemExit("ABORT: Train OOF predictions contain NaN")

    before = reliability(y, p)
    platt = LogisticRegression(C=1e6, max_iter=5000).fit(logit(ptr).reshape(-1, 1), ytr)
    pc = platt.predict_proba(logit(p).reshape(-1, 1))[:, 1]
    after = reliability(y, pc)
    train_oof = reliability(ytr, ptr)
    print(f"  Test  before: ECE {before['ece']:.4f} Brier {before['brier']:.4f} "
          f"mean score {before['mean_score']:.3f} vs base {before['base_rate']:.3f}")
    print(f"  Test  after : ECE {after['ece']:.4f} Brier {after['brier']:.4f} "
          f"mean score {after['mean_score']:.3f}")
    print(f"  Train OOF   : ECE {train_oof['ece']:.4f}")

    auc_before, auc_after = float(roc_auc_score(y, p)), float(roc_auc_score(y, pc))
    print("  gate sweep on recalibrated scores:")
    sweep = C.sweep(y, pc, g, ids, label="platt")

    C.write("calibration.json", {
        "source_predictions": "C:/engagenet/results/test_predictions.npz (committed Test dump)",
        "recalibration": {
            "method": "Platt: logistic regression of label on logit(score), unregularised",
            "fitted_on": (f"Train subset out-of-fold predictions (participant-grouped 5-fold), "
                          f"{len(ytr)} clips; Test labels not used"),
            "coef": float(platt.coef_[0][0]), "intercept": float(platt.intercept_[0]),
        },
        "n_bins": N_BINS,
        "test_before": before, "test_after_platt": after, "train_oof_before": train_oof,
        "auc_before": auc_before, "auc_after": auc_after,
        "gate_sweep_recalibrated": sweep,
        "seed": C.SEED, "versions": C.versions(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
