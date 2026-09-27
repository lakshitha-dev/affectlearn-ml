"""Baselines for the geometry channel on the same Train -> Test protocol and the same 20 features.

    python training/engagenet/review/baselines.py --work C:/engagenet

The paper reported the deployed GBDT (Test AUC 0.922) against a majority baseline only. This adds
the comparators a reviewer asks for, all fitted on the same Train subset (2,177 clips / 91
participants), scored once on Test (2,256 clips / 26 participants), with the same metric set:

* prior: a constant score (AUC 0.5 by construction); accuracy equals the majority rate.
* mean_gaze_y: the single strongest feature, used raw as a score (no fitting) and, separately,
  through a one-feature logistic regression fitted on Train.
* logistic regression: L2 (C = 1), standardised on Train, balanced class weights.
* random forest: 500 trees, balanced class weights, otherwise scikit-learn defaults.
* GBDT: the deployed configuration refitted under the current scikit-learn, plus the committed
  predictions (the numbers the paper quotes) for reference.

All fitted models use balanced class weights, as the GBDT does, so accuracy and kappa at the 0.5
threshold are comparable across rows. Each row carries a participant-bootstrap AUC interval and a
within-participant permutation p, and the GBDT is compared with every baseline through a paired
participant bootstrap of the AUC difference.

A 4-class result is added for comparison with the EngageNet paper's baselines, which report 4-class
accuracy only. It is not directly comparable: EngageNet's baselines use their own gaze, head-pose
and action-unit features and the full training split, while this uses our 20 geometry features and
the 2,177-clip subset.
"""
from __future__ import annotations

import argparse
import pathlib
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _common as C

LEVELS = ["Not-Engaged", "Barely-engaged", "Engaged", "Highly-Engaged"]


def four_class_labels(work: pathlib.Path, split: str, ids: list[str]) -> np.ndarray:
    lp = work / "labels" / C.rungs.LABEL_FILE[split]
    lab = pd.read_csv(lp) if lp.suffix == ".csv" else pd.read_excel(lp)
    m = {re.sub(r"\.mp4$", "", str(k)): str(v) for k, v in zip(lab["chunk"], lab["label"])}
    out = np.array([LEVELS.index(m[c]) for c in ids])
    return out


def cluster_acc_ci(y, pred, g, n=C.N_BOOT, seed=C.SEED):
    rng = np.random.default_rng(seed)
    ids = np.unique(g)
    by = {s: np.flatnonzero(g == s) for s in ids}
    accs = []
    for _ in range(n):
        b = np.concatenate([by[s] for s in rng.choice(ids, size=len(ids), replace=True)])
        accs.append(float((pred[b] == y[b]).mean()))
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(C.WORK))
    a = ap.parse_args()
    work = pathlib.Path(a.work)

    ftr, ytr, gtr, ids_tr = C.load_split("Train", work)
    fte, yte, gte, ids_te = C.load_split("Test", work)
    Xtr = C.rungs.build_matrix(ftr, C.RUNG)
    Xte = C.rungs.build_matrix(fte, C.RUNG)
    names = C.rungs.feature_names(C.RUNG)
    j = names.index("mean_gaze_y")
    print(f"Train {len(ytr)} / {len(set(gtr))} | Test {len(yte)} / {len(set(gte))} | "
          f"{Xtr.shape[1]} features")

    scores: dict[str, np.ndarray] = {}
    scores["prior"] = np.full(len(yte), float(ytr.mean()))
    scores["mean_gaze_y_raw"] = Xte[:, j]
    lr1 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                             max_iter=5000))
    scores["mean_gaze_y_logistic"] = lr1.fit(Xtr[:, [j]], ytr).predict_proba(Xte[:, [j]])[:, 1]
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                            max_iter=5000))
    scores["logistic_regression"] = lr.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=C.SEED,
                                n_jobs=-1)
    scores["random_forest"] = rf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    scores["gbdt_refit"] = C.rungs.gbdt(C.SEED).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    yd, pd_, gd, idd = C.committed_test_predictions(work)
    scores["gbdt_committed"] = C.align(list(idd), pd_, ids_te)
    if not (C.align(list(idd), yd, ids_te) == yte).all():
        raise SystemExit("ABORT: committed dump labels differ from this load")

    rows = {}
    for name, p in scores.items():
        if name == "prior":
            base = float(yte.mean())
            rows[name] = {"n": int(len(yte)), "participants": int(len(set(gte))),
                          "positive_rate": base, "auc": 0.5, "auc_ci95_participant": None,
                          "accuracy": float(max(base, 1 - base)),
                          "majority_baseline": float(max(base, 1 - base)), "kappa": 0.0,
                          "average_precision": base, "ap_over_base_rate": 1.0,
                          "decision_threshold": None,
                          "note": "constant score; AUC 0.5 and kappa 0 by construction"}
            continue
        thr = None if name == "mean_gaze_y_raw" else C.THRESHOLD
        blk = C.auc_block(yte, p, gte, threshold=C.THRESHOLD if thr is not None else np.median(p))
        if thr is None:
            blk["decision_threshold"] = "median score (a raw feature has no 0.5 threshold)"
        rows[name] = blk
        print(f"  {name:22} AUC {blk['auc']:.4f} [{blk['auc_ci95_participant'][0]:.3f}, "
              f"{blk['auc_ci95_participant'][1]:.3f}] acc {blk['accuracy']:.3f} "
              f"kappa {blk['kappa']:.3f} AP/base {blk['ap_over_base_rate']:.2f} "
              f"perm p {blk.get('perm_p_within_participant', float('nan')):.4f}")

    paired = {}
    for name in ("mean_gaze_y_logistic", "logistic_regression", "random_forest", "gbdt_refit"):
        paired[f"gbdt_committed_minus_{name}"] = C.paired_auc_diff(
            yte, scores["gbdt_committed"], scores[name], gte)
        d = paired[f"gbdt_committed_minus_{name}"]
        print(f"  GBDT(committed) - {name:22} dAUC {d['delta_auc']:+.4f} "
              f"[{d['ci95_participant'][0]:+.4f}, {d['ci95_participant'][1]:+.4f}]")

    # ── 4-class ───────────────────────────────────────────────────────────────────────
    y4tr = four_class_labels(work, "Train", ids_tr)
    y4te = four_class_labels(work, "Test", ids_te)
    four = {"levels": LEVELS, "train_counts": np.bincount(y4tr, minlength=4).tolist(),
            "test_counts": np.bincount(y4te, minlength=4).tolist(),
            "test_majority_accuracy": float(np.bincount(y4te).max() / len(y4te))}
    for tag, cw in (("balanced", "balanced"), ("unweighted", None)):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=120, learning_rate=0.06,
                                           min_samples_leaf=8, l2_regularization=1.0,
                                           class_weight=cw, random_state=C.SEED)
        pred = m.fit(Xtr, y4tr).predict(Xte)
        four[tag] = {"accuracy": float(accuracy_score(y4te, pred)),
                     "accuracy_ci95_participant": cluster_acc_ci(y4te, pred, gte),
                     "macro_f1": float(f1_score(y4te, pred, average="macro")),
                     "kappa": float(cohen_kappa_score(y4te, pred)),
                     "confusion_counts": pd.crosstab(pd.Series(y4te, name="true"),
                                                     pd.Series(pred, name="pred"))
                                           .reindex(index=range(4), columns=range(4), fill_value=0)
                                           .values.tolist()}
        print(f"  4-class GBDT ({tag:10}) acc {four[tag]['accuracy']:.4f} "
              f"macro-F1 {four[tag]['macro_f1']:.4f} (majority {four['test_majority_accuracy']:.4f})")
    four["engagenet_paper_best_test_accuracy"] = 0.6761
    four["comparability"] = ("Not directly comparable with the EngageNet paper: different features "
                             "(20 landmark-geometry statistics against the paper's gaze, head-pose "
                             "and action-unit features), different models, and a 2,177-clip "
                             "training subset against the full 7,983-clip split.")

    C.write("baselines.json", {
        "protocol": "fit on Train subset, scored once on Test; same 20 features as the deployed model",
        "train": f"{len(ytr)} clips / {len(set(gtr))} participants",
        "test": f"{len(yte)} clips / {len(set(gte))} participants",
        "features": names, "seed": C.SEED, "n_bootstrap": C.N_BOOT,
        "decision_threshold": C.THRESHOLD,
        "threshold_source": ("rungs.report() thresholds at 0.5; STAGE3_TEST.json was produced by "
                             "that function, and engagenet_lean_gbdt.json records "
                             "threshold_default 0.5"),
        "binary": rows, "paired_vs_gbdt_committed": paired, "four_class": four,
        "versions": C.versions(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
