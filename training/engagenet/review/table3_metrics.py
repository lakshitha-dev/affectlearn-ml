"""EngageNet rows of paper Table 3: accuracy, balanced accuracy and Cohen's kappa at 0.5 (E5).

    python training/engagenet/review/table3_metrics.py --work C:/engagenet

Refits the baselines exactly as baselines.py does (same features, seeds and settings), checks
that the AUCs reproduce baselines.json, then adds balanced accuracy, which baselines.json lacks.
The deployed GBDT uses the committed test predictions, over all clips and over the clips the
browser would score (>= 5 of 10 frames with a face). Aggregates only.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _common as C


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"n": int(len(y)), "auc": float(roc_auc_score(y, p)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "kappa": float(cohen_kappa_score(y, pred)),
            "majority_accuracy": float(max(y.mean(), 1 - y.mean())), "threshold": thr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(C.WORK))
    work = pathlib.Path(ap.parse_args().work)
    ftr, ytr, _, _ = C.load_split("Train", work)
    fte, yte, _, ids_te = C.load_split("Test", work)
    Xtr, Xte = C.rungs.build_matrix(ftr, C.RUNG), C.rungs.build_matrix(fte, C.RUNG)
    j = C.rungs.feature_names(C.RUNG).index("mean_gaze_y")

    scores = {}
    lr1 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                             max_iter=5000))
    scores["mean_gaze_y_logistic"] = lr1.fit(Xtr[:, [j]], ytr).predict_proba(Xte[:, [j]])[:, 1]
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                            max_iter=5000))
    scores["logistic_regression"] = lr.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=C.SEED,
                                n_jobs=-1)
    scores["random_forest"] = rf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    yd, pd_, _, idd = C.committed_test_predictions(work)
    scores["gbdt_deployed"] = np.asarray(C.align(list(idd), pd_, ids_te), dtype=float)

    ref = json.loads((C.REPO / "reports" / "engagenet_review" / "baselines.json")
                     .read_text(encoding="utf-8"))["binary"]
    rows = {}
    for name, p in scores.items():
        rows[name] = metrics(yte, p)
        key = "gbdt_committed" if name == "gbdt_deployed" else name
        if abs(rows[name]["auc"] - ref[key]["auc"]) > 1e-9:
            raise SystemExit(f"ABORT: {name} AUC {rows[name]['auc']} != baselines.json {ref[key]['auc']}")

    found = fte[:, :, C.rungs.NAMES.index("face_found")]
    ok = np.sum(found == 1.0, axis=1) >= 5
    rows["gbdt_deployed_scorable"] = metrics(yte[ok], scores["gbdt_deployed"][ok])

    C.write("table3_metrics.json", {"threshold": 0.5, "rows": rows,
                                    "note": "AUCs reproduce baselines.json exactly"})
    for k, r in rows.items():
        print(f"{k:26s} n {r['n']} AUC {r['auc']:.3f} acc {r['accuracy']:.3f} "
              f"bacc {r['balanced_accuracy']:.3f} kappa {r['kappa']:.3f} (majority {r['majority_accuracy']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
