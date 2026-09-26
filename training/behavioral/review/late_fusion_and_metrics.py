"""DUX: late fusion (E3) and the full threshold metrics for every Table 3 row (E5).

    python training/behavioral/review/late_fusion_and_metrics.py

E3. The platform fuses readings late (it combines the two models' outputs), while
paper-v4 RQ2 tested early fusion (concatenated features). Here the leave-one-session-out
probabilities of the interaction and AFFDEX arms, both raw features, are combined late as their
mean and their maximum, and each is compared with interaction alone by a paired session bootstrap.

E5. For every DUX row of Table 3: AUC with the session interval, accuracy, balanced accuracy and
Cohen's kappa at 0.5, against the majority baseline.

Reads reports/dux_review/predictions/*.npz (written by dux_review.py; identical windows across
arms, checked below). Writes reports/dux_review/late_fusion.json and table3_metrics.json.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))
import dux_review as D  # noqa: E402

PRED = REPO / "reports" / "dux_review" / "predictions"
OUT = REPO / "reports" / "dux_review"


def load(stem):
    d = np.load(PRED / f"{stem}.npz", allow_pickle=False)
    return d["y_true"], d["y_prob"].astype(float), d["groups"], d["window_index"]


def threshold_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "kappa": float(cohen_kappa_score(y, pred)),
            "majority_accuracy": float(max(y.mean(), 1 - y.mean()))}


def main() -> int:
    yi, pi, gi, wi = load("raw_interaction")
    yf, pf, gf, wf = load("raw_facial")
    if not ((yi == yf).all() and (gi == gf).all() and (wi == wf).all()):
        raise SystemExit("ABORT: interaction and facial predictions are not aligned")

    fused = {"late_mean": (pi + pf) / 2.0, "late_max": np.maximum(pi, pf)}
    e3 = {"inputs": "LOSO P(confused), raw features: interaction (80) and AFFDEX (60)",
          "interaction_alone": D.stats(yi, pi, gi)}
    for name, p in fused.items():
        e3[name] = D.stats(yi, p, gi)
        e3[f"interaction_minus_{name}"] = D.paired_diff(yi, pi, p, gi)
    (OUT / "late_fusion.json").write_text(json.dumps(e3, indent=2), encoding="utf-8")
    for name in fused:
        d = e3[f"interaction_minus_{name}"]
        print(f"{name:10s} AUC {e3[name]['auc']:.3f} {[round(x, 3) for x in e3[name]['auc_ci95_session']]}"
              f"  interaction - {name}: {d['delta_auc']:+.3f} {[round(x, 3) for x in d['ci95_session']]}")

    rows = {}
    for label, stem in (("interaction_gbdt_deployed", "raw_interaction"),
                        ("interaction_platform_like", "matched_serving_rule_interaction"),
                        ("random_forest", "baseline_rf_interaction"),
                        ("logistic_regression", "baseline_lr_interaction"),
                        ("affdex_expression", "raw_facial"),
                        ("early_fusion", "raw_fused")):
        y, p, g, _ = load(stem)
        s = D.stats(y, p, g)
        rows[label] = {"n": s["n"], "auc": s["auc"], "auc_ci95_session": s["auc_ci95_session"],
                       **threshold_metrics(y, p)}
        r = rows[label]
        print(f"{label:28s} n {r['n']} AUC {r['auc']:.3f} acc {r['accuracy']:.3f} "
              f"bacc {r['balanced_accuracy']:.3f} kappa {r['kappa']:.3f} (majority {r['majority_accuracy']:.3f})")
    (OUT / "table3_metrics.json").write_text(json.dumps(
        {"threshold": 0.5, "note": "leave-one-session-out predictions, raw features", "rows": rows},
        indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
