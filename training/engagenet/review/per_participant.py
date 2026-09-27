"""Distribution of per-participant Test AUC for the deployed geometry model.

    python training/engagenet/review/per_participant.py --work C:/engagenet

The pooled Test AUC (0.922) mixes between-participant and within-participant separation. This
reports the spread of the within-participant figure: the AUC computed inside each participant who
has both classes, with the count of participants for whom it cannot be computed (one class only).
Only summary statistics are written; the per-subject values already committed in
`reports/engagenet_screen/STAGE3_DIAGNOSTICS.json` are cross-checked, not re-published.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from sklearn.metrics import roc_auc_score

import _common as C


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(C.WORK))
    a = ap.parse_args()
    y, p, g, _ = C.committed_test_predictions(pathlib.Path(a.work))

    per, single = {}, []
    for s in np.unique(g):
        m = g == s
        if len(np.unique(y[m])) == 2:
            per[s] = float(roc_auc_score(y[m], p[m]))
        else:
            single.append({"positive_rate": float(y[m].mean()), "n": int(m.sum())})
    v = np.array(list(per.values()))

    diag = json.loads((C.REPO / "reports" / "engagenet_screen" / "STAGE3_DIAGNOSTICS.json")
                      .read_text(encoding="utf-8"))
    committed = {k: x for k, x in diag["per_subject_auc"].items() if x is not None}
    agrees = all(abs(committed[k] - per[k]) < 1e-9 for k in committed) and set(committed) == set(per)

    q1, med, q3 = np.percentile(v, [25, 50, 75])
    out = {
        "participants": int(len(np.unique(g))), "with_both_classes": int(len(v)),
        "single_class": len(single), "single_class_detail": single,
        "median": float(med), "iqr": [float(q1), float(q3)], "mean": float(v.mean()),
        "min": float(v.min()), "max": float(v.max()),
        "below_0_5": int((v < 0.5).sum()), "below_0_7": int((v < 0.7).sum()),
        "at_or_above_0_9": int((v >= 0.9).sum()),
        "agrees_with_STAGE3_DIAGNOSTICS": bool(agrees),
        "note": "Summary only; per-subject values are in STAGE3_DIAGNOSTICS.json.",
    }
    print(f"  {out['with_both_classes']} of {out['participants']} participants have both classes; "
          f"median {med:.3f} IQR [{q1:.3f}, {q3:.3f}] min {v.min():.3f} "
          f"below 0.5: {out['below_0_5']}; agrees with diagnostics: {agrees}")
    C.write("per_participant.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
