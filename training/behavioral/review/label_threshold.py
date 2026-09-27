"""DUX label-threshold sensitivity for the interaction channel (E4).

    python training/behavioral/review/label_threshold.py

The paper labels a 30 s window confused if its highest human Confusion annotation is at least 1
on DUX's discrete scale (0 / 16.5 / 33 / 50 / 66 / 100), i.e. any annotated confusion. This
re-labels the same 1,419 windows at >= 1, >= 33, >= 50 and >= 66 and re-runs the unchanged
leave-one-session-out GBDT on the same raw interaction features, reporting positives and AUC with
the session-level interval for each.

Features come from the cache written by dux_review.py (reports/dux_review/_cache), so only the
labels change between rows; labels are recomputed with external_datasets.load_dux_confusion.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "training" / "behavioral"))
import dux_review as D  # noqa: E402
from external_datasets import load_dux_confusion  # noqa: E402

THRESHOLDS = [1.0, 33.0, 50.0, 66.0]


def main() -> int:
    dux = str(ROOT / "data" / "external" / "dux")
    cache = ROOT / "reports" / "dux_review" / "_cache"
    base = D.load_all(dux, cache)
    Xb, y1, g, o = D.aggregate(base["Xb"]), base["y"], base["g"], base["o"]   # 80 raw features
    rows = []
    for t in THRESHOLDS:
        wins = load_dux_confusion(dux, threshold=t)
        lab = {(w["participant"], int(w["window_index"])): int(w["label"]) for w in wins}
        y = np.array([lab[(gg, int(oo))] for gg, oo in zip(g, o)], dtype=np.int64)
        if t == 1.0 and not (y == y1).all():
            raise SystemExit("ABORT: >= 1 labels do not reproduce the cached labels")
        prob = D.loso(Xb, y, g)
        s = D.stats(y, prob, g)
        rows.append({"threshold": t, "positives": int(y.sum()), "base_rate": float(y.mean()),
                     "sessions_with_positive": int(len(set(g[y == 1]))),
                     "auc": s["auc"], "auc_ci95_session": s["auc_ci95_session"],
                     "permutation_p_within_session": s["permutation_p_within_session"]})
        r = rows[-1]
        print(f">= {t:4.0f}: positives {r['positives']:4d} ({r['base_rate']:.3f}) sessions "
              f"{r['sessions_with_positive']:2d}  AUC {r['auc']:.3f} "
              f"{[round(x, 3) for x in r['auc_ci95_session']]}  p {r['permutation_p_within_session']:.4f}")
    out = {"_note": __doc__.strip().splitlines()[0], "windows": int(len(g)),
           "sessions": int(len(set(g))), "features": "80 raw interaction features",
           "classifier": "deployed GBDT configuration, leave-one-session-out", "rows": rows}
    (ROOT / "reports" / "dux_review" / "label_threshold.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
