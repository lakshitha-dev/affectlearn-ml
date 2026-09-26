"""DUX: intense confusion against no confusion, with mildly confused windows left out (E12).

    python training/behavioral/review/label_exclusive.py

WHY

label_threshold.py re-labels at >= 50, which turns windows annotated only as mildly confused
(0 < maximum < 50) into negatives. That mixes a harder target with label noise. Here those windows
are removed and the remaining windows are scored two ways:

  deployed_model   the leave-one-session-out predictions of the deployed configuration trained on
                   the paper's label (any confusion), scored on intense (>= 50) against none (0).
  retrained        the same GBDT configuration retrained leave-one-session-out on intense against
                   none only.

Features and the deployed-model predictions come from dux_review.py's cache and prediction files.
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

INTENSE = 50.0


def main() -> int:
    dux = str(ROOT / "data" / "external" / "dux")
    base = D.load_all(dux, ROOT / "reports" / "dux_review" / "_cache")
    Xb, y_any, g, o = D.aggregate(base["Xb"]), base["y"], base["g"], base["o"]
    lab = {(w["participant"], int(w["window_index"])): int(w["label"])
           for w in load_dux_confusion(dux, threshold=INTENSE)}
    y_int = np.array([lab[(gg, int(oo))] for gg, oo in zip(g, o)], dtype=np.int64)
    mild = (y_any == 1) & (y_int == 0)
    keep = ~mild

    pred = np.load(ROOT / "reports" / "dux_review" / "predictions" / "raw_interaction.npz",
                   allow_pickle=False)
    key = {(str(gg), int(ww)): (float(pp), int(yy)) for gg, ww, pp, yy in
           zip(pred["groups"], pred["window_index"], pred["y_prob"], pred["y_true"])}
    p_dep = np.array([key[(str(gg), int(oo))][0] for gg, oo in zip(g, o)])
    y_file = np.array([key[(str(gg), int(oo))][1] for gg, oo in zip(g, o)])
    if not np.array_equal(y_file, y_any):
        raise SystemExit("ABORT: prediction file does not match the cached labels")

    rows = {}
    s = D.stats(y_int[keep], p_dep[keep], g[keep])
    rows["deployed_model"] = s
    prob = D.loso(Xb[keep], y_int[keep], g[keep])
    rows["retrained"] = D.stats(y_int[keep], prob, g[keep])
    out = {"_note": __doc__.strip().splitlines()[0],
           "windows_total": int(len(g)), "mild_removed": int(mild.sum()),
           "windows_kept": int(keep.sum()), "intense_positives": int(y_int[keep].sum()),
           "none_negatives": int((y_int[keep] == 0).sum()),
           "sessions_kept": int(len(set(g[keep]))),
           "intense_threshold": INTENSE, "rows": {
               k: {"auc": v["auc"], "auc_ci95_session": v["auc_ci95_session"],
                   "permutation_p_within_session": v["permutation_p_within_session"]}
               for k, v in rows.items()}}
    (ROOT / "reports" / "dux_review" / "label_exclusive.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"kept {out['windows_kept']} (removed {out['mild_removed']} mild); "
          f"intense {out['intense_positives']} vs none {out['none_negatives']}")
    for k, v in out["rows"].items():
        print(f"  {k:15s} AUC {v['auc']:.3f} {[round(x, 3) for x in v['auc_ci95_session']]} "
              f"p {v['permutation_p_within_session']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
