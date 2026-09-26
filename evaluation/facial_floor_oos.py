"""Choose the facial confidence floor without the test split, then report it on test.

    python evaluation/facial_floor_oos.py --work C:/engagenet

WHY

The deployed facial floor of 0.70 was first chosen by `geometry_gate_calibration.py` on the
EngageNet TEST predictions, so the test precision at 0.70 is not an out-of-sample estimate. This
script chooses the floor on data the test split never touched and reports test precision at it.

RULES (fixed here before the training sweep was looked at)

  primary     the lowest floor in 0.50..0.90 (step 0.05) whose precision on the TRAINING-split
              out-of-fold predictions is at least 0.80; if none reaches 0.80, the floor with the
              highest training precision.
  sensitivity (a) the floor with the highest training lift among floors with at least 10 offers;
              (b) leave-one-test-participant-out: for each test participant, apply the primary rule
              to the other 25 test participants and score the held-out one; offers pooled.

All replays use the deployed gate rule (evaluation/gate_deployed_rule.py: floor on the current
reading, the last two readings in the state, three cycles between offers) and mask windows the
browser would not score (< 5 of 10 frames with a face), in recording order per participant.
Training predictions are the stage-2 participant-grouped 5-fold out-of-fold predictions
(C:/engagenet/results/Train_4_lean_preds.npz). Aggregates only are written.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "evaluation"))
sys.path.insert(0, str(REPO / "training" / "engagenet" / "review"))

import gate_deployed_rule as G  # noqa: E402

TARGET_PRECISION = 0.80
MIN_OFFERS_LIFT = 10


def _split(work: pathlib.Path, split: str, preds: pathlib.Path):
    import _common as C
    from geometry_gate_calibration import temporal_order

    feats, _, _, ids_lab = C.load_split(split, work)
    found = feats[:, :, C.rungs.NAMES.index("face_found")]
    d = np.load(preds, allow_pickle=False)
    y, p, g, ids = d["y_true"], d["y_prob"].astype(float), d["groups"], d["clip_id"]
    n_face = np.asarray(C.align(ids_lab, np.sum(found == 1.0, axis=1), list(ids)))
    order = temporal_order(np.asarray(ids))
    y, p, g = np.asarray(y)[order], p[order], np.asarray(g)[order]
    ok = n_face[order] >= 5
    return y, p, g, ok


def _rows(y, p, g, ok):
    seqs = G.sequences(y, p, g, ok)
    return G.sweep(seqs, len(y), float(y[ok].mean())), seqs


def primary_floor(rows) -> float:
    cands = [r for r in rows if r["floor"] is not None and r["precision"] is not None]
    hit = [r for r in cands if r["precision"] >= TARGET_PRECISION]
    if hit:
        return min(r["floor"] for r in hit)
    return max(cands, key=lambda r: r["precision"])["floor"]


def lift_floor(rows) -> float:
    cands = [r for r in rows if r["floor"] is not None and r["offers"] >= MIN_OFFERS_LIFT]
    return max(cands, key=lambda r: r["lift"])["floor"]


def _rows_fast(y, p, g, ok):
    """Point estimates only (no bootstrap), for the per-participant selection loop."""
    seqs = G.sequences(y, p, g, ok)
    base = float(y[ok].mean())
    rows = []
    for floor in G.FLOORS:
        f, c = G.simulate(seqs, floor)
        prec = c / f if f else None
        rows.append({"floor": floor, "offers": f, "precision": prec,
                     "lift": prec / base if prec is not None else None})
    return rows


def _point(y, p, g, ok, floor):
    seqs = G.sequences(y, p, g, ok)
    return G.simulate(seqs, floor)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="C:/engagenet")
    a = ap.parse_args()
    work = pathlib.Path(a.work)

    ytr, ptr, gtr, oktr = _split(work, "Train", work / "results" / "Train_4_lean_preds.npz")
    train_rows, _ = _rows(ytr, ptr, gtr, oktr)
    yte, pte, gte, okte = _split(work, "Test", work / "results" / "test_predictions.npz")
    test_rows, _ = _rows(yte, pte, gte, okte)

    f_primary, f_lift = primary_floor(train_rows), lift_floor(train_rows)
    at = {r["floor"]: r for r in test_rows if r["floor"] is not None}

    # leave-one-test-participant-out selection on test, scored on the held-out participant
    fired = correct = 0
    chosen = []
    parts = list(dict.fromkeys(gte))
    for q in parts:
        keep = gte != q
        rows_q = _rows_fast(yte[keep], pte[keep], gte[keep], okte[keep])
        fq = primary_floor(rows_q)
        chosen.append(fq)
        m = gte == q
        f, c = _point(yte[m], pte[m], gte[m], okte[m], fq)
        fired += f
        correct += c
    base_te = float(yte[okte].mean())
    lopo = {"offers": fired, "correct": correct,
            "precision": correct / fired if fired else None,
            "lift": (correct / fired / base_te) if fired else None,
            "chosen_floors": {str(k): chosen.count(k) for k in sorted(set(chosen))}}

    out = {"_note": __doc__.strip().splitlines()[0],
           "rules": {"primary": f"lowest floor with training OOF precision >= {TARGET_PRECISION}",
                     "lift": f"max training lift with >= {MIN_OFFERS_LIFT} offers",
                     "lopo": "primary rule on the other 25 test participants"},
           "training_oof": {"windows": int(len(ytr)), "scorable": int(oktr.sum()),
                            "base_rate": float(ytr[oktr].mean()), "rows": train_rows},
           "selected": {"primary": f_primary, "lift": f_lift},
           "test_at_selected": {"primary": at[f_primary], "lift": at[f_lift],
                                "deployed_0_70": at[0.70]},
           "test_lopo_selection": lopo,
           "test_base_rate_scorable": base_te}
    dest = REPO / "reports" / "engagenet_review"
    (dest / "oos_floor.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("training OOF rows:")
    for r in train_rows[1:]:
        print(f"  {r['floor']:.2f} offers {r['offers']:4d} prec "
              f"{r['precision'] and round(r['precision'], 3)} lift {r['lift'] and round(r['lift'], 2)}")
    for k, f in (("primary", f_primary), ("lift", f_lift)):
        r = at[f]
        print(f"{k:8s} floor {f:.2f} -> test offers {r['offers']} prec {round(r['precision'], 3)} "
              f"ci {[round(x, 3) for x in r['ci95']]} lift {round(r['lift'], 2)} "
              f"per_h {r['offers_per_hour']:.2f}")
    print("LOPO:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in lopo.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
