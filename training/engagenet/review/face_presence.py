"""How much of the geometry channel's Test result depends on face absence, under the deployed rule.

    python training/engagenet/review/face_presence.py --work C:/engagenet

THE DEPLOYED RULE

The browser reports a cycle as face-absent when fewer than half of the frames it scores contain a
face (`frontend/src/hooks/use-media-pipe.ts`: `faceRatio < PREPROCESS_CONTRACT.minFaceFrameRatio`,
0.5 in `lib/preprocess.ts`), and the server then produces no reading
(`backend/app/agents/nodes/affect_detection.py:61-69`). EngageNet clips carry 10 frames at 1 fps,
the same window the deployed client scores, so a clip is SCORABLE here iff at least 5 of its 10
frames have face_found == 1. A frame whose face_found is NaN (not decoded) counts as no face, as a
missing frame does in the browser's ratio.

WHAT IS REPORTED

* counts and positive rates of scorable and unscorable clips;
* Test AUC with a participant interval on scorable clips only, and the any-missing-frame mask used
  in STAGE3_DIAGNOSTICS (all 10 frames with a face), which should reproduce 0.913 on 2,001 clips;
* the gate sweep (floors 0.50 to 0.90, 2 consecutive, cooldown 3, recording order per participant)
  over all clips, and with unscorable clips masked the way the deployed gate sees them (no reading:
  they cannot fire or extend a streak, but they still take a cycle of wall time).

The committed predictions are the ones the paper quotes, so no model is refitted.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

import _common as C

sys.path.insert(0, str(C.REPO / "evaluation"))
from gate_calibration import simulate  # noqa: E402

MIN_FACE_FRAMES = 5     # 0.5 x 10 frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(C.WORK))
    a = ap.parse_args()
    work = pathlib.Path(a.work)

    feats, y_lab, g_lab, ids_lab = C.load_split("Test", work)
    found = feats[:, :, C.rungs.NAMES.index("face_found")]
    n_face = np.sum(found == 1.0, axis=1)                    # NaN != 1, so NaN counts as no face
    y, p, g, ids = C.committed_test_predictions(work)
    n_face = C.align(ids_lab, n_face, list(ids))
    if not (C.align(ids_lab, y_lab, list(ids)) == y).all():
        raise SystemExit("ABORT: committed dump labels differ from this load")

    scorable = n_face >= MIN_FACE_FRAMES
    full = n_face >= found.shape[1]
    none = n_face == 0

    def group(mask):
        return {"n": int(mask.sum()), "positive_rate": float(y[mask].mean()) if mask.any() else None,
                "positives": int(y[mask].sum())}

    counts = {
        "frames_per_clip": int(found.shape[1]),
        "rule": f"scorable iff >= {MIN_FACE_FRAMES} of {found.shape[1]} frames have face_found == 1",
        "all_clips": group(np.ones(len(y), bool)),
        "scorable": group(scorable), "unscorable": group(~scorable),
        "all_frames_with_face": group(full), "any_frame_missing_face": group(~full),
        "no_face_in_any_frame": group(none),
        "share_of_test_positives_in_unscorable": float(y[~scorable].sum() / y.sum()),
        "share_of_test_positives_with_any_missing_frame": float(y[~full].sum() / y.sum()),
        "face_frame_histogram": np.bincount(n_face.astype(int), minlength=11).tolist(),
    }
    print(f"  scorable {counts['scorable']} | unscorable {counts['unscorable']} | "
          f"no face at all {counts['no_face_in_any_frame']}")

    auc = {"all_clips": C.auc_block(y, p, g),
           "scorable_only_deployment_rule": C.auc_block(y[scorable], p[scorable], g[scorable]),
           "all_frames_with_face_only": C.auc_block(y[full], p[full], g[full])}
    auc["delta_all_minus_scorable"] = auc["all_clips"]["auc"] - auc["scorable_only_deployment_rule"]["auc"]
    for k in ("all_clips", "scorable_only_deployment_rule", "all_frames_with_face_only"):
        b = auc[k]
        print(f"  AUC {k:32} n={b['n']:>4} {b['auc']:.4f} "
              f"[{b['auc_ci95_participant'][0]:.3f}, {b['auc_ci95_participant'][1]:.3f}]")

    # The masked simulator must equal the committed one when nothing is masked.
    order = C.temporal_order(ids)
    for thr in C.FLOORS:
        ref = simulate(y[order], p[order], g[order], threshold=thr,
                       min_consecutive=C.MIN_CONSECUTIVE, cooldown=C.COOLDOWN)
        mine = C.simulate_masked(y[order], p[order], g[order], threshold=thr)
        if (ref["interventions"], ref["correct"]) != (mine["interventions"], mine["correct"]):
            raise SystemExit(f"ABORT: masked simulator disagrees with gate_calibration at {thr}")

    print("  gate sweep:")
    sweeps = {
        "all_clips": C.sweep(y, p, g, ids, label="all"),
        "scorable_masked_deployment_rule": C.sweep(y, p, g, ids, scorable=scorable,
                                                   label="scorable"),
        "all_frames_with_face_masked": C.sweep(y, p, g, ids, scorable=full, label="full-face"),
    }

    at70 = {k: next(r for r in v["rows"] if r["floor"] == 0.70) for k, v in sweeps.items()}
    unscorable_offers_at70 = at70["all_clips"]["interventions"] - at70[
        "scorable_masked_deployment_rule"]["interventions"]

    C.write("face_presence.json", {
        "source_predictions": "C:/engagenet/results/test_predictions.npz (committed Test dump)",
        "counts": counts, "auc": auc, "gate_sweep": sweeps,
        "at_floor_0_70": at70,
        "offers_lost_to_mask_at_0_70": int(unscorable_offers_at70),
        "masking_semantics": ("An unscorable window yields no reading: it cannot fire and does not "
                              "extend or reset the persistence streak, but it takes one cycle, so "
                              "an active cooldown counts it down. Unmasked, the simulator is "
                              "asserted identical to evaluation/gate_calibration.simulate."),
        "seed": C.SEED, "n_bootstrap": C.N_BOOT, "versions": C.versions(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
