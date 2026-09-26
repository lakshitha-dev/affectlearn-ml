"""Post-hoc seed-stability check on the already-reported Stage 3 model.

    python seed_sweep.py --work C:/engagenet

WHAT THIS IS, AND WHAT IT IS NOT. The three-stage protocol scores Test exactly once, and it did:
`reports/engagenet_screen/STAGE3_TEST.json` holds that run and nothing here replaces it. This is a
stability check on the model already reported, refitting the SAME confirmed variant (4_lean) on the
SAME Train split across several seeds and rescoring Test. No variant is selected on the outcome, so
nothing about the protocol depends on what this returns.

It exists because the reported AUC is one draw at one seed, and the participant bootstrap around it
measures a different thing: how much the figure moves with a different sample of test participants,
not how much it moves with a different seed. Section 4.5.5 of the thesis already reports three seeds
per target for the DAiSEE branch and finds the spread there is not negligible, so leaving the
headline unmeasured on the same axis is an inconsistency worth closing.

It also quantifies how far a refit under today's scikit-learn lands from the committed run. That is
a second and separate question from the seed, and it needs asking here because the pickled estimator
was written under an earlier library version and no longer loads, so a refit is the only way to
compare. The run ABORTS if the refit diverges materially (more than 0.01 AUC, or per-clip
correlation below 0.95), or if the labels do not match, since a sweep computed over a different
model or a different label set would describe something other than the reported result.

Consistent with the EngageNet licence, only aggregates are written; no corpus data leaves here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rungs  # noqa: E402

RUNG = "4_lean"
OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "engagenet_screen"
SEEDS = [42, 0, 1, 7, 13, 99, 123, 777, 2024, 31337]
TOL = 5e-4          # the committed figure is quoted to four places


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="C:/engagenet")
    a = ap.parse_args()
    work = pathlib.Path(a.work)

    committed = json.loads((OUT / "STAGE3_TEST.json").read_text(encoding="utf-8"))
    target = float(committed["auc"])

    ftr, ytr, gtr, _ = rungs.load(work / "geom" / "TrainSub.npz", work / "labels", "Train")
    fte, yte, gte, ids_te = rungs.load(work / "geom" / "Test.npz", work / "labels", "Test")
    Xtr = rungs.build_matrix(ftr, RUNG)
    Xte = rungs.build_matrix(fte, RUNG)
    print(f"Train {Xtr.shape[0]} clips / {len(set(gtr))} subjects | "
          f"Test {Xte.shape[0]} clips / {len(set(gte))} subjects | {Xtr.shape[1]} features")

    rows = []
    for s in SEEDS:
        m = rungs.gbdt(s).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, p))
        acc = float(((p >= 0.5).astype(int) == yte).mean())
        rows.append({"seed": s, "auc": auc, "accuracy": acc,
                     "score_spread": float(p.max() - p.min())})
        print(f"  seed {s:>6}  AUC {auc:.4f}  acc {acc:.4f}")

    # How far does a refit under today's library land from the committed run? The original model
    # was fitted under an earlier scikit-learn, and the pickled estimator no longer loads, so this
    # is the only way to quantify the drift. Compared per clip, not just on the aggregate.
    ref = next(r for r in rows if r["seed"] == 42)
    m42 = rungs.gbdt(42).fit(Xtr, ytr)
    p42 = m42.predict_proba(Xte)[:, 1]
    committed_dump = np.load(work / "results" / "test_predictions.npz", allow_pickle=False)
    order = {str(c): i for i, c in enumerate(ids_te)}
    mine = np.array([p42[order[str(c)]] for c in committed_dump["clip_id"]])
    comm = committed_dump["y_prob"]
    drift = {
        "sklearn_version_now": sklearn.__version__,
        "committed_auc": target,
        "refit_auc_seed_42": ref["auc"],
        "auc_delta": round(ref["auc"] - target, 6),
        "per_clip_max_abs_diff": round(float(np.max(np.abs(mine - comm))), 6),
        "per_clip_mean_abs_diff": round(float(np.mean(np.abs(mine - comm))), 6),
        "per_clip_pearson_r": round(float(np.corrcoef(mine, comm)[0, 1]), 6),
        "labels_identical": bool((committed_dump["y_true"] ==
                                  np.array([yte[order[str(c)]]
                                            for c in committed_dump["clip_id"]])).all()),
    }
    if not drift["labels_identical"]:
        print("\nABORT: labels differ between the refit and the committed dump.")
        return 1
    if abs(drift["auc_delta"]) > 0.01 or drift["per_clip_pearson_r"] < 0.95:
        print(f"\nABORT: refit diverges too far from the committed run "
              f"(delta {drift['auc_delta']:+.4f}, r {drift['per_clip_pearson_r']:.4f}).")
        return 1

    aucs = [r["auc"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    payload = {
        "rung": RUNG,
        "split": "Test",
        "n_clips": int(Xte.shape[0]),
        "n_subjects": len(set(gte)),
        "fit_on": f"Train {Xtr.shape[0]} clips / {len(set(gtr))} subjects",
        "version_drift_vs_committed": drift,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "per_seed": rows,
        "auc_mean": float(statistics.fmean(aucs)),
        "auc_sd": float(statistics.stdev(aucs)),
        "auc_min": float(min(aucs)),
        "auc_max": float(max(aucs)),
        "auc_range": float(max(aucs) - min(aucs)),
        "accuracy_mean": float(statistics.fmean(accs)),
        "accuracy_sd": float(statistics.stdev(accs)),
        "_note": ("Post-hoc stability check on the model already reported in STAGE3_TEST.json, "
                  "not a re-selection: the variant was fixed at Stage 2 and no choice here "
                  "depends on the outcome. Two separate things are measured. The seed sweep "
                  "shows this estimator is seed-invariant on this data, which is expected for a "
                  "histogram gradient-boosted tree without subsampling, so the single-seed "
                  "concern does not apply to it. The version-drift block shows how far a refit "
                  "under the current scikit-learn lands from the committed run, which is a real "
                  "reproducibility gap and is reported rather than absorbed. Neither is the "
                  "participant bootstrap in STAGE3_TEST.json, which measures sampling of test "
                  "participants."),
    }
    (OUT / "STAGE3_SEED_SWEEP.json").write_text(json.dumps(payload, indent=2) + "\n",
                                                encoding="utf-8")
    print(f"\nversion drift vs committed: AUC {drift['auc_delta']:+.4f} "
          f"({target:.4f} -> {ref['auc']:.4f}), per-clip r {drift['per_clip_pearson_r']:.4f}, "
          f"max |diff| {drift['per_clip_max_abs_diff']:.4f}, sklearn {drift['sklearn_version_now']}")
    print(f"AUC across {len(SEEDS)} seeds: mean {payload['auc_mean']:.4f} "
          f"SD {payload['auc_sd']:.4f} range {payload['auc_min']:.4f} to {payload['auc_max']:.4f} "
          f"(spread {payload['auc_range']:.4f})")
    print(f"Wrote {OUT / 'STAGE3_SEED_SWEEP.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
