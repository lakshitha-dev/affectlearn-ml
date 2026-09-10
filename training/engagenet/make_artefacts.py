"""Write the three committed artefacts the thesis quotes but could not verify.

Three sets of figures reached Chapter 4 from a chat transcript and a local
results directory rather than from a committed JSON: the Stage 2 confirmation
row, the Test-split diagnostics, and the serving latency. `verify_numbers.py`
announced each gap on every build. This closes all three, and it does so without
re-extracting video or refitting the model, because everything needed already
exists in the working directory:

    <work>/results/Train_4_lean.json        the Stage 2 run, as scored
    <work>/results/test_predictions.npz     y_true, y_prob, groups, clip_id
    <work>/geom/Test.npz                     the per-frame geometry
    <work>/labels/                           the released label files
    <work>/models/engagenet_lean_gbdt.joblib the fitted estimator
    <work>/models/engagenet_lean_gbdt.onnx   the exported artefact

Nothing here refits anything. That is deliberate: the reported numbers are the
ones the committed ONNX produces, and a refit that landed a thousandth away
would leave the document and its source disagreeing in a way no check could
resolve. The parity block instead *verifies* that the ONNX still reproduces the
committed predictions, which is the property the thesis actually relies on.

    python make_artefacts.py --work C:/engagenet

Nothing derived from EngageNet video is written: the licence forbids
redistributing corpus data, so the outputs are aggregates and per-participant
scores keyed by the corpus's own subject identifiers.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rungs  # noqa: E402

RUNG = "4_lean"
OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "engagenet_screen"


def _onnx_scorer(work: pathlib.Path):
    """P(disengaged) from the exported artefact, in batches."""
    import onnxruntime as ort
    sess = ort.InferenceSession(
        str(work / "models" / "engagenet_lean_gbdt.onnx"),
        providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def predict(batch: np.ndarray) -> np.ndarray:
        raw = sess.run(None, {name: np.ascontiguousarray(batch, dtype=np.float32)})[1]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            return np.asarray([r[1] for r in raw])
        return np.asarray(raw)[:, 1]

    return predict


def _write(name: str, payload: dict) -> None:
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("  wrote %s" % path.name)


# ── Stage 2 ──────────────────────────────────────────────────────────────────────────
def stage2(work: pathlib.Path) -> None:
    """Promote the Stage 2 run into the reports directory under its protocol name.

    The file is copied rather than recomputed. It is the run the thesis reports,
    and rescoring it here would silently substitute a second run for the one the
    document quotes.
    """
    src = work / "results" / ("Train_%s.json" % RUNG)
    d = json.loads(src.read_text(encoding="utf-8"))
    if d.get("stage") != "confirm" or d.get("split") != "Train":
        raise SystemExit("%s is not the Stage 2 confirmation run" % src)
    d["_source"] = "results/Train_%s.json, as scored on %s" % (
        RUNG, time.strftime("%Y-%m-%d", time.localtime(src.stat().st_mtime)))
    d["_note"] = ("Stage 2 of the three-stage protocol: the screening winner "
                  "confirmed on Train, 91 participants disjoint from both other "
                  "splits. Copied from the run directory, not recomputed.")
    _write("STAGE2_TRAIN.json", d)


# ── Stage 3 diagnostics ──────────────────────────────────────────────────────────────
def diagnostics(work: pathlib.Path) -> None:
    """The four qualifications Section 4.9.5 states, computed from the dumps."""
    pred = np.load(work / "results" / "test_predictions.npz", allow_pickle=False)
    y, p, g = pred["y_true"], pred["y_prob"], np.asarray(
        [str(s) for s in pred["groups"]])
    ids = [str(c) for c in pred["clip_id"]]

    out: dict = {
        "rung": RUNG, "split": "Test", "n": int(len(y)),
        "subjects": int(len(np.unique(g))),
        "auc": float(roc_auc_score(y, p)),
    }

    # (a) per-participant AUC. A participant with one class present has no AUC,
    #     which is a property of the split rather than a failure, so it is
    #     recorded as null instead of being dropped silently.
    per: dict[str, float | None] = {}
    for s in sorted(np.unique(g)):
        m = g == s
        per[s] = (float(roc_auc_score(y[m], p[m]))
                  if len(np.unique(y[m])) == 2 else None)
    scored = [v for v in per.values() if v is not None]
    out["per_subject_auc"] = per
    out["per_subject_auc_median"] = float(np.median(scored))
    out["per_subject_auc_min"] = float(np.min(scored))
    out["n_subjects_below_chance"] = int(sum(v < 0.5 for v in scored))

    # (b) leave-one-participant-out, so a claim that no single face carries the
    #     headline is a measurement rather than an assertion.
    loso = [float(roc_auc_score(y[g != s], p[g != s])) for s in np.unique(g)]
    out["loso_auc_range"] = [float(min(loso)), float(max(loso))]

    # (c) the face-detection confound. `face_found` is carried through the
    #     extractor for exactly this purpose. Rebuilt through rungs.load so the
    #     clip ordering is the extractor's, then aligned to the dump by clip id.
    feats, y_lab, g_lab, ids_lab = rungs.load(
        work / "geom" / "Test.npz", work / "labels", "Test")
    found = feats[:, :, rungs.NAMES.index("face_found")]
    share = np.nanmean(np.where(np.isnan(found), 0.0, found), axis=1)
    by_id = dict(zip(ids_lab, share))
    missing = [c for c in ids if c not in by_id]
    if missing:
        raise SystemExit("%d test clips have no geometry row (e.g. %s)"
                         % (len(missing), missing[0]))
    share_aligned = np.asarray([by_id[c] for c in ids])

    out["face_detection_corr"] = float(np.corrcoef(share_aligned, y)[0, 1])
    keep = share_aligned >= 1.0
    out["n_clips_excluded"] = int((~keep).sum())
    out["auc_excluded"] = float(roc_auc_score(y[keep], p[keep]))
    out["baseline_excluded"] = float(max(y[keep].mean(), 1 - y[keep].mean()))
    out["accuracy_excluded"] = float(((p[keep] >= 0.5).astype(int) == y[keep]).mean())
    out["low_engagement_rate_no_face"] = (
        float(y[share_aligned <= 0.0].mean()) if (share_aligned <= 0.0).any() else None)
    out["low_engagement_rate_full_detection"] = float(y[keep].mean())

    # (d) permutation importance on the fitted estimator, over the same features
    #     the model was fitted on, so the "strongest single feature" claim is
    #     checkable. Participant-clustered shuffling would be stricter but is not
    #     what the claim in the chapter is about.
    # Scored through the ONNX rather than the joblib. The pickled estimator was
    # written under a different scikit-learn version and no longer loads, and the
    # ONNX is the better instrument regardless: it is the artefact the platform
    # serves, so an importance computed through it describes the deployed model.
    predict = _onnx_scorer(work)
    X = rungs.build_matrix(feats, RUNG).astype(np.float32)
    names = rungs.feature_names(RUNG)
    order = {c: i for i, c in enumerate(ids_lab)}
    X = X[[order[c] for c in ids]]
    base = roc_auc_score(y, predict(X))
    rng = np.random.default_rng(42)
    imp: dict[str, float] = {}
    for j, nm in enumerate(names):
        drops = []
        for _ in range(20):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - roc_auc_score(y, predict(Xp)))
        imp[nm] = float(np.mean(drops))
    out["permutation_importance"] = imp
    ranked = sorted(imp.items(), key=lambda kv: -kv[1])
    out["strongest_feature"] = ranked[0][0]
    out["strongest_feature_margin_over_second"] = (
        float(ranked[0][1] / ranked[1][1]) if ranked[1][1] > 0 else None)

    # (e) the within-participant univariate AUC of that feature, which is what
    #     rules out camera placement as the explanation.
    j = names.index("mean_gaze_y")
    within = []
    for s in np.unique(g):
        m = g == s
        if len(np.unique(y[m])) == 2:
            within.append(roc_auc_score(y[m], X[m, j]))
    out["mean_gaze_y_univariate_auc"] = float(roc_auc_score(y, X[:, j]))
    out["mean_gaze_y_within_subject_auc_mean"] = float(np.mean(within))
    out["mean_gaze_y_within_subject_auc_range"] = [
        float(np.min(within)), float(np.max(within))]

    out["_note"] = ("Computed from the committed test prediction dump and the "
                    "extracted geometry. Aggregates and per-participant scores "
                    "only; no corpus data is reproduced.")
    _write("STAGE3_DIAGNOSTICS.json", out)


# ── fit report ───────────────────────────────────────────────────────────────────────
def fit_report(work: pathlib.Path) -> None:
    """Serving latency, and whether the ONNX still reproduces the dump."""
    onnx_path = work / "models" / "engagenet_lean_gbdt.onnx"
    feats, _, _, ids_lab = rungs.load(
        work / "geom" / "Test.npz", work / "labels", "Test")
    X = rungs.build_matrix(feats, RUNG).astype(np.float32)
    onnx_prob = _onnx_scorer(work)

    # Single-clip latency, which is the serving shape. Warm up first so the
    # figure is steady-state rather than first-call.
    one = X[:1]
    for _ in range(50):
        onnx_prob(one)
    reps = 2000
    t0 = time.perf_counter()
    for _ in range(reps):
        onnx_prob(one)
    per_clip_ms = (time.perf_counter() - t0) / reps * 1000.0

    ox = onnx_prob(X)

    pred = np.load(work / "results" / "test_predictions.npz", allow_pickle=False)
    order = {c: i for i, c in enumerate(ids_lab)}
    committed = pred["y_prob"]
    reproduced = ox[[order[str(c)] for c in pred["clip_id"]]]
    drift = float(np.max(np.abs(committed - reproduced)))

    _write("fit_report.json", {
        "rung": RUNG,
        "n_features": int(X.shape[1]),
        "onnx_bytes": int(onnx_path.stat().st_size),
        "onnx_opset": 17,
        "latency_ms_per_clip": round(per_clip_ms, 3),
        "latency_reps": reps,
        "committed_prediction_max_abs_drift": drift,
        "reproduces_committed_predictions": bool(drift < 1e-6),
        "_note": ("Latency is single-clip steady state on CPU, which is the "
                  "serving shape. The drift figure checks that the exported "
                  "artefact still reproduces the predictions Chapter 4 is "
                  "computed over; nothing here refits the model. No sklearn "
                  "comparison is reported: the pickled estimator was written "
                  "under a different library version and no longer loads, and "
                  "the ONNX is what the platform serves."),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="C:/engagenet",
                    help="the EngageNet working directory")
    a = ap.parse_args()
    work = pathlib.Path(a.work)
    if not (work / "results").is_dir():
        raise SystemExit("no results directory under %s" % work)
    OUT.mkdir(parents=True, exist_ok=True)
    print("artefacts:")
    stage2(work)
    diagnostics(work)
    fit_report(work)
    print("done. Rebuild the thesis and verify_numbers.py will enforce all three.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
