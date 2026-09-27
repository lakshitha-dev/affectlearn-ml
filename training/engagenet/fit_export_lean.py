"""Fit the deployed geometry model (rung 4_lean) on EngageNet Train and export it to ONNX.

    python training/engagenet/fit_export_lean.py --work C:/engagenet

WHY THIS EXISTS

The deployed `engagenet_lean_gbdt.onnx` was fitted and exported on a Colab runtime that was later
lost. `rungs.py` only produces out-of-fold predictions, and `make_artefacts.py` deliberately refits
nothing, so until now no committed script could rebuild the served model. This script closes that
gap: it fits the same 20 features with the same hyperparameters (`rungs.gbdt`), exports through
skl2onnx, and checks the export against both the committed ONNX and the committed Test dump.

WHAT IT DOES NOT DO

It does not replace the served model. The export is written to `<work>/models/refit/`, never to
`affectlearn/backend/models/`, and the paper's Test figures remain the ones the committed ONNX
produced (`reports/engagenet_screen/STAGE3_TEST.json`). A refit under a different scikit-learn
version is not bit-identical to the original fit; the size of that difference is measured here and
reported, not absorbed.

Aggregates only are written into the repository (`reports/engagenet_review/refit_export.json`).
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
SEED = 42
OPSET = 17
REPORT = (pathlib.Path(__file__).resolve().parents[2] / "reports" / "engagenet_review"
          / "refit_export.json")


def _patch_bool_int_attributes() -> None:
    """Work around skl2onnx 1.20 + onnx 1.22 passing bools into an `ints` attribute.

    skl2onnx emits TreeEnsemble's `nodes_missing_value_tracks_true` as a list of numpy bools, and
    onnx 1.22's `make_attribute` rejects bools in an INTS field. The attribute is defined as 0/1
    ints, so casting is exact; the parity check below (refit ONNX against the refit estimator)
    confirms the exported model is unchanged by it.
    """
    import onnx.helper as oh
    original = oh.make_attribute
    if getattr(original, "_bool_cast", False):
        return

    def make_attribute(key, value, *args, **kwargs):
        if isinstance(value, (list, tuple, np.ndarray)) and any(
                isinstance(v, (bool, np.bool_)) for v in value):
            value = [int(v) if isinstance(v, (bool, np.bool_)) else v for v in value]
        return original(key, value, *args, **kwargs)

    make_attribute._bool_cast = True
    oh.make_attribute = make_attribute


def onnx_predict(path: pathlib.Path, X: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    raw = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(X, dtype=np.float32)})[1]
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return np.asarray([r[1] for r in raw], dtype=float)
    return np.asarray(raw, dtype=float)[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="C:/engagenet")
    a = ap.parse_args()
    work = pathlib.Path(a.work)

    import onnx
    import onnxruntime
    import skl2onnx
    import sklearn
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    ftr, ytr, gtr, _ = rungs.load(work / "geom" / "TrainSub.npz", work / "labels", "Train")
    fte, yte, gte, ids_te = rungs.load(work / "geom" / "Test.npz", work / "labels", "Test")
    Xtr = rungs.build_matrix(ftr, RUNG)
    Xte = rungs.build_matrix(fte, RUNG)
    print(f"Train {Xtr.shape[0]} clips / {len(set(gtr))} participants | "
          f"Test {Xte.shape[0]} clips / {len(set(gte))} participants | {Xtr.shape[1]} features")

    model = rungs.gbdt(SEED).fit(Xtr, ytr)
    p_sk = model.predict_proba(Xte)[:, 1]

    out_dir = work / "models" / "refit"
    out_dir.mkdir(parents=True, exist_ok=True)
    _patch_bool_int_attributes()
    onx = convert_sklearn(model, initial_types=[("input", FloatTensorType([None, Xtr.shape[1]]))],
                          options={id(model): {"zipmap": False}}, target_opset=OPSET)
    onnx_path = out_dir / "engagenet_lean_gbdt_refit.onnx"
    onnx_path.write_bytes(onx.SerializeToString())

    p_refit_onnx = onnx_predict(onnx_path, Xte)
    p_committed_onnx = onnx_predict(work / "models" / "engagenet_lean_gbdt.onnx", Xte)

    dump = np.load(work / "results" / "test_predictions.npz", allow_pickle=False)
    order = {str(c): i for i, c in enumerate(ids_te)}
    idx = np.array([order[str(c)] for c in dump["clip_id"]])
    p_dump = dump["y_prob"].astype(float)
    if not (dump["y_true"] == yte[idx]).all():
        raise SystemExit("ABORT: labels differ between this load and the committed Test dump")

    one = Xte[:1].astype(np.float32)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for _ in range(50):
        sess.run(None, {name: one})
    reps = 2000
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run(None, {name: one})
    latency_ms = (time.perf_counter() - t0) / reps * 1000.0

    def cmp(a, b):
        return {"max_abs_diff": float(np.max(np.abs(a - b))),
                "mean_abs_diff": float(np.mean(np.abs(a - b))),
                "pearson_r": float(np.corrcoef(a, b)[0, 1])}

    report = {
        "rung": RUNG, "seed": SEED, "hyperparameters": rungs.gbdt(SEED).get_params(),
        "n_features": int(Xtr.shape[1]), "feature_names": rungs.feature_names(RUNG),
        "fit_on": f"Train subset {Xtr.shape[0]} clips / {len(set(gtr))} participants",
        "scored_on": f"Test {Xte.shape[0]} clips / {len(set(gte))} participants",
        "versions": {"scikit_learn": sklearn.__version__, "skl2onnx": skl2onnx.__version__,
                     "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__,
                     "numpy": np.__version__, "onnx_opset": OPSET},
        "export_path_outside_repo": str(onnx_path),
        "export_bytes": int(onnx_path.stat().st_size),
        "latency_ms_per_clip": round(latency_ms, 4),
        "test_auc": {
            "committed_dump": float(roc_auc_score(yte[idx], p_dump)),
            "committed_onnx": float(roc_auc_score(yte, p_committed_onnx)),
            "refit_sklearn": float(roc_auc_score(yte, p_sk)),
            "refit_onnx": float(roc_auc_score(yte, p_refit_onnx)),
        },
        "per_clip": {
            "refit_onnx_vs_refit_sklearn": cmp(p_refit_onnx, p_sk),
            "refit_onnx_vs_committed_onnx": cmp(p_refit_onnx, p_committed_onnx),
            "refit_onnx_vs_committed_dump": cmp(p_refit_onnx[idx], p_dump),
            "committed_onnx_vs_committed_dump": cmp(p_committed_onnx[idx], p_dump),
        },
        "_note": ("The served model is the committed ONNX, which reproduces the committed Test "
                  "dump. This refit uses the same data, features and hyperparameters under the "
                  "library versions listed; its per-clip difference from the committed model is "
                  "the version drift, not a different model choice. Aggregates only."),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    ta = report["test_auc"]
    print(f"AUC committed dump {ta['committed_dump']:.4f} | committed ONNX {ta['committed_onnx']:.4f}"
          f" | refit sklearn {ta['refit_sklearn']:.4f} | refit ONNX {ta['refit_onnx']:.4f}")
    pc = report["per_clip"]
    print(f"refit ONNX vs sklearn max |d| {pc['refit_onnx_vs_refit_sklearn']['max_abs_diff']:.2e}; "
          f"vs committed ONNX {pc['refit_onnx_vs_committed_onnx']['max_abs_diff']:.4f}; "
          f"committed ONNX vs dump {pc['committed_onnx_vs_committed_dump']['max_abs_diff']:.2e}")
    print(f"wrote {onnx_path} ({report['export_bytes']} bytes) and {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
