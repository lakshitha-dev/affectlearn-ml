"""Fit the DEPLOYABLE behavioural confusion model on all of DUX and export it to ONNX.

WHY THIS EXISTS
---------------
The AUC 0.747 result came from `train_dux_confusion.py`, which fits a fresh GBDT inside each
leave-one-participant-out fold and throws it away. That is correct for ESTIMATING performance and
useless for SERVING: no artifact survives. Meanwhile the model the backend actually serves
(`behavioral_bilstm.onnx`) was exported from a checkpoint stamped `trained_on: "synthetic"` with
600 fabricated windows from 15 fabricated participants — so the live system's behavioural affect
detection has never seen a real human.

This script closes that gap: one final model, fitted on every DUX window, exported to ONNX.

WHY FITTING ON EVERYTHING IS CORRECT HERE
-----------------------------------------
No held-out score is computed or reported from this fit, so there is nothing to inflate. The honest
performance estimate is the LOPO one already measured (AUC 0.747, kappa 0.274, 46 participants) and
that number stands unchanged. Refitting on all data after cross-validation has fixed the
hyperparameters is standard practice — it is what CV is FOR. The alternative, deploying a model
trained on 45/46 of the data to preserve a holdout nobody will score, would be strictly worse.

WHAT IT OUTPUTS, AND THE CONTRACT CHANGE
----------------------------------------
The served contract changes and this is deliberate:

    old:  (batch, 30, 16) -> 4 logits   [engaged, bored, confused, frustrated]   SYNTHETIC
    new:  (batch, 80)     -> P(confused)                                         REAL, validated

80 = the aggregate features from `aggregate_features.aggregate`. The backend must therefore run
`extract_features` (unchanged, 30x16) and then `aggregate` before inference.

Binary is not a downgrade — it is the honest output space. Measured on DAiSEE and DUX, confusion is
the ONLY one of the four states either channel detects above chance: frustration lands at AUC 0.590
facially and 0.00 recall behaviourally, boredom at 0.561 facially and is absent from DUX entirely,
and engagement has 4 negative examples in 1638 DAiSEE test clips. Emitting four states implies four
detections; there is one.

Run:
    python export_dux_confusion.py
    python export_dux_confusion.py --out-dir ../../models --backend-dir ../../../affectlearn/backend/models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from aggregate_features import aggregate, aggregate_feature_names  # noqa: E402
from external_datasets import load_dux_confusion  # noqa: E402
from feature_engineering import FEATURE_NAMES, extract_features  # noqa: E402
from train_dux_confusion import _gbdt  # noqa: E402  - the SAME estimator LOPO measured


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dux", default="../../data/external/dux")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="../../models")
    ap.add_argument("--backend-dir", default=None,
                    help="if given, also copy the artifacts into the backend's models/ directory")
    a = ap.parse_args()

    win = load_dux_confusion(a.dux)
    if not win:
        raise SystemExit(f"No DUX windows under {a.dux}")
    X = aggregate(np.stack([extract_features(w["events"], 0) for w in win]))
    y = np.array([w["label"] for w in win], dtype=np.int64)
    groups = np.array([w["participant"] for w in win])
    print(f"fitting on ALL DUX: {len(y)} windows, {len(set(groups.tolist()))} participants, "
          f"{int(y.sum())} confused ({y.mean():.3f}), {X.shape[1]} aggregate features")

    clf = _gbdt(a.seed).fit(X, y)
    train_auc = None
    try:
        from sklearn.metrics import roc_auc_score
        train_auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
    except Exception:
        pass
    print(f"  in-sample AUC {train_auc:.4f}  <- NOT a performance estimate, it is fitted on this "
          "data. The honest number is the LOPO AUC 0.7473.")

    # --- skl2onnx compatibility shim -------------------------------------------------------
    # skl2onnx 1.20.0 + onnx 1.22.0 + sklearn 1.9.0 cannot export a HistGradientBoostingClassifier:
    #   TypeError: Field onnx.AttributeProto.ints: Expected an int, got a boolean
    # The converter builds `nodes_missing_value_tracks_true` as a MIXED list of Python `bool` and
    # numpy `uint8`, and ONNX's `ints` field rejects the bools. Note that coercing via
    # `np.asarray(v).dtype == bool` does NOT catch it — the mixed list promotes to uint8 — so the
    # check has to be per-element. Verified to reproduce sklearn to ~7e-08 after coercion.
    import skl2onnx.common._container as _container
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    _real_add_node = _container.ModelComponentContainer.add_node

    def _add_node_bool_safe(self, op_type, inputs, outputs, op_domain="", op_version=1, **attrs):
        for k, v in list(attrs.items()):
            if isinstance(v, (list, tuple)) and any(isinstance(x, (bool, np.bool_)) for x in v):
                attrs[k] = [int(x) for x in v]
        return _real_add_node(self, op_type, inputs, outputs,
                              op_domain=op_domain, op_version=op_version, **attrs)

    _container.ModelComponentContainer.add_node = _add_node_bool_safe
    try:
        onx = to_onnx(clf, initial_types=[("features", FloatTensorType([None, X.shape[1]]))],
                      options={id(clf): {"zipmap": False}}, target_opset=15)
    finally:
        _container.ModelComponentContainer.add_node = _real_add_node

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "behavioral_confusion_gbdt.onnx"
    onnx_path.write_bytes(onx.SerializeToString())

    meta = {
        "model": "HistGradientBoostingClassifier (depth 3, class_weight=balanced)",
        "task": "binary confusion detection from interaction behaviour",
        "labels": ["not_confused", "confused"],
        "input": {"name": "features", "shape": [None, int(X.shape[1])],
                  "pipeline": "extract_features(events) -> (30,16) -> aggregate() -> (80,)"},
        "feature_names_window": list(FEATURE_NAMES),
        "feature_names_aggregate": list(aggregate_feature_names()),
        "trained_on": {
            "corpus": "DUX v0+v1 (Zenodo 10.5281/zenodo.7778612, CC BY 4.0)",
            "label_source": "emotion_manual_Confusion — HUMAN annotation, not a model output",
            "n_windows": int(len(y)), "n_participants": int(len(set(groups.tolist()))),
            "n_confused": int(y.sum()), "positive_rate": float(y.mean()),
            "fit": "all windows (hyperparameters already fixed by LOPO)",
        },
        "honest_performance_lopo": {
            "auc": 0.7473, "cohen_kappa": 0.2738, "accuracy": 0.7301,
            "majority_baseline": 0.8189, "confused_recall": 0.5798,
            "protocol": "leave-one-participant-out over 46 participants, "
                        "within-participant permutation p=0.0005",
        },
        "in_sample_auc_do_not_report": train_auc,
        "deployment": {
            "recommended_threshold": 0.70,
            "threshold_rationale": "precision 0.500 at ~1.5 interventions/hour with "
                                   "ADAPT_MIN_CONSECUTIVE=2 and ADAPT_COOLDOWN_CYCLES=3; "
                                   "see reports/dux_confusion/FINDINGS.md",
        },
        "limitations": [
            "DUX participants used business software, not learning material.",
            "Binary: confusion only. Boredom, frustration and engagement are NOT detected by this "
            "model and must not be inferred from it.",
            "The negative class means 'not annotated as confused', which conflates calm with "
            "unannotated time — emotion_manual_Neutral is never populated in DUX.",
        ],
    }
    (out / "behavioral_confusion_gbdt.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Verify the exported graph reproduces sklearn before anything downstream trusts it.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    name = sess.get_inputs()[0].name
    probe = X[:64].astype(np.float32)
    onnx_p = sess.run(None, {name: probe})[1]
    onnx_p = np.asarray(onnx_p)[:, 1] if np.asarray(onnx_p).ndim == 2 else np.asarray(onnx_p)
    skl_p = clf.predict_proba(probe)[:, 1]
    max_diff = float(np.abs(onnx_p - skl_p).max())
    print(f"  ONNX vs sklearn max abs prob diff over 64 windows: {max_diff:.2e}")
    if max_diff > 1e-4:
        raise SystemExit(f"ONNX export DISAGREES with sklearn (max diff {max_diff:.2e}) — refusing")
    print(f"  outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")
    print(f"\nwrote {onnx_path}  ({onnx_path.stat().st_size/1024:.0f} KB)")
    print(f"wrote {out / 'behavioral_confusion_gbdt.json'}")

    if a.backend_dir:
        bd = Path(a.backend_dir)
        bd.mkdir(parents=True, exist_ok=True)
        for f in (onnx_path, out / "behavioral_confusion_gbdt.json"):
            (bd / f.name).write_bytes(f.read_bytes())
        print(f"copied both into {bd}")
        print("  NOTE: behavioral_inference.py must be updated to aggregate() before inference "
              "and to emit a binary confusion probability. The old (30,16)->4-logit path is NOT "
              "compatible with this artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
