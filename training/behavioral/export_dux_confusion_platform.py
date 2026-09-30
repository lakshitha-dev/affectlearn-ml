"""Fit the confusion GBDT on DUX rebuilt as PLATFORM-LIKE input, and export it for serving.

WHY THIS EXISTS
---------------
The deployed model (`export_dux_confusion.py`) was trained on DUX's native event stream: pointer
at ~110 Hz, clicks double-counted, near-empty windows dropped. The platform sends one pointer
sample per 100 ms tick, one event per press, and scores any window with at least one event. The
model's strongest inputs shrink under that change (mouse_velocity_std max 54.8 -> 4.24), so its
scores do too: on platform-like DUX input it peaks at 0.726, and on the live platform it never
exceeded 0.633 in 30 days, below its 0.70 floor.

This script trains the same estimator on the input-matched windows of
`review/input_matched.load_dux_confusion_matched` (serving window rule), so the model is fitted
on the input distribution the platform actually produces.

WHAT IT DOES NOT FIX
--------------------
DUX is still people filling in business expense forms. Matching the INPUT does not make the
ACTIVITY match: a confused learner reading a lesson moves and clicks far less than a confused
form-filler. No labelled data from the platform exists, so nothing here measures detection on
lesson pages.

HONEST NUMBERS
--------------
The held-out estimates are the ones `review/dux_review.py` already measured on exactly this window
set (leave-one-session-out, 46 sessions): they are read from `reports/dux_review/` and embedded in
the model card, and the script refuses to run if its windows differ from the evaluated ones.

Run from affectlearn-ml/training/behavioral:
    python export_dux_confusion_platform.py --dux ../../data/external/dux \
        --backend-dir ../../../affectlearn/backend/models \
        --fixture ../../../affectlearn/backend/tests/fixtures/behavioral_platform_parity.json
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "review"))
from aggregate_features import aggregate, aggregate_feature_names  # noqa: E402
from export_dux_confusion import to_onnx_bool_safe, verify_onnx  # noqa: E402
from external_datasets import _SCREEN_H, _SCREEN_W  # noqa: E402
from feature_engineering import FEATURE_NAMES, extract_features  # noqa: E402
from input_matched import load_dux_confusion_matched  # noqa: E402
from train_dux_confusion import _gbdt  # noqa: E402  - the SAME estimator LOSO measured

STEM = "behavioral_confusion_gbdt_platform"
REPORTS = HERE.parents[1] / "reports" / "dux_review"
ARM = "matched_serving_rule_interaction"
# The pre-specified floor rule of review/dux_review.py (NESTED_LIFT, NESTED_MIN_OFFERS).
LIFT, MIN_OFFERS = 2.0, 10
WIRE_START_MS = 1_790_000_000_000   # any wall clock; the backend only uses t_wall - start


def load_windows(dux: str) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """The evaluated window set: serving rule, minus windows with no facial rows (as dux_review)."""
    win = load_dux_confusion_matched(dux, threshold=1.0, min_events=1)
    keep = [w for w in win if not np.isnan(aggregate(np.asarray(w["affectiva_seq"])[None])).any()]
    Xb = np.stack([extract_features(w["events"], 0) for w in keep])
    y = np.array([w["label"] for w in keep], dtype=np.int64)
    groups = np.array([w["participant"] for w in keep])
    return keep, aggregate(Xb), y, groups


def reports() -> dict:
    read = lambda name: json.loads((REPORTS / name).read_text(encoding="utf-8"))  # noqa: E731
    matched = read("input_matched.json")
    sweeps = read("gate_sweeps.json")
    sweep = sweeps["sweeps"][f"{ARM}|deployed"]
    return {
        "lopo": matched["matched_serving_rule"],
        "native_model_on_platform_input": matched["native_model_scored_on_matched_inputs"],
        "sweep": sweep,
        "sweep_note": sweeps["note"],
        "nested": read("nested_floor.json")[ARM],
        "calibration": read("calibration.json")[ARM]["uncalibrated"],
    }


def prespecified_floor(sweep: dict) -> float | None:
    """Lowest floor whose gated precision is >= LIFT x base rate with >= MIN_OFFERS offers."""
    base = sweep["base_rate"]
    for row in sweep["rows"] if "rows" in sweep else sweep["results"]:
        if row["threshold"] is None or row["precision"] is None:
            continue
        if row["interventions"] >= MIN_OFFERS and row["precision"] >= LIFT * base:
            return row["threshold"]
    return None


def to_wire(events, start_ms: int) -> list[dict]:
    """ML event rows -> the browser's `behavioral_window` events, as `use-behavioral-signals` sends."""
    out = []
    for r in events.itertuples(index=False):
        ts = float(r.ts)
        if ts != int(ts):
            raise SystemExit(f"non-integer event time {ts}: the wire format carries whole ms")
        t = start_ms + int(ts)
        if r.type == "move":
            out.append({"kind": "mouse_sample", "t_wall": t, "t_mono": float(ts),
                        "x": float(r.x) * _SCREEN_W, "y": float(r.y) * _SCREEN_H})
        elif r.type == "click":
            out.append({"kind": "mouse_click", "t_wall": t, "t_mono": float(ts),
                        "x": 0, "y": 0, "button": 0})
        elif r.type == "key":
            out.append({"kind": "key", "t_wall": t, "t_mono": float(ts),
                        "category": "backspace" if r.key == "Backspace" else "alpha"})
        elif r.type == "scroll":
            out.append({"kind": "scroll", "t_wall": t, "t_mono": float(ts), "delta_y": float(r.dy)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dux", default="../../data/external/dux")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="../../models")
    ap.add_argument("--backend-dir", default=None)
    ap.add_argument("--fixture", default=None,
                    help="write a train/serve parity fixture (wire-format windows + expected P)")
    a = ap.parse_args()

    win, X, y, groups = load_windows(a.dux)
    rep = reports()
    lopo = rep["lopo"]
    if (len(y), int(y.sum()), len(set(groups.tolist()))) != (lopo["n"], lopo["positives"],
                                                              lopo["n_sessions"]):
        raise SystemExit(f"window set {len(y)}/{int(y.sum())}/{len(set(groups.tolist()))} differs "
                         f"from the evaluated one {lopo['n']}/{lopo['positives']}/"
                         f"{lopo['n_sessions']}: re-run review/dux_review.py first")
    print(f"fitting on ALL platform-matched DUX: {len(y)} windows, {lopo['n_sessions']} sessions, "
          f"{int(y.sum())} confused, {X.shape[1]} features")

    clf = _gbdt(a.seed).fit(X, y)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / f"{STEM}.onnx"
    onnx_path.write_bytes(to_onnx_bool_safe(clf, X.shape[1]).SerializeToString())
    verify_onnx(onnx_path, clf, X[:64])

    sweep = rep["sweep"]
    rows = sweep["rows"] if "rows" in sweep else sweep["results"]
    floor = prespecified_floor(sweep)
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
            "input": "platform-matched (training/behavioral/review/input_matched.py): pointer "
                     "resampled to one sample per 100 ms, one click per press, windows scored "
                     "from one event",
            "label_source": "emotion_manual_Confusion >= 1.0, HUMAN annotation",
            "n_windows": int(len(y)), "n_sessions": int(lopo["n_sessions"]),
            "n_confused": int(y.sum()), "positive_rate": float(y.mean()),
            "fit": "all windows (hyperparameters fixed by the deployed model's LOPO)",
        },
        "honest_performance_loso": {
            "source": "reports/dux_review/input_matched.json (matched_serving_rule)",
            "auc": lopo["auc"], "auc_ci95_session": lopo["auc_ci95_session"],
            "cohen_kappa_at_0.5": lopo["kappa_at_0.5"], "average_precision": lopo["average_precision"],
            "base_rate": lopo["base_rate"],
            "permutation_p_within_session": lopo["permutation_p_within_session"],
            "held_out_score_max": lopo["score_max"], "held_out_score_p99": lopo["score_p99"],
            "deployed_model_on_the_same_input_auc": rep["native_model_on_platform_input"]["auc"],
        },
        "gate": {
            "source": f"reports/dux_review/gate_sweeps.json ({ARM}|deployed) and nested_floor.json",
            "rule": rep["sweep_note"],
            "sweep": [{k: r[k] for k in ("threshold", "interventions", "correct", "precision",
                                         "interventions_per_hour", "ci95_session")} for r in rows],
            "prespecified_floor": floor,
            "prespecified_rule": f"lowest floor with gated precision >= {LIFT} x base rate and "
                                 f">= {MIN_OFFERS} offers",
            "prespecified_outcome": ("no floor qualifies on the full out-of-fold predictions"
                                     if floor is None else f"floor {floor}"),
            "nested_held_out": {k: rep["nested"][k] for k in (
                "held_out_offers", "held_out_correct", "held_out_precision",
                "held_out_precision_ci95_session", "lift", "chosen_floor_distribution")},
        },
        "calibration_uncalibrated": {k: rep["calibration"][k] for k in (
            "ece_10_equal_width", "brier", "mean_predicted", "base_rate")},
        "limitations": [
            "DUX participants filled in business expense forms, not lessons: the input is matched "
            "to the platform, the activity is not. Nothing here measures detection on lesson pages.",
            "Under the deployed gate rule its offers are correct about 30% of the time (base rate "
            "18%), lower than the native model on native input.",
            "Scores are not calibrated probabilities (ECE about 0.20): a floor is a rank threshold.",
            "Binary: confusion only. The negative class means 'not annotated as confused'.",
            "DUX sessions are not verified as distinct people.",
        ],
    }
    (out / f"{STEM}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {onnx_path} and {out / (STEM + '.json')}")
    print(f"  LOSO AUC {lopo['auc']:.4f} {lopo['auc_ci95_session']}; pre-specified floor: "
          f"{meta['gate']['prespecified_outcome']}")

    if a.fixture:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path))
        p = np.asarray(sess.run(None, {"features": X.astype(np.float32)})[1])[:, 1]
        rng = np.random.default_rng(a.seed)
        pick = sorted(set(np.argsort(-p)[:6].tolist())
                      | set(rng.choice(np.flatnonzero(y == 1), 4, replace=False).tolist())
                      | set(rng.choice(np.flatnonzero(y == 0), 4, replace=False).tolist()))
        fixture = {
            "model": f"models/{STEM}.onnx",
            "note": "DUX windows (platform-matched) in the browser's wire format, with the "
                    "P(confused) the ML pipeline computes for them. The backend must reproduce it.",
            "capture_started_at_wall": WIRE_START_MS,
            "windows": [{"events": to_wire(win[i]["events"], WIRE_START_MS),
                         "p_confused": float(p[i])} for i in pick],
        }
        Path(a.fixture).parent.mkdir(parents=True, exist_ok=True)
        Path(a.fixture).write_text(json.dumps(fixture), encoding="utf-8")
        print(f"wrote parity fixture ({len(pick)} windows) to {a.fixture}")

    if a.backend_dir:
        bd = Path(a.backend_dir)
        bd.mkdir(parents=True, exist_ok=True)
        for f in (onnx_path, out / f"{STEM}.json"):
            (bd / f.name).write_bytes(f.read_bytes())
        print(f"copied both into {bd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
