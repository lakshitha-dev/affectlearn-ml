"""Shared helpers for the V3-review experiments on the EngageNet geometry channel.

Everything here reuses the committed machinery rather than reimplementing it: the feature builder,
fold assignment, subject bootstrap and within-subject permutation come from `rungs.py`, and the
temporal ordering of clips comes from `evaluation/geometry_gate_calibration.py`. The one piece that
is new is `simulate_masked`, a gate replay that can skip windows the deployed client would never
score. With no mask it is identical to `evaluation/gate_calibration.simulate`, and
`face_presence.py` asserts that before using it.

Outputs are aggregates only. EngageNet's licence forbids redistributing per-clip labels or anything
derived from the video, so nothing keyed by clip is written into `reports/`.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
ENG = HERE.parent
REPO = ENG.parents[1]
sys.path.insert(0, str(ENG))
sys.path.insert(0, str(REPO / "evaluation"))

import rungs  # noqa: E402
from geometry_gate_calibration import temporal_order  # noqa: E402

OUT = REPO / "reports" / "engagenet_review"
WORK = pathlib.Path("C:/engagenet")
RUNG = "4_lean"
SEED = 42
N_BOOT = 2000
THRESHOLD = 0.5            # the threshold rungs.report() uses for accuracy and kappa
FLOORS = tuple(round(0.50 + 0.05 * i, 2) for i in range(9))   # 0.50 .. 0.90
MIN_CONSECUTIVE = 2        # ADAPT_MIN_CONSECUTIVE (deployed)
COOLDOWN = 3               # ADAPT_COOLDOWN_CYCLES (deployed)
CLIP_SECONDS = 10.0
CYCLE_SECONDS = 30.0
NPZ = {"Train": "TrainSub.npz", "Validation": "Validation.npz", "Test": "Test.npz"}


def load_split(split: str, work: pathlib.Path = WORK):
    """feats (n, 10, 11), binary y, subject groups, clip ids, exactly as the protocol loads them."""
    return rungs.load(work / "geom" / NPZ[split], work / "labels", split)


def committed_test_predictions(work: pathlib.Path = WORK):
    d = np.load(work / "results" / "test_predictions.npz", allow_pickle=False)
    return (d["y_true"].astype(int), d["y_prob"].astype(float),
            np.asarray([str(s) for s in d["groups"]]), np.asarray([str(c) for c in d["clip_id"]]))


def align(ids_from, values, ids_to):
    """Reorder `values` (indexed like `ids_from`) into the order of `ids_to`."""
    pos = {c: i for i, c in enumerate(ids_from)}
    missing = [c for c in ids_to if c not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} clips cannot be aligned (e.g. {missing[0]})")
    return np.asarray(values)[[pos[c] for c in ids_to]]


def versions() -> dict:
    import numpy
    import sklearn
    out = {"numpy": numpy.__version__, "scikit_learn": sklearn.__version__}
    for mod in ("onnx", "onnxruntime", "skl2onnx"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    return out


def write(name: str, payload: dict) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(REPO)}")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def auc_block(y, p, g, *, threshold: float = THRESHOLD, permutation: bool = True) -> dict:
    """The metric set rungs.report() uses, returned rather than printed."""
    y, p, g = np.asarray(y), np.asarray(p, dtype=float), np.asarray(g)
    pred = (p >= threshold).astype(int)
    base = float(y.mean())
    lo, hi = rungs.cluster_ci(y, p, g, n=N_BOOT, seed=SEED)
    ap = float(average_precision_score(y, p))
    out = {
        "n": int(len(y)), "participants": int(len(np.unique(g))), "positive_rate": base,
        "auc": float(roc_auc_score(y, p)), "auc_ci95_participant": [lo, hi],
        "accuracy": float((pred == y).mean()), "majority_baseline": float(max(base, 1 - base)),
        "kappa": float(cohen_kappa_score(y, pred)),
        "average_precision": ap, "ap_over_base_rate": ap / base,
        "decision_threshold": threshold,
    }
    if permutation:
        out["perm_p_within_participant"] = float(rungs.perm_p(y, p, g, n=N_BOOT, seed=SEED))
    return out


def paired_auc_diff(y, p_a, p_b, g, *, n: int = N_BOOT, seed: int = SEED) -> dict:
    """AUC(a) - AUC(b) with a participant bootstrap that resamples both arms together."""
    y, p_a, p_b, g = map(np.asarray, (y, p_a, p_b, g))
    rng = np.random.default_rng(seed)
    ids = np.unique(g)
    by = {s: np.flatnonzero(g == s) for s in ids}
    obs = float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b))
    diffs = []
    for _ in range(n):
        b = np.concatenate([by[s] for s in rng.choice(ids, size=len(ids), replace=True)])
        if len(np.unique(y[b])) > 1:
            diffs.append(roc_auc_score(y[b], p_a[b]) - roc_auc_score(y[b], p_b[b]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_auc": obs, "ci95_participant": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


# ── gate replay ──────────────────────────────────────────────────────────────────────────
def simulate_masked(y, prob, groups, *, threshold, scorable=None,
                    min_consecutive: int = MIN_CONSECUTIVE, cooldown: int = COOLDOWN) -> dict:
    """Replay the gate over each participant's windows in stored order.

    With `scorable=None` this is `evaluation/gate_calibration.simulate` line for line: the streak
    is updated before the cooldown check, so it keeps counting during cooldown, as the deployed
    persistence check (which reads the channel's reading history) does.

    With a mask, an unscorable window mirrors the deployed face-absent path
    (`affect_detection.py:61-69`): it yields no reading, so it neither extends nor breaks the
    streak (persistence reads the channel's reading history, which gains no entry), and it can
    never fire. It still takes one cycle of wall time, so an active cooldown still counts it down
    (the cooldown compares cycle numbers).
    """
    y, prob, groups = np.asarray(y), np.asarray(prob, dtype=float), np.asarray(groups)
    ok = np.ones(len(y), dtype=bool) if scorable is None else np.asarray(scorable, dtype=bool)
    fired = correct = 0
    for g in np.unique(groups):
        streak = cool = 0
        for i in np.flatnonzero(groups == g):
            if not ok[i]:
                if cool > 0:
                    cool -= 1
                continue
            over = prob[i] >= threshold
            streak = streak + 1 if over else 0
            if cool > 0:
                cool -= 1
                continue
            if over and streak >= min_consecutive:
                fired += 1
                correct += int(y[i] == 1)
                streak = 0
                cool = cooldown
    return {"interventions": fired, "correct": correct,
            "precision": (correct / fired) if fired else float("nan")}


def masked_ci(y, prob, groups, *, threshold, scorable=None, n: int = N_BOOT,
              seed: int = SEED) -> list:
    """Participant-bootstrap percentile interval for gated precision.

    Same resampling as `gate_calibration.cluster_ci` (participants drawn with replacement, a
    repeated participant kept as a separate sequence, draws with no intervention skipped, the same
    RNG and draw order). Because the gate's state resets at every participant boundary, a draw's
    interventions and correct interventions are exactly the sums of each drawn participant's own
    counts, so those are computed once and summed per draw instead of re-simulating the sequence.
    """
    y, prob, groups = np.asarray(y), np.asarray(prob, dtype=float), np.asarray(groups)
    ok = np.ones(len(y), dtype=bool) if scorable is None else np.asarray(scorable, dtype=bool)
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    fired = np.zeros(len(ids), dtype=int)
    correct = np.zeros(len(ids), dtype=int)
    for k, s in enumerate(ids):
        i = np.flatnonzero(groups == s)
        r = simulate_masked(y[i], prob[i], np.zeros(len(i)), threshold=threshold, scorable=ok[i])
        fired[k], correct[k] = r["interventions"], r["correct"]
    out = []
    for _ in range(n):
        pick = rng.choice(len(ids), size=len(ids), replace=True)
        f = int(fired[pick].sum())
        if f:
            out.append(correct[pick].sum() / f)
    if not out:
        return [float("nan"), float("nan")]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def sweep(y, prob, groups, ids, *, scorable=None, floors=FLOORS, label: str = "") -> dict:
    """Gate sweep in real recording order, with lift and both rate readings."""
    order = temporal_order(np.asarray(ids))
    y, prob, groups = np.asarray(y)[order], np.asarray(prob, dtype=float)[order], np.asarray(groups)[order]
    ok = None if scorable is None else np.asarray(scorable, dtype=bool)[order]
    elig = np.ones(len(y), dtype=bool) if ok is None else ok
    base = float(y[elig].mean())
    n_all = len(y)

    def rates(fired):
        out = {}
        for key, sec in (("as_30s_cycle", CYCLE_SECONDS), ("as_10s_clip", CLIP_SECONDS)):
            hours = n_all * sec / 3600.0
            out[key] = {"interventions_per_hour": fired / hours,
                        "minutes_between": (hours * 60.0 / fired) if fired else None}
        return out

    rows = [{"floor": None, "interventions": int(elig.sum()), "correct": int(y[elig].sum()),
             "precision": base, "lift": 1.0, "ci95_participant": None,
             **rates(int(elig.sum()))}]
    for thr in floors:
        r = simulate_masked(y, prob, groups, threshold=thr, scorable=ok)
        lo, hi = masked_ci(y, prob, groups, threshold=thr, scorable=ok)
        prec = r["precision"]
        rows.append({"floor": thr, "interventions": r["interventions"], "correct": r["correct"],
                     "precision": prec, "lift": (prec / base) if r["interventions"] else None,
                     "ci95_participant": [lo, hi], **rates(r["interventions"])})
        print(f"    {label:>10} floor {thr:.2f}: {r['interventions']:>4} offers, "
              f"precision {prec:.3f} [{lo:.3f}, {hi:.3f}], lift {rows[-1]['lift'] or float('nan'):.2f}")
    return {"n_windows_in_sequence": int(n_all), "n_scorable": int(elig.sum()),
            "participants": int(len(np.unique(groups))), "base_rate_scorable": base,
            "min_consecutive": MIN_CONSECUTIVE, "cooldown_cycles": COOLDOWN, "seed": SEED,
            "n_bootstrap": N_BOOT, "rows": rows}
