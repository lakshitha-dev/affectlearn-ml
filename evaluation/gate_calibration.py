"""Recompute the adaptation-gate threshold sweep from committed out-of-fold predictions.

Why this exists
---------------
The gate calibration is reported in the paper and the thesis as a headline system claim — help
offered roughly once every forty minutes with about half of those offers warranted, against 18% if
help were offered on every positive detection. Until now that sweep existed only as a table in
`reports/dux_confusion/FINDINGS.md` and a rationale string in
`training/behavioral/export_dux_confusion.py`. There was no script, so the numbers could not be
regenerated, and — more importantly — no denominator was ever reported alongside the precision.

The denominator turns out to matter a great deal. At 1.5 interventions per hour over the pooled
window sequence there are only a couple of dozen gated interventions, so a precision of 0.500 rests
on single-digit counts of correct ones. Reporting it without an interval overstates it, and the
findings document already noticed the symptom without naming the cause: neighbouring settings
"wobble ... which is noise on single-digit counts, not signal".

The gate is reimplemented here rather than imported because the backend's version consumes live
session state, but the logic is the same three conditions in the same order: a confidence floor, a
persistence requirement over consecutive windows, and a cooldown that suppresses a second
intervention for a fixed number of cycles after one fires.

Intervals resample participants, not windows, for the reason set out in `dux_cluster_intervals.py`.

    python evaluation/gate_calibration.py --run reports/dux_v1_z
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WINDOW_S = 30.0


def simulate(y: np.ndarray, prob: np.ndarray, groups: np.ndarray, *, threshold: float,
             min_consecutive: int, cooldown: int) -> dict:
    """Fire the gate over each participant's window sequence in order.

    Returns the intervention count and how many landed on a window a human annotated as confused.
    Sequences are per participant because the cooldown and persistence counters are session-scoped
    in the deployed system; running them across a participant boundary would carry one learner's
    state into another's.
    """
    fired = 0
    correct = 0
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        streak = 0
        cool = 0
        for i in idx:
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
    hours = len(y) * WINDOW_S / 3600.0
    return {
        "threshold": threshold,
        "interventions": fired,
        "correct": correct,
        "precision": (correct / fired) if fired else float("nan"),
        "interventions_per_hour": fired / hours if hours else float("nan"),
        "minutes_between": (hours * 60.0 / fired) if fired else float("nan"),
    }


def cluster_ci(y, prob, groups, *, threshold, min_consecutive, cooldown,
               seed: int = 42, n: int = 2000) -> tuple[float, float]:
    """Percentile interval for gated precision, resampling participants with replacement."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ids}
    out = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        # Relabel the draw so repeated participants stay separate sequences.
        ys, ps, gs = [], [], []
        for k, g in enumerate(pick):
            i = idx_by[g]
            ys.append(y[i]); ps.append(prob[i]); gs.append(np.full(len(i), k))
        r = simulate(np.concatenate(ys), np.concatenate(ps), np.concatenate(gs),
                     threshold=threshold, min_consecutive=min_consecutive, cooldown=cooldown)
        if r["interventions"]:
            out.append(r["precision"])
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="reports/dux_v1_z")
    ap.add_argument("--arm", default="behavioural")
    ap.add_argument("--min-consecutive", type=int, default=2)
    ap.add_argument("--cooldown", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    p = Path(a.run) / f"{a.arm}_predictions.npz"
    if not p.exists():
        raise SystemExit(f"not found: {p}")
    d = np.load(p, allow_pickle=True)
    y, prob, groups = d["y_true"], d["y_prob"][:, 1], d["groups"]

    base = float(y.mean())
    hours = len(y) * WINDOW_S / 3600.0
    print(f"  {p}")
    print(f"  {len(y)} windows from {len(np.unique(groups))} participants "
          f"= {hours:.1f} h of sequence | base rate {base:.4f}")
    print(f"  gate: min_consecutive={a.min_consecutive}, cooldown={a.cooldown}\n")

    rows = []
    # The ungated arm: every positive detection intervenes, so precision is the base rate of
    # windows above threshold and the rate is one per window.
    ungated = {
        "threshold": None, "interventions": int(len(y)), "correct": int(y.sum()),
        "precision": base, "interventions_per_hour": len(y) / hours,
        "minutes_between": hours * 60.0 / len(y), "ci95": None,
    }
    rows.append(ungated)
    print(f"  {'thr':>5} {'n':>5} {'ok':>4} {'prec':>6} {'95% CI':>16} {'per h':>7} {'every':>9}")
    print(f"  {'none':>5} {ungated['interventions']:>5} {ungated['correct']:>4} "
          f"{ungated['precision']:>6.3f} {'—':>16} {ungated['interventions_per_hour']:>7.1f} "
          f"{ungated['minutes_between']:>7.1f} m")

    for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        r = simulate(y, prob, groups, threshold=thr,
                     min_consecutive=a.min_consecutive, cooldown=a.cooldown)
        lo, hi = cluster_ci(y, prob, groups, threshold=thr, min_consecutive=a.min_consecutive,
                            cooldown=a.cooldown, seed=a.seed)
        r["ci95"] = [lo, hi]
        rows.append(r)
        ci = f"[{lo:.3f}, {hi:.3f}]" if lo == lo else "—"
        print(f"  {thr:>5.2f} {r['interventions']:>5} {r['correct']:>4} {r['precision']:>6.3f} "
              f"{ci:>16} {r['interventions_per_hour']:>7.1f} {r['minutes_between']:>7.1f} m")

    out = Path(a.out) if a.out else Path(a.run) / "gate_calibration.json"
    out.write_text(json.dumps({
        "arm": a.arm, "n_windows": int(len(y)), "n_participants": int(len(np.unique(groups))),
        "hours_of_sequence": hours, "base_rate": base, "window_seconds": WINDOW_S,
        "min_consecutive": a.min_consecutive, "cooldown_cycles": a.cooldown, "seed": a.seed,
        "sweep": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
