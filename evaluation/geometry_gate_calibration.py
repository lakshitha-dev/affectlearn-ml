"""Threshold sweep for the geometric disengagement channel, before it is allowed to intervene.

WHY THIS EXISTS

`DECISIVE_AFFECT_SOURCES` (`edges.py:143`) exists because a facial channel was once allowed to fire
interventions on its own and, with a probability distribution centred on 0.502 inside a 0.127-wide
band, it triggered more of them than the stronger behavioural channel. The geometric model is a
better detector — Test AUC 0.9225 against the behavioural channel's 0.7473 — but "better AUC"
is not the property the gate depends on. The gate depends on precision at a fixed operating point,
and that has to be measured before the channel is made decisive, exactly as it was for the
behavioural channel in `gate_calibration.py`.

`simulate` and `cluster_ci` are imported from that script rather than reimplemented, so both
channels are calibrated by identical code and the two tables are directly comparable.

TWO DIFFERENCES FROM THE BEHAVIOURAL SWEEP, BOTH LOAD-BEARING

**Ordering.** The gate's persistence and cooldown counters are sequential: they ask what happened
in the *previous* windows. The committed prediction dump is ordered by `sorted(glob(...))`, which
is lexicographic, so chunk 10 sorts between chunk 1 and chunk 2. Simulating a gate over that order
interleaves moments that were minutes apart and silently reports a precision for a sequence that
never occurred. Clips are therefore re-sorted numerically by (subject, video, chunk) here, and the
script refuses to run if the parse fails rather than falling back to the stored order.

**Window length.** EngageNet clips are 10 s; the deployed serving cycle is 30 s. Precision and the
intervention count are properties of the sequence and do not depend on which is assumed, but the
*rate* columns do, by a factor of three. Both readings are reported rather than picking one,
because the honest statement is that the corpus cannot settle the deployed rate.

    python evaluation/geometry_gate_calibration.py \
        --predictions C:/engagenet/results/test_predictions.npz \
        --out reports/engagenet_screen/gate_calibration.json

The predictions live outside the repository by design: they carry per-clip EngageNet labels, which
the end-user licence forbids redistributing. Only the aggregate sweep is written into `reports/`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gate_calibration import cluster_ci, simulate  # noqa: E402

CLIP_RE = re.compile(r"^(subject_\d+)_[^_]+_vid_(\d+)_(\d+)$")

CLIP_SECONDS = 10.0      # EngageNet clip length
CYCLE_SECONDS = 30.0     # deployed serving cycle

THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def temporal_order(clip_ids: np.ndarray) -> np.ndarray:
    """Indices that put clips in real recording order: (subject, video, chunk), numerically.

    Returns an index array. Raises rather than degrading to the stored order, because a silently
    mis-ordered sequence produces a plausible-looking precision for a sequence that never happened.
    """
    keys = []
    for i, c in enumerate(clip_ids):
        m = CLIP_RE.match(str(c))
        if not m:
            raise SystemExit(
                f"cannot parse clip id {c!r} as subject/video/chunk. The gate simulation is "
                f"sequential, so it cannot run on an order it cannot verify."
            )
        keys.append((m.group(1), int(m.group(2)), int(m.group(3)), i))
    keys.sort()
    return np.array([k[3] for k in keys])


def rate_columns(fired: int, n: int, seconds_per_window: float) -> dict:
    hours = n * seconds_per_window / 3600.0
    return {
        "hours_of_sequence": hours,
        "interventions_per_hour": (fired / hours) if hours else float("nan"),
        "minutes_between": (hours * 60.0 / fired) if fired else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="npz with y_true, y_prob, groups, clip_id (kept outside the repo)")
    ap.add_argument("--out", default="reports/engagenet_screen/gate_calibration.json")
    ap.add_argument("--min-consecutive", type=int, default=2,
                    help="ADAPT_MIN_CONSECUTIVE in edges.py")
    ap.add_argument("--cooldown", type=int, default=3, help="ADAPT_COOLDOWN_CYCLES in edges.py")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    d = np.load(a.predictions, allow_pickle=False)
    y, prob, groups, ids = d["y_true"], d["y_prob"], d["groups"], d["clip_id"]
    if prob.ndim == 2:                       # tolerate either dump convention
        prob = prob[:, 1]

    order = temporal_order(ids)
    n_moved = int((order != np.arange(len(order))).sum())
    y, prob, groups = y[order], prob[order], groups[order]

    base = float(y.mean())
    print(f"  {a.predictions}")
    print(f"  {len(y):,} clips from {len(np.unique(groups))} participants | "
          f"base rate {base:.4f}")
    print(f"  re-sorted into temporal order: {n_moved:,} of {len(order):,} clips moved")
    print(f"  gate: min_consecutive={a.min_consecutive}, cooldown={a.cooldown}\n")

    rows = []
    # Ungated: intervene on every positive detection. Precision is then the base rate.
    ungated = {"threshold": None, "interventions": int(len(y)), "correct": int(y.sum()),
               "precision": base, "ci95": None,
               "as_10s_clip": rate_columns(len(y), len(y), CLIP_SECONDS),
               "as_30s_cycle": rate_columns(len(y), len(y), CYCLE_SECONDS)}
    rows.append(ungated)

    hdr = (f"  {'thr':>5} {'fired':>6} {'ok':>5} {'prec':>6} {'95% CI':>16} "
           f"{'/h @10s':>8} {'/h @30s':>8}")
    print(hdr)
    print(f"  {'none':>5} {len(y):>6} {int(y.sum()):>5} {base:>6.3f} {'—':>16} "
          f"{ungated['as_10s_clip']['interventions_per_hour']:>8.1f} "
          f"{ungated['as_30s_cycle']['interventions_per_hour']:>8.1f}")

    for thr in THRESHOLDS:
        r = simulate(y, prob, groups, threshold=thr,
                     min_consecutive=a.min_consecutive, cooldown=a.cooldown)
        lo, hi = cluster_ci(y, prob, groups, threshold=thr,
                            min_consecutive=a.min_consecutive, cooldown=a.cooldown, seed=a.seed)
        # simulate() computes its rate columns against a 30 s window; recompute both readings.
        row = {"threshold": thr, "interventions": r["interventions"], "correct": r["correct"],
               "precision": r["precision"], "ci95": [lo, hi],
               "as_10s_clip": rate_columns(r["interventions"], len(y), CLIP_SECONDS),
               "as_30s_cycle": rate_columns(r["interventions"], len(y), CYCLE_SECONDS)}
        rows.append(row)
        ci = f"[{lo:.3f}, {hi:.3f}]" if lo == lo else "—"
        print(f"  {thr:>5.2f} {r['interventions']:>6} {r['correct']:>5} {r['precision']:>6.3f} "
              f"{ci:>16} {row['as_10s_clip']['interventions_per_hour']:>8.1f} "
              f"{row['as_30s_cycle']['interventions_per_hour']:>8.1f}")

    gated = [r for r in rows if r["threshold"] is not None and r["interventions"]]
    best = max(gated, key=lambda r: r["precision"]) if gated else None
    if best:
        print(f"\n  highest precision at threshold {best['threshold']:.2f}: "
              f"{best['precision']:.3f} on {best['correct']}/{best['interventions']} "
              f"(CI [{best['ci95'][0]:.3f}, {best['ci95'][1]:.3f}])")
        print(f"  behavioural channel for comparison: 0.500 at threshold 0.70 "
              f"(reports/dux_v1_z/gate_calibration.json)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "channel": "facial_geometry", "model": "4_lean",
        "source_predictions": str(a.predictions),
        "n_clips": int(len(y)), "n_participants": int(len(np.unique(groups))),
        "base_rate": base, "clips_reordered": n_moved,
        "clip_seconds": CLIP_SECONDS, "cycle_seconds": CYCLE_SECONDS,
        "min_consecutive": a.min_consecutive, "cooldown_cycles": a.cooldown, "seed": a.seed,
        "recommended_threshold": best["threshold"] if best else None,
        "note": ("Precision and counts are properties of the sequence; the per-hour columns depend "
                 "on whether a window is read as a 10 s EngageNet clip or a 30 s serving cycle, so "
                 "both are reported. Aggregates only - no per-clip EngageNet data is committed."),
        "sweep": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
