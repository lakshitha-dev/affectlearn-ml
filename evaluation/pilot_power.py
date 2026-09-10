"""What the planned learner pilot can and cannot detect, at the sample size available.

Why this exists
---------------
The evaluation protocol commits the pilot's analysis in advance and then hedges it without a
number: effect sizes "are framed as estimation rather than confirmation given the power available
at n = 20 to 30". The limitations section repeats the same thing qualitatively — "underpowered for
anything but a large effect". Neither says what the design can actually detect, and a pre-committed
analysis plan that cannot state its own sensitivity is not really pre-committed.

Sensitivity is a property of the design, not of the data, so it can be established before a single
participant is recruited. That is the whole point of computing it here: the pilot has not run, and
this says what it would have been capable of answering if it had.

Two comparisons, and the asymmetry between them is the finding
--------------------------------------------------------------
The pilot allocates n = 20 to 30 participants across an adaptive and a non-adaptive arm, so each
arm holds 10 to 15. Two different questions are asked of that sample:

  * BETWEEN ARMS — did adaptation help? An independent-samples comparison of gain scores. This is
    the RQ4 test, and it is the one the sample cannot support.
  * WITHIN PARTICIPANT — did learners learn from the material at all? A paired comparison of pre
    against post. Pairing removes between-person variance, so the same participants support a
    considerably smaller effect here.

Reporting both makes clear which of the two questions was ever answerable, rather than leaving a
reader to conclude the whole pilot was too small for anything.

No assumed effect size
----------------------
No effect size for affect-aware adaptation on learning gain is established in the literature
reviewed in Chapter 2, so no target effect is pre-specified and none is assumed here. This script
therefore reports the *minimum detectable* effect at conventional power, plus the power the design
would have across a range of effect sizes. The large-n rows are arithmetic about the design, not a
prediction about what an effect would turn out to be.

    python evaluation/pilot_power.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ALPHA = 0.05
POWER = 0.80

# Totals swept for the between-arms case; allocation is even, so each arm gets half.
TOTALS = (20, 24, 30, 40, 60, 100, 150, 200, 260)
PAIRED_NS = (20, 25, 30)
REFERENCE_D = (0.3, 0.5, 0.8, 1.0, 1.2)


def power_independent(n_per_arm: int, d: float, alpha: float = ALPHA) -> float:
    """Power of a two-tailed independent-samples t test at effect size d.

    Only the upper rejection region is counted. The opposite-tail term is negligible for effects in
    this range — it is the probability of rejecting in the wrong direction — and including it makes
    `nct.cdf(-crit, df, ncp)` underflow to NaN at larger non-centrality, which silently breaks the
    root-finder rather than returning a slightly wrong answer.
    """
    df = 2 * n_per_arm - 2
    ncp = d * np.sqrt(n_per_arm / 2.0)
    crit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(crit, df, ncp))


def power_paired(n: int, dz: float, alpha: float = ALPHA) -> float:
    """Power of a two-tailed paired t test at standardised mean difference dz."""
    df = n - 1
    ncp = dz * np.sqrt(n)
    crit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(crit, df, ncp))


def mdes(power_fn, n: int, target: float = POWER) -> float:
    """Smallest effect size detectable at `target` power — solved rather than tabulated."""
    return float(optimize.brentq(lambda d: power_fn(n, d) - target, 0.01, 4.0, xtol=1e-4))


def build() -> dict:
    between = []
    for total in TOTALS:
        per_arm = total // 2
        between.append({
            "total_n": total,
            "per_arm": per_arm,
            "min_detectable_d": round(mdes(power_independent, per_arm), 4),
        })

    paired = [{"n": n, "min_detectable_dz": round(mdes(power_paired, n), 4)} for n in PAIRED_NS]

    # Power the design would have at the top of the planned range, across a range of effects.
    at_max_planned = [
        {"d": d, "power": round(power_independent(15, d), 4)} for d in REFERENCE_D
    ]

    return {
        "alpha": ALPHA,
        "tails": 2,
        "target_power": POWER,
        "planned_total_n": [20, 30],
        "allocation": "even, two arms",
        "assumed_effect_size": None,
        "note": ("No effect size for affect-aware adaptation on learning gain is established in the "
                 "reviewed literature, so none is assumed; minimum detectable effects are reported "
                 "instead of power against a target."),
        "between_arms": between,
        "within_participant": paired,
        "power_at_15_per_arm": at_max_planned,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/pilot_power/power.json")
    a = ap.parse_args()

    r = build()

    print(f"  two-tailed alpha = {ALPHA}, target power = {POWER:.0%}, even allocation\n")
    print("  BETWEEN ARMS - gain-score comparison (the RQ4 test)")
    print(f"    {'total n':>8} {'per arm':>8} {'min detectable d':>18}")
    for row in r["between_arms"]:
        print(f"    {row['total_n']:>8} {row['per_arm']:>8} {row['min_detectable_d']:>18.2f}")

    print("\n  power at 15 per arm (top of the planned range)")
    for row in r["power_at_15_per_arm"]:
        print(f"    d = {row['d']:<4} {row['power']:>6.1%}")

    print("\n  WITHIN PARTICIPANT - pre against post")
    for row in r["within_participant"]:
        print(f"    n = {row['n']:<4} min detectable dz = {row['min_detectable_dz']:.2f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
