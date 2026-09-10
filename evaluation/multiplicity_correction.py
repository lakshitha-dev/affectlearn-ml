"""Benjamini-Hochberg correction across the scored facial views.

Why this exists
---------------
`facial_subject_intervals.py` scores eleven views of the facial branch — four constructs, two cuts
apiece, plus two extra checkpoints — and each carries a within-subject permutation probability.
Reading those eleven probabilities individually and reporting the smallest as a finding is a
multiple-comparison error, and the error matters here because the paper's specificity argument is
explicitly collective: it claims that only one view clears both a subject-level interval and a
permutation test, and treats that selectivity as evidence the signal is construct-specific rather
than something the pipeline returns for any target. A claim of that shape has to be corrected for
the size of the family it ranges over.

Benjamini-Hochberg rather than Bonferroni, deliberately. The eleven views are heavily dependent —
same 1,638 clips, same 19 subjects, and each checkpoint's `any` and `high` cuts are near-duplicates
of one another — and BH controls the false discovery rate under positive dependence, which is the
situation here. Bonferroni would control the family-wise rate but at a cost that is not warranted
when the tests are this correlated.

Two limits are worth stating with the output. The permutation probabilities are add-one estimates
at n_perm = 1000, so they are censored below at 1/1001 and BH inherits that floor. And the family
is a judgement: correcting over the four constructs rather than the eleven views is defensible, so
this script reports both and the choice should be made on principle before the numbers are seen.

    python evaluation/multiplicity_correction.py \
        --in reports/facial_intervals/facial_subject_intervals.json \
        --out reports/facial_intervals/multiplicity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def benjamini_hochberg(ps: list[float]) -> list[float]:
    """BH-adjusted q-values, returned in the caller's original order.

    The step-up runs from the largest p to the smallest, carrying the running minimum, which is
    what enforces monotonicity: an adjusted value never exceeds the adjusted value of a larger raw
    probability.
    """
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    q = [0.0] * m
    carry = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        carry = min(carry, ps[i] * m / (rank + 1))
        q[i] = carry
    return q


def construct_of(label: str) -> str:
    """The affective construct a view scores, for the alternative four-way family."""
    for c in ("boredom", "confusion", "engagement", "frustration"):
        if c in label.lower():
            return c
    return "other"


def correct(results: list[dict], alpha: float = 0.05) -> dict:
    ps = [r["permutation_p_within_subject"] for r in results]
    qs = benjamini_hochberg(ps)

    rows = []
    for r, q in zip(results, qs):
        lo, hi = r["auc_ci_subject"]
        rows.append({
            "label": r["label"],
            "construct": construct_of(r["label"]),
            "auc": r["auc"],
            "auc_ci_subject": [lo, hi],
            "subject_ci_excludes_chance": bool(lo > 0.5),
            "permutation_p": r["permutation_p_within_subject"],
            "bh_q": q,
            "survives_bh": bool(q < alpha),
            # The paper's criterion is the conjunction, not either half.
            "survives_both": bool(q < alpha and lo > 0.5),
        })
    rows.sort(key=lambda x: x["permutation_p"])

    # The narrower family: one view per construct, the best-ranked by raw p.
    best: dict[str, dict] = {}
    for row in rows:
        best.setdefault(row["construct"], row)
    by_construct = list(best.values())
    cq = benjamini_hochberg([b["permutation_p"] for b in by_construct])
    per_construct = [
        {"construct": b["construct"], "label": b["label"], "permutation_p": b["permutation_p"],
         "bh_q": q, "survives_bh": bool(q < alpha),
         "survives_both": bool(q < alpha and b["subject_ci_excludes_chance"])}
        for b, q in zip(by_construct, cq)
    ]

    return {
        "alpha": alpha,
        "method": "benjamini-hochberg",
        "n_views": len(rows),
        "views": rows,
        "n_survive_bh": sum(r["survives_bh"] for r in rows),
        "n_survive_both": sum(r["survives_both"] for r in rows),
        "n_subject_ci_excludes_chance": sum(r["subject_ci_excludes_chance"] for r in rows),
        "family_of_constructs": per_construct,
        "n_constructs_survive_both": sum(c["survives_both"] for c in per_construct),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src",
                    default="reports/facial_intervals/facial_subject_intervals.json")
    ap.add_argument("--out", default="reports/facial_intervals/multiplicity.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    src = Path(a.src)
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    out = correct(json.loads(src.read_text(encoding="utf-8"))["results"], a.alpha)

    print(f"  {out['n_views']} views, Benjamini-Hochberg at q < {a.alpha}")
    print(f"  {'p':>7} {'BH q':>7}  {'AUC':>6}  {'CI>0.5':>6}  {'both':>4}  view")
    for r in out["views"]:
        print(f"  {r['permutation_p']:7.4f} {r['bh_q']:7.4f}  {r['auc']:6.4f}  "
              f"{'yes' if r['subject_ci_excludes_chance'] else 'no':>6}  "
              f"{'YES' if r['survives_both'] else '-':>4}  {r['label'][:46]}")
    print(f"\n  survive BH:                  {out['n_survive_bh']} of {out['n_views']}")
    print(f"  subject CI excludes chance:  {out['n_subject_ci_excludes_chance']} of {out['n_views']}")
    print(f"  survive BOTH:                {out['n_survive_both']} of {out['n_views']}")
    print(f"  survive both, per-construct family: {out['n_constructs_survive_both']} "
          f"of {len(out['family_of_constructs'])}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
