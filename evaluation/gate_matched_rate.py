"""Gate variants compared at a matched offer rate, and how evenly each spaces its offers.

    python evaluation/gate_matched_rate.py --work C:/engagenet

WHY

The ablation (gate_deployed_rule.py --ablation) compares variants at one floor, where persistence
and the cooldown cut offers without changing precision. That leaves open whether a higher floor
alone would cut offers just as far at equal or better precision. This script answers it:

  curves      precision against offers per hour for each variant (floor only, + persistence,
              + cooldown, deployed rule), sweeping the floor from 0.50 to 0.99 in steps of 0.01.
  matched     the deployed rule at 0.70 is the reference. For floor-only and floor+cooldown, the
              floor on that grid whose offer count is closest to the reference (ties go to the
              higher floor) is compared with it: precision difference (variant minus deployed) with
              a paired participant bootstrap (the same 2,000 draws for both arms).
  spacing     for every compared configuration: the share of offers that come fewer than three
              cycles (90 s) after the previous offer to the same participant, and the longest run
              of offers in consecutive cycles.

Channels, masking and ordering are those of the ablation (gate_deployed_rule._ablation_channels):
geometry on EngageNet Test with windows the browser would not score masked, and the interaction
channel on DUX, raw and platform-like. Aggregates only are written.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "evaluation"))

import gate_deployed_rule as G  # noqa: E402

GRID = [round(0.50 + 0.01 * i, 2) for i in range(50)]
REFERENCE_FLOOR = 0.70
VARIANTS = G.VARIANTS                                    # name -> (persistence, cooldown)


def offer_positions(y, p, ok, floor, persistence, cooldown):
    """Indices of the windows at which the rule makes an offer (same logic as G._one_sequence)."""
    pos = []
    history: list[int] = []
    last = None
    for k in range(len(p)):
        if ok is not None and not ok[k]:
            continue
        state = 1 if p[k] >= 0.5 else 0
        history.append(state)
        if state != 1 or p[k] < floor:
            continue
        if persistence:
            recent = history[-G.MIN_CONSECUTIVE:]
            if len(recent) < G.MIN_CONSECUTIVE or any(s != 1 for s in recent):
                continue
        if cooldown and last is not None and k - last < G.COOLDOWN:
            continue
        pos.append(k)
        last = k
    return pos


def spacing(seqs, floor, persistence, cooldown) -> dict:
    close = total = longest = 0
    for y, p, ok in seqs:
        pos = offer_positions(y, p, ok, floor, persistence, cooldown)
        total += len(pos)
        run = 1
        for a, b in zip(pos, pos[1:]):
            gap = b - a
            close += int(gap < G.COOLDOWN)
            run = run + 1 if gap == 1 else 1
            longest = max(longest, run)
        if pos:
            longest = max(longest, 1)
    return {"offers": total, "share_within_90s": (close / total) if total else None,
            "longest_consecutive_run": longest}


def curve(seqs, hours, persistence, cooldown) -> list[dict]:
    rows = []
    for f in GRID:
        fired, correct = G.simulate(seqs, f, persistence, cooldown)
        rows.append({"floor": f, "offers": fired, "correct": correct,
                     "precision": (correct / fired) if fired else None,
                     "offers_per_hour": fired / hours})
    return rows


def matched_floor(rows, target_offers) -> float:
    cands = [r for r in rows if r["offers"] > 0]
    return min(cands, key=lambda r: (abs(r["offers"] - target_offers), -r["floor"]))["floor"]


def paired(seqs, arm_a, arm_b) -> dict:
    """Precision of arm_a minus arm_b over the same participant draws. Arms are (floor, pers, cool)."""
    rng = np.random.default_rng(G.SEED)
    diffs = []
    for _ in range(G.N_BOOT):
        d = rng.integers(0, len(seqs), len(seqs))
        s = [seqs[i] for i in d]
        fa, ca = G.simulate(s, *arm_a)
        fb, cb = G.simulate(s, *arm_b)
        if fa and fb:
            diffs.append(ca / fa - cb / fb)
    fa, ca = G.simulate(seqs, *arm_a)
    fb, cb = G.simulate(seqs, *arm_b)
    return {"difference": ca / fa - cb / fb,
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_of_draws_above_zero": float(np.mean(np.array(diffs) > 0)),
            "n_draws_used": len(diffs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="C:/engagenet")
    a = ap.parse_args()

    out = {"_note": __doc__.strip().splitlines()[0], "grid": [GRID[0], GRID[-1], 0.01],
           "reference": {"variant": "deployed_rule", "floor": REFERENCE_FLOOR},
           "n_bootstrap": G.N_BOOT, "seed": G.SEED, "channels": {}}
    for name, (seqs, n, base) in G._ablation_channels(pathlib.Path(a.work)).items():
        hours = n * G.WINDOW_S / 3600.0
        curves = {v: curve(seqs, hours, *pc) for v, pc in VARIANTS.items()}
        ref_rows = {r["floor"]: r for r in curves["deployed_rule"]}
        ref = ref_rows[REFERENCE_FLOOR]
        ref_arm = (REFERENCE_FLOOR, True, True)
        block = {"windows": n, "hours": hours, "base_rate": base, "curves": curves,
                 "reference": {**ref, "spacing": spacing(seqs, *ref_arm)}, "matched": {}}
        for v in ("floor_only", "floor_cooldown"):
            pers, cool = VARIANTS[v]
            f = matched_floor(curves[v], ref["offers"])
            row = next(r for r in curves[v] if r["floor"] == f)
            block["matched"][v] = {**row, "spacing": spacing(seqs, f, pers, cool),
                                   "precision_minus_deployed": paired(seqs, (f, pers, cool), ref_arm)}
        out["channels"][name] = block
        print(f"== {name}: reference deployed@0.70 offers {ref['offers']} "
              f"prec {ref['precision']:.3f} per_h {ref['offers_per_hour']:.2f}")
        for v, m in block["matched"].items():
            d = m["precision_minus_deployed"]
            print(f"   {v:15s} floor {m['floor']:.2f} offers {m['offers']} prec "
                  f"{m['precision']:.3f} diff {d['difference']:+.3f} "
                  f"{[round(x, 3) for x in d['ci95']]} P(>0) {d['share_of_draws_above_zero']:.3f} "
                  f"within90s {m['spacing']['share_within_90s']:.3f} "
                  f"longest {m['spacing']['longest_consecutive_run']}")
    dest = REPO / "reports" / "gate_deployed_rule" / "matched_rate.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
