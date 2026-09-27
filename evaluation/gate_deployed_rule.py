"""Gate sweep under the rule the deployed gate applies, for both detection channels.

    python evaluation/gate_deployed_rule.py --work C:/engagenet

WHY THIS EXISTS

`evaluation/gate_calibration.py:simulate` (the sweep behind paper-v3 Table 5) is stricter than the
deployed gate in two ways: it requires EVERY reading in the persistence run to clear the floor, and
its cooldown lets the next offer come four cycles after an offer. The deployed gate
(affectlearn backend/app/agents/edges.py:passes_adaptation_gate) requires only the CURRENT reading to
clear the floor; persistence asks that the channel's last two readings share the current STATE
(argmax, i.e. P >= 0.5 for these binary models); and the cooldown lets the next offer come three
cycles (90 s of server time) after an offer.

This script replays that rule for:

* the geometry channel on the EngageNet Test split (committed predictions, recording order per
  participant), over all clips and with the windows the browser would not score masked
  (< 5 of 10 frames with a face: no reading, so no history entry, but a cycle of time still passes);
* the interaction channel on DUX (leave-one-session-out predictions written by
  training/behavioral/review/dux_review.py), raw features and platform-like inputs.

Offers on a window whose label is positive count as correct. Intervals resample participants
(sessions for DUX) with 2,000 draws, keeping each sequence intact. Offer rates read each window as
one 30 s cycle. Aggregates only are written.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training" / "engagenet" / "review"))
sys.path.insert(0, str(REPO / "evaluation"))

FLOORS = [round(0.50 + 0.05 * i, 2) for i in range(9)]
MIN_CONSECUTIVE = 2
COOLDOWN = 3
WINDOW_S = 30.0
N_BOOT = 2000
SEED = 42


def _one_sequence(y, p, ok, floor, persistence=True, cooldown=True):
    """Offers and correct offers for one participant's ordered windows.

    `persistence` and `cooldown` switch those checks off for the ablation (--ablation); with both
    on this is the deployed rule and reproduces sweep.json exactly.
    """
    fired = correct = 0
    history: list[int] = []
    last = None
    for k in range(len(p)):
        if ok is not None and not ok[k]:
            continue                                   # no reading: no history entry, time passes
        state = 1 if p[k] >= 0.5 else 0
        history.append(state)
        if state != 1 or p[k] < floor:
            continue
        if persistence:
            recent = history[-MIN_CONSECUTIVE:]
            if len(recent) < MIN_CONSECUTIVE or any(s != 1 for s in recent):
                continue
        if cooldown and last is not None and k - last < COOLDOWN:
            continue
        fired += 1
        correct += int(y[k] == 1)
        last = k
    return fired, correct


def simulate(seqs, floor, persistence=True, cooldown=True):
    fired = correct = 0
    for y, p, ok in seqs:
        f, c = _one_sequence(y, p, ok, floor, persistence, cooldown)
        fired += f
        correct += c
    return fired, correct


def sweep(seqs, n_windows, base_rate, rng_seed=SEED, persistence=True, cooldown=True):
    rng = np.random.default_rng(rng_seed)
    draws = [rng.integers(0, len(seqs), len(seqs)) for _ in range(N_BOOT)]
    hours = n_windows * WINDOW_S / 3600.0
    rows = [{"floor": None, "offers": n_windows, "precision": base_rate, "lift": 1.0,
             "ci95": None, "offers_per_hour": n_windows / hours}]
    for floor in FLOORS:
        fired, correct = simulate(seqs, floor, persistence, cooldown)
        prec = correct / fired if fired else None
        boot = []
        for d in draws:
            f, c = simulate([seqs[i] for i in d], floor, persistence, cooldown)
            if f:
                boot.append(c / f)
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None
        rows.append({"floor": floor, "offers": fired, "correct": correct, "precision": prec,
                     "lift": (prec / base_rate) if prec is not None else None, "ci95": ci,
                     "offers_per_hour": fired / hours,
                     "minutes_between": (hours * 60.0 / fired) if fired else None})
    return rows


def sequences(y, p, groups, ok=None):
    out = []
    for g in dict.fromkeys(groups):                     # first-appearance order, stable
        idx = np.flatnonzero(groups == g)
        out.append((y[idx], p[idx], None if ok is None else ok[idx]))
    return out


def engagenet(work: pathlib.Path) -> dict:
    import _common as C
    from geometry_gate_calibration import temporal_order

    feats, y_lab, _, ids_lab = C.load_split("Test", work)
    found = feats[:, :, C.rungs.NAMES.index("face_found")]
    n_face = np.sum(found == 1.0, axis=1)
    y, p, g, ids = C.committed_test_predictions(work)
    n_face = C.align(ids_lab, n_face, list(ids))
    order = temporal_order(np.asarray(ids))
    y, p, g = np.asarray(y)[order], np.asarray(p, float)[order], np.asarray(g)[order]
    ok = (np.asarray(n_face)[order] >= 5)
    res = {}
    for name, mask in (("all_clips", None), ("scorable_deployment_rule", ok)):
        seqs = sequences(y, p, g, mask)
        base = float(y[mask].mean()) if mask is not None else float(y.mean())
        res[name] = {"windows": int(len(y)), "scorable": int(mask.sum()) if mask is not None
                     else int(len(y)), "participants": int(len(set(g))), "base_rate": base,
                     "rows": sweep(seqs, len(y), base)}
    return res


def dux() -> dict:
    pred_dir = REPO / "reports" / "dux_review" / "predictions"
    res = {}
    for stem in ("raw_interaction", "matched_serving_rule_interaction",
                 "transductive_interaction"):
        d = np.load(pred_dir / f"{stem}.npz", allow_pickle=False)
        y, p, g = d["y_true"], d["y_prob"].astype(float), d["groups"]
        o = np.lexsort((d["window_index"], g))          # time order within each session
        y, p, g = y[o], p[o], g[o]
        seqs = sequences(y, p, g)
        res[stem] = {"windows": int(len(y)), "sessions": int(len(set(g))),
                       "base_rate": float(y.mean()), "rows": sweep(seqs, len(y), float(y.mean()))}
    return res


VARIANTS = {                                  # (persistence, cooldown)
    "floor_only": (False, False),
    "floor_persistence": (True, False),
    "floor_cooldown": (False, True),
    "deployed_rule": (True, True),
}


def _ablation_channels(work: pathlib.Path) -> dict:
    """The three series of Fig. 2, as (sequences, n_windows, base_rate)."""
    import _common as C
    from geometry_gate_calibration import temporal_order

    feats, _, _, ids_lab = C.load_split("Test", work)
    found = feats[:, :, C.rungs.NAMES.index("face_found")]
    y, p, g, ids = C.committed_test_predictions(work)
    n_face = C.align(ids_lab, np.sum(found == 1.0, axis=1), list(ids))
    order = temporal_order(np.asarray(ids))
    y, p, g = np.asarray(y)[order], np.asarray(p, float)[order], np.asarray(g)[order]
    ok = np.asarray(n_face)[order] >= 5
    out = {"geometry_scorable": (sequences(y, p, g, ok), len(y), float(y[ok].mean()))}
    pred_dir = REPO / "reports" / "dux_review" / "predictions"
    for name, stem in (("interaction_raw", "raw_interaction"),
                       ("interaction_platform_like", "matched_serving_rule_interaction")):
        d = np.load(pred_dir / f"{stem}.npz", allow_pickle=False)
        yy, pp, gg = d["y_true"], d["y_prob"].astype(float), d["groups"]
        o = np.lexsort((d["window_index"], gg))
        out[name] = (sequences(yy[o], pp[o], gg[o]), len(yy), float(yy.mean()))
    return out


def ablation(work: pathlib.Path) -> dict:
    """Which of the gate's checks change precision and offer rate beyond the floor alone."""
    res = {}
    for name, (seqs, n, base) in _ablation_channels(work).items():
        block = {"windows": n, "base_rate": base, "variants": {}}
        for v, (pers, cool) in VARIANTS.items():
            block["variants"][v] = sweep(seqs, n, base, persistence=pers, cooldown=cool)
        # paired difference at 0.70: deployed rule minus floor only, same participant draws
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(N_BOOT):
            d = rng.integers(0, len(seqs), len(seqs))
            s = [seqs[i] for i in d]
            f_full, c_full = simulate(s, 0.70, True, True)
            f_base, c_base = simulate(s, 0.70, False, False)
            if f_full and f_base:
                diffs.append(c_full / f_full - c_base / f_base)
        block["paired_precision_diff_at_0_70"] = {
            "deployed_minus_floor_only": float(np.mean(diffs)) if diffs else None,
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
            if diffs else None, "n_draws_used": len(diffs)}
        res[name] = block
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="C:/engagenet")
    ap.add_argument("--ablation", action="store_true",
                    help="write ablation.json (floor only / + persistence / + cooldown / both)")
    a = ap.parse_args()
    if a.ablation:
        out = {"_note": "Gate ablation under the deployed rule; see ablation() docstring.",
               "variants": {k: {"persistence": v[0], "cooldown": v[1]} for k, v in VARIANTS.items()},
               "n_bootstrap": N_BOOT, "seed": SEED, "channels": ablation(pathlib.Path(a.work))}
        dest = REPO / "reports" / "gate_deployed_rule"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "ablation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        for ch, block in out["channels"].items():
            for v, rows in block["variants"].items():
                r = next(x for x in rows if x["floor"] == 0.70)
                pr = None if r["precision"] is None else round(r["precision"], 3)
                print(f"{ch:26s} {v:18s} @0.70 offers {r['offers']:4d} prec {pr} "
                      f"ci {r['ci95'] and [round(x, 3) for x in r['ci95']]} "
                      f"per_h {r['offers_per_hour']:.2f}")
            print(f"{ch:26s} paired diff (deployed - floor only) {block['paired_precision_diff_at_0_70']}")
        return 0
    out = {"_note": __doc__.strip().splitlines()[0],
           "rule": {"floor_applies_to": "current reading only",
                    "persistence": f"last {MIN_CONSECUTIVE} readings share the actionable state",
                    "cooldown_cycles": COOLDOWN, "window_s": WINDOW_S},
           "n_bootstrap": N_BOOT, "seed": SEED,
           "engagenet_geometry": engagenet(pathlib.Path(a.work)),
           "dux_interaction": dux()}
    dest = REPO / "reports" / "gate_deployed_rule"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for ch, block in (("geometry", out["engagenet_geometry"]), ("interaction", out["dux_interaction"])):
        for name, r in block.items():
            row = next(x for x in r["rows"] if x["floor"] == 0.70)
            print(f"{ch:11s} {name:32s} base {r['base_rate']:.3f}  @0.70 offers {row['offers']:4d} "
                  f"prec {row['precision'] if row['precision'] is None else round(row['precision'], 3)} "
                  f"ci {row['ci95'] and [round(v, 3) for v in row['ci95']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
