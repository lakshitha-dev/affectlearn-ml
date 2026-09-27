"""How much of the per-learner normalisation gain survives being made causal?

`per_participant_z` gave +0.079 AUC on DUX, and that number is an UPPER BOUND: it centres each
window using statistics drawn from the participant's whole session, including windows that come
AFTER it. The deployed platform cannot do that — at window t it has only windows 1..t-1.

So the +0.079 splits into two parts, and only one of them is real:
  * genuine personalisation   — knowing this learner's own baseline helps
  * temporal leakage          — knowing this learner's FUTURE helps

This script measures the split by scoring three representations on identical folds:
  raw           no normalisation
  transductive  per_participant_z          (whole session — the optimistic bound)
  causal        per_participant_z_causal   (expanding past only — the deployable number)

The causal arm's cold-start rows fall back to POPULATION statistics fitted on the TRAINING FOLD
ONLY, recomputed inside every fold. Fitting them once over all data would quietly reintroduce the
leak this script exists to measure.

Whatever the causal number turns out to be, that is the one the thesis should quote for a deployed
system. The transductive number is reportable only as an ablation, clearly labelled.

Run:  python dux_normalisation_audit.py
      python dux_normalisation_audit.py --v1-only --out ../../reports/dux_confusion/normalisation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2] / "evaluation"))
from aggregate_features import aggregate  # noqa: E402
from dux_sensitivity import per_participant_z, per_participant_z_causal  # noqa: E402
from external_datasets import load_dux_confusion  # noqa: E402
from feature_engineering import extract_features  # noqa: E402
from train_dux_confusion import _bootstrap_auc, _gbdt, _permutation_p  # noqa: E402

from sklearn.metrics import roc_auc_score  # noqa: E402


def lopo_causal(X, y, groups, order, seed: int):
    """LOPO where the causal normalisation is refitted inside each fold.

    The population fallback used by cold-start windows must be fitted on the TRAINING participants
    only. That makes normalisation part of the fold, which is the whole point — a transform fitted
    once outside the loop is a leak however innocent it looks.
    """
    prob = np.full(len(y), np.nan)
    for p in np.unique(groups):
        te = groups == p
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        Xn = per_participant_z_causal(X, groups, order, fit_mask=tr)
        prob[te] = _gbdt(seed).fit(Xn[tr], y[tr]).predict_proba(Xn[te])[:, 1]
    return prob


def lopo_fixed(X, y, groups, seed: int):
    """LOPO on an already-transformed matrix (raw or transductive)."""
    prob = np.full(len(y), np.nan)
    for p in np.unique(groups):
        te = groups == p
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        prob[te] = _gbdt(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return prob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dux", default="../../data/external/dux")
    ap.add_argument("--window-s", type=int, default=30)
    ap.add_argument("--min-history", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resamples", type=int, default=2000,
                    help="bootstrap and permutation draws per arm. The quantity of interest here is "
                         "the AUC DIFFERENCE between representations, which is a point comparison "
                         "on shared folds and does not need 2000 draws; 500 is enough to size the "
                         "interval. Lower it if the run is slow.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    win = load_dux_confusion(a.dux, window_ms=a.window_s * 1000)
    if not win:
        raise SystemExit(f"No DUX windows under {a.dux}")
    Xb = np.stack([extract_features(w["events"], 0, window_length_ms=a.window_s * 1000,
                                   bin_length_ms=1_000) for w in win])
    y = np.array([w["label"] for w in win], dtype=np.int64)
    groups = np.array([w["participant"] for w in win])
    order = np.array([w["window_index"] for w in win])
    A = aggregate(Xb)

    print(f"windows {len(y)}  participants {len(set(groups.tolist()))}  "
          f"confused {int(y.sum())} ({100 * y.mean():.1f}%)  window {a.window_s}s")
    print(f"cold-start fallback: first {a.min_history} windows per participant use TRAINING-FOLD "
          "population stats\n")

    arms = {
        "raw": lambda: lopo_fixed(A, y, groups, a.seed),
        "transductive (whole session)": lambda: lopo_fixed(per_participant_z(A, groups), y,
                                                           groups, a.seed),
        "causal (expanding past)": lambda: lopo_causal(A, y, groups, order, a.seed),
    }
    res = {}
    print(f"  {'representation':30s} {'AUC':>6s} {'95% CI':>16s} {'perm p':>8s}")
    print("  " + "-" * 64)
    for name, fn in arms.items():
        prob = fn()
        keep = ~np.isnan(prob)
        yk, pk, gk = y[keep], prob[keep], groups[keep]
        auc = roc_auc_score(yk, pk)
        lo, hi = _bootstrap_auc(yk, pk, a.seed, n=a.resamples)
        p = _permutation_p(yk, pk, gk, a.seed, n=a.resamples)
        res[name] = {"auc": float(auc), "auc_ci95": [lo, hi], "permutation_p": float(p)}
        print(f"  {name:30s} {auc:6.3f} [{lo:.3f}, {hi:.3f}] {p:8.4f}"
              f"{' *' if p < 0.05 else '  '}")

    raw = res["raw"]["auc"]
    trans = res["transductive (whole session)"]["auc"]
    caus = res["causal (expanding past)"]["auc"]
    print(f"\n  total transductive gain over raw : {trans - raw:+.3f}")
    print(f"  of which SURVIVES being causal   : {caus - raw:+.3f}   <- the deployable gain")
    print(f"  of which was temporal leakage    : {trans - caus:+.3f}")
    if caus <= raw:
        print("\n  The gain does NOT survive. Personalisation as implemented was leakage; the")
        print("  honest headline is the RAW number and the z-scored figure must not be quoted")
        print("  as achievable in deployment.")
    elif trans - caus > caus - raw:
        print("\n  MOST of the gain was leakage. Quote the causal number; report the transductive")
        print("  one only as an upper bound with the leak named explicitly.")
    else:
        print("\n  The gain largely survives. Personalisation is real and deployable — quote the")
        print("  causal number as the headline and cite the WESAD baseline-prefix precedent.")
    print("\n  Note: all three arms share the same folds and the same classifier, so the only")
    print("  thing varying is the representation.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"window_s": a.window_s, "min_history": a.min_history, "n": int(len(y)),
             "n_participants": int(len(set(groups.tolist()))), "positives": int(y.sum()),
             "arms": res, "deployable_gain": caus - raw, "leakage": trans - caus},
            indent=2), encoding="utf-8")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
