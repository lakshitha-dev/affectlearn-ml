"""Recompute DUX arm statistics from committed predictions, resampling participants.

Why this exists
---------------
Two corrections are applied here that the original run did not make, and both matter enough that
the numbers in the paper should come from this script rather than from the stored JSON.

The first is the resampling unit. `train_dux_confusion._bootstrap_auc` originally resampled
windows, which treats 1,419 correlated observations as 1,419 independent ones. That is exactly the
error this project criticises in the DAiSEE literature — where clip-level resampling was shown to
flip eight of eleven results — and it cannot stand in our own headline while that criticism does.
A participant's windows are drawn together here, so the interval answers "how much does this
depend on which people we sampled" rather than "which moments of these fixed people".

The second is average precision. The corpus is 18.1% positive, and AUC is a ranking measure that
can look tolerable on imbalanced data where precision does not. Reporting AP against the base rate
states the same comparison in the units a deployed detector would actually operate in.

Everything is computed from the committed `*_predictions.npz` files, so this is post-processing
rather than a re-run: no raw corpus, no refitting, seconds rather than minutes.

    python evaluation/dux_cluster_intervals.py --run reports/dux_v1_z
    python evaluation/dux_cluster_intervals.py --run reports/dux_v1_matched
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

ARMS = ("behavioural", "facial", "fused")


def cluster_bootstrap(y, prob, groups, seed: int = 42, n: int = 4000) -> tuple[float, float]:
    """95% interval for AUC, resampling participants with replacement."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ids}
    out = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        b = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(y[b])) > 1:
            out.append(roc_auc_score(y[b], prob[b]))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def window_bootstrap(y, prob, seed: int = 42, n: int = 4000) -> tuple[float, float]:
    """The naive interval, retained so the paper can report what the correction costs."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    out = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) > 1:
            out.append(roc_auc_score(y[b], prob[b]))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def permutation_p(y, prob, groups, seed: int = 42, n: int = 2000) -> float:
    """Within-participant label permutation, so identity alone cannot beat the null."""
    rng = np.random.default_rng(seed)
    observed = roc_auc_score(y, prob)
    ids = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ids}
    hits = 0
    for _ in range(n):
        yp = y.copy()
        for g in ids:
            i = idx_by[g]
            yp[i] = rng.permutation(y[i])
        if len(np.unique(yp)) > 1 and roc_auc_score(yp, prob) >= observed:
            hits += 1
    return (hits + 1) / (n + 1)


def score(path: Path, seed: int) -> dict:
    d = np.load(path, allow_pickle=True)
    y, prob, groups = d["y_true"], d["y_prob"][:, 1], d["groups"]
    pred = (prob >= 0.5).astype(np.int64)
    lo, hi = cluster_bootstrap(y, prob, groups, seed)
    wlo, whi = window_bootstrap(y, prob, seed)
    base = float(y.mean())
    return {
        "n": int(len(y)),
        "n_participants": int(len(np.unique(groups))),
        "positives": int(y.sum()),
        "base_rate": base,
        "auc": float(roc_auc_score(y, prob)),
        "auc_ci95_participant": [lo, hi],
        "auc_ci95_window": [wlo, whi],
        "ci_width_ratio": float((hi - lo) / (whi - wlo)) if whi > wlo else float("nan"),
        "ci_excludes_chance": bool(lo > 0.5),
        "average_precision": float(average_precision_score(y, prob)),
        "ap_over_base_rate": float(average_precision_score(y, prob) / base),
        "permutation_p_within_participant": permutation_p(y, prob, groups, seed),
        "kappa": float(cohen_kappa_score(y, pred)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="directory holding *_predictions.npz")
    ap.add_argument("--out", default=None, help="default: <run>/cluster_intervals.json")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    run = Path(a.run)
    results = {}
    for arm in ARMS:
        p = run / f"{arm}_predictions.npz"
        if p.exists():
            results[arm] = score(p, a.seed)

    if not results:
        raise SystemExit(f"no *_predictions.npz under {run}")

    print(f"  {run}")
    print(f"  {'arm':13} {'AUC':>7} {'participant CI':>17} {'window CI':>16} "
          f"{'AP':>6} {'x base':>7} {'perm p':>8}")
    for arm, r in results.items():
        lo, hi = r["auc_ci95_participant"]
        wl, wh = r["auc_ci95_window"]
        print(f"  {arm:13} {r['auc']:7.4f} [{lo:.3f}, {hi:.3f}]{'*' if r['ci_excludes_chance'] else ' '}"
              f"    [{wl:.3f}, {wh:.3f}] {r['average_precision']:6.3f} "
              f"{r['ap_over_base_rate']:6.2f}x {r['permutation_p_within_participant']:8.4f}")
    print(f"\n  * interval excludes chance | base rate "
          f"{next(iter(results.values()))['base_rate']:.4f}")

    out = Path(a.out) if a.out else run / "cluster_intervals.json"
    out.write_text(json.dumps({"seed": a.seed, "arms": results}, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
