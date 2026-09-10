"""Sensitivity of the DUX confusion result to window length and per-learner normalisation.

The headline number in `train_dux_confusion.py` uses a 30 s window and raw features, because that
is what the deployed platform computes - it is fixed in advance, not chosen here. This script asks
whether that choice is costing anything, and reports the whole grid rather than the best cell.

READ THIS BEFORE QUOTING ANY NUMBER FROM HERE. Eight configurations are evaluated on 46 positive
windows. The best cell in a grid that size is optimistically biased by construction: with 8
independent draws you expect the maximum to sit well above the true value even if every config is
identical. So the grid is a ROBUSTNESS CHECK - does the effect survive reasonable choices? - and
never a model-selection procedure. If a different cell beats 30 s, the honest report is "the result
is stable across window lengths" or "the result depends on window length", not "we achieve AUC X".

PER-LEARNER NORMALISATION is z-scoring each feature against that participant's OWN windows. It is
legitimate rather than leakage: it uses no labels, and at serve time the platform genuinely has the
learner's own session history to normalise against. It is stated separately because it changes what
the model sees - deviation from personal baseline rather than absolute value - which the affective
computing literature repeatedly finds matters more than the raw signal.

Run:  python dux_sensitivity.py
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
from external_datasets import load_dux_confusion  # noqa: E402
from feature_engineering import extract_features  # noqa: E402
from train_dux_confusion import (_bootstrap_auc, _permutation_p,  # noqa: E402
                                 leave_one_participant_out)

from sklearn.metrics import roc_auc_score  # noqa: E402

WINDOWS_S = [10, 20, 30, 60]


def per_participant_z(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Z-score every feature within each participant's own windows — TRANSDUCTIVE.

    Uses the participant's WHOLE session, including windows that occur after the one being
    normalised. That is fine for asking "does deviation-from-personal-baseline carry signal?", but
    it is NOT what the platform can do at serve time, because at window t the future does not exist
    yet. Any number produced with this function is therefore an UPPER BOUND. Use
    `per_participant_z_causal` for the deployable estimate and report both.

    std < 1e-9 -> divide by 1.0, leaving a constant channel at 0 rather than producing inf. A
    participant who never scrolled has a constant scroll column, and that must not become NaN.
    """
    out = np.empty_like(X, dtype=np.float64)
    for p in np.unique(groups):
        m = groups == p
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        out[m] = (X[m] - mu) / np.where(sd < 1e-9, 1.0, sd)
    return out


def per_participant_z_causal(X: np.ndarray, groups: np.ndarray, order: np.ndarray,
                             fit_mask: np.ndarray | None = None,
                             min_history: int = 4) -> np.ndarray:
    """Z-score each window against ONLY that participant's EARLIER windows.

    This is the deployable version of `per_participant_z`, and the difference between the two is a
    real effect size, not a technicality: the transductive form lets window t be centred using
    statistics that include windows t+1..T, which the running platform cannot see.

    Each window is normalised with an EXPANDING mean/std over the participant's preceding windows
    in `order` (their chronological window index). Until `min_history` windows exist, a per-
    participant estimate would be noise, so those windows fall back to POPULATION statistics
    computed over `fit_mask` only — the training folds. Passing the full mask would leak the test
    fold back in through the cold-start path, which is exactly the bug this function exists to
    avoid.

    This mirrors the platform's genuine cold-start problem (WESAD's baseline-prefix approach, and
    Ben-Shakhar's individual-responsivity correction, are the established precedents), so the number
    it produces is the one that should be quoted as achievable in deployment.
    """
    fit = np.ones(len(X), dtype=bool) if fit_mask is None else fit_mask
    pop_mu = X[fit].mean(axis=0)
    pop_sd = np.where(X[fit].std(axis=0) < 1e-9, 1.0, X[fit].std(axis=0))

    out = np.empty_like(X, dtype=np.float64)
    for p in np.unique(groups):
        idx = np.flatnonzero(groups == p)
        idx = idx[np.argsort(order[idx], kind="stable")]      # chronological within participant
        seen = X[idx]
        # Expanding stats EXCLUDING the current row: cumulative sums shifted by one.
        csum = np.cumsum(seen, axis=0)
        csq = np.cumsum(seen ** 2, axis=0)
        for k, row_i in enumerate(idx):
            if k < min_history:
                out[row_i] = (X[row_i] - pop_mu) / pop_sd
                continue
            n = k                                             # rows strictly before this one
            mu = csum[k - 1] / n
            var = np.maximum(csq[k - 1] / n - mu ** 2, 0.0)
            sd = np.where(var < 1e-18, 1.0, np.sqrt(var))
            out[row_i] = (X[row_i] - mu) / sd
    return out


def build(dux_dir: str, window_s: int, threshold: float):
    window_ms = window_s * 1000
    win = load_dux_confusion(dux_dir, threshold=threshold, window_ms=window_ms)
    if not win:
        raise SystemExit(f"No DUX windows under {dux_dir}")
    # bin_length_ms stays 1000 so a longer window means MORE bins, not coarser ones - the temporal
    # resolution of the sequence is held constant and only its extent varies.
    Xb = np.stack([extract_features(w["events"], 0, window_length_ms=window_ms,
                                    bin_length_ms=1_000) for w in win])
    y = np.array([w["label"] for w in win], dtype=np.int64)
    groups = np.array([w["participant"] for w in win])
    return aggregate(Xb), y, groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dux", default="../../data/external/dux")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print(f"{'window':>7s} {'norm':>12s} {'n':>5s} {'pos':>4s} {'AUC':>6s} "
          f"{'95% CI':>16s} {'perm p':>8s}")
    print("-" * 66)
    rows = []
    for ws in WINDOWS_S:
        X, y, groups = build(a.dux, ws, a.threshold)
        for norm in ("raw", "per-learner z"):
            Xn = per_participant_z(X, groups) if norm != "raw" else X
            prob = leave_one_participant_out(Xn, y, groups, a.seed)
            keep = ~np.isnan(prob)
            yk, pk, gk = y[keep], prob[keep], groups[keep]
            if len(np.unique(yk)) < 2:
                print(f"{ws:6d}s {norm:>12s}  single-class pool, skipped")
                continue
            auc = roc_auc_score(yk, pk)
            # Cluster bootstrap over PARTICIPANTS. This call previously omitted `gk` and so
            # resampled windows, which understated every interval in the grid; the signature
            # changed when `train_dux_confusion` moved to the cluster form and this call site
            # was not updated with it.
            lo, hi = _bootstrap_auc(yk, pk, gk, a.seed)
            p = _permutation_p(yk, pk, gk, a.seed, n=1000)
            star = " *" if p < 0.05 else ""
            print(f"{ws:6d}s {norm:>12s} {len(yk):5d} {int(yk.sum()):4d} {auc:6.3f} "
                  f"[{lo:.3f}, {hi:.3f}] {p:8.4f}{star}")
            rows.append({"window_s": ws, "norm": norm, "n": int(len(yk)),
                         "positives": int(yk.sum()), "auc": float(auc),
                         "auc_ci95": [lo, hi], "permutation_p": float(p)})

    print("\n* = beats the within-participant permutation null at p < 0.05")
    best = max(rows, key=lambda r: r["auc"])
    ref = next((r for r in rows if r["window_s"] == 30 and r["norm"] == "raw"), None)
    print(f"\nhighest cell: {best['window_s']}s / {best['norm']} -> AUC {best['auc']:.3f}")
    if ref:
        print(f"deployed config (30s / raw): AUC {ref['auc']:.3f}")
        print(f"  spread across the grid: {best['auc'] - min(r['auc'] for r in rows):+.3f}")
        if best["auc_ci95"][0] <= ref["auc"]:
            print("  the best cell's CI contains the deployed cell's point estimate, so the grid")
            print("  shows STABILITY, not an improvement. Report 30s and cite this as robustness.")
    n_sig = sum(1 for r in rows if r["permutation_p"] < 0.05)
    print(f"\n{n_sig}/{len(rows)} configurations beat the null. A result that survives most of the")
    print("grid is far harder to dismiss as a lucky window length than a single tuned number.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"grid": rows, "windows_s": WINDOWS_S}, indent=2),
                               encoding="utf-8")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
