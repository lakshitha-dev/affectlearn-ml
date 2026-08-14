"""Significance tests with effect sizes — model comparison and pilot outcomes.

Two distinct jobs, deliberately in one place so the right test is easy to find:

1. MODEL COMPARISON on the same test set (e.g. Bi-LSTM vs GBDT, or pretrained vs scratch).
   The correct test is McNemar's, not two independent-sample tests: the two models are scored on
   the SAME items, so their errors are paired and an unpaired test overstates significance.

2. PILOT OUTCOMES (RQ4): pre/post learning gain within a learner (paired), and adaptive vs
   control between learners (independent).

EVERY test reports an effect size, and every function returns the n it used. With a pilot of
n = 20-30 the honest framing is ESTIMATION, not confirmation: a non-significant result is a
legitimate, reportable finding, and a p-value alone from that sample is close to uninformative.
This is why `interpret()` leads with the effect size and its confidence interval.

Run:
    python statistical_tests.py --compare a.npz b.npz          # McNemar, paired
    python statistical_tests.py --paired pre.csv post.csv      # learning gain
    python statistical_tests.py --groups adaptive.csv control.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from predictions import Predictions  # noqa: E402


# ----------------------------------------------------------------- model comparison

def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray, exact: bool | None = None) -> dict:
    """McNemar's test on two models' per-item correctness over the SAME items.

    Only the discordant pairs carry information: b01 = A wrong & B right, b10 = A right & B wrong.
    Uses the exact binomial test when the discordant count is small (< 25), where the chi-square
    approximation is unreliable — which is the common case at pilot scale.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} and {len(b)}")

    both = int(np.sum(a & b))
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    neither = int(np.sum(~a & ~b))
    n_disc = only_a + only_b

    if exact is None:
        exact = n_disc < 25

    if n_disc == 0:
        p, stat, method = 1.0, 0.0, "no discordant pairs"
    elif exact:
        p = float(stats.binomtest(only_b, n_disc, 0.5).pvalue)
        stat, method = float(min(only_a, only_b)), "exact binomial"
    else:
        stat = (abs(only_a - only_b) - 1) ** 2 / n_disc      # Yates-corrected
        p = float(stats.chi2.sf(stat, df=1))
        stat, method = float(stat), "chi-square (Yates)"

    # Odds ratio over discordant pairs = how many times more often B fixes A than breaks it.
    odds = float("inf") if only_a == 0 and only_b > 0 else (
        0.0 if only_b == 0 and only_a > 0 else (only_b / only_a if only_a else float("nan"))
    )
    return {
        "test": "mcnemar", "method": method, "n": int(len(a)),
        "both_correct": both, "only_a_correct": only_a, "only_b_correct": only_b,
        "neither_correct": neither, "n_discordant": n_disc,
        "statistic": stat, "p_value": p,
        "accuracy_a": float(a.mean()), "accuracy_b": float(b.mean()),
        "accuracy_delta": float(b.mean() - a.mean()),
        "odds_ratio_b_over_a": odds,
    }


def compare_predictions(a: Predictions, b: Predictions) -> dict:
    """McNemar between two prediction records over the same items, in the same order."""
    if len(a) != len(b):
        raise ValueError(f"records differ in length ({len(a)} vs {len(b)}) — not the same test set")
    if not np.array_equal(a.y_true, b.y_true):
        raise ValueError(
            "y_true differs between the two records — they are not the same items in the same "
            "order, so the pairing is invalid and McNemar would be meaningless"
        )
    return mcnemar(a.y_true == a.y_pred, b.y_true == b.y_pred)


# ----------------------------------------------------------------- effect sizes

def cohens_d_paired(pre: np.ndarray, post: np.ndarray) -> float:
    """Cohen's d_z for paired samples (mean difference / sd of differences)."""
    d = np.asarray(post, float) - np.asarray(pre, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges' g — Cohen's d with the small-sample correction. Preferred at n = 20-30."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    d = (x.mean() - y.mean()) / pooled
    return float(d * (1 - 3 / (4 * (nx + ny) - 9)))


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation — the effect size that belongs with Mann-Whitney U."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x) or not len(y):
        return 0.0
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def bootstrap_ci(x, y=None, stat_fn=None, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI. With `y`, resamples the difference in means (paired if equal n)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if stat_fn is None:
        stat_fn = (lambda a: a.mean()) if y is None else None
    vals = []
    if y is None:
        for _ in range(n_boot):
            vals.append(stat_fn(rng.choice(x, len(x), replace=True)))
    else:
        y = np.asarray(y, float)
        paired = len(x) == len(y)
        for _ in range(n_boot):
            if paired:
                idx = rng.integers(0, len(x), len(x))
                vals.append(y[idx].mean() - x[idx].mean())
            else:
                vals.append(rng.choice(y, len(y), True).mean() - rng.choice(x, len(x), True).mean())
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ----------------------------------------------------------------- pilot outcomes

def paired_outcome(pre, post, label: str = "learning gain") -> dict:
    """Pre/post within the same learners. Paired t-test + Wilcoxon + d_z + bootstrap CI."""
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    if len(pre) != len(post):
        raise ValueError(f"paired test needs equal lengths, got {len(pre)} and {len(post)}")
    diff = post - pre
    out = {
        "test": "paired", "measure": label, "n": int(len(pre)),
        "mean_pre": float(pre.mean()), "mean_post": float(post.mean()),
        "mean_difference": float(diff.mean()),
        "cohens_dz": cohens_d_paired(pre, post),
    }
    if len(pre) >= 2:
        t = stats.ttest_rel(post, pre)
        out["t_statistic"], out["p_value_ttest"] = float(t.statistic), float(t.pvalue)
        # Wilcoxon needs at least one non-zero difference.
        if np.any(diff != 0):
            w = stats.wilcoxon(post, pre)
            out["wilcoxon_statistic"], out["p_value_wilcoxon"] = float(w.statistic), float(w.pvalue)
        lo, hi = bootstrap_ci(pre, post)
        out["difference_ci95"] = [lo, hi]
    return out


def independent_outcome(group_a, group_b, names=("adaptive", "control"),
                        label: str = "outcome") -> dict:
    """Between-group comparison (the A/B arm). Mann-Whitney U + Welch t + Hedges' g."""
    a, b = np.asarray(group_a, float), np.asarray(group_b, float)
    out = {
        "test": "independent", "measure": label,
        "n_a": int(len(a)), "n_b": int(len(b)), "names": list(names),
        "mean_a": float(a.mean()) if len(a) else None,
        "mean_b": float(b.mean()) if len(b) else None,
        "hedges_g": hedges_g(a, b),
        "rank_biserial": rank_biserial(a, b),
    }
    if len(a) >= 2 and len(b) >= 2:
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        out["u_statistic"], out["p_value_mannwhitney"] = float(u.statistic), float(u.pvalue)
        t = stats.ttest_ind(a, b, equal_var=False)     # Welch: unequal variances by default
        out["t_statistic"], out["p_value_welch"] = float(t.statistic), float(t.pvalue)
        lo, hi = bootstrap_ci(b, a)
        out["difference_ci95"] = [lo, hi]
    return out


# ----------------------------------------------------------------- reporting

def _magnitude(effect: float) -> str:
    e = abs(effect)
    return "negligible" if e < 0.2 else "small" if e < 0.5 else "moderate" if e < 0.8 else "large"


def interpret(result: dict) -> str:
    """Effect size first, p-value second — the honest order at pilot sample sizes."""
    lines = []
    if result["test"] == "mcnemar":
        lines += [
            f"McNemar ({result['method']}), n = {result['n']}",
            f"  accuracy: A {result['accuracy_a']:.4f} -> B {result['accuracy_b']:.4f} "
            f"(delta {result['accuracy_delta']:+.4f})",
            f"  discordant pairs: B fixed {result['only_b_correct']}, B broke "
            f"{result['only_a_correct']}  (total {result['n_discordant']})",
            f"  p = {result['p_value']:.4f}",
        ]
        if result["n_discordant"] < 10:
            lines.append("  NOTE: very few discordant pairs — this test cannot resolve a "
                         "difference either way. Do not read the p-value as evidence of parity.")
    else:
        eff = result.get("cohens_dz", result.get("hedges_g", 0.0))
        name = "Cohen's d_z" if "cohens_dz" in result else "Hedges' g"
        lines.append(f"{result['measure']} ({result['test']})")
        if "difference_ci95" in result:
            lo, hi = result["difference_ci95"]
            lines.append(f"  difference: {result.get('mean_difference', (result.get('mean_a') or 0) - (result.get('mean_b') or 0)):+.4f}"
                         f"   95% CI [{lo:+.4f}, {hi:+.4f}]")
        lines.append(f"  {name} = {eff:+.3f} ({_magnitude(eff)})")
        for k in ("p_value_ttest", "p_value_wilcoxon", "p_value_mannwhitney", "p_value_welch"):
            if k in result:
                lines.append(f"  {k:22s} {result[k]:.4f}")
        n = result.get("n") or min(result.get("n_a", 0), result.get("n_b", 0))
        if n and n < 40:
            lines.append(f"  NOTE: n = {n}. Report this as ESTIMATION, not confirmation — the "
                         "interval above is the finding, not the p-value.")
    return "\n".join(lines)


def _read_column(path: str) -> np.ndarray:
    """One numeric value per line, or a single-column CSV with an optional header."""
    text = Path(path).read_text(encoding="utf-8").strip().splitlines()
    vals = []
    for line in text:
        tok = line.split(",")[0].strip()
        try:
            vals.append(float(tok))
        except ValueError:
            continue        # header or blank
    return np.array(vals, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="two prediction records over the same test set -> McNemar")
    ap.add_argument("--paired", nargs=2, metavar=("PRE", "POST"))
    ap.add_argument("--groups", nargs=2, metavar=("A", "B"))
    ap.add_argument("--label", default="outcome")
    ap.add_argument("--out", default=None, help="also write the result as JSON here")
    a = ap.parse_args()

    if a.compare:
        result = compare_predictions(Predictions.load(a.compare[0]),
                                    Predictions.load(a.compare[1]))
    elif a.paired:
        result = paired_outcome(_read_column(a.paired[0]), _read_column(a.paired[1]), a.label)
    elif a.groups:
        result = independent_outcome(_read_column(a.groups[0]), _read_column(a.groups[1]),
                                     label=a.label)
    else:
        ap.error("pick one of --compare / --paired / --groups")

    print(interpret(result))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
