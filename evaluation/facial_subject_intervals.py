"""Attach subject-level confidence intervals to every DAiSEE facial result.

WHY THIS EXISTS
---------------
Every DUX behavioural figure in this project carries three forms of inference: a bootstrap 95%
interval, a within-participant permutation test, and a stated majority baseline. **No DAiSEE
facial figure carries any of them.** Table 4.6 compares confusion (AUC 0.6414) against frustration
(0.5899) and boredom (0.5608) as bare point estimates, and §4.5.2 leans on that comparison to
argue the pipeline responds to a specific construct rather than to anything handed to it.

That asymmetry is the paper's weakest point, and it is closable without a GPU: the per-clip
probability dumps for seven facial checkpoints already exist, each with `y_true`, `y_prob` and
DAiSEE clip ids.

WHY SUBJECT-LEVEL, NOT CLIP-LEVEL
---------------------------------
The DAiSEE test split holds 1,638 clips from only **19 subjects**, a number that appears nowhere in
the thesis. Clips of one person share a face, a camera and a session, so a clip-level bootstrap
treats correlated observations as independent and reports an interval far tighter than the data
supports. On the engagement HIGH cut that difference decides the conclusion: the clip-level
interval excludes chance while the subject-level interval contains it.

So intervals here resample **subjects**, and the script also reports the clip-level interval beside
it so the gap is visible rather than asserted. DAiSEE encodes the subject in the first six digits
of the ClipID (`5000441001` -> subject `500044`), verified against the label CSVs.

WHAT IT ALSO REPORTS
--------------------
* the majority baseline beside every accuracy, per this project's own reporting rule
* a **permutation test that shuffles labels within subject**, so the null preserves each person's
  positive rate — a model that had learned only who the subject was could beat a naive null while
  detecting nothing
* a **leave-one-subject-out sensitivity**, because one subject supplies 25 of the 85 disengaged
  clips on the engagement HIGH cut, and a headline that rests on one face should say so

Usage:
    python evaluation/facial_subject_intervals.py \
        --predictions-dir "G:/My Drive/affectlearn-ml/reports/facial_confusion" \
        --out reports/facial_intervals/facial_subject_intervals.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

# Which binary view to score for each 4-level checkpoint. ANY = {1,2,3} vs {0} asks "present at
# all"; HIGH = {2,3} vs {0,1} asks "present at intensity". The cut must always be named: on
# engagement, ANY leaves four negatives in the whole split and HIGH leaves 85.
CUTS = {"any": (1, 2, 3), "high": (2, 3)}


def subject_of(clip_id) -> str:
    return str(clip_id)[:6]


def bootstrap_auc(y, p, groups, n_boot=4000, seed=0):
    """Percentile bootstrap over GROUPS. Pass one group per row for the clip-level variant."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue                      # a resample with one class has no AUC
        vals.append(roc_auc_score(y[idx], p[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def permutation_p(y, p, subjects, n_perm=2000, seed=0):
    """Shuffle labels WITHIN each subject, so the null keeps each person's positive rate.

    Mirrors the within-participant test used for the DUX arms, for the same reason: appearance is
    strongly identifying, and a naive shuffle would let a model that had learned identity alone
    look significant.
    """
    rng = np.random.default_rng(seed)
    observed = roc_auc_score(y, p)
    idx_by = {s: np.flatnonzero(subjects == s) for s in np.unique(subjects)}
    hits = 0
    for _ in range(n_perm):
        yp = y.copy()
        for idx in idx_by.values():
            yp[idx] = rng.permutation(yp[idx])
        if len(np.unique(yp)) > 1 and roc_auc_score(yp, p) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)      # add-one, so p is never reported as exactly zero


def score(y, p, subjects, label, n_boot, n_perm):
    pred = (p >= 0.5).astype(int)
    counts = np.bincount(y, minlength=2)
    baseline = float(counts.max() / len(y))
    acc = float((pred == y).mean())
    auc = float(roc_auc_score(y, p))

    lo_s, hi_s = bootstrap_auc(y, p, subjects, n_boot=n_boot, seed=0)
    clip_groups = np.arange(len(y)).astype(str)
    lo_c, hi_c = bootstrap_auc(y, p, clip_groups, n_boot=n_boot, seed=0)
    pval = permutation_p(y, p, subjects, n_perm=n_perm)

    # the subject carrying the most positives, and what happens without them
    pos_by = {s: int(((subjects == s) & (y == 1)).sum()) for s in np.unique(subjects)}
    big = max(pos_by, key=pos_by.get)
    keep = subjects != big
    auc_wo = (float(roc_auc_score(y[keep], p[keep]))
              if len(np.unique(y[keep])) > 1 else float("nan"))

    m = {
        "label": label, "n": int(len(y)), "positives": int(counts[1]),
        "subjects": int(len(np.unique(subjects))),
        "subjects_with_positives": int(sum(1 for v in pos_by.values() if v)),
        "auc": auc,
        "auc_ci_subject": [lo_s, hi_s], "auc_ci_clip": [lo_c, hi_c],
        "ci_width_subject": hi_s - lo_s, "ci_width_clip": hi_c - lo_c,
        "subject_ci_excludes_chance": bool(lo_s > 0.5),
        "clip_ci_excludes_chance": bool(lo_c > 0.5),
        "permutation_p_within_subject": pval,
        "accuracy": acc, "majority_baseline": baseline,
        "beats_baseline": bool(acc > baseline),
        "kappa": float(cohen_kappa_score(y, pred)),
        "positive_recall": float((pred[y == 1] == 1).mean()) if counts[1] else None,
        "largest_positive_subject": big,
        "largest_subject_positive_share": pos_by[big] / max(int(counts[1]), 1),
        "auc_without_largest_subject": auc_wo,
    }
    print(f"  {label}")
    print(f"    n={m['n']} pos={m['positives']} subjects={m['subjects']} "
          f"({m['subjects_with_positives']} carry a positive)")
    print(f"    AUC {auc:.4f}   subject CI [{lo_s:.4f}, {hi_s:.4f}] "
          f"{'excludes' if lo_s > 0.5 else '** CONTAINS **'} chance")
    print(f"                 clip CI    [{lo_c:.4f}, {hi_c:.4f}] "
          f"({m['ci_width_subject']/max(m['ci_width_clip'],1e-9):.2f}x wider at subject level)")
    print(f"    within-subject permutation p = {pval:.4f}")
    print(f"    accuracy {acc:.4f} vs baseline {baseline:.4f}"
          f"{'' if m['beats_baseline'] else '   ** BELOW BASELINE **'}")
    print(f"    largest subject {big} holds {m['largest_subject_positive_share']:.0%} of positives"
          f" -> without it AUC {auc_wo:.4f}")
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()

    pdir = Path(args.predictions_dir)
    files = sorted(pdir.glob("*_test_predictions.npz"))
    if not files:
        raise SystemExit(f"no *_test_predictions.npz under {pdir}")

    results = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        y_raw, prob = d["y_true"], d["y_prob"]
        subjects = np.array([subject_of(c) for c in d["ids"]])
        stem = f.stem.replace("_test_predictions", "")
        n_classes = prob.shape[1] if prob.ndim == 2 else 2

        if n_classes == 2:
            # already binary: score as stored, and name the cut from the filename
            cut = "any" if "anycut" in stem else "as-stored"
            p = prob[:, 1] if prob.ndim == 2 else prob
            results.append(score(y_raw.astype(int), p, subjects, f"{stem} [{cut}]",
                                 args.n_boot, args.n_perm))
        else:
            # 4-level: collapse by SUMMING probabilities, which is the calibrated way to obtain a
            # group probability -- mapping the argmax discards evidence spread across siblings.
            for cut, levels in CUTS.items():
                y = np.isin(y_raw, levels).astype(int)
                p = prob[:, list(levels)].sum(axis=1)
                # score the RARE class: a recall of 0.99 on a 95%-prevalence class says nothing
                if y.mean() > 0.5:
                    y, p = 1 - y, 1.0 - p
                results.append(score(y, p, subjects, f"{stem} [{cut} cut, rare class positive]",
                                     args.n_boot, args.n_perm))
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"predictions_dir": str(pdir),
                               "n_boot": args.n_boot, "n_perm": args.n_perm,
                               "results": results}, indent=2))
    print(f"  wrote {out}  ({len(results)} scored views)")

    wider = [r for r in results
             if r["clip_ci_excludes_chance"] and not r["subject_ci_excludes_chance"]]
    if wider:
        print("\n  RESULTS THAT LOOK SIGNIFICANT AT CLIP LEVEL BUT NOT AT SUBJECT LEVEL:")
        for r in wider:
            print(f"    {r['label']}: clip [{r['auc_ci_clip'][0]:.4f}, {r['auc_ci_clip'][1]:.4f}] "
                  f"vs subject [{r['auc_ci_subject'][0]:.4f}, {r['auc_ci_subject'][1]:.4f}]")
        print("  Reporting these at clip level would be a false positive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
