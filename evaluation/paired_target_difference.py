"""Paired participant-bootstrap difference in AUC between two targets scored on the same clips.

WHY THIS EXISTS
---------------
Section 4.5.5 compares the four facial targets under subject-grouped cross-validation and says two
of them are statistically indistinguishable. An unpaired reading of that claim would compare two
intervals and note that they overlap, which is the wrong test: the two targets are scored on the
SAME 7,919 clips from the SAME 101 participants, so their errors are paired, and the interval on
the DIFFERENCE is narrower than the overlap of the two marginal intervals suggests.

This script computes that difference directly. Both arms are resampled together — the same
bootstrap draw of participants is applied to both targets before either AUC is recomputed — so the
correlation between the two targets is preserved in every replicate and the resulting interval is
on the paired difference rather than on two independent estimates.

RESAMPLING UNIT is the participant, never the clip, for the reason Section 4.5.3 sets out: clips of
one person share a face, a camera and a recording session, and treating them as independent
observations understates the interval by roughly half.

Run:
    python paired_target_difference.py \
        --a ../.features/engagement_high_cv.npz     --a-label "Engagement, HIGH cut" \
        --b ../.features/confusion_any_cv_cv.npz    --b-label "Confusion, ANY cut" \
        --out ../reports/facial_grouped_cv/paired_differences.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load(path: str) -> dict:
    """Load a prediction dump, keyed so the two arms can be aligned clip-for-clip."""
    d = np.load(path, allow_pickle=True)
    missing = {"y_true", "y_prob", "subject", "clip_id"} - set(d.files)
    if missing:
        raise SystemExit(f"{path} is missing {sorted(missing)}; cannot pair on it")
    return {
        "y": d["y_true"].astype(np.int64),
        "p": d["y_prob"].astype(np.float64),
        "subject": d["subject"].astype(str),
        "clip_id": d["clip_id"].astype(str),
    }


def align(a: dict, b: dict) -> tuple[dict, dict]:
    """Reorder b onto a's clip order, refusing rather than guessing if they do not correspond.

    A silent mismatch here would produce a paired statistic over unpaired rows, which is worse than
    no statistic at all — it would look like a tighter interval while measuring nothing.
    """
    if len(a["clip_id"]) != len(b["clip_id"]):
        raise SystemExit(f"row counts differ: {len(a['clip_id'])} vs {len(b['clip_id'])}")
    if set(a["clip_id"]) != set(b["clip_id"]):
        raise SystemExit("clip id sets differ; the two dumps are not over the same clips")
    order = {c: i for i, c in enumerate(b["clip_id"])}
    idx = np.array([order[c] for c in a["clip_id"]])
    b = {k: v[idx] for k, v in b.items()}
    if not np.array_equal(a["subject"], b["subject"]):
        raise SystemExit("subject ids disagree after alignment; refusing to pair")
    return a, b


def paired_bootstrap(a: dict, b: dict, n_boot: int, seed: int) -> dict:
    """Resample participants ONCE per replicate and score both targets on that same draw."""
    rng = np.random.default_rng(seed)
    subjects = a["subject"]
    uniq = np.unique(subjects)
    idx_by = {s: np.flatnonzero(subjects == s) for s in uniq}

    diffs, a_vals, b_vals, skipped = [], [], [], 0
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[s] for s in pick])
        # Either target can degenerate to one class in a resample; both must be scorable or the
        # replicate carries no difference and is dropped from both arms together.
        if len(np.unique(a["y"][idx])) < 2 or len(np.unique(b["y"][idx])) < 2:
            skipped += 1
            continue
        av = roc_auc_score(a["y"][idx], a["p"][idx])
        bv = roc_auc_score(b["y"][idx], b["p"][idx])
        a_vals.append(av)
        b_vals.append(bv)
        diffs.append(av - bv)

    if not diffs:
        raise SystemExit("every bootstrap replicate degenerated; nothing to report")

    diffs = np.asarray(diffs)
    auc_a = float(roc_auc_score(a["y"], a["p"]))
    auc_b = float(roc_auc_score(b["y"], b["p"]))
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {
        "n": int(len(a["y"])),
        "n_participants": int(len(uniq)),
        "positives_a": int(a["y"].sum()),
        "positives_b": int(b["y"].sum()),
        "auc_a": auc_a,
        "auc_b": auc_b,
        "difference": auc_a - auc_b,
        "difference_ci95_participant": [lo, hi],
        # A paired interval containing zero is the whole point of the comparison: it says the two
        # targets are not separated by this evaluation, which is a stronger statement than
        # "their marginal intervals overlap".
        "ci_contains_zero": bool(lo <= 0.0 <= hi),
        "bootstrap_mean_difference": float(diffs.mean()),
        "n_boot": n_boot,
        "n_boot_used": int(len(diffs)),
        "n_boot_skipped": skipped,
        "seed": seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="prediction dump for the first target")
    ap.add_argument("--b", required=True, help="prediction dump for the second target")
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a, b = align(load(args.a), load(args.b))
    res = paired_bootstrap(a, b, args.n_boot, args.seed)
    res["target_a"] = args.a_label
    res["target_b"] = args.b_label
    res["source_a"] = args.a
    res["source_b"] = args.b

    print(f"{args.a_label}  AUC {res['auc_a']:.4f}  ({res['positives_a']} positives)")
    print(f"{args.b_label}  AUC {res['auc_b']:.4f}  ({res['positives_b']} positives)")
    print(f"n = {res['n']} clips from {res['n_participants']} participants, paired")
    lo, hi = res["difference_ci95_participant"]
    print(f"\npaired difference {res['difference']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("interval contains zero: " + ("yes — indistinguishable on this evaluation"
                                        if res["ci_contains_zero"] else "NO — separated"))
    if res["n_boot_skipped"]:
        print(f"({res['n_boot_skipped']} of {args.n_boot} replicates dropped as single-class)")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
