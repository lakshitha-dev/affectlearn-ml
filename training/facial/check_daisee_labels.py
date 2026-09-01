"""Is a binary CONFUSION split viable on DAiSEE? Decide before spending any GPU time.

WHY THIS RUNS FIRST
-------------------
Switching Model A from Engagement to Confusion is a one-line config change, but it is only worth
doing if DAiSEE actually contains enough confused clips. It might not: DAiSEE's Engagement labels
are famously concentrated in the two high levels, and confusion is plausibly rarer still.

The failure mode this guards against is already measured in this project. Collapsing a 4-level
DAiSEE-shaped target to binary moved accuracy 0.724 -> 0.934 and weighted-F1 0.702 -> 0.908 while
Cohen's kappa COLLAPSED 0.462 -> 0.025 and minority recall was 0.020 (2 of 102). A 93% headline that
detects nothing is the single easiest way to fail a viva, and it looks like success right up to the
moment someone asks for per-class recall.

So this script answers one question per split: after merging levels into
{0,1} = not-confused and {2,3} = confused, what fraction is positive?

  >= 20%   comfortable. Train it, report AUC + per-class recall as well as accuracy.
  10-20%   viable but imbalanced. Mandatory: class weighting, AUC and minority recall as headline
           metrics, and the majority baseline printed next to every number.
  5-10%    marginal. Expect low positive recall. Consider level >= 1 as the cut instead.
  < 5%     do NOT collapse to binary on this cut. Accuracy becomes meaningless and a 4-level
           ordinal or a different threshold is the honest option.

It also reports the alternative cut ({0} vs {1,2,3}, i.e. ANY confusion), because on a rare state
that is often the better-balanced and more defensible threshold.

INPUT: DAiSEE's label CSVs (TrainLabels.csv / ValidationLabels.csv / TestLabels.csv). These are a
few hundred KB — download just them, not the 17 GB of video.

Run:
    python check_daisee_labels.py --labels-dir /path/to/DAiSEE/Labels
    python check_daisee_labels.py --labels-dir /content/drive/MyDrive/affectlearn-ml/data/DAiSEE/Labels
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

LABEL_COLS = ("Boredom", "Engagement", "Confusion", "Frustration")
SPLIT_FILES = {"Train": "TrainLabels.csv", "Validation": "ValidationLabels.csv",
               "Test": "TestLabels.csv"}


def read_split(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            # DAiSEE ships a trailing space in the "Frustration " header — strip every key.
            rows.append({k.strip(): (v or "").strip() for k, v in row.items()})
    return rows


def verdict(pos_frac: float) -> tuple[str, str]:
    if pos_frac >= 0.20:
        return "COMFORTABLE", "train it; still report AUC and per-class recall"
    if pos_frac >= 0.10:
        return "VIABLE, IMBALANCED", "class weighting + AUC/minority recall as headline"
    if pos_frac >= 0.05:
        return "MARGINAL", "expect low positive recall; consider the ANY-confusion cut"
    return "DO NOT COLLAPSE", "accuracy is meaningless here; use ordinal or the other cut"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", required=True,
                    help="directory holding TrainLabels.csv / ValidationLabels.csv / TestLabels.csv")
    ap.add_argument("--target", default="Confusion", choices=LABEL_COLS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = Path(a.labels_dir)
    if not d.exists():
        raise SystemExit(f"{d} does not exist. Download DAiSEE's Labels folder (a few hundred KB) "
                         "— the videos are not needed for this check.")

    results = {"target": a.target, "splits": {}}
    print(f"DAiSEE label audit — target = {a.target}\n")

    for split, fname in SPLIT_FILES.items():
        p = d / fname
        if not p.exists():
            print(f"  {split:11s} MISSING ({fname})")
            continue
        rows = read_split(p)
        if a.target not in rows[0]:
            raise SystemExit(f"{p} has no '{a.target}' column. Columns: {sorted(rows[0])}")

        levels = collections.Counter(int(r[a.target]) for r in rows if r[a.target] != "")
        n = sum(levels.values())
        # Two candidate binarisations. `high` is the direct analogue of the engagement merge;
        # `any` is usually better balanced when the state is rare.
        high = levels[2] + levels[3]
        anyc = levels[1] + levels[2] + levels[3]

        print(f"  {split} — {n} clips")
        for lv in range(4):
            c = levels.get(lv, 0)
            print(f"      level {lv}: {c:5d}  {100 * c / n:5.1f}%")
        for name, pos in (("HIGH  {2,3} vs {0,1}", high), ("ANY   {1,2,3} vs {0}", anyc)):
            frac = pos / n
            v, advice = verdict(frac)
            base = max(pos, n - pos) / n
            print(f"      {name}: {pos:5d} positive ({100 * frac:5.1f}%)  "
                  f"majority baseline {base:.3f}  -> {v}")
            print(f"          {advice}")
        results["splits"][split] = {
            "n": n, "levels": {str(k): v for k, v in sorted(levels.items())},
            "binary_high": {"positive": high, "fraction": high / n,
                            "majority_baseline": max(high, n - high) / n},
            "binary_any": {"positive": anyc, "fraction": anyc / n,
                           "majority_baseline": max(anyc, n - anyc) / n},
        }
        print()

    # Refuse to exit 0 on an empty audit. A caller that treats "no splits found" as a pass would
    # go on to spend GPU hours on a wrong labels path, and the empty JSON looks like a clean run.
    if not results["splits"]:
        raise SystemExit(
            f"No label CSVs found in {d}. Expected any of {sorted(SPLIT_FILES.values())}. "
            "Check the path points at DAiSEE's Labels directory.")
    if "Test" not in results["splits"]:
        raise SystemExit(
            f"Found {sorted(results['splits'])} but no TestLabels.csv. The reportable number comes "
            "from the test split, so this audit cannot decide whether the target is viable.")

    if True:
        t = results["splits"]["Test"]
        print("=" * 74)
        print("DECISION — read off the TEST split, since that is what gets reported")
        print("=" * 74)
        for cut in ("binary_high", "binary_any"):
            f = t[cut]["fraction"]
            v, advice = verdict(f)
            print(f"  {cut:12s} {100 * f:5.1f}% positive -> {v}: {advice}")
        better = max(("binary_high", "binary_any"), key=lambda c: min(t[c]["fraction"],
                                                                     1 - t[c]["fraction"]))
        print(f"\n  Better-balanced cut: {better}")
        print("  Whichever cut is used, print the majority baseline beside every accuracy. The")
        print("  measured trap in this project: binary collapse took accuracy 0.724 -> 0.934 while")
        print("  kappa fell 0.462 -> 0.025 and minority recall was 0.020.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
