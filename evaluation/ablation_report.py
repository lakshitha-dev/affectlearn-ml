"""Unimodal vs fusion ablation — the RQ2 comparison, with the prior stated before the number.

Takes prediction records for face-only, behaviour-only and fused models over the SAME items and
reports whether fusion beats the best single modality, by how much, and whether that difference
survives a paired test.

THE PRIOR MATTERS AND IS PRINTED FIRST. D'Mello & Kory's meta-analysis of 90 systems found
multimodal beat the best unimodal in 85% of them, by an average of 9.83% (MEDIAN 6.60% — the mean
is skewed by a long right tail, so the median is the better expectation for a single new system).
Crucially the gain was "three times lower when systems were trained on natural (4.59%) versus acted
data (12.7%)". Reported gains near 19 points come from small, controlled, acted corpora and should
not be expected to transfer. Printing the expectation before the result is what stops a modest gain
being written up as a disappointment, or a large one being accepted without suspicion.

These figures are quoted verbatim from the paper's abstract. An earlier version of this file carried
8.12 / 4.39 / 12.1, which were wrong; if those numbers appear anywhere else they are stale.

A NOTE ON WHAT THIS CAN AND CANNOT SHOW HERE. A genuine fusion result requires paired data — the
same learner, the same window, both modalities, one label. The facial branch is trained on DAiSEE
clips and the behavioural branch on platform windows or public keystroke corpora; no public
dataset pairs mouse+keyboard+scroll with learning-affect labels, so until jointly labelled pilot
data exists this script measures a mechanism, not an answer to RQ2. It refuses to run on records
whose items do not match, precisely so that a non-paired comparison cannot be produced by
accident.

Run:
    python ablation_report.py --face face.npz --behaviour beh.npz --fused fused.npz
    python ablation_report.py --face f.npz --behaviour b.npz --fused fu.npz --out-dir reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import confusion_matrix as cmx  # noqa: E402
from predictions import Predictions  # noqa: E402
from statistical_tests import compare_predictions, interpret  # noqa: E402

# D'Mello & Kory (2015), ACM Computing Surveys 47(3) art. 43 — verbatim from the abstract:
# "MM systems were consistently (85% of systems) more accurate than their best unimodal
#  counterparts, with an average improvement of 9.83% (median of 6.60%). However, improvements were
#  three times lower when systems were trained on natural (4.59%) versus acted data (12.7%)."
PRIOR_ALL = 9.83
PRIOR_MEDIAN = 6.60          # prefer this as the expectation for one new system
PRIOR_NATURALISTIC = 4.59
PRIOR_ACTED = 12.7
PRIOR_SHARE_BEATING_UNIMODAL = 85.0


def _require_same_items(records: dict[str, Predictions]) -> None:
    """Refuse to compare records that are not the same items in the same order."""
    names = list(records)
    ref_name, ref = names[0], records[names[0]]
    for name in names[1:]:
        other = records[name]
        if len(other) != len(ref):
            raise ValueError(
                f"'{name}' has {len(other)} items but '{ref_name}' has {len(ref)} — these are not "
                "the same evaluation set, so no ablation is valid over them"
            )
        if not np.array_equal(other.y_true, ref.y_true):
            raise ValueError(
                f"'{name}' and '{ref_name}' disagree on y_true — the records are not aligned "
                "item-for-item. A fusion gain computed across different items is meaningless."
            )
        if other.labels != ref.labels:
            raise ValueError(f"'{name}' and '{ref_name}' have different label sets")


def build(face: Predictions, behaviour: Predictions, fused: Predictions) -> dict:
    records = {"face_only": face, "behaviour_only": behaviour, "fused": fused}
    _require_same_items(records)

    arms = {name: cmx.compute(p) for name, p in records.items()}
    unimodal = {k: arms[k] for k in ("face_only", "behaviour_only")}
    best_uni = max(unimodal, key=lambda k: unimodal[k]["weighted_f1"])

    gain_wf1 = arms["fused"]["weighted_f1"] - arms[best_uni]["weighted_f1"]
    gain_acc = arms["fused"]["accuracy"] - arms[best_uni]["accuracy"]

    out = {
        "n": len(fused),
        "labels": fused.labels,
        "prior": {
            "source": "D'Mello & Kory (2015), meta-analysis of multimodal affect detection",
            "mean_gain_over_best_unimodal_pct": PRIOR_ALL,
            "naturalistic_pct": PRIOR_NATURALISTIC,
            "acted_pct": PRIOR_ACTED,
            "expectation": (
                f"Expect roughly {PRIOR_NATURALISTIC}% on naturalistic data, not "
                f"{PRIOR_ACTED}% — the acted figure does not transfer."
            ),
        },
        "arms": {
            name: {
                "accuracy": m["accuracy"],
                "weighted_f1": m["weighted_f1"],
                "macro_f1": m["macro_f1"],
                "cohen_kappa": m["cohen_kappa"],
                "per_class_recall": {k: v["recall"] for k, v in m["per_class"].items()},
                "zero_recall_classes": m["zero_recall_classes"],
            }
            for name, m in arms.items()
        },
        "best_unimodal": best_uni,
        "fusion_gain": {
            "weighted_f1_absolute": float(gain_wf1),
            "weighted_f1_points": float(gain_wf1 * 100),
            "accuracy_absolute": float(gain_acc),
            "vs_naturalistic_prior": float(gain_wf1 * 100 - PRIOR_NATURALISTIC),
        },
        # Paired tests: fusion against EACH unimodal arm on identical items.
        "mcnemar_vs_best_unimodal": compare_predictions(records[best_uni], fused),
        "mcnemar_vs_face": compare_predictions(face, fused),
        "mcnemar_vs_behaviour": compare_predictions(behaviour, fused),
    }

    # Fusion can raise the aggregate while losing a class — worth catching explicitly.
    lost = [
        c for c in fused.labels
        if c in arms["fused"]["zero_recall_classes"]
        and c not in arms[best_uni]["zero_recall_classes"]
    ]
    if lost:
        out["classes_lost_by_fusion"] = lost
    return out


def format_report(r: dict) -> str:
    p = r["prior"]
    lines = [
        "MULTIMODAL FUSION ABLATION",
        "=" * 66,
        f"n = {r['n']} items, {len(r['labels'])} classes: {', '.join(r['labels'])}",
        "",
        "PRIOR EXPECTATION (stated before the result, on purpose)",
        f"  {p['source']}",
        f"  mean gain over best unimodal: {p['mean_gain_over_best_unimodal_pct']}%",
        f"  naturalistic {p['naturalistic_pct']}%  vs  acted {p['acted_pct']}%",
        f"  -> {p['expectation']}",
        "",
        "ARMS",
        f"  {'arm':16s} {'acc':>7s} {'wF1':>7s} {'macF1':>7s} {'kappa':>7s}",
    ]
    for name, a in r["arms"].items():
        mark = "  *" if name == r["best_unimodal"] else ""
        lines.append(f"  {name:16s} {a['accuracy']:>7.4f} {a['weighted_f1']:>7.4f} "
                     f"{a['macro_f1']:>7.4f} {a['cohen_kappa']:>7.4f}{mark}")
    lines.append(f"  (* = best unimodal, the baseline fusion must beat)")

    g = r["fusion_gain"]
    lines += [
        "",
        "FUSION GAIN over the best single modality",
        f"  weighted-F1: {g['weighted_f1_absolute']:+.4f}  ({g['weighted_f1_points']:+.2f} points)",
        f"  accuracy:    {g['accuracy_absolute']:+.4f}",
        f"  vs the {PRIOR_NATURALISTIC}% naturalistic prior: {g['vs_naturalistic_prior']:+.2f} points",
        "",
        "PAIRED TEST (fusion vs best unimodal, same items)",
    ]
    lines += ["  " + ln for ln in interpret(r["mcnemar_vs_best_unimodal"]).splitlines()]

    lines += ["", "PER-CLASS RECALL"]
    header = f"  {'class':16s}" + "".join(f"{n[:12]:>14s}" for n in r["arms"])
    lines.append(header)
    for c in r["labels"]:
        row = f"  {c:16s}" + "".join(
            f"{r['arms'][n]['per_class_recall'][c]:>14.3f}" for n in r["arms"]
        )
        lines.append(row)

    if r.get("classes_lost_by_fusion"):
        lines += ["", "  *** fusion drove these classes to ZERO recall even though the best "
                  f"unimodal arm detected them: {', '.join(r['classes_lost_by_fusion'])}. An "
                  "aggregate gain that costs a whole class is not an improvement. ***"]

    if g["weighted_f1_absolute"] <= 0:
        lines += ["", "  Fusion did NOT beat the best single modality. That is a legitimate, "
                  "reportable finding — report it, with the prior above as context."]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--face", required=True)
    ap.add_argument("--behaviour", "--behavioral", dest="behaviour", required=True)
    ap.add_argument("--fused", required=True)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    report = build(Predictions.load(a.face), Predictions.load(a.behaviour),
                   Predictions.load(a.fused))
    text = format_report(report)
    print(text)

    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ablation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / "ablation_summary.txt").write_text(text, encoding="utf-8")
        print(f"\nwrote {out / 'ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
