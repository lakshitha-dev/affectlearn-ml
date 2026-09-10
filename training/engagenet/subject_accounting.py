"""Subject and clip accounting for the three official EngageNet splits.

Written because the per-split participant counts recovered from the released
identifiers sum to 128 while the corpus paper reports 127. A difference of one
has two readings - a subject counted twice because it appears in two splits, or
a genuine 128th subject the published total omits - and only the first is
testable. This tests it, by intersecting the three subject sets pairwise.

It also records the extracted-versus-scored clip counts, so the one test clip
that is extracted and not scored is attributable to a label value rather than
left as an unexplained discrepancy. That matters here more than it would
elsewhere: Section 4.4.1 documents a silent 506-row loss on DAiSEE caused by a
join that failed on a filename extension, and an unexplained count is exactly
what that failure looks like from the outside.

The output holds identifiers and counts only. The challenge release forbids
redistributing corpus data, so nothing derived from the video is written.

    python subject_accounting.py [--work C:/engagenet] [--out ../../reports/engagenet_screen]
"""

import argparse
import datetime
import io
import json
import os
import re

import numpy as np
import pandas as pd

import rungs

SPLITS = (("Train", "TrainSub.npz"), ("Validation", "Validation.npz"), ("Test", "Test.npz"))
PUBLISHED_PARTICIPANTS = 127


def _stem(s):
    return re.sub(r"\.mp4$", "", str(s))


def accounting(work):
    """Return the record. Reads the same files, and joins them the same way, as `rungs.load`."""
    rec = {
        "_note": ("Subject accounting for the three official splits, as extracted and as "
                  "scored. Written because the per-split counts sum to one more than the "
                  "published participant total, and a thesis should say which number it is "
                  "using and why. Identifiers and counts only, no corpus data."),
        "_generator": "training/engagenet/subject_accounting.py",
        "_captured": datetime.date.today().isoformat(),
        "published_participants": PUBLISHED_PARTICIPANTS,
        "splits": {},
    }
    extracted, scored = {}, {}
    for split, npz in SPLITS:
        lp = os.path.join(work, "labels", rungs.LABEL_FILE[split])
        lab = pd.read_csv(lp) if lp.endswith(".csv") else pd.read_excel(lp)
        label_of = {_stem(k): str(v) for k, v in zip(lab["chunk"], lab["label"])}

        ids = [_stem(c) for c in
               np.load(os.path.join(work, "geom", npz), allow_pickle=False)["clip_id"]]
        kept = [c for c in ids if label_of.get(c) not in (None, rungs.SNP)]
        extracted[split] = {rungs.subject_of(c) for c in ids}
        scored[split] = {rungs.subject_of(c) for c in kept}

        rec["splits"][split] = {
            "clips_extracted": len(ids),
            "clips_scored": len(kept),
            "dropped_subject_not_present": sum(1 for c in ids if label_of.get(c) == rungs.SNP),
            "dropped_no_label_row": sum(1 for c in ids if c not in label_of),
            "subjects_extracted": len(extracted[split]),
            "subjects_scored": len(scored[split]),
        }

    rec["pairwise_overlap"] = {"%s/%s" % (a, b): len(scored[a] & scored[b])
                               for a in scored for b in scored if a < b}
    rec["splits_are_disjoint"] = all(v == 0 for v in rec["pairwise_overlap"].values())
    rec["subjects_union"] = len(set().union(*extracted.values()))
    rec["subjects_summed"] = sum(len(v) for v in extracted.values())
    rec["subjects_union_scored"] = len(set().union(*scored.values()))
    rec["subjects_summed_scored"] = sum(len(v) for v in scored.values())
    # A union below the sum would mean a subject spans two splits, and the leakage argument
    # of Section 4.9 would not hold. Assert it here rather than leave it to a reader.
    assert rec["subjects_union"] == rec["subjects_summed"], "splits share a subject"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="C:/engagenet")
    ap.add_argument("--out", default=os.path.join("..", "..", "reports", "engagenet_screen"))
    a = ap.parse_args()

    rec = accounting(a.work)
    path = os.path.join(a.out, "subject_accounting.json")
    io.open(path, "w", encoding="utf-8", newline="\n").write(json.dumps(rec, indent=2) + "\n")

    print("wrote %s" % path)
    for split, _ in SPLITS:
        e = rec["splits"][split]
        print("  %-11s %5d extracted, %5d scored, %3d subjects"
              % (split, e["clips_extracted"], e["clips_scored"], e["subjects_scored"]))
    print("  union %d, summed %d, disjoint %s, published %d"
          % (rec["subjects_union"], rec["subjects_summed"],
             rec["splits_are_disjoint"], rec["published_participants"]))


if __name__ == "__main__":
    main()
