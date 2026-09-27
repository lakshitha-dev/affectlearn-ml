"""Record where the EngageNet inputs came from and how the training subset was drawn.

    python training/engagenet/review/provenance.py --work C:/engagenet

Metadata only: file sizes, modification times (rclone preserves the source's), row counts, label
counts, the download script's remote, and the per-participant clip counts of the training subset.
No label content keyed by clip is written.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import pathlib
import re

import pandas as pd

import _common as C


def stat(p: pathlib.Path) -> dict:
    s = p.stat()
    return {"path": str(p), "bytes": int(s.st_size),
            "mtime": dt.datetime.fromtimestamp(s.st_mtime).isoformat(timespec="seconds")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(C.WORK))
    a = ap.parse_args()
    work = pathlib.Path(a.work)

    labels = {}
    for split, name in C.rungs.LABEL_FILE.items():
        p = work / "labels" / name
        d = pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p)
        labels[split] = {**stat(p), "rows": int(len(d)), "columns": list(map(str, d.columns)),
                         "label_counts": {str(k): int(v) for k, v in d["label"].value_counts().items()}}

    script = (work / "download_all.sh").read_text(encoding="utf-8")
    remote = re.search(r'R="([^"]+)"', script)

    files = [ln.strip() for ln in (work / "trainsub_files.txt").read_text().splitlines() if ln.strip()]
    per = collections.Counter(C.rungs.subject_of(re.sub(r"\.mp4$", "", f)) for f in files)
    train_lab = pd.read_excel(work / "labels" / C.rungs.LABEL_FILE["Train"])
    train_lab = train_lab[train_lab["label"] != C.rungs.SNP]
    full_per = collections.Counter(C.rungs.subject_of(re.sub(r"\.mp4$", "", str(c)))
                                   for c in train_lab["chunk"])
    capped = sum(1 for s in per if per[s] == min(full_per[s], 27))
    low = train_lab["label"].isin(C.rungs.LOW)
    sub_set = {re.sub(r"\.mp4$", "", f) for f in files}
    sub_mask = train_lab["chunk"].map(lambda c: re.sub(r"\.mp4$", "", str(c)) in sub_set)

    C.write("provenance.json", {
        "label_files": labels,
        "download": {"script": stat(work / "download_all.sh"),
                     "rclone_remote": remote.group(1) if remote else None,
                     "log": stat(work / "dl_all.log"),
                     "note": ("download_all.sh copies video from the remote only; the label files "
                              "are not fetched by it, so their origin is not recorded by any script.")},
        "train_subset": {
            "file_list": stat(work / "trainsub_files.txt"), "clips": len(files),
            "participants": len(per),
            "max_clips_per_participant": max(per.values()),
            "participants_at_cap_27": sum(1 for v in per.values() if v == 27),
            "participants_with_all_their_clips_or_27": capped,
            "clip_count_histogram": dict(sorted(collections.Counter(per.values()).items())),
            "positive_rate_full_train_excl_snp": float(low.mean()),
            "positive_rate_subset": float(low[sub_mask].mean()),
            "generator_committed": False,
            "note": ("Each participant contributes min(own clip count, 27) clips. The script that "
                     "drew the list (and any seed) is not in the repository."),
        },
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
