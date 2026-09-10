"""Run geometry extraction across several worker processes, one shard each.

WHY PROCESSES AND NOT THREADS

MediaPipe releases the GIL unevenly and OpenCV decode does not, so threads buy almost nothing here.
Separate processes scale cleanly because each holds its own landmarker and its own decoder.

WHY --threads 1 PER WORKER

MediaPipe threads internally and OpenCV will happily take every core it can see. With W workers each
spawning T threads the box is oversubscribed W*T ways, and the measured result of that on a 2-core
VM was two extractors running at 8.7 s/clip against 1.66 s/clip for one alone - a 6x LOSS. Each
worker is pinned to one thread and parallelism comes from the worker count instead, which is the
arrangement that actually scales with cores.

WHY ONE npz PER SHARD

A lost run should cost the shard in flight, not the whole job. Shards are written independently and
`rungs.load` accepts a list of npz paths, so nothing has to be merged before training. Combined with
`--resume` in the extractor, re-running this script continues where it stopped.

    python extract_parallel.py --clips-dir D:/engagenet/Validation \
                               --out-dir  D:/engagenet/geom/Validation --workers 6
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=str(HERE / "face_landmarker.task"))
    ap.add_argument("--limit", type=int, default=None, help="cap total clips, for a smoke test")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    a = ap.parse_args()

    clips = sorted(pathlib.Path(a.clips_dir).glob("*.mp4"))
    if a.limit:
        clips = clips[:a.limit]
    if not clips:
        raise SystemExit(f"no .mp4 under {a.clips_dir}")

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Round-robin rather than contiguous blocks: clip length varies by subject (two are encoded at
    # 1000 fps), and contiguous blocks would hand one worker all the slow ones.
    shards = [clips[i::a.workers] for i in range(a.workers)]

    procs = []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        lst = out / f"shard{i}.txt"
        lst.write_text("\n".join(str(c) for c in shard), encoding="utf-8")
        cmd = [sys.executable, str(HERE / "extract_geometry.py"),
               "--file-list", str(lst), "--out", str(out / f"shard{i}.npz"),
               "--model", a.model, "--resume", "--threads", "1",
               "--checkpoint-every", str(a.checkpoint_every)]
        log = open(out / f"shard{i}.log", "a", encoding="utf-8")
        procs.append((i, len(shard), subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
        print(f"  worker {i}: {len(shard)} clips -> shard{i}.npz")

    print(f"\n  {len(procs)} workers, {len(clips)} clips total. Waiting...\n")
    t0 = time.time()
    while any(p.poll() is None for _, _, p in procs):
        time.sleep(20)
        done = sum(_count(out / f"shard{i}.npz") for i, _, _ in procs)
        el = time.time() - t0
        rate = done / el if el and done else 0
        eta = (len(clips) - done) / rate / 60 if rate else float("nan")
        print(f"    {done}/{len(clips)}  {el/60:.1f}m elapsed  "
              f"{(1/rate if rate else 0):.2f}s/clip  eta {eta:.0f}m", flush=True)

    bad = [i for i, _, p in procs if p.returncode not in (0, None)]
    total = sum(_count(out / f"shard{i}.npz") for i, _, _ in procs)
    print(f"\n  done in {(time.time()-t0)/60:.1f} min: {total}/{len(clips)} clips")
    if bad:
        print(f"  WORKERS THAT FAILED: {bad} - check shard<N>.log")
        return 1
    return 0


def _count(npz):
    try:
        import numpy as np
        return int(len(np.load(npz, allow_pickle=False)["clip_id"]))
    except Exception:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
