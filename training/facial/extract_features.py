"""Cache frozen-backbone frame features, so the temporal head can be trained in seconds.

WHY THIS EXISTS
---------------
`train_cached_features.py` is the only training path in this project that does the things a small
imbalanced corpus demands: five seeds with reported variance, `GroupKFold` on participant, early
stopping on validation AUC rather than weighted F1, subject-level bootstrap intervals, and a
leave-largest-subject-out sensitivity check. It can afford all of that because it never touches
pixels — with the backbone frozen, each clip's 16x512 features are constant, so they are computed
once and reused.

It reads a feature cache that **nothing in this repository writes.** The caches under `.features/`
are committed artefacts whose generator was never committed, which means the cached-feature results
already reported are not reproducible from the repository either. This script closes that hole and
is a prerequisite for running the same path on any new corpus.

BATCHNORM IS THE WHOLE SUBTLETY
-------------------------------
`requires_grad_(False)` freezes weights. It does not freeze BatchNorm's running mean and variance,
which keep updating on every forward pass while the module is in training mode. `train_cnn_lstm.py`
never calls `.eval()` on the backbone, so its "frozen" ResNet-18 quietly adapted its normalisation
statistics to the corpus over training — worth 0.058 AUC on the deployed confusion model, and
reported as such rather than discovered by a reader.

This script forces `.eval()`. That is the correct behaviour for a genuinely frozen encoder, and it
means features cached here are **not** bit-identical to what the pixel path saw. That is a
deliberate divergence, not an oversight: one of the two had to be wrong, and this is the one that
matches what the artefact claims about itself.

CONTRACT
--------
One `<split>.npz` per split, matching what `train_cached_features.load_split` expects:

    feats      (N, 16, 512) float16   frozen-backbone features, in clip frame order
    feats_flip (N, 16, 512) float16   the same clips horizontally mirrored
    level      (N,)         int8      ordinal label, or -1 when no label CSV is supplied
    clip_id    (N,)         str       stem of the source .npy
    subject    (N,)         str       participant id, for GroupKFold and subject bootstraps

`feats_flip` exists because mirroring is the one augmentation that survives caching: a horizontal
flip of a face is still a plausible face, so the flipped features are a second view the head can
train on at no extra cost. It is applied to the fit half only, by the caller.

float16 halves the cache with no measurable effect downstream — the head casts back to float32 on
load, and these are activations rather than gradients.

    python training/facial/extract_features.py \
        --clips-dir  "G:/My Drive/affectlearn-ml/daisee_preprocessed" \
        --out-dir    .features \
        --labels-dir "G:/My Drive/affectlearn-ml/daisee_labels" \
        --target Engagement
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import FrameCNN  # noqa: E402

SPLITS = ("Train", "Validation", "Test")
SPLIT_CSV = {"Train": "TrainLabels.csv", "Validation": "ValidationLabels.csv",
             "Test": "TestLabels.csv"}


def subject_of(clip_id: str) -> str:
    """DAiSEE encodes the participant in the first six digits of the ClipID.

    A corpus that names participants differently should pass `--subject-mode` rather than have this
    guessed at, because every grouped fold and every subject-level interval downstream depends on
    it being right.
    """
    return str(clip_id)[:6]


def levels_from_csv(labels_dir: Path, split: str, target: str,
                    clip_ids: list[str]) -> np.ndarray:
    """Ordinal label per clip, joined on ClipID.

    The `.avi` re-append is the join contract the rest of the pipeline uses: preprocessed files are
    named by stem, the label CSVs key on the original filename. Header whitespace is stripped
    because at least one DAiSEE column ships as `"Frustration "`.
    """
    path = Path(labels_dir) / SPLIT_CSV[split]
    table: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            table[row["ClipID"].strip()] = int(row[target])
    missing = [c for c in clip_ids if f"{c}.avi" not in table]
    if missing:
        raise SystemExit(
            f"{len(missing)} cached clips have no label in {path.name} "
            f"(first few: {missing[:5]}). Refusing to write a cache with silent gaps."
        )
    return np.array([table[f"{c}.avi"] for c in clip_ids], dtype=np.int8)


@torch.no_grad()
def encode(clips: list[Path], enc: FrameCNN, device: str, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (feats, feats_flip), each (N, T, 512) float16."""
    feats, flips = [], []
    for i in range(0, len(clips), batch):
        chunk = clips[i:i + batch]
        arr = np.stack([np.load(c) for c in chunk])            # (B, T, 3, H, W)
        x = torch.from_numpy(arr).float().to(device)
        b, t = x.shape[:2]

        for source, sink in ((x, feats), (torch.flip(x, dims=[-1]), flips)):
            f = enc(source.reshape(b * t, *source.shape[2:]))   # (B*T, 512)
            sink.append(f.reshape(b, t, -1).cpu().numpy().astype(np.float16))

        done = min(i + batch, len(clips))
        print(f"\r    {done}/{len(clips)} clips", end="", flush=True)
    print()
    return np.concatenate(feats), np.concatenate(flips)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", required=True,
                    help="root holding <Split>/<clip>.npy of shape (T, 3, H, W)")
    ap.add_argument("--out-dir", default=".features")
    ap.add_argument("--labels-dir", default=None,
                    help="omit to write level = -1 and supply labels at train time")
    ap.add_argument("--target", default="Engagement")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--out-dim", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    # eval() is the point of this script, not a detail: see the module docstring.
    enc = FrameCNN(out_dim=a.out_dim, backbone=a.backbone).to(a.device).eval()
    enc.requires_grad_(False)
    print(f"  {a.backbone} encoder on {a.device}, BN in eval mode, weights frozen")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in a.splits:
        clips = sorted((Path(a.clips_dir) / split).glob("*.npy"))
        if not clips:
            print(f"  {split}: no .npy clips found, skipping")
            continue
        print(f"  {split}: {len(clips)} clips")

        clip_id = [c.stem for c in clips]
        subject = [subject_of(c) for c in clip_id]
        feats, flips = encode(clips, enc, a.device, a.batch)
        level = (levels_from_csv(Path(a.labels_dir), split, a.target, clip_id)
                 if a.labels_dir else np.full(len(clip_id), -1, dtype=np.int8))

        dest = out_dir / f"{split}.npz"
        np.savez_compressed(
            dest,
            feats=feats, feats_flip=flips, level=level,
            clip_id=np.array(clip_id), subject=np.array(subject),
        )
        mb = dest.stat().st_size / 1e6
        print(f"    wrote {dest}  {feats.shape} + flip, "
              f"{len(set(subject))} subjects, {mb:.1f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
