"""DAiSEE dataset loader.

Expects clips pre-processed by preprocess.py and saved as .npy files
(shape: T×3×96×96, float32, ImageNet-normalised) under:
  <preprocessed_dir>/Train/
  <preprocessed_dir>/Validation/
  <preprocessed_dir>/Test/

Label CSVs live in the original DAiSEE/Labels/ directory.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ── augmentation helpers ──────────────────────────────────────────────────────

def _random_hflip(clip: np.ndarray) -> np.ndarray:
    """Horizontally flip all frames with 50 % probability. clip: (T,3,H,W)."""
    if np.random.rand() < 0.5:
        return clip[:, :, :, ::-1].copy()
    return clip


def _random_brightness(clip: np.ndarray, delta: float = 0.2) -> np.ndarray:
    """Additive brightness jitter in normalised space."""
    shift = np.random.uniform(-delta, delta)
    return np.clip(clip + shift, -3.0, 3.0)  # stay within sane normalised range


def _random_contrast(clip: np.ndarray, lo: float = 0.8, hi: float = 1.2) -> np.ndarray:
    """Multiplicative contrast jitter in normalised space."""
    factor = np.random.uniform(lo, hi)
    return np.clip(clip * factor, -3.0, 3.0)


# ── dataset ───────────────────────────────────────────────────────────────────

class DAiSEEDataset(Dataset):
    """PyTorch dataset for pre-processed DAiSEE clips.

    Args:
        preprocessed_dir: Root directory containing Train/Validation/Test sub-dirs
                          with .npy clip files.
        label_csv:        Path to TrainLabels.csv / ValidationLabels.csv / TestLabels.csv.
        split:            'Train', 'Validation', or 'Test'.
        target:           Which affect label to use as the target.
                          One of: Engagement, Boredom, Confusion, Frustration.
        binary_cut:       None keeps the raw 0-3 ordinal label. 'any' collapses to
                          {1,2,3} vs {0}; 'high' collapses to {2,3} vs {0,1}. Which cut
                          is chosen changes the task completely, so it is recorded on the
                          dataset and must be reported alongside any metric -- on
                          Engagement the ANY cut leaves four negatives in the whole test
                          split while the HIGH cut leaves 85.
        augment:          Apply random horizontal flip + brightness jitter (train only).
    """

    LABEL_COLS = ("Boredom", "Engagement", "Confusion", "Frustration")
    # Positive level sets. 'any' asks "was this state present at all", 'high' asks "was it
    # present at intensity". They are NOT interchangeable: on Engagement the ANY cut puts
    # 1,634 of 1,638 test clips in the positive class, so a model can score 0.9969 accuracy
    # against a 0.9976 baseline with a negative kappa. See reports/facial_confusion.
    BINARY_CUTS = {"any": (1, 2, 3), "high": (2, 3)}

    def __init__(
        self,
        preprocessed_dir: str,
        label_csv: str,
        split: str = "Train",
        target: str = "Engagement",
        binary_cut: str | None = None,
        augment: bool = False,
    ):
        assert target in self.LABEL_COLS, f"target must be one of {self.LABEL_COLS}"
        if binary_cut is not None and binary_cut not in self.BINARY_CUTS:
            raise ValueError(
                f"binary_cut must be None or one of {tuple(self.BINARY_CUTS)}, got {binary_cut!r}"
            )
        self.target = target
        self.binary_cut = binary_cut
        self.num_classes = 2 if binary_cut else 4
        self.augment = augment
        self.clips:  list[Path] = []
        self.labels: list[int]  = []

        label_map: dict[str, int] = {}
        with open(label_csv, newline="") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items()}   # DAiSEE CSV has a "Frustration " header
                level = int(row[target])
                label_map[row["ClipID"].strip()] = (
                    int(level in self.BINARY_CUTS[binary_cut]) if binary_cut else level
                )

        clip_dir = Path(preprocessed_dir) / split
        for npy in sorted(clip_dir.glob("*.npy")):
            key = npy.stem + ".avi"
            if key in label_map:
                self.clips.append(npy)
                self.labels.append(label_map[key])

        if not self.clips:
            raise RuntimeError(
                f"No clips found in {clip_dir}. "
                "Run preprocess.py first (or check the split / label_csv path)."
            )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip = np.load(self.clips[idx])   # (T, 3, H, W) float32

        if self.augment:
            clip = _random_hflip(clip)
            clip = _random_brightness(clip, delta=0.3)
            clip = _random_contrast(clip)

        return torch.from_numpy(clip.copy()), self.labels[idx]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency per-class weights for FocalLoss alpha.

        Returns a float32 tensor of shape (num_classes,) summing to num_classes.
        """
        n = self.num_classes
        counts = np.bincount(self.labels, minlength=n).astype(np.float64)
        counts = np.maximum(counts, 1.0)          # avoid division by zero
        w = 1.0 / np.sqrt(counts)                 # sqrt-inverse-freq (softer than 1/counts)
        w = w / w.sum() * n                       # normalise so weights sum to num_classes
        return torch.tensor(w, dtype=torch.float32)
