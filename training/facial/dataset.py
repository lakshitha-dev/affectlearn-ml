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
        augment:          Apply random horizontal flip + brightness jitter (train only).
    """

    LABEL_COLS = ("Boredom", "Engagement", "Confusion", "Frustration")

    def __init__(
        self,
        preprocessed_dir: str,
        label_csv: str,
        split: str = "Train",
        target: str = "Engagement",
        augment: bool = False,
    ):
        assert target in self.LABEL_COLS, f"target must be one of {self.LABEL_COLS}"
        self.augment = augment
        self.clips:  list[Path] = []
        self.labels: list[int]  = []

        label_map: dict[str, int] = {}
        with open(label_csv, newline="") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items()}   # DAiSEE CSV has a "Frustration " header
                label_map[row["ClipID"].strip()] = int(row[target])

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
        counts = np.bincount(self.labels, minlength=4).astype(np.float64)
        counts = np.maximum(counts, 1.0)          # avoid division by zero
        w = 1.0 / np.sqrt(counts)                 # sqrt-inverse-freq (softer than 1/counts)
        w = w / w.sum() * 4.0                     # normalise so weights sum to 4
        return torch.tensor(w, dtype=torch.float32)
