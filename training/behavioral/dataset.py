"""Build the windowed Bi-LSTM dataset from raw-event windows.

Turns a list of {participant, label_idx, events} into feature sequences via the
SHARED feature_engineering.extract_features, applies global z-score (saved for
serving), and splits BY PARTICIPANT (never by window — avoids leakage).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from feature_engineering import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    N_FEATURES,
    extract_features,
)

# Splits live in a torch-free module so they can be unit-tested without the GPU stack.
# Re-exported here so existing `from dataset import split_by_participant` call sites are
# unaffected. Prefer the stratified variant under codebook labelling — see splits.py.
from splits import (  # noqa: F401
    assert_all_classes_present,
    group_labels,
    split_by_participant,
    split_by_participant_stratified,
)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def windows_to_arrays(windows):
    if windows and "features" in windows[0]:
        # Real Phase A windows already carry the backend-computed aggregate feature window
        # (train/serve parity by construction — see phase_a.py). Do NOT re-extract.
        X = np.stack([np.asarray(w["features"], dtype=np.float32) for w in windows])
        _check_feature_schema(X, windows)
    else:
        X = np.stack([extract_features(w["events"], 0) for w in windows])   # (N, T, F)
    y = np.array([w["label_idx"] for w in windows], dtype=np.int64)
    pid = np.array([w["participant"] for w in windows])
    return X, y, pid


def _check_feature_schema(X: np.ndarray, windows) -> None:
    """Reject persisted windows whose feature width does not match the current extractor.

    Persisted `features` are passed through un-re-extracted, so a window captured under an older
    FEATURE_SCHEMA_VERSION flows straight into the model and fails ~200 frames deep inside torch
    with `input.size(-1) must be equal to input_size`. That error names neither the cause nor the
    file, and it is the exact "two schemas silently mixed in one training set" hazard the schema
    stamp exists to prevent — but the stamp is only useful if something CHECKS it.

    Mixing widths is unrecoverable: the columns mean different things, so there is no padding or
    truncation that would be honest. Re-export from the platform, or re-run the rehearsal
    generator, so every window carries the current schema.
    """
    width = int(X.shape[-1])
    if width == N_FEATURES:
        return
    seen = {int(w.get("feature_schema_version", 0)) for w in windows}
    raise SystemExit(
        f"FEATURE SCHEMA MISMATCH: persisted windows are {width} features wide but this "
        f"extractor produces {N_FEATURES} (schema v{FEATURE_SCHEMA_VERSION}).\n"
        f"  schema versions stamped on these windows: {sorted(seen) or 'none (pre-versioning)'}\n"
        f"  current feature order: {', '.join(FEATURE_NAMES)}\n"
        "  These windows predate the current feature schema. Padding or truncating would silently\n"
        "  misalign columns, so re-generate the data instead:\n"
        "    - platform export -> re-run scripts/export_research.py after redeploying the backend\n"
        "    - rehearsal        -> re-run scripts/rehearse_codebook_export.py"
    )


def fit_zscore(X_train: np.ndarray):
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(X, mean, std):
    return ((X - mean) / std).astype(np.float32)
