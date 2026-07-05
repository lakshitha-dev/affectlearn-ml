"""Build the windowed Bi-LSTM dataset from raw-event windows.

Turns a list of {participant, label_idx, events} into feature sequences via the
SHARED feature_engineering.extract_features, applies global z-score (saved for
serving), and splits BY PARTICIPANT (never by window — avoids leakage).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from feature_engineering import extract_features, FEATURE_NAMES


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
    else:
        X = np.stack([extract_features(w["events"], 0) for w in windows])   # (N, T, F)
    y = np.array([w["label_idx"] for w in windows], dtype=np.int64)
    pid = np.array([w["participant"] for w in windows])
    return X, y, pid


def split_by_participant(pid: np.ndarray, seed: int = 42, ratios=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    users = np.array(sorted(set(pid.tolist())))
    rng.shuffle(users)
    n = len(users)
    n_tr = max(1, int(round(ratios[0] * n)))
    n_va = max(1, int(round(ratios[1] * n)))
    tr, va, te = users[:n_tr], users[n_tr:n_tr + n_va], users[n_tr + n_va:]
    if len(te) == 0:                       # tiny-N safety: borrow one from train
        te = tr[-1:]; tr = tr[:-1]
    sel = lambda group: np.isin(pid, group)
    return sel(tr), sel(va), sel(te), (tr.tolist(), va.tolist(), te.tolist())


def fit_zscore(X_train: np.ndarray):
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(X, mean, std):
    return ((X - mean) / std).astype(np.float32)
