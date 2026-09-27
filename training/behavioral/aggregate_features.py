"""Flatten a (n_bins, N_FEATURES) window into per-channel summary statistics.

A gradient-boosted tree cannot consume a sequence, so each channel is summarised over the
window's bins. This is the standard treatment for tabular time series, and it is what makes the
GBDT arm a fair comparison rather than a handicapped one: the Bi-LSTM sees the ordering, the GBDT
sees the distribution plus a coarse trend term, and the held-out comparison decides whether the
ordering was worth anything on this data.

WHY A GBDT ARM EXISTS AT ALL. Botelho et al. (2017) showed an LSTM beating classical ML on these
four affect states — on roughly 2.5 million windows. At a few hundred to a few thousand windows
the ordering commonly flips, because gradient boosting on tabular summaries is far more
sample-efficient than a recurrent net. Running both and reporting the comparison is more honest
than assuming either, and the GBDT's feature importances are direct evidence for RQ1: they say
WHICH behaviours carry the signal, which no aggregate F1 can.

Output is NAMED, so an importance table is readable rather than a list of column indices.
"""

from __future__ import annotations

import numpy as np

from feature_engineering import FEATURE_NAMES

# Per-channel statistics. `trend` is the late-window mean minus the early-window mean — a crude
# direction-of-travel term that gives the trees a little of what the LSTM gets for free.
_STATS = ("mean", "std", "min", "max", "trend")


def aggregate_feature_names() -> list[str]:
    return [f"{ch}_{stat}" for ch in FEATURE_NAMES for stat in _STATS]


def aggregate_window(X: np.ndarray) -> np.ndarray:
    """(n_bins, n_channels) -> (n_channels * len(_STATS),) float32."""
    X = np.asarray(X, dtype=np.float64)
    n_bins = X.shape[0]
    third = max(1, n_bins // 3)
    early = X[:third].mean(axis=0)
    late = X[-third:].mean(axis=0)
    parts = np.stack(
        [X.mean(axis=0), X.std(axis=0), X.min(axis=0), X.max(axis=0), late - early],
        axis=1,                                   # (n_channels, n_stats)
    )
    return parts.reshape(-1).astype(np.float32)   # channel-major, matching the names above


def aggregate(X: np.ndarray) -> np.ndarray:
    """(N, n_bins, n_channels) -> (N, n_channels * len(_STATS)) float32."""
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"expected (N, n_bins, n_channels), got shape {X.shape}")
    return np.stack([aggregate_window(w) for w in X]).astype(np.float32)
