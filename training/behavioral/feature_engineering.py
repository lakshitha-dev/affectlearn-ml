"""Mouse / keyboard / scroll feature extraction for behavioral affect.

SHARED CONTRACT: this same module is imported by the backend at serve time
(`model_inference.py`). The frontend sends RAW events (ts, type, x, y, key, dy);
the backend calls extract_features() for both offline training and live inference,
so train/serve feature computation is guaranteed identical.

Input  : a DataFrame of raw events within one 30-second window.
Output : (n_bins, n_features) float32 array — one feature vector per 1-second bin.
"""

import numpy as np
import pandas as pd

# Locked feature order — DO NOT reorder (the input tensor column order depends on it).
FEATURE_NAMES = [
    "mouse_entropy", "mouse_velocity_mean", "mouse_velocity_std", "click_count",
    "idle_time_pct", "hover_dwell_mean",
    "keystroke_count", "typing_rhythm_std", "backspace_pct", "pause_count",
    "scroll_velocity_mean", "scroll_direction_changes", "section_dwell_time",
]
N_FEATURES = len(FEATURE_NAMES)
_SAMPLE_HZ = 10                 # frontend capture rate
_LOW_VEL = 0.05                 # normalized units/sec, "hovering" threshold


def _spatial_entropy(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=6, range=[[0, 1], [0, 1]])
    p = h.flatten()
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _bin_features(ev: pd.DataFrame) -> np.ndarray:
    f = {k: 0.0 for k in FEATURE_NAMES}

    mv = ev[ev["type"] == "move"].sort_values("ts")
    if len(mv) >= 2:
        xy = mv[["x", "y"]].to_numpy(dtype=float)
        ts = mv["ts"].to_numpy(dtype=float)
        d = np.diff(xy, axis=0)
        dist = np.sqrt((d ** 2).sum(axis=1))
        dt = np.clip(np.diff(ts) / 1000.0, 1e-3, None)
        vel = dist / dt
        f["mouse_velocity_mean"] = float(vel.mean())
        f["mouse_velocity_std"] = float(vel.std())
        f["mouse_entropy"] = _spatial_entropy(xy)
        low = vel < _LOW_VEL
        runs, c = [], 0
        for v in low:
            if v:
                c += 1
            elif c:
                runs.append(c); c = 0
        if c:
            runs.append(c)
        f["hover_dwell_mean"] = float(np.mean(runs) / _SAMPLE_HZ) if runs else 0.0
    f["idle_time_pct"] = float(np.clip(1.0 - len(mv) / _SAMPLE_HZ, 0.0, 1.0))
    f["click_count"] = float((ev["type"] == "click").sum())

    ks = ev[ev["type"] == "key"].sort_values("ts")
    f["keystroke_count"] = float(len(ks))
    if len(ks) >= 2:
        iki = np.diff(ks["ts"].to_numpy(dtype=float)) / 1000.0
        f["typing_rhythm_std"] = float(iki.std())
        f["pause_count"] = float((iki > 0.5).sum())
    if len(ks) >= 1:
        f["backspace_pct"] = float((ks["key"] == "Backspace").mean())

    sc = ev[ev["type"] == "scroll"].sort_values("ts")
    if len(sc):
        dy = sc["dy"].to_numpy(dtype=float)
        f["scroll_velocity_mean"] = float(np.abs(dy).mean())
        s = np.sign(dy)
        s = s[s != 0]
        f["scroll_direction_changes"] = float((np.diff(s) != 0).sum()) if len(s) >= 2 else 0.0
    f["section_dwell_time"] = float(np.clip(1.0 - len(sc) / _SAMPLE_HZ, 0.0, 1.0))

    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)


def extract_features(raw_events: pd.DataFrame, window_start: int,
                     window_length_ms: int = 30_000, bin_length_ms: int = 1_000) -> np.ndarray:
    """Raw events within one window -> (n_bins, N_FEATURES) float32."""
    n_bins = window_length_ms // bin_length_ms
    out = np.zeros((n_bins, N_FEATURES), dtype=np.float32)
    if raw_events is None or len(raw_events) == 0:
        return out
    for b in range(n_bins):
        lo = window_start + b * bin_length_ms
        hi = lo + bin_length_ms
        ev = raw_events[(raw_events["ts"] >= lo) & (raw_events["ts"] < hi)]
        if len(ev):
            out[b] = _bin_features(ev)
    return out
