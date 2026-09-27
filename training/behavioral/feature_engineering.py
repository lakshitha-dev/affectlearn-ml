"""Mouse / keyboard / scroll / attention feature extraction for behavioral affect.

SHARED CONTRACT: this same module is copied into the backend for serve time
(`app/services/feature_engineering.py`). The frontend sends RAW events; the backend calls
extract_features() for both offline training and live inference, so train/serve feature
computation is guaranteed identical. Change one copy, change the other, then retrain and
re-export `behavioral_bilstm.onnx` + `behavioral_feature_stats.json` together.

Input  : a DataFrame of raw events within one 30-second window
         (columns: ts [window-relative ms], type, x, y, key, dy).
Output : (n_bins, N_FEATURES) float32 array — one feature vector per 1-second bin.

SCHEMA v2 — WHY IT CHANGED
--------------------------
v1 had thirteen features weighted 6 mouse / 4 keyboard / 3 scroll. That is inverted for a
text-reading platform, where the learner barely touches the mouse, types almost nothing outside a
handful of free-text exercises, and expresses essentially all reading behaviour through SCROLL and
DWELL. Published work reaches ~70% binary disengagement detection from scroll alone on reading
tasks (Biedermann et al., LAK '23, N = 565), so the channel that carried the signal was the one
with the fewest — and least correct — features.

Three v1 features did not measure what their names claimed:

  * `idle_time_pct` was `1 - (mouse-move count)/10`. Not a clock, and it assumed exactly 10 Hz
    polling. On keyboard-only data it read 1.0 in every populated bin no matter how furiously the
    participant typed, and 0.0 in empty bins (via the zero-fill), making it a bimodal
    "is this bin empty?" flag rather than an idle measure. It is now a real time measure.
  * `section_dwell_time` was `1 - (scroll count)/10` — no section id, no dwell clock, nothing to
    do with either. Renamed `scroll_inactivity_pct`, which is what it actually computes. A real
    per-section dwell needs section boundaries plumbed into the window payload; that is deferred
    rather than faked.
  * `scroll_direction_changes` counted sign flips, which conflates deliberate re-reading with
    ordinary jitter. `scroll_back_runs` counts maximal UPWARD runs instead — re-reading is the
    canonical confusion signal in reading research, and a run is the unit that expresses it.

And one channel was collected but thrown away: the frontend has emitted `visibility` events since
Story 4.3 (the type comment even says "useful for idle_time_pct calc") and the backend adapter
dropped them before extraction. Leaving the tab is among the strongest disengagement signals
available and it costs nothing, so v2 adds `blur_time_pct` and `tab_switch_count`.

Deferred deliberately (needs section metadata in the window payload, not fakeable from events):
real per-section dwell, and reading-rate deviation against the section's expected duration.
"""

import numpy as np
import pandas as pd

# Bump whenever FEATURE_NAMES or any feature's math changes. Persisted alongside each window so
# old and new windows are never silently mixed in one training set.
FEATURE_SCHEMA_VERSION = 2

# PERFORMANCE. The DataFrame is converted to numpy ONCE and every per-bin slice is taken by
# integer index, because pandas boolean indexing dominates this function otherwise: profiling a
# realistic 480-event window showed ~665 `DataFrame.__getitem__` calls and 449 ms per window,
# against a sub-200 ms per-cycle budget. The columnar arrays below cost a few milliseconds for
# the same result. Do not reintroduce per-bin `ev[ev["type"] == ...]` filtering.
_MOVE, _CLICK, _KEY, _SCROLL, _VIS = 0, 1, 2, 3, 4
_TYPE_CODE = {"move": _MOVE, "click": _CLICK, "key": _KEY, "scroll": _SCROLL,
              "visibility": _VIS}
_INPUT_CODES = (_MOVE, _CLICK, _KEY, _SCROLL)   # what counts as "activity" for idle time

# Locked feature order — DO NOT reorder (the input tensor column order depends on it).
FEATURE_NAMES = [
    # mouse
    "mouse_entropy", "mouse_velocity_mean", "mouse_velocity_std", "click_count",
    "hover_dwell_mean",
    # keyboard
    "keystroke_count", "typing_rhythm_std", "backspace_pct", "pause_count",
    # scroll — the reading channel
    "scroll_velocity_mean", "scroll_direction_changes", "scroll_back_runs",
    "scroll_inactivity_pct",
    # attention / time
    "idle_time_pct", "blur_time_pct", "tab_switch_count",
]
N_FEATURES = len(FEATURE_NAMES)

_SAMPLE_HZ = 10                 # frontend capture rate
_LOW_VEL = 0.05                 # normalized units/sec, "hovering" threshold
# How long a single instantaneous event is treated as "activity" when integrating idle time.
# Events carry no duration, so without this every bin reads as almost entirely idle.
_ACTIVITY_MS = 200


def _spatial_entropy(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=6, range=[[0, 1], [0, 1]])
    p = h.flatten()
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def _run_count(mask: np.ndarray) -> int:
    """Number of maximal runs of True — a run is one contiguous episode, not one event."""
    if not len(mask):
        return 0
    return int(np.sum(mask & ~np.concatenate([[False], mask[:-1]])))


def _idle_fraction(ts: np.ndarray, lo: float, bin_ms: float) -> float:
    """Proportion of [lo, lo+bin_ms) NOT covered by activity.

    Each event is treated as covering `_ACTIVITY_MS` from its timestamp; overlapping covers are
    merged, so a dense burst counts once. An empty bin is fully idle (1.0) — which is the
    behaviour v1 got backwards via its zero-fill.
    """
    if not len(ts):
        return 1.0
    hi = lo + bin_ms
    covered = 0.0
    cur_start = None
    cur_end = None
    for t in np.sort(ts):
        s = max(float(t), lo)
        e = min(float(t) + _ACTIVITY_MS, hi)
        if e <= s:
            continue
        if cur_end is None:
            cur_start, cur_end = s, e
        elif s <= cur_end:                  # overlaps the open interval — extend it
            cur_end = max(cur_end, e)
        else:                               # gap: bank the closed interval
            covered += cur_end - cur_start
            cur_start, cur_end = s, e
    if cur_end is not None:
        covered += cur_end - cur_start
    return float(np.clip(1.0 - covered / bin_ms, 0.0, 1.0))


def _to_columns(raw_events: pd.DataFrame):
    """DataFrame -> sorted numpy columns, ONCE. Returns (ts, code, x, y, dy, is_backspace).

    Unknown event types get code -1 and are ignored by every feature, which keeps a future
    frontend event kind from silently polluting a channel.
    """
    ts = raw_events["ts"].to_numpy(dtype=np.float64)
    types = raw_events["type"].to_numpy()
    code = np.full(len(ts), -1, dtype=np.int8)
    for name, c in _TYPE_CODE.items():
        code[types == name] = c
    x = raw_events["x"].to_numpy(dtype=np.float64)
    y = raw_events["y"].to_numpy(dtype=np.float64)
    dy = raw_events["dy"].to_numpy(dtype=np.float64)
    is_bs = raw_events["key"].to_numpy() == "Backspace"
    # Stable sort keeps same-timestamp events in arrival order, matching the pandas version.
    order = np.argsort(ts, kind="stable")
    return ts[order], code[order], x[order], y[order], dy[order], is_bs[order]


def _bin_features(ts, code, x, y, dy, is_bs, lo: float, bin_ms: float) -> np.ndarray:
    """Features for ONE bin, from numpy slices. Arrays may be empty.

    `blur_time_pct` is filled in by the caller — it is the only scope that can carry the
    visibility state across bin boundaries.
    """
    f = {k: 0.0 for k in FEATURE_NAMES}

    # ---- mouse ----
    m = code == _MOVE
    if int(m.sum()) >= 2:
        mts = ts[m]
        xy = np.column_stack((x[m], y[m]))
        d = np.diff(xy, axis=0)
        dist = np.sqrt((d ** 2).sum(axis=1))
        dt = np.clip(np.diff(mts) / 1000.0, 1e-3, None)
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
                runs.append(c)
                c = 0
        if c:
            runs.append(c)
        f["hover_dwell_mean"] = float(np.mean(runs) / _SAMPLE_HZ) if runs else 0.0
    f["click_count"] = float(np.count_nonzero(code == _CLICK))

    # ---- keyboard ----
    k = code == _KEY
    n_keys = int(k.sum())
    f["keystroke_count"] = float(n_keys)
    if n_keys >= 2:
        iki = np.diff(ts[k]) / 1000.0
        f["typing_rhythm_std"] = float(iki.std())
        f["pause_count"] = float((iki > 0.5).sum())
    if n_keys >= 1:
        f["backspace_pct"] = float(is_bs[k].mean())

    # ---- scroll (the reading channel) ----
    s_mask = code == _SCROLL
    n_scroll = int(s_mask.sum())
    if n_scroll:
        sdy = dy[s_mask]
        f["scroll_velocity_mean"] = float(np.abs(sdy).mean())
        sign = np.sign(sdy)
        nz = sign[sign != 0]
        f["scroll_direction_changes"] = float((np.diff(nz) != 0).sum()) if len(nz) >= 2 else 0.0
        # Upward RUNS, not events: one re-read episode is one signal, however many wheel ticks
        # it took. Positive dy is downward (wheel deltaY convention), so negative is scroll-back.
        f["scroll_back_runs"] = float(_run_count(sdy < 0))
    f["scroll_inactivity_pct"] = float(np.clip(1.0 - n_scroll / _SAMPLE_HZ, 0.0, 1.0))

    # ---- attention / time ----
    # Idle counts EVERY input channel, so a keyboard-only learner is not called idle.
    act = np.isin(code, _INPUT_CODES)
    f["idle_time_pct"] = _idle_fraction(ts[act], lo, bin_ms)
    f["tab_switch_count"] = float(np.count_nonzero(code == _VIS))

    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)


def _blur_per_bin(vis_ts: np.ndarray, vis_hidden: np.ndarray, window_start: int,
                  n_bins: int, bin_ms: int) -> np.ndarray:
    """Fraction of each bin spent hidden/unfocused.

    Needs the whole window, not one bin: the page can go hidden in bin 3 and come back in bin 9,
    so the state has to be carried forward. `dy` on a visibility event is 1.0 for hidden and 0.0
    for visible (set by the adapter). The window is assumed to START visible — the frontend
    carries hidden time across window boundaries in its own summary, so an already-hidden window
    is under-counted here rather than over-counted, which is the safe direction.
    """
    out = np.zeros(n_bins, dtype=np.float32)
    if len(vis_ts) == 0:
        return out

    events = [(float(t), bool(h)) for t, h in zip(vis_ts, vis_hidden)]
    for b in range(n_bins):
        lo = window_start + b * bin_ms
        hi = lo + bin_ms
        # State entering this bin = the last transition at or before `lo` (default: visible).
        hidden = False
        for t, h in events:
            if t <= lo:
                hidden = h
            else:
                break
        hidden_ms = 0.0
        cursor = lo
        for t, h in events:
            if t <= lo or t >= hi:
                continue
            if hidden:
                hidden_ms += t - cursor
            cursor = t
            hidden = h
        if hidden:
            hidden_ms += hi - cursor
        out[b] = float(np.clip(hidden_ms / bin_ms, 0.0, 1.0))
    return out


def extract_features(raw_events: pd.DataFrame, window_start: int,
                     window_length_ms: int = 30_000, bin_length_ms: int = 1_000) -> np.ndarray:
    """Raw events within one window -> (n_bins, N_FEATURES) float32."""
    n_bins = window_length_ms // bin_length_ms
    out = np.zeros((n_bins, N_FEATURES), dtype=np.float32)
    empty = raw_events is None or len(raw_events) == 0

    idle_i = FEATURE_NAMES.index("idle_time_pct")
    inact_i = FEATURE_NAMES.index("scroll_inactivity_pct")
    blur_i = FEATURE_NAMES.index("blur_time_pct")

    if empty:
        # A silent window is fully idle with no scroll activity — NOT all-zeros. v1 returned
        # zeros here, which told the model the opposite of the truth.
        out[:, idle_i] = 1.0
        out[:, inact_i] = 1.0
        return out

    # One pandas -> numpy conversion for the whole window; every per-bin slice below is an
    # integer range, so pandas never enters the loop (see the PERFORMANCE note at the top).
    ts, code, x, y, dy, is_bs = _to_columns(raw_events)

    # Bin boundaries as index positions into the sorted timestamps: O(log n) per edge instead of
    # a full boolean scan of the frame per bin.
    edges = window_start + np.arange(n_bins + 1) * bin_length_ms
    starts = np.searchsorted(ts, edges[:-1], side="left")
    stops = np.searchsorted(ts, edges[1:], side="left")

    for b in range(n_bins):
        s, e = starts[b], stops[b]
        # Called for EVERY bin, including empty ones, so idle/inactivity are correct throughout.
        out[b] = _bin_features(
            ts[s:e], code[s:e], x[s:e], y[s:e], dy[s:e], is_bs[s:e],
            float(edges[b]), float(bin_length_ms),
        )

    vis = code == _VIS
    out[:, blur_i] = _blur_per_bin(
        ts[vis], dy[vis] >= 0.5, window_start, n_bins, bin_length_ms
    )
    return out
