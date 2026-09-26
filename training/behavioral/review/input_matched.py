"""DUX windows rebuilt to look like what the platform actually sends at serve time.

Why this exists
---------------
`external_datasets.load_dux_confusion` feeds the shared extractor with DUX's native event stream.
That stream differs from the platform's in three ways the parity tests cannot see, because they
check the arithmetic of the extractor and not the distribution of its inputs:

1. POINTER RATE. DUX logs `MouseMovement` at roughly 110 Hz (median gap 9 ms). The platform's
   client (`use-behavioral-signals.ts`, `aggregatorTick`) emits at most one `mouse_sample` per
   100 ms tick, carrying the MOST RECENT position, and only if the pointer moved since the last
   tick. Velocity, hover dwell and entropy all depend on that rate.
2. CLICKS. DUX logs `MouseButtonDown` and `MouseClick` for one physical click (plus
   `MouseDoubleClick` for a double click), and the native loader maps all three to "click", so
   `click_count` is roughly doubled. The platform records one event per `mousedown`.
3. EMPTY WINDOWS. The native loader drops windows with fewer than 5 events. The platform scores
   any window with at least one pointer, click, key or scroll event
   (`behavioral_inference._is_idle`).

What this loader does, exactly
------------------------------
* Pointer: within each session, time is cut into 100 ms slots from the session's first row. For
  every slot holding at least one `MouseMovement`, ONE sample is kept: the last position in the
  slot, timestamped at slot start + 99 ms (the end of the slot, which keeps it inside the same
  30 s window because windows are whole multiples of 100 ms). Consecutive samples are therefore
  exactly 100 ms apart, as at serve time.
* Clicks: `MouseButtonDown` is a click. A `MouseClick` is a click only when no unmatched
  `MouseButtonDown` occurred in the preceding 1,000 ms (DUX occasionally omits the down event:
  940 downs against 1,023 clicks in v0). `MouseDoubleClick` is dropped, because its two presses
  are already logged as downs, and the platform's `mousedown` listener would record two presses.
* Keys, scroll, labels and facial channels are built exactly as in `load_dux_confusion`
  (keys reduced to "a" or "Backspace"; scroll `dy` = within-session diff of the absolute
  `yPosition` over scroll rows; label = max human Confusion annotation in the window >= threshold;
  facial = AFFDEX rows of the whole window).
* Windows: kept when they contain at least `min_events` mapped events (default 1, the serving
  rule). Pass `min_events=5` to reproduce the native drop rule on the matched stream.

Nothing here changes `external_datasets.py`; the native loader is untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from external_datasets import (DUX_AFFECTIVA, DUX_MANUAL_CONFUSION, WINDOW_MS,  # noqa: E402
                               _SCREEN_H, _SCREEN_W, _bin_per_second, _is_backspace)

SAMPLE_MS = 100            # platform aggregator tick (use-behavioral-signals.ts SAMPLE_INTERVAL_MS)
CLICK_PAIR_MS = 1_000      # a MouseClick within this long after an unmatched down is the same click

_KEEP_TYPES = ("MouseMovement", "MouseButtonDown", "MouseClick", "MouseDoubleClick",
               "KeyPressed", "Scroll")


def _dedupe_clicks(types: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Boolean mask over rows: True for rows that are one physical click."""
    keep = np.zeros(len(types), dtype=bool)
    pending: list[float] = []                      # unmatched MouseButtonDown times
    for i, (ty, t) in enumerate(zip(types, ts)):
        if ty == "MouseButtonDown":
            keep[i] = True
            pending.append(t)
        elif ty == "MouseClick":
            # match the most recent unmatched down within the pairing window
            pending = [p for p in pending if t - p <= CLICK_PAIR_MS]
            if pending:
                pending.pop()                      # consumed by this click
            else:
                keep[i] = True                     # down was not logged: count the click itself
    return keep


def _resample_moves(g: pd.DataFrame, t0: int) -> pd.DataFrame:
    """One pointer sample per 100 ms slot that saw movement: the latest position in the slot."""
    mv = g[g["type"] == "MouseMovement"]
    if mv.empty:
        return mv
    slot = ((mv["ts"] - t0) // SAMPLE_MS).astype(np.int64)
    last = mv.groupby(slot, sort=True).tail(1).copy()
    last_slot = ((last["ts"] - t0) // SAMPLE_MS).astype(np.int64)
    last["ts"] = t0 + last_slot * SAMPLE_MS + (SAMPLE_MS - 1)
    return last


def load_dux_confusion_matched(dux_dir: str, threshold: float = 1.0,
                               window_ms: int = WINDOW_MS, min_events: int = 1) -> list[dict]:
    """Input-matched counterpart of `external_datasets.load_dux_confusion` (same output schema)."""
    d = Path(dux_dir)
    files = [p for p in (d / "v0.csv", d / "v1.csv") if p.exists()] if d.exists() else []
    if not files:
        return []
    aff_cols = [f"emotion_affectiva_{c}" for c in DUX_AFFECTIVA]
    cols = ["session", "timestamp", "type", "key", "x", "y", "yPosition",
            DUX_MANUAL_CONFUSION] + aff_cols

    windows: list[dict] = []
    for f in files:
        raw = pd.read_csv(f, sep="\t", usecols=cols, low_memory=False)
        raw["conf"] = pd.to_numeric(raw[DUX_MANUAL_CONFUSION], errors="coerce").fillna(0.0)
        for c in aff_cols:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw["ts"] = pd.to_numeric(raw["timestamp"], errors="coerce")
        raw = raw.dropna(subset=["ts"])

        for sess, g_all in raw.groupby("session"):
            g_all = g_all.sort_values("ts", kind="stable")
            t0 = int(g_all["ts"].min())
            widx_all = ((g_all["ts"] - t0) // window_ms).astype(int)
            lab = {int(w): float(sub["conf"].max() >= threshold)
                   for w, sub in g_all.groupby(widx_all)}
            facial = {int(w): sub[aff_cols].mean().to_numpy(dtype=np.float64)
                      for w, sub in g_all.groupby(widx_all)}
            facial_seq = {int(w): _bin_per_second(sub, aff_cols, t0, int(w), window_ms)
                          for w, sub in g_all.groupby(widx_all)}

            g = g_all[g_all["type"].isin(_KEEP_TYPES)].copy()
            if g.empty:
                continue

            # --- pointer: resample to the platform's 10 Hz tick ---
            moves = _resample_moves(g, t0)
            # --- clicks: one per physical press ---
            ck = g[g["type"].isin(("MouseButtonDown", "MouseClick"))]
            ck = ck[_dedupe_clicks(ck["type"].to_numpy(), ck["ts"].to_numpy(dtype=np.float64))]
            other = g[g["type"].isin(("KeyPressed", "Scroll"))]

            ev = pd.concat([moves, ck, other]).sort_values("ts", kind="stable")
            is_bs = _is_backspace(ev["key"])
            ev["type"] = ev["type"].map({"MouseMovement": "move", "MouseButtonDown": "click",
                                         "MouseClick": "click", "KeyPressed": "key",
                                         "Scroll": "scroll"})
            ev["x"] = np.clip(pd.to_numeric(ev["x"], errors="coerce").fillna(0.0) / _SCREEN_W, 0, 1)
            ev["y"] = np.clip(pd.to_numeric(ev["y"], errors="coerce").fillna(0.0) / _SCREEN_H, 0, 1)
            ev["ypos"] = pd.to_numeric(ev["yPosition"], errors="coerce")
            ev["key"] = np.where(is_bs, "Backspace", np.where(ev["type"] == "key", "a", ""))
            ev = ev.reset_index(drop=True)
            ev["dy"] = 0.0
            scroll_rows = ev.index[ev["type"] == "scroll"]
            if len(scroll_rows):
                ev.loc[scroll_rows, "dy"] = ev.loc[scroll_rows, "ypos"].diff().fillna(0.0).to_numpy()

            widx = ((ev["ts"] - t0) // window_ms).astype(int)
            for w, sub in ev.groupby(widx):
                if len(sub) < min_events:
                    continue
                e = sub[["ts", "type", "x", "y", "key", "dy"]].copy()
                e["ts"] = (e["ts"] - t0) - int(w) * window_ms
                windows.append({
                    "participant": f"dux_{f.stem}_{sess}",
                    "window_index": int(w),
                    "label": int(lab.get(int(w), 0.0)),
                    "affectiva": facial.get(int(w)),
                    "affectiva_seq": facial_seq.get(int(w)),
                    "events": e.reset_index(drop=True),
                })
    return windows
