"""Adapters: public human-interaction datasets -> the SHARED feature-window format.

Stage-0 pretraining (see plan + DATASET_CARD.md). Converts EmoSurv (keystroke-only) and DUX
(mouse + keyboard + scroll) into the same per-window `events` DataFrames that
`feature_engineering.extract_features` consumes — so the SAME extractor produces pretrain,
fine-tune, and serve features (train/serve parity). Each emitted window is labeled with a
BINARY PROXY (`neutral` vs `emotion`); public emotion labels do not map to the 4 learning-
affect states, so we only pretrain the encoder on this proxy.

Emitted window: {participant, proxy_label (0/1), events(DataFrame[ts,type,x,y,key,dy])}, with
`ts` relative to the window start (0..WINDOW_MS), matching synthetic_data.generate_dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROXY_LABELS = ["neutral", "emotion"]     # index 0, 1
WINDOW_MS = 30_000
_SCREEN_W, _SCREEN_H = 1920.0, 1080.0     # DUX x,y are pixels; normalise to [0,1] like serve-time
_NEUTRAL_CUTOFF = 90.0                     # DUX affectiva Neutral (0-100); window is "emotion" if mean < cutoff
_MIN_EVENTS = 5                            # skip near-empty windows

# DUX event type -> our type
_DUX_TYPE = {
    "MouseMovement": "move", "MouseClick": "click", "MouseDoubleClick": "click",
    "MouseButtonDown": "click", "KeyPressed": "key", "Scroll": "scroll",
}


def _windows_from_session(df: pd.DataFrame, participant: str, neutral: np.ndarray) -> list[dict]:
    """Slice one session's mapped events into 30s windows + a proxy label per window.

    `df` columns: ts (abs ms), type, x, y, key, dy. `neutral` = per-row affectiva Neutral (0-100)
    or NaN (EmoSurv has no such signal — label comes from its own emotion column instead).
    """
    if len(df) == 0:
        return []
    t0 = int(df["ts"].min())
    widx = ((df["ts"] - t0) // WINDOW_MS).astype(int)
    out = []
    for w, sub in df.groupby(widx):
        if len(sub) < _MIN_EVENTS:
            continue
        ev = sub.copy()
        ev["ts"] = (ev["ts"] - t0) - int(w) * WINDOW_MS       # -> [0, WINDOW_MS)
        ev = ev[["ts", "type", "x", "y", "key", "dy"]].reset_index(drop=True)
        neu = neutral[sub.index]
        neu = neu[~np.isnan(neu)]
        proxy = 1 if (len(neu) and float(neu.mean()) < _NEUTRAL_CUTOFF) else 0
        out.append({"participant": participant, "proxy_label": proxy, "events": ev})
    return out


def load_dux(dux_dir: str) -> list[dict]:
    """DUX (Zenodo) TSV files (v0.csv / v1.csv). Mouse + keyboard + scroll + affectiva labels."""
    d = Path(dux_dir)
    files = [p for p in (d / "v0.csv", d / "v1.csv") if p.exists()] if d.exists() else []
    if not files:
        return []
    cols = ["session", "timestamp", "type", "key", "x", "y", "yPosition",
            "emotion_affectiva_Neutral"]
    windows = []
    for f in files:
        raw = pd.read_csv(f, sep="\t", usecols=cols, low_memory=False)
        raw = raw[raw["type"].isin(_DUX_TYPE)].copy()
        # backspace check reads the ORIGINAL key token before we overwrite `type`/`key`
        is_bs = raw["key"].astype(str) == "BACK_SPACE"
        raw["type"] = raw["type"].map(_DUX_TYPE)             # overwrite in place (no dup column)
        raw["ts"] = pd.to_numeric(raw["timestamp"], errors="coerce")
        raw["x"] = np.clip(pd.to_numeric(raw["x"], errors="coerce").fillna(0.0) / _SCREEN_W, 0, 1)
        raw["y"] = np.clip(pd.to_numeric(raw["y"], errors="coerce").fillna(0.0) / _SCREEN_H, 0, 1)
        # scroll delta rides yPosition; backspace token is BACK_SPACE
        raw["dy"] = np.where(raw["type"] == "scroll",
                             pd.to_numeric(raw["yPosition"], errors="coerce").fillna(0.0), 0.0)
        raw["key"] = np.where(is_bs, "Backspace", np.where(raw["type"] == "key", "a", ""))
        raw["neutral"] = pd.to_numeric(raw["emotion_affectiva_Neutral"], errors="coerce")
        raw = raw.dropna(subset=["ts"])
        for sess, g in raw.groupby("session"):
            g = g.sort_values("ts").reset_index(drop=True)
            windows += _windows_from_session(
                g[["ts", "type", "x", "y", "key", "dy"]], f"dux_{sess}", g["neutral"].to_numpy())
    return windows


def _norm(col) -> str:
    """Normalise a column header for tolerant matching (EmoSurv headers vary: 'key Down' etc.)."""
    return str(col).strip().lower().replace(" ", "").replace("_", "")


def load_emosurv(emosurv_dir: str) -> list[dict]:
    """EmoSurv (IEEE DataPort) keystroke-only. Keyboard features populate; mouse/scroll stay 0.

    Real EmoSurv per-keystroke schema (Fixed/Free Text Typing Dataset.csv, ';'-separated,
    European decimal comma): userId, emotionIndex (H/S/A/C/N), index, keyCode, keyDown, keyUp,
    D1U1, D1U2, D1D2, ... We do NOT use `keyDown` — the CSV export mangles the absolute
    timestamp into 3-sig-fig scientific notation (e.g. `1,58E+12`), destroying ms precision.
    Instead we reconstruct the key-down timeline by cumulatively summing `D1D2` (the down-to-
    down inter-key interval, in ms). `Key Code` -> key; `emotionIndex` -> proxy (N -> 0, else
    1). Requires a free IEEE DataPort account. LICENSE: non-commercial research only, NO
    redistribution — keep the files local (gitignored), never commit them.
    """
    d = Path(emosurv_dir)
    if not d.exists():
        return []
    # the per-keystroke typing files (skip the Frequency / Participants files)
    files = [f for f in d.glob("*.csv") if "typing" in f.name.lower()] or list(d.glob("*.csv"))
    windows = []
    for f in sorted(files):
        try:
            raw = pd.read_csv(f, sep=None, engine="python", dtype=str)   # auto-detect ',' or ';'
        except Exception:
            continue
        cmap = {_norm(c): c for c in raw.columns}
        if not all(k in cmap for k in ("userid", "emotionindex", "keycode", "d1d2")):
            continue  # not the per-keystroke EmoSurv schema (e.g. Frequency file) — skip
        ren = {cmap["userid"]: "userid", cmap["emotionindex"]: "emotion",
               cmap["keycode"]: "keycode", cmap["d1d2"]: "gap"}
        if "index" in cmap:
            ren[cmap["index"]] = "kindex"
        raw = raw.rename(columns=ren)
        # down-to-down interval in ms (handle European decimal comma); cap pauses at 5s
        raw["gap"] = (pd.to_numeric(raw["gap"].astype(str).str.replace(",", ".", regex=False),
                                    errors="coerce").fillna(0.0).clip(lower=0, upper=5_000))
        if "kindex" in raw.columns:
            raw["kindex"] = pd.to_numeric(raw["kindex"], errors="coerce")
        bs = raw["keycode"].astype(str).str.strip().str.lower().isin(
            ["backspace", "back_space", "back", "8"])
        raw["type"] = "key"; raw["x"] = 0.0; raw["y"] = 0.0; raw["dy"] = 0.0  # keystroke-only
        raw["key"] = np.where(bs, "Backspace", "a")
        for (uid, emo), g in raw.groupby(["userid", "emotion"]):
            g = g.sort_values("kindex") if "kindex" in g.columns else g
            g = g.reset_index(drop=True)
            gaps = g["gap"].to_numpy(dtype=float)
            ts = np.concatenate([[0.0], np.cumsum(gaps)[:-1]]) if len(gaps) else np.array([])
            g = g.assign(ts=ts)
            neutral_val = 100.0 if str(emo).strip().upper() in ("N", "NEUTRAL") else 0.0
            windows += _windows_from_session(
                g[["ts", "type", "x", "y", "key", "dy"]], f"emosurv_{uid}",
                np.full(len(g), neutral_val))
    return windows


def load_external_windows(cfg: dict) -> list[dict]:
    """Combine all available public datasets into one proxy-labeled window list."""
    ext = cfg.get("external", {})
    windows = []
    if ext.get("dux_path"):
        windows += load_dux(str(Path(__file__).parent / ext["dux_path"]))
    if ext.get("emosurv_path"):
        windows += load_emosurv(str(Path(__file__).parent / ext["emosurv_path"]))
    return windows
