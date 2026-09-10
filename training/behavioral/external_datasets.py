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

import warnings

import numpy as np
import pandas as pd

PROXY_LABELS = ["neutral", "emotion"]     # index 0, 1

# EmoSurv's OWN five induced-emotion labels, recovered rather than collapsed.
#
# These are real human affect labels from 83 participants, and until now `load_emosurv` threw them
# away to build the binary arousal proxy — which meant the only genuinely subject-independent
# affect data in the project was being reduced to one bit. The proxy is still produced (encoder
# pretraining needs it), but the five-class index now rides alongside it.
#
# READ THIS BEFORE REPORTING ANYTHING FROM IT. These are BASIC emotions from a typing task, NOT
# the four learning-centred states (bored / confused / engaged / frustrated) the platform targets.
# A result here supports the PREMISE that keystroke dynamics carry affective signal. It is not an
# answer to RQ1, and the two must never be conflated in the write-up.
EMOSURV_LABELS = ["Neutral", "Happy", "Sad", "Angry", "Calm"]
_EMOSURV_CODE = {
    "N": 0, "NEUTRAL": 0,
    "H": 1, "HAPPY": 1,
    "S": 2, "SAD": 2,
    "A": 3, "ANGRY": 3, "ANGER": 3,
    "C": 4, "CALM": 4,
}
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
        is_bs = _is_backspace(raw["key"])
        raw["type"] = raw["type"].map(_DUX_TYPE)             # overwrite in place (no dup column)
        raw["ts"] = pd.to_numeric(raw["timestamp"], errors="coerce")
        raw["x"] = np.clip(pd.to_numeric(raw["x"], errors="coerce").fillna(0.0) / _SCREEN_W, 0, 1)
        raw["y"] = np.clip(pd.to_numeric(raw["y"], errors="coerce").fillna(0.0) / _SCREEN_H, 0, 1)
        # `yPosition` is an ABSOLUTE scroll offset, not a delta — verified on v0.csv, where the
        # scroll sequence climbs monotonically and then resets to 0 (e.g. 1, 126, 127, ... 251, 0).
        # Feeding it straight into `dy` made `scroll_velocity_mean` average positions (~103 px) and
        # pinned `scroll_direction_changes` near zero, since an absolute offset is never negative.
        # The delta is therefore the within-session diff over SCROLL ROWS ONLY, computed after the
        # per-session sort below.
        raw["ypos"] = pd.to_numeric(raw["yPosition"], errors="coerce")
        raw["key"] = np.where(is_bs, "Backspace", np.where(raw["type"] == "key", "a", ""))
        raw["neutral"] = pd.to_numeric(raw["emotion_affectiva_Neutral"], errors="coerce")
        raw = raw.dropna(subset=["ts"])
        for sess, g in raw.groupby("session"):
            g = g.sort_values("ts").reset_index(drop=True)
            g["dy"] = 0.0
            scroll_rows = g.index[g["type"] == "scroll"]
            if len(scroll_rows):
                # .diff() leaves the session's first scroll event as NaN -> 0.0 (no prior offset).
                g.loc[scroll_rows, "dy"] = g.loc[scroll_rows, "ypos"].diff().fillna(0.0).to_numpy()
            # Namespaced by file for the same reason as `load_dux_confusion`: v0 and v1 both number
            # sessions from 1, so a bare session id fuses two different people into one group.
            windows += _windows_from_session(
                g[["ts", "type", "x", "y", "key", "dy"]], f"dux_{f.stem}_{sess}",
                g["neutral"].to_numpy())
    return windows


# The twelve AFFDEX/Affectiva channels DUX logs per row, and the human-annotation block that
# parallels them. `emotion_manual_*` is the only INDEPENDENT ground truth anywhere in the corpora
# held locally: it is a human judgement, not another model's output, so a behavioural model scored
# against it is not being trained to imitate a facial classifier.
DUX_AFFECTIVA = ["Anger", "Confusion", "Contempt", "Disgust", "Engagement", "Fear", "Joy",
                 "Neutral", "Sadness", "Sentimentality", "Surprise", "Valence"]

# Human-annotation density, measured per file. v1 is the one to use: it is the release with the
# emotional triggers ENABLED, and it carries ~9x the confusion annotation of v0.
#
#   channel          v0 (10 sessions)   v1 (36 sessions)
#   Confusion              4,202             38,183      <- the only densely annotated state
#   Joy                    1,104             11,676
#   Anger                    398              7,099      <- closest AFFDEX channel to frustration
#   Judgement                977              3,471
#   Surprise                   2              3,709
#   Fear                       0              1,344
#   Contempt / Disgust      96 / 114        763 / 418
#   Engagement                 0                489      <- present in v1, but far too sparse
#   Sadness                   13                234
#   Sentimentality             0                 42
#   Neutral                    0                  0      <- never annotated in either file
#
# Two consequences for the write-up. `emotion_manual_Engagement` is identically zero in v0 and
# covers 489 rows of 590,738 in v1, so DUX cannot supply a usable human ENGAGEMENT label either way
# — engagement has to come from the platform or from DAiSEE. And `Neutral` is never annotated at
# all, so the negative class is "not annotated as confused", which is an ABSENCE of annotation
# rather than a positive judgement of calm. That asymmetry belongs in the limitations.
DUX_MANUAL_CONFUSION = "emotion_manual_Confusion"


def _bin_per_second(sub: pd.DataFrame, aff_cols: list[str], t0: int, w: int,
                    window_ms: int) -> np.ndarray:
    """The window's AFFDEX rows as (n_bins, n_channels), one bin per second.

    This exists so the facial arm can be given the SAME treatment as the behavioural one. The
    behavioural extractor emits (30, 16) and `aggregate` then takes five statistics per channel;
    reducing the facial channels to a single per-window mean instead — as `affectiva` does — hands
    the behavioural arm every within-window dynamic and denies the facial arm all of them. A
    modality comparison built on that asymmetry is confounded with temporal aggregation, so the
    binned form is offered alongside and `train_dux_confusion.py --facial-aggregate` uses it.

    Seconds with no rows are forward-filled, then back-filled: an expression channel is a
    continuous quantity that persists between samples, so carrying the last observation is the
    honest interpolation. A window with no usable rows at all returns zeros.
    """
    n_bins = max(1, window_ms // 1000)
    A = sub[aff_cols].to_numpy(dtype=np.float64)
    off = (sub["ts"].to_numpy() - t0) - w * window_ms
    b = np.clip((off // 1000).astype(int), 0, n_bins - 1)

    seq = np.full((n_bins, len(aff_cols)), np.nan, dtype=np.float64)
    for i in range(n_bins):
        m = b == i
        if m.any():
            with warnings.catch_warnings():           # all-NaN channel in a bin is expected
                warnings.simplefilter("ignore", RuntimeWarning)
                seq[i] = np.nanmean(A[m], axis=0)

    df = pd.DataFrame(seq).ffill().bfill()
    return df.to_numpy(dtype=np.float64) if not df.isna().all().all() else np.zeros_like(seq)


def load_dux_confusion(dux_dir: str, threshold: float = 1.0,
                       window_ms: int = WINDOW_MS) -> list[dict]:
    """DUX windows labelled by the HUMAN Confusion annotation, with the facial channels alongside.

    Returns windows carrying three things measured on the SAME 30 s of the SAME session:
      * `events`     — mouse/key/scroll, for the behavioural arm (the shared extractor consumes it)
      * `affectiva`  — the 12 AFFDEX channels, bin-averaged, for the facial arm
      * `label`      — 1 if a human annotated any Confusion in the window, else 0

    That triple is what makes a genuine unimodal-vs-fused ablation possible here: both arms predict
    an INDEPENDENT human label, so neither is fitted to the other's output. Scoring the behavioural
    arm against `emotion_affectiva_*` instead would be distillation dressed up as fusion.

    `threshold` is on the window's MAXIMUM annotation intensity. The scale is discrete
    (0/16.5/33/50/66/100); >=1 takes any annotated confusion, and because 16.5 never occurs as a
    window maximum in v0, >=1 and >=33 give the identical 46 positives.

    Participant ids are namespaced by FILE (`dux_v0_3`, not `dux_3`). v0 and v1 both number their
    sessions from 1 and they are different recordings, so a bare session id would silently fuse two
    unrelated people into one participant group — which would put the same "participant" on both
    sides of a leave-one-participant-out split and inflate every score. Namespacing is the only
    thing standing between adding v1 and a corrupted result.
    """
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
        # Annotation and facial rows must be read BEFORE the event-type filter drops rows: the
        # label is carried on every row of the session, including types we do not model.
        raw["conf"] = pd.to_numeric(raw[DUX_MANUAL_CONFUSION], errors="coerce").fillna(0.0)
        for c in aff_cols:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw["ts"] = pd.to_numeric(raw["timestamp"], errors="coerce")
        raw = raw.dropna(subset=["ts"])

        for sess, g_all in raw.groupby("session"):
            g_all = g_all.sort_values("ts")
            t0 = int(g_all["ts"].min())
            widx_all = ((g_all["ts"] - t0) // window_ms).astype(int)
            # Per-window label + facial vector from ALL rows, so filtering events cannot move a
            # window's label or blank its facial channels.
            lab = {int(w): float(sub["conf"].max() >= threshold)
                   for w, sub in g_all.groupby(widx_all)}
            facial = {int(w): sub[aff_cols].mean().to_numpy(dtype=np.float64)
                      for w, sub in g_all.groupby(widx_all)}
            facial_seq = {int(w): _bin_per_second(sub, aff_cols, t0, int(w), window_ms)
                          for w, sub in g_all.groupby(widx_all)}

            g = g_all[g_all["type"].isin(_DUX_TYPE)].copy()
            if g.empty:
                continue
            is_bs = _is_backspace(g["key"])
            g["type"] = g["type"].map(_DUX_TYPE)
            g["x"] = np.clip(pd.to_numeric(g["x"], errors="coerce").fillna(0.0) / _SCREEN_W, 0, 1)
            g["y"] = np.clip(pd.to_numeric(g["y"], errors="coerce").fillna(0.0) / _SCREEN_H, 0, 1)
            g["ypos"] = pd.to_numeric(g["yPosition"], errors="coerce")
            g["key"] = np.where(is_bs, "Backspace", np.where(g["type"] == "key", "a", ""))
            g = g.reset_index(drop=True)
            g["dy"] = 0.0
            scroll_rows = g.index[g["type"] == "scroll"]
            if len(scroll_rows):
                g.loc[scroll_rows, "dy"] = g.loc[scroll_rows, "ypos"].diff().fillna(0.0).to_numpy()

            widx = ((g["ts"] - t0) // window_ms).astype(int)
            for w, sub in g.groupby(widx):
                if len(sub) < _MIN_EVENTS:
                    continue
                ev = sub[["ts", "type", "x", "y", "key", "dy"]].copy()
                ev["ts"] = (ev["ts"] - t0) - int(w) * window_ms       # -> [0, window_ms)
                windows.append({
                    "participant": f"dux_{f.stem}_{sess}",
                    "window_index": int(w),
                    "label": int(lab.get(int(w), 0.0)),
                    "affectiva": facial.get(int(w)),
                    "affectiva_seq": facial_seq.get(int(w)),
                    "events": ev.reset_index(drop=True),
                })
    return windows


def _norm(col) -> str:
    """Normalise a column header for tolerant matching (EmoSurv headers vary: 'key Down' etc.)."""
    return str(col).strip().lower().replace(" ", "").replace("_", "")


# EmoSurv writes backspace as the LITERAL two-character text `\b` — a backslash followed by 'b',
# not the 0x08 control character. Verified against the shipped files: 1708 occurrences in
# `Fixed Text Typing Dataset.csv` and 2939 in `Free Text Typing Dataset.csv`, i.e. 6.2% of all
# keystrokes.
#
# The previous matcher (`["backspace", "back_space", "back", "8"]`) therefore scored ZERO real
# hits, leaving `backspace_pct` structurally 0.0 across the entire corpus — and worse, its `"8"`
# entry matched the DIGIT 8 being typed, so its only 4 "hits" were false positives. Correction
# rate is among the most discriminative keystroke-affect features, so this silently discarded the
# single most useful signal EmoSurv provides.
_BACKSPACE_TOKENS = frozenset({
    "\\b",          # EmoSurv: literal backslash-b (the real one)
    "\x08",         # genuine ASCII BS, in case another corpus emits the control char
    "back_space",   # DUX: verified 565 occurrences in v0.csv
    "backspace",
})


def _is_backspace(keycode_series) -> "pd.Series":
    """Boolean mask of backspace keystrokes. Deliberately does NOT match the digit `8`."""
    s = keycode_series.astype(str).str.strip()
    return s.isin(_BACKSPACE_TOKENS) | s.str.lower().isin(_BACKSPACE_TOKENS)


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
        bs = _is_backspace(raw["keycode"])
        raw["type"] = "key"; raw["x"] = 0.0; raw["y"] = 0.0; raw["dy"] = 0.0  # keystroke-only
        raw["key"] = np.where(bs, "Backspace", "a")
        for (uid, emo), g in raw.groupby(["userid", "emotion"]):
            g = g.sort_values("kindex") if "kindex" in g.columns else g
            g = g.reset_index(drop=True)
            gaps = g["gap"].to_numpy(dtype=float)
            ts = np.concatenate([[0.0], np.cumsum(gaps)[:-1]]) if len(gaps) else np.array([])
            g = g.assign(ts=ts)
            token = str(emo).strip().upper()
            emo_idx = _EMOSURV_CODE.get(token)
            if emo_idx is None:
                continue                      # unknown label: better dropped than guessed
            neutral_val = 100.0 if emo_idx == 0 else 0.0
            batch = _windows_from_session(
                g[["ts", "type", "x", "y", "key", "dy"]], f"emosurv_{uid}",
                np.full(len(g), neutral_val))
            for w in batch:
                # The five-class label, carried alongside the binary proxy. `session_group` is the
                # (participant, emotion) passage these windows were sliced from: windows inside one
                # passage are consecutive 30s slices of the SAME continuous typing and are not
                # independent, so anything grouping at window level would leak.
                w["emotion_index"] = emo_idx
                w["emotion_label"] = EMOSURV_LABELS[emo_idx]
                w["session_group"] = f"emosurv_{uid}::{EMOSURV_LABELS[emo_idx]}"
            windows += batch
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
