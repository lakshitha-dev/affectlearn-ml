"""Synthetic Phase-A-like behavioral data generator.

PURPOSE: validate the training pipeline and produce a demo model BEFORE the real
Phase A pilot exists. Each simulated participant has a baseline "speed" factor;
each affect class has characteristic mouse/keyboard/scroll behaviour:

  Engaged     : focused cursor (low entropy/velocity), steady typing, some idle (reading)
  Bored       : wandering cursor (high entropy/velocity), little typing, navigation clicks
  Confused    : lots of scroll direction changes (re-reading), long hovers, high backspace
  Frustrated  : jerky high-variance movement, rage-clicks, irregular typing, high backspace

NOTE: a model trained on this is only as good as these hand-coded priors. It is NOT a
substitute for real pilot data — it proves the pipeline and gives a working demo model.
"""

import numpy as np
import pandas as pd

LABELS = ["Engaged", "Bored", "Confused", "Frustrated"]
WINDOW_MS = 30_000
HZ = 10                      # 10 moves/sec nominal

# per-class behaviour priors (per 1-second bin)
PROFILES = {
    "Engaged":    dict(moves=7, jitter=0.012, focus=True,  clicks=0.05, keys=2.0, bs=0.05, irreg=0.02, scroll=0.3, flip=0.05),
    "Bored":      dict(moves=9, jitter=0.10,  focus=False, clicks=0.15, keys=0.3, bs=0.10, irreg=0.06, scroll=0.5, flip=0.10),
    "Confused":   dict(moves=7, jitter=0.045, focus=False, clicks=0.10, keys=1.2, bs=0.28, irreg=0.10, scroll=0.9, flip=0.55),
    "Frustrated": dict(moves=10, jitter=0.07, focus=False, clicks=0.55, keys=1.5, bs=0.33, irreg=0.18, scroll=0.4, flip=0.10),
}
# rough Phase-A-like imbalance (engaged dominates)
WINDOWS_PER_PARTICIPANT = {"Engaged": 18, "Bored": 8, "Confused": 8, "Frustrated": 6}


def _window_events(cls: str, rng: np.random.Generator, spd: float) -> pd.DataFrame:
    p = PROFILES[cls]
    rows = []
    focus = rng.uniform(0.4, 0.6, size=2)
    pos = focus.copy()
    n_bins = WINDOW_MS // 1000
    for b in range(n_bins):
        t0 = b * 1000
        nm = max(0, int(rng.normal(p["moves"], 1.5)))
        for i in range(nm):
            if p["focus"]:
                pos = np.clip(focus + rng.normal(0, p["jitter"] * spd, 2), 0, 1)
            else:
                step = p["jitter"] * spd * (3.0 if (cls == "Frustrated" and rng.random() < 0.3) else 1.0)
                pos = np.clip(pos + rng.normal(0, step, 2), 0, 1)
            rows.append((t0 + int(i * 1000 / max(nm, 1)), "move", pos[0], pos[1], "", 0.0))
        for _ in range(rng.poisson(p["clicks"] * spd)):
            rows.append((t0 + rng.integers(0, 1000), "click", 0.0, 0.0, "", 0.0))
        nk = rng.poisson(p["keys"] * spd)
        if nk:
            if p["irreg"] > 0.1:                       # irregular timing
                kts = np.sort(rng.integers(0, 1000, size=nk))
            else:                                       # steady timing
                kts = np.linspace(0, 1000, nk, endpoint=False).astype(int)
            for kt in kts:
                key = "Backspace" if rng.random() < p["bs"] else "a"
                rows.append((t0 + int(kt), "key", 0.0, 0.0, key, 0.0))
        ns = rng.poisson(p["scroll"] * spd)
        d = 1.0
        for _ in range(ns):
            if rng.random() < p["flip"]:
                d = -d
            rows.append((t0 + rng.integers(0, 1000), "scroll", 0.0, 0.0, "", d * rng.uniform(20, 60)))
    return pd.DataFrame(rows, columns=["ts", "type", "x", "y", "key", "dy"])


def generate_dataset(n_participants: int = 15, seed: int = 42):
    """Return list of dicts: {participant, label, label_idx, events(DataFrame)}."""
    rng = np.random.default_rng(seed)
    windows = []
    for pid in range(n_participants):
        spd = rng.uniform(0.7, 1.4)                    # per-participant baseline speed
        for cls, n in WINDOWS_PER_PARTICIPANT.items():
            for _ in range(n):
                windows.append(dict(
                    participant=f"P{pid:02d}", label=cls, label_idx=LABELS.index(cls),
                    events=_window_events(cls, rng, spd),
                ))
    rng.shuffle(windows)
    return windows


if __name__ == "__main__":
    ws = generate_dataset(3)
    print(f"generated {len(ws)} windows from 3 participants; sample events:")
    print(ws[0]["label"], ws[0]["events"].head())
