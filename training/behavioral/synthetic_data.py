"""Synthetic Phase-A-like behavioral data generator (REALISTIC mode).

PURPOSE: validate the training pipeline and produce a believable demo model BEFORE
the real Phase A pilot exists. Each affect class has characteristic mouse/keyboard/
scroll behaviour, but realism knobs (config `synthetic:`) make it messy like real data:

  class_overlap     : blends each class toward the global mean (classes are NOT
                      cleanly separable in reality)
  participant_noise : every participant has a different baseline "style"
  label_noise       : some self-reports are wrong (lazy / mis-clicked)
  lazy_fraction     : some participants mostly just click "Engaged"

NOTE: a model trained on this is only as good as these hand-coded priors. It is NOT a
substitute for real pilot data — it proves the pipeline and gives a believable demo
(~50-65%, the literature range for behavioural-only affect), not a real result.
"""

import numpy as np
import pandas as pd

LABELS = ["Engaged", "Bored", "Confused", "Frustrated"]
WINDOW_MS = 30_000

# per-class behaviour priors (per 1-second bin) — the "clean" centres
PROFILES = {
    "Engaged":    dict(moves=7,  jitter=0.012, clicks=0.05, keys=2.0, bs=0.05, irreg=0.02, scroll=0.3, flip=0.05),
    "Bored":      dict(moves=9,  jitter=0.10,  clicks=0.15, keys=0.3, bs=0.10, irreg=0.06, scroll=0.5, flip=0.10),
    "Confused":   dict(moves=7,  jitter=0.045, clicks=0.10, keys=1.2, bs=0.28, irreg=0.10, scroll=0.9, flip=0.55),
    "Frustrated": dict(moves=10, jitter=0.07,  clicks=0.55, keys=1.5, bs=0.33, irreg=0.18, scroll=0.4, flip=0.10),
}
NUMERIC = ["moves", "jitter", "clicks", "keys", "bs", "irreg", "scroll", "flip"]
WINDOWS_PER_PARTICIPANT = {"Engaged": 18, "Bored": 8, "Confused": 8, "Frustrated": 6}


def _window_events(eff: dict, rng: np.random.Generator, spd: float) -> pd.DataFrame:
    rows = []
    focus = rng.uniform(0.4, 0.6, size=2)
    pos = focus.copy()
    for b in range(WINDOW_MS // 1000):
        t0 = b * 1000
        nm = max(0, int(rng.normal(eff["moves"], 2.0)))   # more within-class variance
        for i in range(nm):
            if eff["focus"]:
                pos = np.clip(focus + rng.normal(0, eff["jitter"] * spd, 2), 0, 1)
            else:
                big = 3.0 if rng.random() < 0.15 else 1.0   # occasional jerk
                pos = np.clip(pos + rng.normal(0, eff["jitter"] * spd * big, 2), 0, 1)
            rows.append((t0 + int(i * 1000 / max(nm, 1)), "move", pos[0], pos[1], "", 0.0))
        for _ in range(rng.poisson(max(eff["clicks"], 0) * spd)):
            rows.append((t0 + rng.integers(0, 1000), "click", 0.0, 0.0, "", 0.0))
        nk = rng.poisson(max(eff["keys"], 0) * spd)
        if nk:
            kts = np.sort(rng.integers(0, 1000, nk)) if eff["irreg"] > 0.1 \
                else np.linspace(0, 1000, nk, endpoint=False).astype(int)
            for kt in kts:
                key = "Backspace" if rng.random() < eff["bs"] else "a"
                rows.append((t0 + int(kt), "key", 0.0, 0.0, key, 0.0))
        d = 1.0
        for _ in range(rng.poisson(max(eff["scroll"], 0) * spd)):
            if rng.random() < eff["flip"]:
                d = -d
            rows.append((t0 + rng.integers(0, 1000), "scroll", 0.0, 0.0, "", d * rng.uniform(20, 60)))
    return pd.DataFrame(rows, columns=["ts", "type", "x", "y", "key", "dy"])


def generate_dataset(n_participants: int = 15, seed: int = 42, realism: dict | None = None):
    """Return list of {participant, label, label_idx, true_label, events}.

    `label` is the (noisy) self-report used for training; `true_label` is the latent
    behaviour class (kept only for analysis).
    """
    realism = realism or {}
    overlap = float(realism.get("class_overlap", 0.55))
    pnoise = float(realism.get("participant_noise", 0.45))
    lnoise = float(realism.get("label_noise", 0.15))
    lazy_frac = float(realism.get("lazy_fraction", 0.13))

    mean_prof = {k: float(np.mean([PROFILES[c][k] for c in LABELS])) for k in NUMERIC}
    rng = np.random.default_rng(seed)
    windows = []
    for pid in range(n_participants):
        spd = rng.uniform(0.7, 1.4)
        pmult = {k: float(np.exp(rng.normal(0, pnoise))) for k in NUMERIC}   # participant style
        lazy = rng.random() < lazy_frac
        for cls, n in WINDOWS_PER_PARTICIPANT.items():
            for _ in range(n):
                eff = {}
                for k in NUMERIC:
                    base = (1 - overlap) * PROFILES[cls][k] + overlap * mean_prof[k]
                    eff[k] = max(0.0, base * pmult[k] * float(np.exp(rng.normal(0, 0.25))))
                eff["focus"] = (cls == "Engaged") and (rng.random() > overlap * 0.8)
                ev = _window_events(eff, rng, spd)

                # noisy self-report label
                if lazy and rng.random() < 0.7:
                    obs = "Engaged"
                elif rng.random() < lnoise:
                    obs = LABELS[int(rng.integers(0, 4))]
                else:
                    obs = cls
                windows.append(dict(participant=f"P{pid:02d}", label=obs,
                                    label_idx=LABELS.index(obs), true_label=cls, events=ev))
    rng.shuffle(windows)
    return windows


if __name__ == "__main__":
    ws = generate_dataset(3, realism=dict(class_overlap=0.55, label_noise=0.15))
    print(f"generated {len(ws)} windows; label-noise example:",
          sum(w["label"] != w["true_label"] for w in ws), "windows have noisy labels")
