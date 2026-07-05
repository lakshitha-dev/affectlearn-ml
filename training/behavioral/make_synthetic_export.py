"""Generate a synthetic Phase-A research export to validate the REAL training path.

Runs the synthetic generator, computes the backend's aggregate feature window for each
window (the SAME extract_features the platform persists at serve time), and emits a
research-export JSON — behavioral_affect_detected {features} + self_report {affect}
events — in the exact shape `phase_a.load_phase_a_windows` consumes.

This exercises the real Phase-A pipeline (export -> loader -> trainer) end-to-end BEFORE
real pilot data exists (guide §2 / Step 2), and is the fixture used to prove the platform
can train. It is NOT the real result — that comes from actual Phase A data.

Usage:  python make_synthetic_export.py --out ../../data/phase_a/export.json --participants 15
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_data import generate_dataset          # noqa: E402
from feature_engineering import extract_features      # noqa: E402


def build_export(participants: int, seed: int) -> dict:
    windows = generate_dataset(participants, seed, {})
    items, t_by_pid = [], {}
    for w in windows:
        p = w["participant"]
        t = t_by_pid.get(p, 0) + 30_000            # one 30 s window per step, per participant
        t_by_pid[p] = t
        feats = extract_features(w["events"], 0).round(6).tolist()   # what the backend persists
        items.append({"eventType": "behavioral_affect_detected", "sessionId": p,
                      "learnerId": p, "timestamp": t, "payload": {"features": feats}})
        # self-report 1 s after the window it describes -> nearest-preceding join maps it back
        items.append({"eventType": "self_report", "sessionId": p, "timestamp": t + 1000,
                      "payload": {"affect": w["label"].lower(), "skipped": False, "omitted": False}})
    return {"items": items, "_note": "SYNTHETIC pipeline-validation export, not real Phase A data"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "../../data/phase_a/export.json"))
    ap.add_argument("--participants", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    export = build_export(args.participants, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(export, open(out, "w"))
    n_win = sum(1 for e in export["items"] if e["eventType"] == "behavioral_affect_detected")
    print(f"wrote {len(export['items'])} events ({n_win} windows) -> {out.resolve()}")


if __name__ == "__main__":
    main()
