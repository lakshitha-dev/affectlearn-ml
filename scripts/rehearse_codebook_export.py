"""Rehearse the export -> codebook-label -> train chain before any real data exists.

Simulates one learner completing the whole seeded curriculum in order, emitting events in the
EXACT shape the research API returns, then runs the real codebook labeller over them. This is
the offline half of the 12 Aug dry run: it proves the loader, the section attribution, and the
class arithmetic work end to end, and it prints the yield to expect from a genuine pass.

The behavioural FEATURES here are noise — this rehearses plumbing and yield, not learnability.
Never train a reported model on this output.

Usage:
    python scripts/rehearse_codebook_export.py
    python scripts/rehearse_codebook_export.py --agreement 0.6 --out data/phase_a/rehearsal.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "behavioral"))
from codebook_labels import load_codebook_windows, summarise  # noqa: E402
# Width and schema version come from the extractor, never hardcoded: a stale rehearsal export is
# indistinguishable from real data until it hits the model, and mixing widths is unrecoverable.
from feature_engineering import FEATURE_SCHEMA_VERSION, N_FEATURES  # noqa: E402

N_BINS = 30
WINDOW_MS = 30_000
T0 = 1_755_000_000_000  # fixed epoch so runs are reproducible


def build_events(codebook_path: Path, agreement: float, seed: int):
    rng = random.Random(seed)
    book = json.loads(codebook_path.read_text(encoding="utf-8"))
    sections = sorted(book["sections"].values(), key=lambda r: r["canonical_order"])

    other = {
        "engaged": ["bored", "confused", "frustrated"],
        "bored": ["engaged", "confused", "frustrated"],
        "confused": ["engaged", "bored", "frustrated"],
        "frustrated": ["engaged", "bored", "confused"],
        "neutral": ["engaged", "bored"],
    }

    items: list[dict] = []
    sections_by_id: dict[str, dict] = {}
    t = T0
    seq = 0

    for i, sec in enumerate(sections):
        sid = f"sec-{i:03d}"
        sections_by_id[sid] = {"title": sec["section"], "course": sec["course"]}
        minutes = sec.get("estimated_duration_minutes") or 4
        n_windows = max(1, int(minutes * 60_000 / WINDOW_MS))

        for _ in range(n_windows):
            items.append(
                {
                    "eventType": "behavioral_affect_detected",
                    "learnerId": "bootstrap-subject",
                    "sessionId": f"sess-{i // 12}",  # ~4 sittings across the curriculum
                    "timestamp": t,
                    "sequenceNumber": seq,
                    "payload": {
                        "features": [
                            [round(rng.gauss(0, 1), 6) for _ in range(N_FEATURES)]
                            for _ in range(N_BINS)
                        ],
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    },
                }
            )
            seq += 1
            t += WINDOW_MS

        # Section marked complete promptly on finishing it (the protocol rule).
        items.append(
            {
                "eventType": "section_completed",
                "learnerId": "bootstrap-subject",
                "sessionId": None,
                "timestamp": t,
                "payload": {"section_id": sid, "created": True},
            }
        )
        t += 1_000

        # Self-report prompt: SECTIONS_PER_PROMPT=1, omission disabled.
        designed = sec["affect"]
        reported = designed if rng.random() < agreement else rng.choice(other[designed])
        items.append(
            {
                "eventType": "self_report",
                "learnerId": "bootstrap-subject",
                "sessionId": f"sess-{i // 12}",
                "timestamp": t,
                "payload": {
                    "affect": reported,
                    "skipped": False,
                    "omitted": False,
                    "prompt_index": i,
                },
            }
        )
        t += 5_000

    return {"items": items, "sections": sections_by_id}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codebook", default="data/codebook.json")
    ap.add_argument("--out", default="data/phase_a/rehearsal.json")
    ap.add_argument(
        "--agreement",
        type=float,
        default=0.7,
        help="Probability the simulated self-report matches the designed affect.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    codebook_path = Path(args.codebook)
    if not codebook_path.exists():
        print(
            f"ERROR: {codebook_path} not found — run scripts/generate_codebook.py first",
            file=sys.stderr,
        )
        return 2

    export = build_events(codebook_path, args.agreement, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(export, indent=1), encoding="utf-8")

    book = json.loads(codebook_path.read_text(encoding="utf-8"))
    codebook = {t: r["affect"] for t, r in book["sections"].items()}

    n_win = sum(
        1 for e in export["items"] if e["eventType"] == "behavioral_affect_detected"
    )
    print(f"Simulated a full curriculum pass: {n_win} behavioural windows\n")

    print(f"--- strict (self-report must agree, p={args.agreement}) ---")
    strict = load_codebook_windows(export, codebook)
    print(summarise(strict))

    print(f"\n--- design labels only (no manipulation check) ---")
    loose = load_codebook_windows(export, codebook, require_agreement=False)
    print(summarise(loose))

    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
