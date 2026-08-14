"""Generate the machine-readable affect codebook from the course source.

The source of truth for a section's intended affect is the `# affect: <state>` comment
directly above each `section(...)` call in
`affectlearn/backend/app/db/course_content/*.py`. `deploy/azure/AFFECT_SECTION_CODEBOOK.md`
is the human-readable mirror of the same facts.

Sections are joined to their intended affect BY TITLE (section ids are UUIDs minted at seed
time, so they are not stable across databases). This script therefore also verifies that
every section title is unique across all courses — if it were not, a title join would be
ambiguous and silently mislabel windows.

Method: the authoritative section list comes from importing the course builders and walking
the ORM objects; the affect values come from a regex over the same source files. The two are
then joined and cross-checked, so a section that loses its `# affect:` comment (or a comment
that drifts away from its section) fails loudly instead of vanishing from the codebook.

Usage (from the repo root of affectlearn-ml):
    python scripts/generate_codebook.py
    python scripts/generate_codebook.py --backend ../affectlearn/backend --out data/codebook.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Course modules to read, in the order a participant meets them.
MODULES = ["warmup", "building", "foundations", "multiagent"]

VALID_AFFECTS = ("engaged", "bored", "confused", "frustrated", "neutral")

# `# affect: <state>` (optionally followed by a parenthetical such as "(baseline)"), then the
# `section(` call and its first double-quoted argument — the section title. The title may sit
# on the same line as `section(` or on the next one.
SECTION_RE = re.compile(
    r"#\s*affect:\s*(?P<affect>" + "|".join(VALID_AFFECTS) + r")\b[^\n]*\n"
    r"\s*section\(\s*\n?\s*\"(?P<title>(?:[^\"\\]|\\.)*)\"",
    re.MULTILINE,
)


def walk_sections(build_fn) -> list[dict]:
    """Walk a built Course ORM object -> ordered section records."""
    course = build_fn()
    out: list[dict] = []
    for mod in course.modules:
        for lsn in mod.lessons:
            for sec in lsn.sections:
                out.append(
                    {
                        "course": course.title,
                        "module": mod.title,
                        "lesson": lsn.title,
                        "section": sec.title,
                        "estimated_duration_minutes": sec.estimated_duration_minutes,
                    }
                )
    return out


def parse_affects(source: Path) -> list[tuple[str, str]]:
    """Regex the `# affect:` markers out of a course source file -> [(title, affect)]."""
    text = source.read_text(encoding="utf-8")
    return [(m.group("title"), m.group("affect")) for m in SECTION_RE.finditer(text)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backend",
        default="../affectlearn/backend",
        help="Path to the affectlearn backend package root (contains `app/`).",
    )
    ap.add_argument("--out", default="data/codebook.json", help="Output JSON path.")
    args = ap.parse_args()

    backend = Path(args.backend).resolve()
    content_dir = backend / "app" / "db" / "course_content"
    if not content_dir.is_dir():
        print(f"ERROR: course content not found at {content_dir}", file=sys.stderr)
        return 2

    # Import the builders from the backend package (needs the backend root on sys.path).
    sys.path.insert(0, str(backend))
    import importlib

    records: list[dict] = []
    affect_pairs: list[tuple[str, str]] = []
    for name in MODULES:
        mod = importlib.import_module(f"app.db.course_content.{name}")
        records.extend(walk_sections(mod.build))
        affect_pairs.extend(parse_affects(content_dir / f"{name}.py"))

    # --- Cross-checks -------------------------------------------------------------------
    errors: list[str] = []

    titles = [r["section"] for r in records]
    dupes = [t for t, n in Counter(titles).items() if n > 1]
    if dupes:
        errors.append(
            "Section titles are NOT unique across courses, so a title join would be "
            f"ambiguous: {dupes}. Give them distinct titles, or switch the loader to a "
            "(course, section) composite key."
        )

    affect_by_title = dict(affect_pairs)
    if len(affect_by_title) != len(affect_pairs):
        adupes = [t for t, n in Counter(t for t, _ in affect_pairs).items() if n > 1]
        errors.append(f"Duplicate `# affect:` markers for the same title: {adupes}")

    missing = [t for t in titles if t not in affect_by_title]
    if missing:
        errors.append(f"Sections with no `# affect:` marker: {missing}")

    orphan = [t for t in affect_by_title if t not in set(titles)]
    if orphan:
        errors.append(
            f"`# affect:` markers whose title matches no section (comment drifted away "
            f"from its section, or the title changed): {orphan}"
        )

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # --- Emit ---------------------------------------------------------------------------
    for i, r in enumerate(records):
        r["affect"] = affect_by_title[r["section"]]
        r["canonical_order"] = i  # serial position across the whole seeded curriculum

    dist = Counter(r["affect"] for r in records)
    codebook = {
        "generated_from": "backend/app/db/course_content/*.py (`# affect:` comments)",
        "join_key": "section_title",
        "valid_affects": list(VALID_AFFECTS),
        "model_classes": ["engaged", "bored", "confused", "frustrated"],
        "note": (
            "`neutral` is the warm-up baseline and is NOT a model class. Windows in neutral "
            "sections are the per-person resting baseline, not training labels."
        ),
        "counts": dict(sorted(dist.items(), key=lambda kv: -kv[1])),
        "n_sections": len(records),
        "sections": {r["section"]: r for r in records},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(codebook, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out} — {len(records)} sections")
    for affect, n in codebook["counts"].items():
        mins = sum(
            r["estimated_duration_minutes"] or 0 for r in records if r["affect"] == affect
        )
        print(f"  {affect:12s} {n:3d} sections  ~{mins:3d} min  ~{mins * 2:3d} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
