"""Codebook labelling: research export -> section-labelled feature windows.

An ALTERNATIVE to `phase_a.load_phase_a_windows`, not a replacement — both remain importable
so either labelling scheme can be reported.

WHY THIS EXISTS
---------------
`phase_a.py` labels windows from self-reports only: one report labels the single window that
preceded it. On the live instrument (a prompt every N section completions, a fraction randomly
omitted, `neutral`/skips discarded) a full pass of the curriculum produces ~160 behavioural
windows but well under a dozen labels. That is unusable for a single-subject bootstrap, where
the content supports exactly one genuine pass.

The course content was built as an elicitation instrument: every section is designed to induce
a specific affect (bored sections stripped of all interactivity, confused sections carrying a
resolvable contradiction, frustrated sections gated behind tricky-distractor quizzes), recorded
per section in `AFFECT_SECTION_CODEBOOK.md` / the `# affect:` comments. This module uses that
DESIGNED affect as the label for every window inside a section, and demotes the self-report to
a per-section MANIPULATION CHECK: a section's windows are kept only if the learner's own report
for that section agrees with the design.

The labels are therefore WEAK (design-assigned), not self-reported. That is a real limitation
and must be stated wherever these results are reported. In exchange, per-class agreement rates
(`summarise`) become a reportable validation of the elicitation design in their own right.

SECTION ATTRIBUTION
-------------------
Windows carry a timestamp; sections must be reconstructed. Two facts constrain how:

  * `section_started` is NOT emitted when the learner begins a section. Both `section_started`
    and `section_completed` are emitted together at completion time
    (`routes/section_progress.py:124-133`), so `section_started` cannot bound a section's start.
  * `payload.time_spent_seconds` is always null — the lesson page posts only `{sectionId}`
    (`use-progress.ts:97`), so the optional duration is never populated.

What remains reliable is the ORDER and TIME of completions. Sections are therefore bounded by
consecutive completions: a section's interval is `(previous completion, its own completion]`.
This is exact only if each section is marked complete promptly, on finishing it, before moving
on — a protocol requirement for collection, not an assumption the data can verify. Windows
before the first completion of a learner's day are assigned to that first section, bounded by
`max_lead_ms` so an idle gap before starting does not absorb unrelated windows.

Section events carry `learner_id` but `session_id` is null on them, so attribution is keyed by
learner, not session; window timestamps are matched into the learner's completion timeline.

SPLITTING
---------
`participant` is set to the SECTION TITLE, not the learner. `dataset.split_by_participant` then
produces a SECTION-INDEPENDENT split: no section's windows appear in both train and test, so the
model cannot be scored on a section it memorised. At n=1 this is the strongest available control
— but it is NOT subject-independent, so the resulting score is an optimistic upper bound and must
be reported as such.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

# Class order — identical to synthetic_data.LABELS, phase_a.LABELS, and the backend
# BEHAVIORAL_CLASS_ORDER. Do not reorder.
LABELS = ["Engaged", "Bored", "Confused", "Frustrated"]
_AFFECT_IDX = {"engaged": 0, "bored": 1, "confused": 2, "frustrated": 3}

# `neutral` is the warm-up baseline: a real self-report value and a real section design, but not
# a model class. Neutral windows are returned separately as the per-person resting baseline.
NEUTRAL = "neutral"

# A window this far before the FIRST section completion still belongs to that first section.
# Longer leads are dropped rather than attributed to a section the learner had not reached.
DEFAULT_MAX_LEAD_MS = 10 * 60 * 1000  # 10 minutes

# The longest plausible time spent on ONE section. Every interval — not just the first — is
# bounded by this, because consecutive completions are the only bound the data provides and the
# gap between two SITTINGS is arbitrarily large. Without it, every window emitted after a
# sitting's last completion (tail reading, tab left open overnight) is attributed to the NEXT
# sitting's first section: silent, certain, and class-specific.
#
# Set generously relative to the content: sections are estimated at 4-5 minutes, and a learner
# stuck on a confusing one can legitimately take several times that, so windows inside a genuine
# slow read must survive. 20 minutes keeps those and still severs an inter-sitting gap.
DEFAULT_MAX_SECTION_MS = 20 * 60 * 1000  # 20 minutes


def _get(ev, *names):
    """Read the first present key (handles camelCase API export or snake_case DB dump)."""
    for n in names:
        if n in ev and ev[n] is not None:
            return ev[n]
    return None


def load_codebook(path: str | Path) -> dict[str, str]:
    """Load `codebook.json` -> `{section_title: affect}`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {title: rec["affect"] for title, rec in data["sections"].items()}


def _section_timeline(events) -> dict[str, list[tuple[int, str]]]:
    """`{learner_id: [(completed_ts, section_id), ...]}`, ascending, de-duplicated.

    Only the FIRST completion of a section counts. A re-completion (`created: false`) is an
    idempotent re-post, not a second pass through the material.
    """
    seen: dict[tuple[str, str], int] = {}
    for ev in events:
        if _get(ev, "eventType", "event_type") != "section_completed":
            continue
        payload = _get(ev, "payload") or {}
        sid = payload.get("section_id") or payload.get("sectionId")
        ts = _get(ev, "timestamp")
        learner = _get(ev, "learnerId", "learner_id")
        if sid is None or ts is None:
            continue
        key = (str(learner), str(sid))
        if key not in seen or int(ts) < seen[key]:
            seen[key] = int(ts)

    timeline: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for (learner, sid), ts in seen.items():
        timeline[learner].append((ts, sid))
    for learner in timeline:
        timeline[learner].sort()
    return dict(timeline)


def _assign_sections(windows, timeline, max_lead_ms, max_section_ms):
    """Attach each window to the section whose completion interval contains it.

    A section's interval is `(start, end]` where `end` is its own completion timestamp and
    `start` is the LATER of (a) the previous completion and (b) `end - max_section_ms`. Bounding
    on (b) as well as (a) is what keeps a multi-sitting export correct: (a) alone leaves every
    interval open-ended backwards, so a window emitted after one sitting finished lands on the
    next sitting's first section.

    `windows` is `[(learner, ts, session, features)]`. Returns
    `[(learner, ts, session, features, section_id)]`; unattributable windows are dropped.
    """
    out = []
    for learner, ts, session, feats in windows:
        marks = timeline.get(str(learner))
        if not marks:
            continue
        # First completion at or after this window == the section being worked on.
        lo, hi = 0, len(marks)
        while lo < hi:
            mid = (lo + hi) // 2
            if marks[mid][0] < ts:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(marks):
            continue  # after the last completion — nothing left to attribute it to
        end_ts, sid = marks[lo]
        if lo > 0:
            start_ts = max(marks[lo - 1][0], end_ts - max_section_ms)
        else:
            start_ts = end_ts - max_lead_ms
        if ts <= start_ts:
            continue  # over-long lead-in, or an inter-sitting gap — not safely attributable
        out.append((learner, ts, session, feats, sid))
    return out


def load_codebook_windows(
    export,
    codebook,
    sections_by_id: dict[str, dict] | None = None,
    require_agreement: bool = True,
    max_lead_ms: int = DEFAULT_MAX_LEAD_MS,
    max_section_ms: int = DEFAULT_MAX_SECTION_MS,
):
    """Export -> `{"windows": [...], "baseline": [...], "stats": {...}}`.

    Args:
        export: the export dict (`{"items": [...], "sections": {...}}`) or a bare event list.
        codebook: `{section_title: affect}` from `load_codebook`.
        sections_by_id: `{section_id: {"title": ...}}`. Defaults to `export["sections"]`.
        require_agreement: keep a section's windows only where the learner's self-report for
            that section matches the designed affect. Setting this False keeps every window and
            makes the labels purely design-assigned — higher yield, weaker claim.
        max_lead_ms: how far before the first completion a window may still be attributed.
        max_section_ms: the longest plausible single section. Bounds EVERY interval, which is
            what makes a multi-sitting export safe (see `_assign_sections`).

    `windows` entries match what `dataset.windows_to_arrays` consumes, with
    `participant = section_title` (see module docstring, SPLITTING).
    """
    if isinstance(export, dict):
        events = export.get("items", [])
        sections_by_id = sections_by_id or export.get("sections") or {}
    else:
        events = export
        sections_by_id = sections_by_id or {}

    title_of = {str(k): (v or {}).get("title") for k, v in sections_by_id.items()}

    behavioural = []  # (learner, ts, session, features)
    reports = []  # (learner, ts, affect)
    for ev in events:
        etype = _get(ev, "eventType", "event_type")
        ts = _get(ev, "timestamp")
        payload = _get(ev, "payload") or {}
        if etype == "behavioral_affect_detected":
            feats = payload.get("features")
            if feats is None or ts is None:
                continue  # error / empty cycles carry no feature window
            behavioural.append(
                (
                    str(_get(ev, "learnerId", "learner_id")),
                    int(ts),
                    _get(ev, "sessionId", "session_id"),
                    feats,
                )
            )
        elif etype == "self_report":
            if payload.get("skipped") or payload.get("omitted") or ts is None:
                continue
            affect = str(payload.get("affect") or "").lower()
            if not affect:
                continue
            reports.append(
                (str(_get(ev, "learnerId", "learner_id")), int(ts), affect)
            )

    timeline = _section_timeline(events)
    attributed = _assign_sections(behavioural, timeline, max_lead_ms, max_section_ms)

    # Manipulation check: attribute each self-report to the section it closed, the same way
    # windows are attributed. A report fired by the prompt AFTER completing section S falls in
    # the interval of the NEXT section, so step back one boundary to reach S.
    report_for_section: dict[tuple[str, str], str] = {}
    for learner, ts, affect in reports:
        marks = timeline.get(learner)
        if not marks:
            continue
        prior = [m for m in marks if m[0] <= ts]
        if not prior:
            continue
        report_for_section[(learner, prior[-1][1])] = affect

    windows, baseline = [], []
    stats = Counter()
    agree = defaultdict(lambda: [0, 0])  # designed affect -> [agreed, total_checked]

    for learner, ts, session, feats, sid in attributed:
        title = title_of.get(str(sid))
        if not title:
            stats["dropped_unknown_section_id"] += 1
            continue
        designed = codebook.get(title)
        if designed is None:
            stats["dropped_section_not_in_codebook"] += 1
            continue

        if designed == NEUTRAL:
            baseline.append(dict(participant=title, features=feats, section_id=str(sid)))
            stats["baseline_windows"] += 1
            continue

        reported = report_for_section.get((learner, str(sid)))
        if reported is not None:
            agree[designed][1] += 1
            if reported == designed:
                agree[designed][0] += 1

        if require_agreement:
            if reported is None:
                stats["dropped_no_selfreport"] += 1
                continue
            if reported != designed:
                stats["dropped_selfreport_disagreed"] += 1
                continue

        idx = _AFFECT_IDX[designed]
        windows.append(
            dict(
                participant=title,  # section-independent splits — see module docstring
                label=LABELS[idx],
                label_idx=idx,
                features=feats,
                section_id=str(sid),
                session_id=session,
                learner_id=learner,
                timestamp=ts,
            )
        )
        stats["kept"] += 1

    stats["behavioural_events"] = len(behavioural)
    stats["attributed_to_section"] = len(attributed)
    stats["self_reports"] = len(reports)
    stats["sections_visited"] = len({sid for *_, sid in attributed})

    return {
        "windows": windows,
        "baseline": baseline,
        "stats": dict(stats),
        "agreement": {k: {"agreed": v[0], "checked": v[1]} for k, v in agree.items()},
    }


def summarise(result: dict) -> str:
    """Human-readable report — print this after every export."""
    lines = ["Codebook labelling", "=" * 60]
    s = result["stats"]
    for k in (
        "behavioural_events",
        "attributed_to_section",
        "sections_visited",
        "self_reports",
        "baseline_windows",
        "dropped_unknown_section_id",
        "dropped_section_not_in_codebook",
        "dropped_no_selfreport",
        "dropped_selfreport_disagreed",
        "kept",
    ):
        if k in s:
            lines.append(f"  {k:34s} {s[k]:6d}")

    dist = Counter(w["label"] for w in result["windows"])
    lines += ["", "Class distribution (trainable windows):"]
    total = sum(dist.values()) or 1
    for label in LABELS:
        n = dist.get(label, 0)
        flag = "  <-- EMPTY" if n == 0 else ""
        lines.append(f"  {label:12s} {n:5d}  {100 * n / total:5.1f}%{flag}")

    lines += ["", "Manipulation check (self-report agrees with designed affect):"]
    if not result["agreement"]:
        lines.append("  no self-reports could be attributed to a section")
    for affect, a in sorted(result["agreement"].items()):
        rate = f"{100 * a['agreed'] / a['checked']:5.1f}%" if a["checked"] else "   n/a"
        lines.append(f"  {affect:12s} {a['agreed']:3d}/{a['checked']:3d}  {rate}")

    n_sections = len({w["participant"] for w in result["windows"]})
    lines += ["", f"Distinct sections contributing windows: {n_sections}"]
    if n_sections < 8:
        lines.append(
            "  WARNING: too few sections for a meaningful section-independent split."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Label an export via the affect codebook.")
    ap.add_argument("--export", default="../../data/phase_a/export.json")
    ap.add_argument("--codebook", default="../../data/codebook.json")
    ap.add_argument(
        "--no-agreement",
        action="store_true",
        help="Keep all windows (design labels only, no manipulation check).",
    )
    a = ap.parse_args()

    export = json.loads(Path(a.export).read_text(encoding="utf-8"))
    result = load_codebook_windows(
        export, load_codebook(a.codebook), require_agreement=not a.no_agreement
    )
    print(summarise(result))
