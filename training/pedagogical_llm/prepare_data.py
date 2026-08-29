"""Regenerate the pedagogical fine-tuning corpus against the CURRENTLY SERVED prompts.

WHY THIS EXISTS (and why the old data cannot simply be reused)
--------------------------------------------------------------
`data/pedagogical/{train,val}.jsonl` already holds 3,690 + 410 usable chat examples. They are clean,
well balanced across the nine `ACTION_TYPES`, and span 24 topics. Regenerating them is not busywork —
the served prompts have changed underneath them:

    agent                live prompt   dataset prompt   match
    Pedagogical Strategist  1,028 ch      1,028 ch       identical
    Content Adapter         1,089 ch        461 ch       DIVERGED

The Content Adapter's system prompt gained two things the old corpus predates:
  * a prohibition on answering the section's own assessment questions (a real production leak: a
    delivered hint answered both halves of an exercise, so a learner could paste it into the box);
  * plain-text-only and brevity rules (a delivered breakdown shipped literal `**bold**` and was cut
    off mid-sentence at the old 256-token cap).

Its 1,345 targets were written under the OLD prompt, so they contain markdown and long multi-step
breakdowns — precisely the behaviour the served prompt now forbids. Fine-tuning on them would teach
the model to violate its own system prompt.

The USER prompt diverged too: `content_context` now carries `lesson` and a fenced section body, where
the old corpus had only `content_topic` / `content_difficulty` / `learner_skill_level`.

Regenerating only the inputs while keeping the old targets would be worse than doing nothing: the
model would be shown a section body and trained on a reply that ignores it, i.e. explicitly taught to
disregard the grounding it is handed.

THE PARITY GUARANTEE
--------------------
Prompts are IMPORTED FROM THE BACKEND, never copied. Copy-paste parity is exactly how the corpus
drifted from production in the first place; a shared import cannot drift. `--check-parity` asserts it
and is run before any GPU time is spent.

TEACHER
-------
GPT-4o, through the same OpenAI-compatible settings the platform serves with, so the corpus is a
distillation of the system actually in production rather than of some other prompt shape.

GENERALISATION
--------------
Section bodies are sampled from the REAL seeded courses across every domain (Agentic AI and Java),
and whole domains are held out for evaluation. Narrowing to one subject would teach the model that
subject instead of the transferable skill of grounding a hint in whatever text it is given — the
failure this corpus's 24-topic spread was designed to avoid.

USAGE
-----
    python prepare_data.py --check-parity                 # no API calls, verifies imports
    python prepare_data.py --dry-run --limit 5            # prints prompts, no API calls
    python prepare_data.py --out ../../data/pedagogical_v2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --- backend import (the parity guarantee) -----------------------------------------------------
# The backend is a sibling checkout, not an installed package, so its path is added explicitly.
_HERE = Path(__file__).resolve()
_BACKEND = _HERE.parents[3] / "affectlearn" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# `app.core.config` instantiates `Settings()` at import time and requires DATABASE_URL / JWT_SECRET.
# This script never opens a connection or signs a token — it only reads prompt strings and renders
# authored course content — so placeholders are supplied to satisfy the import. Real values are NOT
# read from the environment on purpose: a data-prep run must never be able to touch a live database.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://prepare-data:unused@localhost/unused")
os.environ.setdefault("JWT_SECRET", "prepare-data-not-a-real-secret")

try:
    from app.agents.fallbacks import ACTION_TYPES, GENERATIVE_ACTIONS, SELECTIVE_ACTIONS
    from app.agents.nodes import content_adapter as ca
    from app.agents.nodes import pedagogical as ped
except ImportError as exc:  # pragma: no cover - operator-facing
    raise SystemExit(
        f"Cannot import the backend agents from {_BACKEND}.\n"
        f"  {exc}\n"
        "This script deliberately imports the LIVE prompts rather than copying them, so the\n"
        "corpus cannot drift from what production serves. Run it from a checkout where\n"
        "affectlearn/backend is a sibling of affectlearn-ml."
    ) from exc


# --- affect / profile sampling space -----------------------------------------------------------
# Mirrors what `learner_profiler` actually puts in the state, so a training prompt is shaped like a
# serving prompt rather than like an idealised one.
AFFECT_STATES = ("confused", "frustrated", "bored", "engaged")
SKILL_LEVELS = ("beginner", "intermediate", "advanced")

# Held-out domains: whole subjects the model never sees during training, so "does this generalise to
# course content it has not met" is MEASURED rather than assumed. Course titles are matched by
# prefix against the seeded catalogue.
DEFAULT_HELDOUT_COURSES = ("Java Essentials",)


@dataclass
class Section:
    """One authored section, flattened to what a prompt needs."""

    course: str
    lesson: str
    title: str
    body: str
    n_words: int = field(default=0)

    def __post_init__(self) -> None:
        self.n_words = len(self.body.split())


def load_sections() -> list[Section]:
    """Every seeded section, rendered exactly as the serving prompt would render it.

    Reuses `content_context_service._render_body`, so the training body and the serving body are
    produced by one function — including its exclusion of assessment blocks, which is what stops the
    corpus teaching the model to answer quiz questions.
    """
    from app.db.course_content import building, foundations, java, multiagent, warmup
    from app.services import content_context_service as ccs

    out: list[Section] = []
    for mod in (warmup, building, foundations, multiagent, java):
        course = mod.build()
        for m in getattr(course, "modules", []) or []:
            for lesson in getattr(m, "lessons", []) or []:
                for sec in getattr(lesson, "sections", []) or []:
                    body = ccs._render_body(list(sec.content_blocks or []))
                    if body.strip():
                        out.append(
                            Section(course.title, lesson.title, sec.title, body)
                        )
    return out


def check_parity() -> None:
    """Fail loudly if the imported prompts are not the ones production serves."""
    assert ped._SYSTEM_PROMPT.strip(), "strategist system prompt is empty"
    assert ca._SYSTEM_PROMPT.strip(), "adapter system prompt is empty"
    assert "NEVER answer" in ca._SYSTEM_PROMPT, (
        "adapter prompt is missing the assessment-answer prohibition — this corpus would train "
        "the model to leak exercise answers (the production bug fixed in PR #73)"
    )
    assert "PLAIN TEXT ONLY" in ca._SYSTEM_PROMPT, (
        "adapter prompt is missing the plain-text rule — the corpus would teach markdown that the "
        "UI renders as literal asterisks (PR #81)"
    )
    sections = load_sections()
    courses = sorted({s.course for s in sections})
    print(f"  strategist system prompt : {len(ped._SYSTEM_PROMPT)} chars")
    print(f"  adapter system prompt    : {len(ca._SYSTEM_PROMPT)} chars")
    print(f"  ACTION_TYPES             : {len(ACTION_TYPES)} "
          f"({len(GENERATIVE_ACTIONS)} generative, {len(SELECTIVE_ACTIONS)} selective)")
    print(f"  sections available       : {len(sections)} across {len(courses)} courses")
    for c in courses:
        n = sum(1 for s in sections if s.course == c)
        held = " [HELD OUT]" if c.startswith(DEFAULT_HELDOUT_COURSES) else ""
        print(f"      {n:3d}  {c}{held}")
    print("  parity OK — prompts imported from the backend, not copied")


# --- prompt construction ------------------------------------------------------------------------


def strategist_messages(sec: Section, affect: str, rng: random.Random) -> list[dict]:
    """A strategist example, built with the SERVED prompt builder."""
    profile = {
        "skill_level": rng.choice(SKILL_LEVELS),
        "topic_mastery": {sec.title: round(rng.uniform(0.1, 0.9), 2)},
        "format_preferences": {},
        "affect_history": [affect] * rng.randint(1, 3),
    }
    content_context = {
        "topic": sec.title,
        "lesson": sec.lesson,
        "body": sec.body,
        "difficulty": "unknown",
    }
    human = ped._build_human_prompt(
        affect, round(rng.uniform(0.55, 0.95), 2), profile, content_context
    )
    return [
        {"role": "system", "content": ped._SYSTEM_PROMPT},
        {"role": "user", "content": human},
    ]


def adapter_messages(sec: Section, action: str, rng: random.Random) -> list[dict]:
    """An adapter example, built with the SERVED prompt builder."""
    profile = {"skill_level": rng.choice(SKILL_LEVELS)}
    content_context = {
        "topic": sec.title,
        "lesson": sec.lesson,
        "body": sec.body,
        "difficulty": "unknown",
    }
    human = ca._build_human_prompt(action, profile, content_context)
    return [
        {"role": "system", "content": ca._SYSTEM_PROMPT},
        {"role": "user", "content": human},
    ]


# --- teacher ------------------------------------------------------------------------------------


def make_teacher(model: str, api_key: str | None, base_url: str | None):
    """OpenAI-compatible chat client. Returns a `complete(messages) -> str` callable."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
    )

    def complete(messages: list[dict]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            # Matches the served cap so the teacher cannot produce replies the student is never
            # allowed to emit.
            max_tokens=ca.settings.VLLM_MAX_TOKENS,
            temperature=0.7,
        )
        return (resp.choices[0].message.content or "").strip()

    return complete


def valid_strategy(text: str) -> dict | None:
    """Parse and validate a strategist reply. Returns None if unusable.

    Applies the SAME vocabulary check the node applies at serve time, so a malformed teacher reply
    never becomes a training target.
    """
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("action_type") not in ACTION_TYPES:
        return None
    if obj.get("urgency") not in ("low", "medium", "high"):
        return None
    return obj


def build(
    sections: list[Section],
    per_section: int,
    teacher,
    rng: random.Random,
    dry_run: bool,
    limit: int | None,
) -> list[dict]:
    """Generate examples for the given sections. Skips unusable teacher replies rather than
    keeping them — a malformed target is worse than a missing one."""
    rows: list[dict] = []
    skipped = 0
    shown = 0  # dry-run counter: `rows` stays empty there, so `limit` needs its own tally
    generative = list(GENERATIVE_ACTIONS)

    for i, sec in enumerate(sections):
        produced = shown if dry_run else len(rows)
        if limit is not None and produced >= limit:
            break
        for _ in range(per_section):
            # --- strategist ---
            affect = rng.choice(AFFECT_STATES)
            msgs = strategist_messages(sec, affect, rng)
            if dry_run:
                print(f"\n--- STRATEGIST [{sec.title}] affect={affect} ---")
                print(msgs[-1]["content"][:600])
                shown += 1
            else:
                reply = teacher(msgs)
                obj = valid_strategy(reply)
                if obj is None:
                    skipped += 1
                else:
                    rows.append({
                        "messages": msgs + [{"role": "assistant",
                                             "content": json.dumps(obj, ensure_ascii=False)}],
                        "meta": {"agent": "strategist", "course": sec.course,
                                 "section": sec.title, "affect": affect},
                    })

            # --- adapter ---
            action = rng.choice(generative)
            msgs = adapter_messages(sec, action, rng)
            if dry_run:
                print(f"\n--- ADAPTER [{sec.title}] action={action} ---")
                print(msgs[-1]["content"][:600])
                shown += 1
            else:
                reply = teacher(msgs)
                # Same sanitiser the node applies, so the corpus can never contain markdown the
                # served prompt forbids.
                reply = ca._strip_markdown(reply)
                if not reply:
                    skipped += 1
                else:
                    rows.append({
                        "messages": msgs + [{"role": "assistant", "content": reply}],
                        "meta": {"agent": "adapter", "course": sec.course,
                                 "section": sec.title, "action": action},
                    })
        if not dry_run and (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(sections)} sections · {len(rows)} examples · {skipped} skipped")

    if skipped:
        print(f"  skipped {skipped} unusable teacher replies")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(_HERE.parents[2] / "data" / "pedagogical_v2"))
    p.add_argument("--model", default="gpt-4o", help="teacher model")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--per-section", type=int, default=6,
                   help="strategist+adapter pairs per section (6 -> ~12 examples/section)")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--limit", type=int, default=None,
                   help="cap examples PER SPLIT (train/val/heldout each) — for smoke tests")
    p.add_argument("--heldout", nargs="*", default=list(DEFAULT_HELDOUT_COURSES),
                   help="course-title prefixes held out for generalisation evaluation")
    p.add_argument("--check-parity", action="store_true", help="verify prompts, make no API calls")
    p.add_argument("--dry-run", action="store_true", help="print prompts, make no API calls")
    args = p.parse_args()

    if args.check_parity:
        check_parity()
        return

    rng = random.Random(args.seed)
    sections = load_sections()
    if not sections:
        raise SystemExit("no sections found — is the backend importable and seeded?")

    heldout = [s for s in sections if s.course.startswith(tuple(args.heldout))] if args.heldout else []
    trainable = [s for s in sections if s not in heldout]
    rng.shuffle(trainable)

    # Validation is a SUBJECT-INDEPENDENT slice of the trainable sections; held-out COURSES are a
    # separate, harder test of generalisation to unseen material.
    n_val = max(1, len(trainable) // 10)
    val_secs, train_secs = trainable[:n_val], trainable[n_val:]

    print(f"  sections: {len(train_secs)} train · {len(val_secs)} val · {len(heldout)} held-out")
    if args.dry_run:
        print("  DRY RUN — no API calls")

    teacher = None if args.dry_run else make_teacher(args.model, args.api_key, args.base_url)
    out = Path(args.out)

    for name, secs in (("train", train_secs), ("val", val_secs), ("heldout", heldout)):
        if not secs:
            continue
        print(f"\n  building {name} ...")
        rows = build(secs, args.per_section, teacher, rng, args.dry_run, args.limit)
        if not args.dry_run:
            n = write_jsonl(out / f"{name}.jsonl", rows)
            print(f"  wrote {n} examples -> {out / (name + '.jsonl')}")

    if not args.dry_run:
        meta = {
            "teacher_model": args.model,
            "per_section": args.per_section,
            "seed": args.seed,
            "heldout_courses": args.heldout,
            "strategist_system_prompt_chars": len(ped._SYSTEM_PROMPT),
            "adapter_system_prompt_chars": len(ca._SYSTEM_PROMPT),
            "note": "prompts imported from the backend at generation time; see module docstring",
        }
        (out / "generation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  wrote {out / 'generation_meta.json'}")


if __name__ == "__main__":
    main()
