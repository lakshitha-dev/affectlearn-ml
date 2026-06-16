"""Generate fine-tuning pairs for the pedagogical LLM (Llama 3 8B).

This produces a chat-format SFT dataset for the TWO tasks the fine-tuned model performs
in production, using the EXACT system/user prompts the backend sends so the adapter is
drop-in (see affectlearn/backend/app/agents/nodes/pedagogical.py and content_adapter.py):

  1. DECISION  — Pedagogical Strategist: affect + profile + content context
                 -> JSON {action_type, reason, urgency}
  2. CONTENT   — Content Adapter: action + context -> warm, conversational learner text

Targets are synthesized from the locked ACTION_TYPES vocabulary and the acceptance-
criteria mappings in epics.md (Story 5.1 lines 911-929, Story 5.2 lines 942-952), with
context-sensitive variation (first vs sustained confusion, moderate vs extended
frustration, sustained engagement -> harder, etc.).

NOTE (research honesty): this is *synthetic, template-grounded* data — appropriate to
validate the fine-tuning pipeline and the RQ3 mechanism (a fine-tuned LLM can make
pedagogical decisions) for the pilot. It can be upgraded later with human-authored or
teacher-distilled pairs without changing the schema. Output is deterministic (seeded).

Usage:
    python prepare_data.py --out ../../data/pedagogical --n-decision 2600 --n-content 1500 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ── Locked vocabulary (mirrors affectlearn/backend/app/agents/fallbacks.py) ──────────
ACTION_TYPES = (
    "no_action", "show_hint", "show_alternative", "show_breakdown",
    "show_encouragement", "simplify", "suggest_break", "skip_ahead", "increase_difficulty",
)
GENERATIVE_ACTIONS = (
    "show_hint", "show_alternative", "show_breakdown",
    "show_encouragement", "suggest_break", "simplify",
)
AFFECT_STATES = ("bored", "confused", "engaged", "frustrated")
SKILL_LEVELS = ("beginner", "intermediate", "advanced")
DIFFICULTIES = ("easy", "medium", "hard")
TOPICS = (
    "variables and types", "for and while loops", "functions and scope", "recursion",
    "pointers and references", "Big-O notation", "binary search", "linked lists",
    "hash maps", "SQL joins", "database normalization", "REST APIs", "HTTP status codes",
    "derivatives", "definite integrals", "matrix multiplication", "probability basics",
    "supply and demand", "opportunity cost", "the scientific method", "photosynthesis",
    "Newton's second law", "object-oriented design", "git branching",
)

# ── Exact production prompts (copied from the backend nodes; keep in sync) ───────────
DECISION_SYSTEM = (
    "You are the Pedagogical Strategist for an adaptive e-learning platform. Given a "
    "learner's current affective state, their profile, and the content they are studying, "
    "decide the single best teaching intervention.\n\n"
    "Reply with ONLY a JSON object: "
    '{"action_type": <action>, "reason": <short string>, "urgency": "low"|"medium"|"high"}.\n'
    f"action_type MUST be one of: {', '.join(ACTION_TYPES)}.\n\n"
    "Guidance:\n"
    "- confused: show_hint (first/mild), show_alternative (sustained), or show_breakdown "
    "(deep confusion on conceptual content).\n"
    "- frustrated: show_encouragement or simplify (moderate); suggest_break "
    "(high/extended, especially late sessions).\n"
    "- bored: skip_ahead or increase_difficulty to re-engage.\n"
    "- engaged: no_action, or increase_difficulty only if engagement is sustained.\n"
    "Use the learner profile (skill level, mastery, recent affect history) and content "
    "difficulty to choose. Prefer no_action over an unhelpful interruption."
)

CONTENT_SYSTEM = (
    "You are the Content Adapter for an adaptive e-learning platform. You write short, "
    "warm, conversational messages directly to a learner who is studying. Always use the "
    "second person ('you'), sound human and encouraging, and NEVER sound clinical or "
    "robotic. Never mention the system, the detection mechanism, or words like "
    "'difficulty reduced' or 'intervention'. Reply with ONLY the message text the learner "
    "should see — no labels, no preamble, no quotation marks."
)

CONTENT_INSTRUCTION = {
    "show_hint": (
        "Give a brief hint: 1-2 sentences offering a simplified explanation or an analogy "
        "that nudges the learner toward the idea without giving the full answer."
    ),
    "show_alternative": (
        "Explain the current concept again from a different angle — a full alternative "
        "explanation that may click better than the original framing."
    ),
    "show_breakdown": (
        "Break the concept down into a clear, numbered step-by-step walkthrough so the "
        "learner can follow it one piece at a time."
    ),
    "show_encouragement": (
        "Offer one warm, brief line of encouragement to keep the learner motivated."
    ),
    "suggest_break": (
        "Gently suggest the learner take a short break to rest and recharge. Keep it "
        "caring and brief."
    ),
    "simplify": (
        "Re-explain the current concept more gently, at a lower cognitive load — simpler "
        "words and smaller steps, with a brief encouraging tone."
    ),
}


def decision_user_prompt(affect_state, conf, profile, content_context) -> str:
    """Mirrors pedagogical._build_human_prompt exactly."""
    topic = content_context.get("topic", "unknown")
    difficulty = content_context.get("difficulty", "unknown")
    recent = (profile.get("affect_history") or [])[-5:]
    return (
        f"affect_state: {affect_state}\n"
        f"affect_confidence: {conf}\n"
        f"skill_level: {profile.get('skill_level', 'unknown')}\n"
        f"topic_mastery: {profile.get('topic_mastery', {}).get(topic, 'unknown')}\n"
        f"format_preferences: {profile.get('format_preferences', {})}\n"
        f"recent_affect_history: {recent}\n"
        f"content_topic: {topic}\n"
        f"content_difficulty: {difficulty}"
    )


def content_user_prompt(action_type, profile, content_context) -> str:
    """Mirrors content_adapter._build_human_prompt exactly."""
    topic = content_context.get("topic", "unknown")
    difficulty = content_context.get("difficulty", "unknown")
    instruction = CONTENT_INSTRUCTION.get(action_type, "Help the learner with this topic.")
    return (
        f"content_topic: {topic}\n"
        f"content_difficulty: {difficulty}\n"
        f"learner_skill_level: {profile.get('skill_level', 'unknown')}\n\n"
        f"Task: {instruction}"
    )


# ── Decision target logic (context-sensitive, AC-grounded) ───────────────────────────
def _consecutive_tail(history, state) -> int:
    n = 0
    for s in reversed(history):
        if s == state:
            n += 1
        else:
            break
    return n


def decide(rng, affect_state, conf, profile, content_context) -> dict:
    """Choose {action_type, reason, urgency} the way the AC guidance intends."""
    history = profile.get("affect_history") or []
    streak = _consecutive_tail(history, affect_state)
    difficulty = content_context.get("difficulty", "medium")
    topic = content_context.get("topic", "this topic")
    mastery = profile.get("topic_mastery", {}).get(topic, 0.5)

    if affect_state == "confused":
        if streak >= 3 and difficulty == "hard":
            return {"action_type": "show_breakdown", "urgency": "high",
                    "reason": f"deep, sustained confusion on hard material ({topic}); a step-by-step breakdown will help"}
        if streak >= 2:
            return {"action_type": "show_alternative", "urgency": "medium",
                    "reason": f"confusion on {topic} is persisting; a different explanation may click better"}
        return {"action_type": "show_hint", "urgency": "medium",
                "reason": f"first sign of confusion on {topic}; a light hint should unblock the learner"}

    if affect_state == "frustrated":
        if streak >= 3 or (streak >= 2 and conf >= 0.8):
            return {"action_type": "suggest_break", "urgency": "high",
                    "reason": "extended frustration; a short break will help the learner reset"}
        if rng.random() < 0.45:
            return {"action_type": "show_encouragement", "urgency": "medium",
                    "reason": f"moderate frustration on {topic}; brief encouragement keeps momentum"}
        return {"action_type": "simplify", "urgency": "high",
                "reason": f"frustration on {topic}; reducing the cognitive load should ease the struggle"}

    if affect_state == "bored":
        if mastery >= 0.7 and profile.get("skill_level") == "advanced":
            return {"action_type": "increase_difficulty", "urgency": "low",
                    "reason": f"strong mastery of {topic} and visible boredom; a harder challenge will re-engage"}
        return {"action_type": "skip_ahead", "urgency": "low",
                "reason": f"boredom on familiar {topic}; moving ahead will restore engagement"}

    # engaged
    if streak >= 3 and difficulty != "hard" and mastery >= 0.6:
        return {"action_type": "increase_difficulty", "urgency": "low",
                "reason": f"sustained engagement on {topic}; raising the challenge keeps it stimulating"}
    return {"action_type": "no_action", "urgency": "low",
            "reason": "the learner is engaged and progressing; do not interrupt a working flow"}


# ── Content generation targets (warm, varied, topic-grounded templates) ──────────────
def content_text(rng, action_type, topic, skill) -> str:
    if action_type == "show_hint":
        return rng.choice([
            f"Here's a nudge: think of {topic} as something you already do every day, then ask what's really changing step to step.",
            f"Try this — picture a simple, concrete example of {topic} first, and let the bigger rule fall out of that.",
            f"Quick hint: focus on what stays the same in {topic} versus what changes, and the rest tends to click.",
        ])
    if action_type == "show_alternative":
        return rng.choice([
            f"Let's come at {topic} from a different angle. Instead of the formal definition, imagine you're explaining it to a friend — what's the one thing they'd need to picture? Start there, and build outward until the whole idea feels natural.",
            f"Here's another way to see {topic}. Forget the textbook framing for a second: think about the problem it was invented to solve, then watch how each piece exists to answer that problem. Seen that way, it usually makes a lot more sense.",
        ])
    if action_type == "show_breakdown":
        return rng.choice([
            f"Let's take {topic} one step at a time:\n1. Start with the core idea on its own and make sure that part feels solid.\n2. Add the next piece, checking exactly how it connects to the first.\n3. Work a tiny example end to end so you can see it in action.",
            f"Here's {topic} broken into small steps:\n1. Name what you're starting with and what you want at the end.\n2. Do just the first transformation and pause to check it.\n3. Repeat one move at a time until you reach the goal — no leaps.",
        ])
    if action_type == "show_encouragement":
        return rng.choice([
            "You're doing great — wrestling with the tricky parts is exactly how this starts to stick. Keep going!",
            "Nice work pushing through this. The fact that it feels hard means you're right at the edge of learning something new.",
            "You've got this. Every bit of effort here is paying off, even when it doesn't feel like it yet.",
        ])
    if action_type == "suggest_break":
        return rng.choice([
            "You've been at this a while and working hard. How about a short break to stretch and rest your eyes? A few minutes away can make the next part feel much easier.",
            "This is a good moment to pause. Step away for a few minutes, grab some water, and come back fresh — the material will still be here, and you'll see it more clearly.",
        ])
    if action_type == "simplify":
        return rng.choice([
            f"Let's slow right down with {topic}. Here's the same idea in plainer terms, in smaller steps — no rush. We'll build it back up once this part feels comfortable.",
            f"No problem — let's take {topic} more gently. Strip it back to the simplest version that's still true, get that feeling solid, and we'll add detail only when you're ready.",
        ])
    return "Keep going — you're making real progress."


def build_profile(rng):
    skill = rng.choice(SKILL_LEVELS)
    topic = rng.choice(TOPICS)
    return skill, topic, {
        "skill_level": skill,
        "topic_mastery": {topic: round(rng.uniform(0.1, 0.95), 2)},
        "format_preferences": rng.choice([{}, {"visual": True}, {"examples": True}, {"text": True}]),
        "affect_history": [],
    }


def make_decision_example(rng) -> dict:
    skill, topic, profile = build_profile(rng)
    affect = rng.choice(AFFECT_STATES)
    streak_len = rng.choice([0, 1, 1, 2, 3, 4])
    other = [a for a in AFFECT_STATES if a != affect]
    history = [rng.choice(other) for _ in range(rng.randint(0, 3))] + [affect] * streak_len
    profile["affect_history"] = history[-8:]
    conf = round(rng.uniform(0.55, 0.98), 2)
    content_context = {"topic": topic, "difficulty": rng.choice(DIFFICULTIES)}

    target = decide(rng, affect, conf, profile, content_context)
    return {
        "messages": [
            {"role": "system", "content": DECISION_SYSTEM},
            {"role": "user", "content": decision_user_prompt(affect, conf, profile, content_context)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
    }


def make_content_example(rng) -> dict:
    skill, topic, profile = build_profile(rng)
    action = rng.choice(GENERATIVE_ACTIONS)
    content_context = {"topic": topic, "difficulty": rng.choice(DIFFICULTIES)}
    return {
        "messages": [
            {"role": "system", "content": CONTENT_SYSTEM},
            {"role": "user", "content": content_user_prompt(action, profile, content_context)},
            {"role": "assistant", "content": content_text(rng, action, topic, skill)},
        ]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../data/pedagogical")
    ap.add_argument("--n-decision", type=int, default=2600)
    ap.add_argument("--n-content", type=int, default=1500)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    examples = [make_decision_example(rng) for _ in range(args.n_decision)]
    examples += [make_content_example(rng) for _ in range(args.n_content)]
    rng.shuffle(examples)

    n_val = int(len(examples) * args.val_frac)
    val, train = examples[:n_val], examples[n_val:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):>5} examples -> {path}")
    print(f"total={len(examples)} (decision={args.n_decision}, content={args.n_content}, seed={args.seed})")


if __name__ == "__main__":
    main()
