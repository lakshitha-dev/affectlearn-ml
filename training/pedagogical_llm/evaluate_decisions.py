"""Evaluate the fine-tuned pedagogical model on the held-out decision set.

Runs generation over the DECISION examples in val.jsonl and reports, vs the synthesized
targets:
  - json_valid_rate   : fraction of outputs that parse as the required JSON object
  - in_vocab_rate     : fraction whose action_type is in the locked ACTION_TYPES
  - action_accuracy   : exact action_type match vs target
  - urgency_accuracy   : exact urgency match vs target (over json-valid outputs)
  - cohen_kappa        : agreement on action_type (chance-corrected) vs target

This gives a quick, automatable signal that the adapter learned the decision contract.
(Human expert review / Cohen's Kappa against expert labels — architecture's eval intent —
is a later step; this uses the synthetic targets as the reference.)

Usage:
    python evaluate_decisions.py --model ../../models/pedagogical_merged --data ../../data/pedagogical/val.jsonl
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ACTION_TYPES = (
    "no_action", "show_hint", "show_alternative", "show_breakdown",
    "show_encouragement", "simplify", "suggest_break", "skip_ahead", "increase_difficulty",
)


def parse_strategy(text: str):
    """Mirror of pedagogical._parse_strategy (lenient extraction of the JSON object)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or obj.get("action_type") not in ACTION_TYPES:
        return None
    return obj


def cohen_kappa(y_true, y_pred, labels) -> float:
    idx = {l: i for i, l in enumerate(labels)}
    n = len(y_true)
    if n == 0:
        return 0.0
    po = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    from collections import Counter
    tc, pc = Counter(y_true), Counter(y_pred)
    pe = sum((tc.get(l, 0) / n) * (pc.get(l, 0) / n) for l in idx)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="../../data/pedagogical/val.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="0 = all decision examples")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    rows = []
    for line in open(args.data, encoding="utf-8"):
        msgs = json.loads(line)["messages"]
        if msgs[-1]["content"].lstrip().startswith("{"):  # decision examples only
            rows.append(msgs)
    if args.limit:
        rows = rows[:args.limit]

    json_valid = in_vocab = action_hit = urgency_hit = 0
    t_actions, p_actions = [], []
    for msgs in rows:
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        target = json.loads(msgs[-1]["content"])
        pred = parse_strategy(gen)
        if pred is not None:
            json_valid += 1
            in_vocab += 1
            t_actions.append(target["action_type"]); p_actions.append(pred["action_type"])
            if pred["action_type"] == target["action_type"]:
                action_hit += 1
            if pred.get("urgency") == target.get("urgency"):
                urgency_hit += 1

    n = len(rows)
    print(json.dumps({
        "n_decision": n,
        "json_valid_rate": round(json_valid / n, 4) if n else 0,
        "in_vocab_rate": round(in_vocab / n, 4) if n else 0,
        "action_accuracy": round(action_hit / n, 4) if n else 0,
        "urgency_accuracy": round(urgency_hit / json_valid, 4) if json_valid else 0,
        "cohen_kappa_action": round(cohen_kappa(t_actions, p_actions, ACTION_TYPES), 4),
    }, indent=2))


if __name__ == "__main__":
    main()
