"""Three-way evaluation of the pedagogical agents: rule-based vs GPT-4o vs fine-tuned.

THIS IS THE THESIS OUTPUT. The fine-tuned model is a research artifact, not a deployment, so this
comparison is the deliverable — not the checkpoint.

WHAT IS COMPARED
----------------
  rules      `fallbacks.rule_based_strategy` / `rule_based_content` — what production serves today
             whenever the LLM is unreachable. The floor.
  gpt4o      the teacher, and what production currently serves. The ceiling.
  finetuned  the artifact.

A distilled 8B is very unlikely to beat its own teacher. The defensible claim is "comparable at a
fraction of the inference cost, self-hostable, with no third-party transfer of participant data" —
so this script is built to measure COMPARABILITY, not to manufacture a win.

REPORTING RULES (inherited from reports/facial_confusion/FINDINGS.md §7, and they exist because this
project has already been burned by a headline accuracy that detected nothing)
  1. Never print bare accuracy — always print the majority baseline beside it.
  2. Report per-class recall, not just aggregates.
  3. Cohen's kappa second, never as the headline.
  4. Report held-out DOMAINS separately from in-domain: generalising to unseen course content is the
     platform's actual claim, and in-domain numbers cannot support it.

STRATEGIST metrics: exact `action_type` agreement with the teacher, macro-F1, per-class recall,
Cohen's kappa, majority baseline.

ADAPTER metrics: no reference text exists to score against, so this measures the properties the
SYSTEM PROMPT actually demands — each is a real production defect that has occurred:
  * grounding    — does the reply use terms from the section it was given? (hints once said
                   "connect it to something you already know", grounded in nothing)
  * word budget  — under 80 words (a breakdown once overran and was cut off mid-sentence)
  * no markdown  — literal `**bold**` reached a learner
  * no answer leak — a hint once answered both halves of the section's own exercise

Per-example outputs are written to `predictions.jsonl` so every metric is recomputable later on a
CPU with no model, no GPU and no API key — the durable-artifact principle from the facial work.

USAGE
-----
    python evaluate_decisions.py --data ../../data/pedagogical_v2 --split heldout \
        --model ./out/merged --gpt4o --rules --out ../../reports/pedagogical_llm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve()
_BACKEND = _HERE.parents[3] / "affectlearn" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# See prepare_data.py: Settings() is constructed at import and demands these. Nothing here connects.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://evaluate:unused@localhost/unused")
os.environ.setdefault("JWT_SECRET", "evaluate-not-a-real-secret")

from app.agents import fallbacks  # noqa: E402
from app.agents.nodes import content_adapter as ca  # noqa: E402

WORD_BUDGET = 80
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "is", "are", "was",
    "it", "this", "that", "you", "your", "for", "on", "as", "with", "be", "can", "not", "at",
    "by", "from", "so", "we", "they", "have", "has", "will", "would", "when", "what", "which",
}


# --- metrics -------------------------------------------------------------------------------------


def _content_terms(text: str) -> set[str]:
    """Distinctive lowercase tokens, used to test whether a reply engages with its section."""
    return {w for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", text.lower()) if w not in _STOP}


def grounding_score(reply: str, section_body: str) -> float:
    """Fraction of the reply's distinctive terms that also appear in the section.

    A blunt proxy, and deliberately so: it is computable without human raters and it separates the
    two cases that actually matter — a reply about THIS section versus generic study advice. It does
    not measure pedagogical quality, and nothing here claims it does.
    """
    r, s = _content_terms(reply), _content_terms(section_body)
    return (len(r & s) / len(r)) if r else 0.0


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def cohen_kappa(y_true: list[str], y_pred: list[str]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    po = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    tc, pc = Counter(y_true), Counter(y_pred)
    pe = sum((tc[k] / n) * (pc[k] / n) for k in set(tc) | set(pc))
    return (po - pe) / (1 - pe) if pe != 1 else 0.0


def per_class_recall(y_true: list[str], y_pred: list[str]) -> dict[str, str]:
    out = {}
    for lab in sorted(set(y_true)):
        tot = sum(1 for t in y_true if t == lab)
        hit = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        out[lab] = f"{hit}/{tot} = {hit / tot:.3f}" if tot else "n/a"
    return out


# --- systems under test ----------------------------------------------------------------------


def rules_strategy(affect: str) -> str:
    return fallbacks.rule_based_strategy(affect).get("action_type", "no_action")


def rules_content(action: str) -> str:
    return fallbacks.rule_based_content(action).get("text", "")


def make_openai(model: str):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY"))

    def run(messages: list[dict]) -> str:
        r = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=ca.settings.VLLM_MAX_TOKENS, temperature=0.7,
        )
        return (r.choices[0].message.content or "").strip()

    return run


def make_local(path: str):
    """Load the merged (or adapter) checkpoint for local generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    def run(messages: list[dict]) -> str:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=ca.settings.VLLM_MAX_TOKENS,
                do_sample=True, temperature=0.7, pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()

    return run


# --- evaluation ------------------------------------------------------------------------------


def evaluate(rows: list[dict], systems: dict, out_dir: Path, split: str) -> dict:
    strat = [r for r in rows if r.get("meta", {}).get("agent") == "strategist"]
    adapt = [r for r in rows if r.get("meta", {}).get("agent") == "adapter"]
    print(f"\n  {split}: {len(strat)} strategist · {len(adapt)} adapter examples")

    report: dict = {"split": split, "n_strategist": len(strat), "n_adapter": len(adapt)}
    preds: list[dict] = []

    # ---- strategist ----
    y_true = []
    for r in strat:
        try:
            y_true.append(json.loads(r["messages"][-1]["content"])["action_type"])
        except Exception:
            y_true.append("no_action")

    if y_true:
        maj = Counter(y_true).most_common(1)[0]
        report["strategist_majority_baseline"] = round(maj[1] / len(y_true), 4)
        report["strategist_majority_class"] = maj[0]

    for name, fn in systems.items():
        if not strat:
            break
        y_pred = []
        for r in strat:
            msgs = r["messages"][:-1]
            if name == "rules":
                affect = ""
                m = re.search(r"affect_state:\s*(\w+)", msgs[-1]["content"])
                if m:
                    affect = m.group(1)
                pred = rules_strategy(affect)
            else:
                try:
                    obj = json.loads(fn(msgs))
                    pred = obj.get("action_type", "invalid")
                except Exception:
                    pred = "invalid"
            y_pred.append(pred)
            preds.append({"agent": "strategist", "system": name, "true": y_true[len(y_pred) - 1],
                          "pred": pred, "meta": r.get("meta", {})})

        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        report[f"strategist_{name}"] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1(y_true, y_pred), 4),
            "cohen_kappa": round(cohen_kappa(y_true, y_pred), 4),
            "invalid_json": sum(1 for p in y_pred if p == "invalid"),
            "per_class_recall": per_class_recall(y_true, y_pred),
        }

    # ---- adapter ----
    for name, fn in systems.items():
        if not adapt:
            break
        g, w, md, leak = [], [], 0, 0
        for r in adapt:
            msgs = r["messages"][:-1]
            action = r.get("meta", {}).get("action", "show_hint")
            reply = rules_content(action) if name == "rules" else fn(msgs)
            body = ""
            m = re.search(r"reading this section:\n---\n(.*?)\n---", msgs[-1]["content"], re.S)
            if m:
                body = m.group(1)
            g.append(grounding_score(reply, body))
            w.append(len(reply.split()))
            if any(t in reply for t in ("**", "__", "`")):
                md += 1
            preds.append({"agent": "adapter", "system": name, "action": action,
                          "reply": reply, "words": len(reply.split()),
                          "grounding": round(g[-1], 4), "meta": r.get("meta", {})})

        n = len(adapt)
        report[f"adapter_{name}"] = {
            "grounding_mean": round(sum(g) / n, 4),
            "words_mean": round(sum(w) / n, 1),
            "over_word_budget_pct": round(100 * sum(1 for x in w if x > WORD_BUDGET) / n, 1),
            "markdown_violations": md,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"predictions_{split}.jsonl", "w", encoding="utf-8") as fh:
        for p_ in preds:
            fh.write(json.dumps(p_, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="heldout", choices=["train", "val", "heldout"])
    p.add_argument("--model", default=None, help="merged/adapter checkpoint for the fine-tuned system")
    p.add_argument("--gpt4o", action="store_true", help="include the GPT-4o teacher")
    p.add_argument("--gpt4o-model", default="gpt-4o")
    p.add_argument("--rules", action="store_true", help="include the rule-based floor")
    p.add_argument("--out", default=str(_HERE.parents[2] / "reports" / "pedagogical_llm"))
    args = p.parse_args()

    path = Path(args.data) / f"{args.split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} — run prepare_data.py first")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    systems = {}
    if args.rules:
        systems["rules"] = None
    if args.gpt4o:
        systems["gpt4o"] = make_openai(args.gpt4o_model)
    if args.model:
        systems["finetuned"] = make_local(args.model)
    if not systems:
        raise SystemExit("nothing to evaluate — pass at least one of --rules / --gpt4o / --model")

    out_dir = Path(args.out)
    report = evaluate(rows, systems, out_dir, args.split)

    print("\n" + json.dumps(report, indent=2))
    (out_dir / f"report_{args.split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  report -> {out_dir / f'report_{args.split}.json'}")
    print("  NOTE: quote the majority baseline beside every accuracy, and report held-out")
    print("        domains separately from in-domain (FINDINGS.md §7).")


if __name__ == "__main__":
    main()
