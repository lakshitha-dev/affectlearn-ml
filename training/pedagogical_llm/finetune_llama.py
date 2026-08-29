"""QLoRA fine-tune of Llama-3.1-8B-Instruct on the pedagogical corpus.

SCOPE: this produces a RESEARCH ARTIFACT, not a production model. Nothing here is deployed —
the platform continues to serve GPT-4o. The deliverable is a reproducible checkpoint plus the
three-way comparison in `evaluate_decisions.py` (rule-based fallback vs GPT-4o vs fine-tuned).

WHERE THIS RUNS
---------------
Colab (T4 16GB or A100). 4-bit NF4 quantisation puts an 8B base inside a T4; without it, it does
not fit. Nothing in this repo has a GPU, so this script is written to be uploaded and run there —
it is deliberately dependency-light and takes every path from CLI flags.

BASE MODEL IS GATED
-------------------
`meta-llama/Llama-3.1-8B-Instruct` requires accepting Meta's licence on Hugging Face and an approved
token (`huggingface-cli login`). Approval takes hours to days. `--model` accepts any chat model with
a tokeniser chat template, so an ungated 7B can be substituted without touching this file.

WHAT IS TRAINED
---------------
Both agents in one model, exactly as production runs them: the corpus interleaves Pedagogical
Strategist examples (JSON action decisions) and Content Adapter examples (learner-facing prose). They
share a base and are separated at inference by their system prompt, so training them jointly matches
how they are served. `meta.agent` in each row records which is which for per-agent evaluation.

LOSS IS COMPUTED ON THE ASSISTANT TURN ONLY. The prompts are long (a fenced section body) and
identical across many examples; training on them would spend most of the gradient budget teaching the
model to reproduce its own input.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path


def _dataset_fingerprint(paths: list[Path]) -> str:
    """Content hash of the training files, stamped into the checkpoint.

    Without this a checkpoint cannot be tied to the corpus that produced it — and this corpus is
    regenerated whenever the served prompts change, so "which data was this trained on" is a real
    question with a real answer that is otherwise unrecoverable.
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="directory holding train.jsonl / val.jsonl")
    p.add_argument("--out", required=True, help="output directory for the LoRA adapter")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8, help="effective batch = batch-size x this")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="must fit system + fenced section body + reply; 2048 covers the corpus")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--no-4bit", action="store_true", help="disable quantisation (needs ~40GB VRAM)")
    args = p.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    data_dir = Path(args.data)
    train_path, val_path = data_dir / "train.jsonl", data_dir / "val.jsonl"
    if not train_path.exists():
        raise SystemExit(f"missing {train_path} — run prepare_data.py first")

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path) if val_path.exists() else []
    print(f"  train={len(train_rows)}  val={len(val_rows)}")

    agents = {}
    for r in train_rows:
        agents[r.get("meta", {}).get("agent", "?")] = agents.get(r.get("meta", {}).get("agent", "?"), 0) + 1
    print(f"  per-agent: {agents}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        # Llama ships no pad token. Reusing EOS is standard; the collator masks padding out of the
        # loss, so this does not teach the model to emit EOS mid-sequence.
        tok.pad_token = tok.eos_token

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Attention AND MLP projections. Attention-only adapts too little for a task that changes
        # output FORMAT (strict JSON, plain text, a word budget) rather than just tone.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    def to_ds(rows: list[dict]) -> Dataset:
        # TRL applies the tokeniser's chat template to `messages` itself. `meta` is dropped here so
        # it cannot be mistaken for a trainable field.
        return Dataset.from_list([{"messages": r["messages"]} for r in rows])

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_rows else "no",
        bf16=True,
        gradient_checkpointing=True,
        max_length=args.max_seq_len,
        seed=args.seed,
        report_to=[],
        # Loss on the completion only — see the module docstring.
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=to_ds(train_rows),
        eval_dataset=to_ds(val_rows) if val_rows else None,
        peft_config=peft_config,
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    # Provenance. A LoRA adapter is meaningless without the base it attaches to and the corpus it
    # saw; recording both is what makes the artifact citable and reproducible.
    stamp = {
        "base_model": args.model,
        "dataset_dir": str(data_dir),
        "dataset_sha256_16": _dataset_fingerprint(
            [p for p in (train_path, val_path) if p.exists()]
        ),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "per_agent": agents,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": peft_config.target_modules},
        "epochs": args.epochs,
        "lr": args.lr,
        "effective_batch": args.batch_size * args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "quantised_4bit": not args.no_4bit,
        "seed": args.seed,
    }
    (Path(args.out) / "training_meta.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    print(f"  adapter + training_meta.json -> {args.out}")


if __name__ == "__main__":
    main()
