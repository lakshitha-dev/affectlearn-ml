"""QLoRA fine-tuning for the pedagogical LLM (Llama 3 8B).

4-bit QLoRA SFT over the chat-format dataset from prepare_data.py. Trains an adapter
that makes Llama 3 8B emit the backend's strategy JSON + warm content text on the SAME
prompts the backend sends, so the merged model is a drop-in vLLM target.

Base model defaults to `NousResearch/Meta-Llama-3-8B-Instruct` — the ungated mirror of
the exact Llama 3 8B Instruct weights, so no HF token / Meta license is needed. Pass
`--base-model meta-llama/Meta-Llama-3-8B-Instruct` (with HF_TOKEN set) to use the gated
official repo instead.

Designed to run on a Colab Pro GPU (T4/L4/A100). Pin deps in the notebook:
    transformers>=4.43 peft>=0.12 trl>=0.9 bitsandbytes>=0.43 accelerate>=0.33 datasets

Usage:
    python finetune_llama.py --data-dir ../../data/pedagogical --output-dir ../../models/pedagogical_lora
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

LLAMA_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--data-dir", default="../../data/pedagogical")
    ap.add_argument("--output-dir", default="../../models/pedagogical_lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--merge", action="store_true", help="merge adapter into base after training")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Render each chat example to a single training string with the model's chat template.
    def to_text(batch):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch["messages"]
        ]}

    ds = load_dataset(
        "json",
        data_files={"train": str(data_dir / "train.jsonl"), "validation": str(data_dir / "val.jsonl")},
    )
    ds = ds.map(to_text, batched=True, remove_columns=ds["train"].column_names)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LLAMA_LORA_TARGETS,
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] adapter saved to {args.output_dir}")

    if args.merge:
        from merge_weights import merge  # local import; only when requested
        merge(args.base_model, args.output_dir, f"{args.output_dir}-merged")


if __name__ == "__main__":
    main()
