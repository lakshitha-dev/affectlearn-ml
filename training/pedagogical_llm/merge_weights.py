"""Merge the LoRA adapter into base Llama 3 8B for vLLM serving.

vLLM can serve a base model + LoRA adapter dynamically, but the simplest pilot path is a
single merged model directory. This loads the base in fp16, applies the trained adapter,
merges, and writes a standalone model + tokenizer that vLLM serves directly:

    python -m vllm.entrypoints.openai.api_server --model <merged-dir> --port 8000

Then point the backend at it: settings.VLLM_ENDPOINT=http://<host>:8000 and
settings.VLLM_MODEL=<merged-dir name or served id>.

Usage:
    python merge_weights.py --base NousResearch/Meta-Llama-3-8B-Instruct \
        --adapter ../../models/pedagogical_lora --out ../../models/pedagogical_merged
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base_model: str, adapter_dir: str, out_dir: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    print(f"[done] merged model saved to {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--adapter", default="../../models/pedagogical_lora")
    ap.add_argument("--out", default="../../models/pedagogical_merged")
    args = ap.parse_args()
    merge(args.base, args.adapter, args.out)


if __name__ == "__main__":
    main()
