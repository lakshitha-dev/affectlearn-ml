"""Merge the trained LoRA adapter into the base model and export a standalone checkpoint.

WHY MERGE AT ALL, given the model is not being deployed?

1. **Citability.** A LoRA adapter is not a model — it is a diff against a specific base revision.
   A merged checkpoint is one artifact a thesis can name, hash and archive.
2. **Evaluation honesty.** `evaluate_decisions.py` can run against the merged model exactly as a
   server would, with no PEFT wrapper in the path. If merging changed behaviour, that would be
   worth discovering during evaluation rather than after publication.
3. **Optionality.** If the project later decides to self-host (removing the third-party data
   transfer to OpenAI, which is an ethics consideration for participant data), vLLM serves a merged
   checkpoint directly.

MEMORY
------
Merging cannot use the 4-bit quantised weights training ran on: dequantising and re-merging loses
precision, and PEFT refuses `merge_and_unload()` on a 4-bit model. The base is therefore reloaded in
bf16, which needs roughly 16GB — an A100, a CPU with enough RAM (slow but works), or a high-RAM
Colab runtime. This is the one step a T4 cannot do.

USAGE
-----
    python merge_weights.py --adapter ./out/adapter --out ./out/merged
    python merge_weights.py --adapter ./out/adapter --out ./out/merged --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="directory produced by finetune_llama.py")
    p.add_argument("--out", required=True, help="destination for the merged model")
    p.add_argument("--base", default=None,
                   help="base model id; defaults to the one recorded in training_meta.json")
    p.add_argument("--device", default="auto", choices=["auto", "cpu"],
                   help="'cpu' merges without a GPU — slow, but avoids needing 16GB VRAM")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(args.adapter)
    meta_path = adapter / "training_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    base = args.base or meta.get("base_model")
    if not base:
        raise SystemExit(
            "base model unknown: pass --base, or train with a finetune_llama.py that writes "
            "training_meta.json. Merging against the wrong base silently produces a broken model."
        )
    print(f"  base    : {base}")
    print(f"  adapter : {adapter}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # NOT quantised — see the memory note in the module docstring.
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=dtype,
        device_map=None if args.device == "cpu" else "auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    print("  merging adapter into base ...")
    model = model.merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)

    # Tokeniser travels with the model: the adapter directory holds the one training used, which is
    # the one that defines the chat template the model was fitted to.
    tok_src = adapter if (adapter / "tokenizer_config.json").exists() else base
    AutoTokenizer.from_pretrained(str(tok_src)).save_pretrained(str(out))

    (out / "merge_meta.json").write_text(
        json.dumps({"base_model": base, "adapter_dir": str(adapter),
                    "dtype": args.dtype, "training_meta": meta}, indent=2),
        encoding="utf-8",
    )
    print(f"  merged model -> {out}")


if __name__ == "__main__":
    main()
