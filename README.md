# affectlearn-ml

Separate ML training repository for the AffectLearn FYRP. Hosts training scripts, datasets, Colab notebooks, and model artifacts for the four models consumed by the main `affectlearn/` runtime:

| Model | Purpose | Trained in |
|---|---|---|
| CNN-LSTM | Facial affect detection on DAiSEE | `training/facial/` |
| Bi-LSTM | Behavioral affect detection on pilot data | `training/behavioral/` |
| Late Fusion | Multimodal fusion + ablation | `training/fusion/` |
| Llama 3 8B (LoRA) | Pedagogical reasoning + content generation | `training/pedagogical_llm/` |

## Layout

```
affectlearn-ml/
  training/
    facial/             # CNN-LSTM on DAiSEE
    behavioral/         # Bi-LSTM on pilot behavioral data
    fusion/             # Late fusion training
    pedagogical_llm/    # Llama 3 8B LoRA fine-tuning
  models/               # Exported artifacts (.pt / .onnx / merged weights)
  data/                 # Datasets (gitignored)
  notebooks/            # Google Colab notebooks
  evaluation/           # Ablation, CV, statistical tests
```

## Handoff to the main repo

- **`training/facial/preprocess.py`** must remain byte-identical to `affectlearn/backend/app/services/model_inference.py` preprocessing (MediaPipe crop, 16-frame sampling, 96×96, ImageNet normalization). Any change here must ship with a matching change in the backend.
- Exported artifacts (`models/*.pt`, merged LoRA weights) are uploaded to Azure Blob Storage; the backend / vLLM pull from there.

See `_bmad-output/planning-artifacts/architecture.md` lines 913–954 for the canonical spec and `ml-training-guide-daisee.md` Section 5 for setup instructions.
