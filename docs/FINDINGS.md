# Facial Engagement (CNN-LSTM) — Training Findings

Bring-up and tuning of the DAiSEE engagement model (`training/facial/`), trained on
Google Colab (A100). Target: weighted F1 ≥ 0.60 on the DAiSEE test set.

## TL;DR

- **Best model: test weighted F1 = 0.509** (val 0.5195) — ResNet18 backbone (layer4
  fine-tuned) + LSTM, class-balanced sampling, corrected focal loss.
- Started from a model that **could not train at all** (val F1 ≈ 0.009 due to a focal-loss
  bug). Fixing the pipeline took it to 0.456, and tuning to **0.509**.
- **0.60 was not reached.** Two very different capacity regimes (9.2M vs 0.79M trainable
  params) both plateau at val ≈ 0.52 → this is the **representational ceiling** of
  per-frame-CNN + LSTM on DAiSEE's High↔Very-High distinction, not an optimisation gap.

## Bugs fixed (this branch)

| # | File | Problem | Fix |
|---|------|---------|-----|
| 1 | `losses.py` | `FocalLoss` computed `pt = exp(-ce)` on the **alpha-weighted** CE, so the focal term was wrong. With inverse-freq alpha this zeroed the loss on the majority classes → total class collapse, **val F1 ≈ 0**. | Compute `ce` **unweighted**; apply `alpha` as a per-sample multiplier on the focal term. Added optional `label_smoothing`. |
| 2 | `dataset.py` | Inverse-frequency `class_weights` were extreme (75× ratio), amplifying the collapse. | Softened to **sqrt-inverse-frequency** (~8× ratio). Added contrast augmentation. |
| 3 | `preprocess.py` | One corrupt `.avi` in `DAiSEE.zip` (bad CRC-32) aborted the entire 8k-clip preprocessing run. | `_clips_from_zip` now logs and **skips** unreadable entries. |
| 4 | `model.py` | Only a from-scratch 4-block CNN (weak features, ~1.3M params). | Added selectable `backbone: scratch \| resnet18` (ImageNet-pretrained; inputs are already ImageNet-normalised). |
| 5 | `train_cnn_lstm.py` | No way to counter class imbalance or overfitting. | Added **class-balanced sampling**, **backbone freezing** (full or layer4-only), and `label_smoothing` wiring. (Note: ONNX export needs `pip install onnxscript`; for a self-contained `.onnx`, export with `dynamo=False`.) |

## Tuning runs (DAiSEE test set)

| Run | Config | Test wF1 | "Very High" recall |
|-----|--------|----------|--------------------|
| 1 | from-scratch CNN | 0.456 | 15% |
| 2 | ResNet18, full fine-tune | 0.4575 | 15% |
| **3** | **ResNet18 (layer4) + balanced sampling** | **0.509** ⭐ | 40% |
| 4 | full freeze + label-smoothing + heavy reg | 0.4946 | 38% |

`config.yaml` on this branch reproduces **run #3** (the best). The fix for the class collapse
lifted "Very High" recall from 15% → 40% and weighted F1 from 0.456 → 0.509.

## Why 0.60 is out of reach here

- The dominant error is **High ↔ Very High**, which is genuinely fuzzy/noisy in DAiSEE labels.
- Runs #3 (9.2M trainable) and #4 (0.79M trainable) both top out at val ≈ 0.52, so adding or
  removing capacity does not move the ceiling — it is representational.
- Consistent with published DAiSEE 4-class engagement results (~50–60% accuracy; we hit 52%).

## To push past ~0.51 (future work, not tuning)

1. **Ordinal modelling** (CORAL/CORN) — treat VL<L<H<VH as ordered; targets the exact error.
2. **Stronger temporal model** — small 3D-CNN or temporal transformer instead of CNN+LSTM.
3. **Label de-noising / merging** to a 2–3 class engagement scheme if the use-case allows.

## Reproduce

1. Colab + GPU, mount Drive with the DAiSEE data at the `paths:` in `config.yaml`.
2. Run `notebooks/02_cnn_lstm_training.ipynb` top-to-bottom (preprocess → train → eval),
   or `python training/facial/train_cnn_lstm.py --config training/facial/config.yaml`
   after preprocessing.
3. Best checkpoint + self-contained ONNX are written to the `checkpoints` path on Drive.
