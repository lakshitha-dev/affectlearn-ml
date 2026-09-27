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

1. ~~**Ordinal modelling** (CORAL/CORN)~~ — **tried (2026-06-09), no benefit.** CORN ordinal
   regression scored test wF1 0.471, tied with focal 0.473. The High↔Very-High wall does not
   yield to ordinal modelling. See "Serve-alignment + ordinal experiment" below.
2. **Stronger temporal model** — small 3D-CNN or temporal transformer instead of CNN+LSTM.
3. **Label de-noising / merging** to a 2–3 class engagement scheme if the use-case allows.

## Reproduce

1. Colab + GPU, mount Drive with the DAiSEE data at the `paths:` in `config.yaml`.
2. Run `notebooks/02_cnn_lstm_training.ipynb` top-to-bottom (preprocess → train → eval),
   or `python training/facial/train_cnn_lstm.py --config training/facial/config.yaml`
   after preprocessing.
3. Best checkpoint + self-contained ONNX are written to the `checkpoints` path on Drive.

## Serve-alignment + ordinal experiment (2026-06-09)

**Motivation.** Training cropped faces with **OpenCV Haar + 10% padding + centre-crop
fallback**, but the browser (`affectlearn/frontend/src/lib/preprocess.ts`, story 4-2) crops
with **MediaPipe `blaze_face_short_range`, raw bbox, no padding, dropping faceless frames**.
Train/serve preprocessing must match or the model sees a distribution it never trained on.

**Measured gap** (same 1638 Test clips, same `cnn_lstm_best.pt`, `evaluation/crop_gap_eval.py`):

| Serve cropper | test wF1 | macro F1 | acc |
|---|---|---|---|
| Haar (matched train) | 0.509 | 0.284 | 0.520 |
| MediaPipe (real production) | **0.480** | 0.245 | 0.540 |

So the deployed Haar-trained model fed real MediaPipe crops scores ~0.48, not 0.509.

**Fix attempt — re-preprocess all splits with MediaPipe and retrain**
(`training/facial/mediapipe_retrain_pipeline.py`, two variants):

| Model | train→serve | val wF1 | test wF1 | macro F1 | per-class [VL,L,H,VH] |
|---|---|---|---|---|---|
| focal | MP→MP | 0.507 | **0.473** | 0.271 | [.00, .12, .59, .38] |
| CORN ordinal | MP→MP | 0.505 | 0.471 | 0.262 | [.00, .08, .58, .39] |

**Conclusions.**
- Aligning the cropper did **not** recover weighted-F1 (0.473 is within run-noise of 0.480).
  The mismatch is **not** a recoverable distribution-shift bug — MP's tight face-only crop is
  intrinsically a touch harder than Haar's looser, context-including crop.
- **CORN ordinal loss gave no benefit** — triple-confirms the ~0.51 wall is representational
  (High↔Very-High label noise), not loss/preprocessing.
- Only upside: the MP-aligned focal model is **better balanced** (macro F1 0.271 vs 0.245,
  non-zero "Low" recall). It is now the canonical `cnn_lstm_best.pt` (train==serve==browser);
  the prior Haar model is preserved as `cnn_lstm_haar_baseline_f509_pre_mp.pt` on Drive.
- `preprocess.py` now uses the MediaPipe cropper (serve-aligned) — a **correctness/hygiene**
  change, not an accuracy win. 0.60 still needs an architectural change.
