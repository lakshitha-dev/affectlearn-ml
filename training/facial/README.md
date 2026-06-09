# training/facial — CNN-LSTM facial engagement

Trains a CNN-LSTM to classify **engagement** (4 levels: Very Low / Low / High / Very High)
from short face-video clips in the [DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html)
dataset. Output is consumed by the `affectlearn/` runtime as an ONNX model.

## Files

| File | Role |
|---|---|
| `preprocess.py` | Video → 16-frame face crops (96×96, ImageNet-normalised) → `.npy`. ⚠️ Must stay byte-identical to the backend's `model_inference.py` preprocessing. |
| `dataset.py` | `DAiSEEDataset` — loads `.npy` clips + labels, augmentation, class weights. |
| `model.py` | `CNNLSTMModel` — per-frame CNN encoder (`backbone: scratch \| resnet18`) + LSTM + classifier head. |
| `losses.py` | `FocalLoss` (per-sample alpha, optional label smoothing) for class imbalance. |
| `train_cnn_lstm.py` | Training loop: AdamW + cosine LR, mixed precision, early stopping, best-checkpoint save, ONNX export. |
| `config.yaml` | All paths + model/training hyperparameters. |

## Pipeline

```
DAiSEE.zip ──preprocess.py──> .npy clips ──DAiSEEDataset──> CNN-LSTM ──> cnn_lstm_best.pt / .onnx
```

## Quickstart (Colab, GPU)

1. Mount Drive and point the `paths:` in `config.yaml` at your DAiSEE data + output dirs.
2. Install deps: `pip install -q mediapipe opencv-python-headless scikit-learn pyyaml tqdm onnxscript`
3. Run end-to-end via `notebooks/02_cnn_lstm_training.ipynb`, or from the CLI:

```bash
# 1. preprocess all splits (resumable; skips done clips & corrupt entries)
python preprocess.py --config config.yaml

# 2. train -> eval -> export best checkpoint + ONNX
python train_cnn_lstm.py --config config.yaml
```

Best checkpoint (`cnn_lstm_best.pt`) and a self-contained `cnn_lstm_best.onnx` are written to
the `checkpoints` path. For a single-file ONNX, export with `dynamo=False` (the default dynamo
exporter externalises weights to a `.onnx.data` sidecar).

## Key config knobs (`config.yaml`)

| Key | Default | Notes |
|---|---|---|
| `model.backbone` | `resnet18` | `scratch` (4-block CNN) or ImageNet-pretrained `resnet18`. |
| `training.balanced_sampling` | `true` | sqrt-weighted `WeightedRandomSampler` to counter imbalance. |
| `training.freeze_backbone` | `true` | Freeze ResNet18 except `layer4` (anti-overfit). |
| `training.freeze_backbone_full` | _(off)_ | Freeze the entire backbone; train LSTM+head only. |
| `training.label_smoothing` | _(0)_ | Extra regularisation in the focal loss. |
| `training.focal_gamma` | `2.0` | Focal-loss focusing parameter. |

## Current results

Best model: **test weighted F1 = 0.509** (ResNet18 layer4-finetuned + balanced sampling).
The High↔Very-High boundary is the limiting factor; weighted F1 ≈ 0.51 is the representational
ceiling for this CNN-LSTM on DAiSEE. See [`docs/FINDINGS.md`](../../docs/FINDINGS.md) for the
full bug-fix history, tuning runs, and paths to push past 0.51.
