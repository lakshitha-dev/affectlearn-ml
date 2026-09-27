# Behavioral Bi-LSTM — Dataset & Pretraining Card

Transfer-learning setup (guide "Stage 0"): **pretrain the Bi-LSTM encoder on public human
interaction data** on a binary proxy task, then **fine-tune the real 4-class head on Phase A
pilot data**. This card documents what the pretraining data is and — importantly — the honesty
rules for reporting.

## Pipeline
`pretrain_bilstm.py` (proxy pretrain on public data) → `behavioral_bilstm_pretrained.pt`
→ `train_bilstm.py` (`use_synthetic:false`) loads the pretrained encoder + a fresh 4-class head,
fine-tunes on the Phase A export, and runs a **from-scratch-vs-pretrained ablation**.

## Public datasets
| Dataset | Source | Modalities | Labels | Access |
|---|---|---|---|---|
| **DUX** | Zenodo `10.5281/zenodo.7778612`, i-com 2023, CC-BY-4.0 | mouse + keyboard + **scroll** | per-event Affectiva emotion scores (+manual) | open (`v0.csv` 63 MB subset, `v1.csv` 255 MB full) |
| **EmoSurv** | IEEE DataPort `10.21227/eae6-pk42`, Maalej & Kallel 2020 | **keyboard only** | Anger/Happiness/Calmness/Sadness/Neutral | free IEEE account (manual download) |

Adapters (`external_datasets.py`) convert both into the SAME `extract_features` window format
(13 features) → train/serve parity. Put files under `data/external/{dux,emosurv}/`.

## Proxy task
Binary **neutral vs emotion** (DUX: window mean Affectiva-Neutral `< 90` → emotion; EmoSurv:
`Neutral` label → 0, any other emotion → 1). Public emotion labels do **not** map to the four
learning-affect states — the proxy's only job is to teach the LSTM the **temporal structure of
human keyboard/mouse/scroll input**, which the fine-tune then re-purposes.

## Coverage gaps (state honestly)
- **EmoSurv has no mouse/scroll** → only keyboard features populate; mouse/scroll stay zero.
- Platform-specific features (`section_dwell_time`, `hover_dwell_mean`) are not meaningfully
  present in public data → they are effectively learned only at fine-tune.
- DUX/EmoSurv are **induced-lab emotions in a different task**, not voluntary learning.

## Honesty rules (thesis integrity — non-negotiable)
1. **Reported result = the REAL fine-tuned model** evaluated on **held-out real participants**.
2. Pretraining is justified **only** by the ablation `delta` (pretrained macro-F1 − scratch
   macro-F1). If delta ≤ 0, report that and deploy the from-scratch model.
3. **Never** cite the proxy/pretrain accuracy as a finding.
4. Split **by participant** everywhere; keep pretrain → checkpoint → fine-tune **sequential**;
   never mix public data into the real train/test split.
5. Disclose the pretraining data + proxy + coverage gaps in the methods chapter (transfer
   learning from a related corpus is a legitimate, citable contribution).

## Reproduce
```bash
# 1) acquire data:  DUX -> data/external/dux/ (Zenodo);  EmoSurv -> data/external/emosurv/ (IEEE)
python pretrain_bilstm.py            # -> models/behavioral_bilstm_pretrained.pt
# 2) after the pilot, with a real Phase A export at config data.phase_a_export:
python train_bilstm.py               # use_synthetic:false + pretrain.enabled -> fine-tune + ablation
```
