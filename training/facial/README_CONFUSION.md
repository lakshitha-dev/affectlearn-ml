# Retraining Model A on DAiSEE **Confusion**

Purpose: put Model A and Model B on the **same construct** so they can ever be combined. Today
Model A outputs engagement level and Model B outputs confusion — they answer different questions, so
they cannot be fused at any amount of data.

Model B's confusion result for comparison: **AUC 0.734 raw / 0.741 causal, κ 0.274**, 46
participants, leave-one-participant-out, DUX human annotations
(`reports/dux_confusion/FINDINGS.md`).

---

## Step 1 — the label audit. Do this before anything else.

Costs 30 seconds and needs **only DAiSEE's `Labels/` folder** (a few hundred KB), not the 17 GB of
video. It decides whether the whole plan is viable.

```bash
python check_daisee_labels.py --labels-dir /path/to/DAiSEE/Labels
```

It prints, per split, the four Confusion levels and both candidate binary cuts:

| Cut | Meaning |
|---|---|
| **HIGH** `{2,3}` vs `{0,1}` | direct analogue of the engagement merge |
| **ANY** `{1,2,3}` vs `{0}` | any annotated confusion — usually better balanced when the state is rare |

Read the verdict off the **Test** split, because that is what gets reported.

**Why this gate exists.** This project has already measured the failure it prevents: collapsing a
DAiSEE-shaped 4-level target to binary moved accuracy **0.724 → 0.934** and weighted-F1 0.702 →
0.908, while Cohen's **κ collapsed 0.462 → 0.025** and minority recall was **0.020 (2 of 102)**. A
93% headline that detects nothing looks exactly like success until someone asks for per-class recall.

`colab_train_confusion.py` refuses to train if the best cut has under 5% positives on Test.

> On synthetic DAiSEE-shaped distributions the HIGH cut came out marginal (~7%) and the ANY cut
> comfortable (~29%). **That was invented data used to test the script — it is not a prediction.**
> Only your real CSVs decide.

---

## Step 2 — run it in Colab

Mount Drive, then launch detached. Do **not** hold an interactive cell open: this project's notes
record the Colab bridge dropping mid-session on long runs.

```python
from google.colab import drive; drive.mount('/content/drive')
```

```bash
!mkdir -p /content/drive/MyDrive/affectlearn-ml/logs
!cd /content/affectlearn-ml/training/facial && nohup python colab_train_confusion.py --stage all \
    > /content/drive/MyDrive/affectlearn-ml/logs/confusion_run.log 2>&1 &
```

Poll it:

```bash
!tail -40 /content/drive/MyDrive/affectlearn-ml/logs/confusion_run.log
```

Stages are separable so a dropped session never restarts from zero:

| Stage | Cost | Notes |
|---|---|---|
| `--stage audit` | seconds, no GPU | the gate above |
| `--stage preprocess` | **slow** | skipped entirely if clips are cached |
| `--stage train` | GPU hours | writes `cnn_lstm_confusion.pt` + `.onnx` |
| `--stage evaluate` | minutes | writes the durable predictions `.npz` |

---

## Two things that make this much cheaper than the first run

**Preprocessing is target-independent.** The `.npy` clips are decoded video frames and carry no
label, so the clips produced for the Engagement model are *byte-identical* to what Confusion needs.
The runner looks for a cached copy at `data/daisee_preprocessed/` on Drive, copies it to local disk
(training I/O must not come off Drive — per-batch `.npy` reads over Drive are pathologically slow),
and caches the result after a fresh run so no future target ever redoes it.

**The test-prediction dump is the real artifact.** `--stage evaluate` writes per-clip probabilities
to `<target>_test_predictions.npz`. After that, **every** metric — 4-class, either binary cut, any
threshold, AUC, κ, confusion matrices — is recomputable offline forever with no GPU, no Drive and no
DAiSEE. This also closes the gap that made the Model A binary recompute require Colab at all: there
was previously no per-clip probability dumper anywhere in the repo.

---

## Safety fix applied to `train_cnn_lstm.py`

Checkpoints were hardcoded to `cnn_lstm_best.pt` / `.onnx` with no target in the name, so **training
any non-Engagement target would have silently overwritten the deployed Engagement model on Drive**,
including the 47 MB ONNX the backend serves.

Now: Engagement keeps the historic filenames (so existing artifacts and `predict.py` paths still
resolve), and every other target gets its own name — matching the convention `predict_camera.py`
already expected (`cnn_lstm_confusion.pt`, `cnn_lstm_boredom.pt`, `cnn_lstm_frustration.pt`).
Checkpoints are also stamped with `target_affect`, and `--stage evaluate` refuses to score a
checkpoint whose stamp disagrees with the config.

---

## Reporting rules for whatever comes out

1. **Never quote bare accuracy.** Print the majority baseline beside every accuracy — the runner
   does this automatically for both cuts.
2. **AUC and confused-class recall are the headline.** κ second.
3. **Quote the Test number, not the Validation one.** `val_f1` in the checkpoint is the score the
   model was *selected* on and is optimistically biased; the runner prints both so the gap is
   visible.
4. **Say which cut you used** and why. Reporting only the better-looking one without naming the
   alternative is selection on the outcome.
5. This still does **not** license a fusion claim. Model A on DAiSEE and Model B on DUX are
   different corpora and different people, so there is no paired data. Same construct means fusion
   becomes *possible once platform data exists* — not that it has been demonstrated.
