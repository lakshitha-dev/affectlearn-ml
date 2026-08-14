# Model A on DAiSEE: the engagement target is structurally unusable, and what to do instead

**Date:** 13 Aug 2026
**Run:** Colab T4, driven over the `colab-mcp` bridge, Drive-mounted DAiSEE
**Artifacts on Drive:** `affectlearn-ml/reports/facial_confusion/`
**Scripts:** `training/facial/check_daisee_labels.py`, `colab_train_confusion.py`,
`train_cnn_lstm.py` (checkpoint-naming fix)

---

## 1. The binary engagement recompute — done, and it needed no GPU

`cnn_lstm_best.pt` carried its test confusion matrix inside the checkpoint under a `test` key all
along, so the long-pending "binary recompute on Colab" task was pure arithmetic. The checkpoint is
the MediaPipe-focal model (`trained_on: mediapipe_crops`, `loss_type: focal`); its stored block is
identical to `mp_focal.test` in `eval_crop_gap/mediapipe_retrain_result.json`.

Test split, n = 1638:

| Metric | 4-level | Binary (engaged vs not) |
|---|---|---|
| Accuracy | 0.4921 | 0.9353 |
| Weighted F1 | 0.4731 | 0.9230 |
| Macro F1 | 0.2712 | 0.5488 |
| Cohen κ | 0.0234 | 0.1029 |
| **Majority baseline** | — | **0.9481** |
| Disengaged recall | — | **0.0941** (8 of 85) |

Binary confusion matrix (rows true): `[[8, 77], [29, 1524]]`.

**The binary model does not beat "always predict engaged" — it is 0.0128 BELOW it.** Accuracy appears
to nearly double while the model finds 8 of 85 disengaged clips.

**Correction to an earlier claim in this project.** A synthetic demonstration had predicted κ would
*collapse* under the binary merge (0.462 → 0.025). On the real model κ actually *rose*, 0.0234 →
0.1029. Both values are near-worthless, but the direction was wrong, and the honest headline is
simpler: **binary accuracy sits below the majority baseline.** Quote that, not the κ story.

## 2. Why Model A has been stuck at wF1 ≈ 0.47 — it is the labels, not the architecture

Full DAiSEE label audit, Test split, both candidate binary cuts:

| Affect | ANY cut `{1,2,3}` | baseline | HIGH cut `{2,3}` | baseline |
|---|---|---|---|---|
| **Engagement** | **99.8%** | 0.998 | 95.1% | 0.951 |
| Boredom | 53.9% | 0.539 | 21.1% | 0.789 |
| **Confusion** | **32.7%** | 0.673 | 8.8% | 0.912 |
| Frustration | 22.2% | 0.778 | 4.5% | 0.955 |

Engagement level counts on Test: **L0 = 4, L1 = 84, L2 = 882, L3 = 814.**

**Engagement is very nearly a constant on DAiSEE.** Four clips in the entire test split are at the
lowest level. No architecture, loss function, sampler or backbone can learn a class with four test
examples, and the per-class F1 of `[0.00, 0.12, 0.59, 0.38]` recorded throughout this project is the
direct signature of that. This reframes a long-running "model underperformance" narrative as a
**dataset property**, which is a legitimate Chapter 4 finding rather than a failure.

On the clips that actually exist (the zip is missing some), the Confusion ANY cut is well balanced:

| Split | usable n | levels | ANY positive | baseline |
|---|---|---|---|---|
| Train | 4852 | 3331 / 1061 / 400 / 60 | 1521 (31.3%) | 0.687 |
| Validation | 1429 | 942 / 322 / 153 / 12 | 487 (34.1%) | 0.659 |
| Test | 1638 | 1135 / 368 / 116 / 19 | 503 (30.7%) | 0.693 |

`Test usable = 1638` **exactly matches** the n in every previous evaluation, so a new confusion model
is scored on the same clips as the engagement model. The comparison is clean.

## 3. Decision: retarget Model A to CONFUSION, ANY cut

Chosen because it is (a) learnable — 31% positive rather than 0.2%, and (b) the **same construct
Model B already detects**, which is the precondition for ever combining them. Boredom is better
balanced (53.9%) but is the weakest facial state in the literature (AUC .610, κ .04) and would not
pair with Model B, which cannot detect boredom at all.

## 4. Three fixes required to make the run possible

**A checkpoint-clobbering bug (the dangerous one).** `train_cnn_lstm.py` hardcoded
`cnn_lstm_best.pt` / `.onnx` with no target in the filename. Training any non-Engagement target
would have **silently overwritten the deployed engagement model on Drive**, including the 47.9 MB
ONNX the backend serves. Now Engagement keeps the historic names (nothing existing breaks) and other
targets get their own, matching the convention `predict_camera.py` already expected. Checkpoints are
stamped with `target_affect`, and evaluation refuses a checkpoint whose stamp disagrees with config.

**cv2 5.0.0 removed `CascadeClassifier`.** It is absent from the top level *and* `cv2.objdetect` is
gone entirely, so `preprocess.py`'s Haar path cannot run on current Colab. `crop_face` also ignores
its own `detector` argument, so there is no injection point. Replaced with MediaPipe BlazeFace
(`crops_mp.py`) at identical geometry — 10% padding, largest detection, centre-crop fallback.
Verified 16/16 frames detected on a real clip. This is an improvement rather than a workaround: the
deployed checkpoint is `trained_on: mediapipe_crops` and the browser serves MediaPipe crops, so it
**removes** the train/serve mismatch this project measured at 0.029 wF1
(haar→haar 0.509 vs haar→mp 0.480).

**`dataset.py` only supported the HIGH cut.** `binary=True` mapped `v < 2`, i.e. 8.8% positive for
confusion — near-unlearnable. Added a `binary_cut` parameter with an `any` option. Patched in a
working copy; the Drive original is untouched.

## 5. Also discovered about the existing checkpoints

| Checkpoint | target | epoch | val wF1 | assessment |
|---|---|---|---|---|
| `cnn_lstm_frustration.pt` | Frustration | **31** | 0.6354 | **the only converged model — never tested** |
| `cnn_lstm_boredom.pt` | Boredom | 5 | 0.3469 | likely majority-class |
| `cnn_lstm_confusion.pt` | Confusion | **2** | 0.5551 | dead run, ≈ majority-class |
| `cnn_lstm_engagement.pt` | Engagement | 2 | 0.5195 | dead run |
| `cnn_lstm_best.pt` | Engagement | — | 0.5074 | the deployed model |

None carry a `target_affect` stamp (all pre-date the fix). `eval_all_checkpoints.py` scores each on
the test split and dumps per-clip probabilities — worth doing because the frustration model is
converged and untested, and frustration is 22.2% positive.

## 6. The durable artifact, and why it matters most

Every evaluation writes per-clip probabilities to `<stem>_test_predictions.npz`. After that, **any**
metric — 4-level, either binary cut, any threshold, AUC, κ, confusion matrices — is recomputable
offline, forever, with no GPU, no Drive and no 17 GB dataset. There was previously no per-clip
probability dumper anywhere in the repo, which is precisely why the binary recompute had been
blocked on Colab for so long. It never needed to be.

## 7. Reporting rules

1. Never quote bare accuracy; print the majority baseline beside it.
2. AUC and positive-class recall are the headline; κ second.
3. Quote the Test number, not Validation — `val_f1` is what the checkpoint was selected on.
4. Name which binary cut was used, and say the other exists.
5. **This does not license a fusion claim.** Model A trains on DAiSEE and Model B on DUX — different
   corpora, different people, no paired moments. Sharing a construct makes fusion *possible once
   platform data exists*; it does not demonstrate it. See `reports/dux_confusion/FINDINGS.md`, where
   fusion measured on genuinely paired data gave **no gain** (0.728 fused vs 0.747 behavioural).

## 8. RESULT — facial confusion, and a frozen-vs-finetuned ablation

Binary ANY-cut Confusion, MediaPipe crops, held-out DAiSEE test split (n = 1638 — the same clips as
every previous evaluation in this project).

| Run | Trainable params | Best val wF1 | **Test AUC** | κ | Accuracy | Baseline |
|---|---|---|---|---|---|---|
| **backbone frozen** (adopted) | ~800k | 0.6256 | **0.6414** | **0.2119** | 0.6813 | 0.6929 |
| layer4 unfrozen | 11,965,506 | 0.6077 | 0.5735 | — | 0.6636 | 0.6929 |

Confused-class detail for the adopted run: recall 0.3917, precision 0.4770, F1 0.4301; matrix
`[[919, 216], [306, 197]]`. Accuracy sits just below the majority baseline, which is the expected
cost of class-balanced training — AUC and κ are the reportable metrics.

**Unfreezing layer4 made it WORSE, and the mechanism is visible.** Training loss collapsed 0.1678 →
0.0528 while validation wF1 declined over the same epochs: 12M trainable parameters memorised 4,852
training clips. Early stop at epoch 15. This turns `freeze_backbone_full: true` from an untested
default into an evidence-backed design decision.

**External validity check.** Bosch et al. (IUI 2015, N = 137, real classroom, in the wild) report
facial confusion at **AUC 0.649**. This run lands at 0.6414 — within 0.008. So 0.6414 is not
underperformance; it is where facial confusion detection actually sits.

### Every facial checkpoint, finally tested on the same split

| Checkpoint | Target | Epoch | AUC (any cut) | κ | Verdict |
|---|---|---|---|---|---|
| **confusion_anycut** (new) | Confusion | 18 | **0.6414** | **0.2119** | real signal |
| confusion (old) | Confusion | 2 | 0.5893 | 0.0695 | ~chance |
| boredom | Boredom | 5 | 0.5608 | 0.0353 | ~chance |
| frustration | Frustration | 31 | 0.5596 | 0.0831 | **~chance** |
| engagement | Engagement | 2 | *0.7942* | **−0.0010** | **MEANINGLESS** |

Two warnings that must travel with this table.

**The frustration checkpoint does not detect frustration.** It was the only converged model in the
project (epoch 31, val wF1 0.6354), but on test its per-class recall is `[0.92, 0.14, 0.05, 0.00]` —
it answers level 0 almost always. The 0.6354 was a majority-class artefact.

**Never quote the engagement AUC of 0.7942.** The ANY cut on engagement is 1,634 positive vs **4
negative**. The model predicts "engaged" for essentially everything, scores 0.9969 accuracy against a
0.9976 baseline, and **κ is negative**. An AUC computed over four negatives is noise, not skill.

## 9. Head-to-head, and what it means

| | Model A (face) | Model B (behaviour) |
|---|---|---|
| AUC | 0.6414 | **0.7473** |
| Cohen κ | 0.2119 | **0.2738** |
| Confused recall | 0.3917 | **0.5798** |

**Behaviour beats face for confusion on every prevalence-robust metric.** Note the two accuracies
(0.6813 vs 0.7301) are NOT comparable — the tasks differ in prevalence (30.7% vs 18.1%), so the
baselines differ (0.693 vs 0.819). Compare AUC and κ, never accuracy across the two.

**Of the four target states, exactly one is detected: confusion, now by both channels.** Engagement is
unreachable (4 disengaged clips in DAiSEE; no usable DUX label), boredom is unreachable for Model B
entirely (absent from AFFDEX) and ~chance for Model A, and frustration is undetected by both. The
honest thesis position is to report these measured limits rather than claim four working states.

## 10. Frustration with the ANY cut — a negative result that validates the positive one

Same architecture, same frozen backbone, same ANY cut, same balanced sampling, same pipeline as the
confusion run. Only the target changed. DAiSEE test, n = 1638, frustration 21.9% positive.

| Metric | Value |
|---|---|
| AUC | **0.5899** |
| Cohen κ | **0.0655** |
| Accuracy | 0.7692 (baseline 0.7808) |
| Frustrated recall | **0.0864** (31 of 359) |
| Weighted F1 | 0.7076 |
| Best val wF1 (selected on) | 0.6515 |

The ANY cut did improve on the old 4-class checkpoint (0.5596 → 0.5899) but the result is unusable:
κ 0.0655 is negligible and 9% recall cannot drive an intervention whose whole purpose is to catch the
moment. Validation predicted it — 0.6515 against a 0.630 majority-class weighted-F1 is **+0.021**,
where confusion managed **+0.103** on the identical setup.

### Why this negative result is load-bearing

| Target | AUC | κ | Verdict |
|---|---|---|---|
| **Confusion** | **0.6414** | **0.2119** | works |
| Frustration | 0.5899 | 0.0655 | ~chance |
| Boredom | 0.5608 | 0.0353 | ~chance |
| Engagement | — | −0.0010 | worse than chance |

**The pipeline discriminates rather than manufacturing positives.** A recipe that returned ~0.64 on
whatever target it was given would make the confusion result meaningless. It returns 0.59 on
frustration and 0.56 on boredom under identical conditions, so the confusion number is measuring
something specific to confusion. This is the strongest validity argument available for the headline
result, and it exists only because the failures were run rather than assumed.

## 11. FINAL STATE — one of four states is detected

| State | Model A (face) | Model B (behaviour) | Detected? |
|---|---|---|---|
| **Confused** | AUC 0.6414, κ 0.212 | **AUC 0.7473, κ 0.274** | **yes, both channels** |
| Frustrated | AUC 0.5899, κ 0.066 | Anger-proxy recall 0.00 | no |
| Bored | AUC 0.5608, κ 0.035 | impossible — absent from AFFDEX | no |
| Engaged | κ −0.0010 | no usable label (489 / 590,738) | no |

Engagement and boredom are unreachable as a **property of the available data**, not as a modelling
failure: DAiSEE has 4 disengaged clips in 1638, and AFFDEX has no boredom channel at all. Frustration
is reachable in principle (21.9% positive) but is not detected by either channel at a usable level.

**The defensible claim:** four states are targeted; **confusion is detected well enough to drive
adaptation** on both channels, with behaviour outperforming face (0.747 vs 0.641); the other three are
reported as measured limits. Combined with the gate calibration in
`reports/dux_confusion/FINDINGS.md` (precision 0.500, one intervention per ~40 min), that is a
complete and honest system result.
