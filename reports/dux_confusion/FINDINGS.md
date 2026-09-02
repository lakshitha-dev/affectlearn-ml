# Model B trained on public data: human-annotated confusion in DUX

**Date:** 13 Aug 2026
**Scripts:** `training/behavioral/train_dux_confusion.py`, `dux_sensitivity.py`,
`dux_normalisation_audit.py`, `train_dux_multistate.py`
**Loader:** `external_datasets.load_dux_confusion`
**Artifacts:** `reports/dux_v1_z/` (headline), `reports/dux_confusion/`, `reports/dux_confusion_z/`

---

## HEADLINE — v0 + v1, 46 participants

This supersedes the v0-only section below, which is kept because the contrast between the two is
itself the most instructive thing in this report.

**1,419 windows, 46 participants, 257 confused (18.1%), majority baseline 0.8189 accuracy.**
Leave-one-participant-out, per-learner z-scored, within-participant permutation null.

| Arm | AUC | 95% CI | perm *p* | κ | macro-F1 | confused recall |
|---|---|---|---|---|---|---|
| **behavioural** | **0.747** | [0.716, 0.778] | **0.0005** | **0.274** | 0.630 | **0.580** |
| facial (12 AFFDEX) | 0.506 | [0.470, 0.546] | 0.108 | −0.001 | 0.481 | 0.350 |
| fused | 0.728 | [0.695, 0.759] | 0.0005 | 0.239 | 0.612 | 0.553 |

### The three results, in order of importance

**1. The behavioural channel detects human-annotated confusion: AUC 0.747, κ 0.274.**
The confidence interval [0.716, 0.778] excludes chance, and κ 0.274 sits inside the published range
for sensor-free affect detection (Baker et al. 2012 report mean κ 0.30 across four states; published
frustration detectors go as low as κ 0.23). Confused recall 0.580 means the model catches more than
half of the confused windows. Quadrupling participants moved AUC 0.678 → 0.747 and roughly halved
the interval width, which is what a real effect does when given more data.

**2. The facial channel is indistinguishable from chance: AUC 0.506, κ −0.001, p = 0.108.**
Its confidence interval [0.470, 0.546] contains 0.5. The apparent facial signal in the v0-only run
(AUC 0.562–0.610 on 10 participants) was noise, and the larger sample destroyed it. Twelve
commercial AFFDEX channels carry no usable information about the confusion a human annotator marked.

**3. Fusion does not help — it costs 0.019 AUC and 0.035 κ.** Which now has an obvious explanation
rather than being a puzzle: there is nothing to fuse with. Adding a chance-level channel to a
working one dilutes it. This is the honest RQ2 answer from the only paired data available, and it is
consistent with D'Mello & Kory's natural-data gain being only 4.59% — small enough that a null is
unremarkable.

### Why finding 2 matters more than it looks

The project's premise is that facial analysis alone is insufficient for learning-centred affect.
This is direct evidence for it, from a commercial classifier that costs real money, on the state that
matters most pedagogically. Two supporting measurements:

* `emotion_affectiva_Confusion` — the channel *named* for the state — scores **AUC 0.476 against the
  human label, worse than chance** (v0 measurement). A vendor's label for a state is not the state.
* For calibration, Bosch et al. (IUI 2015, N=137, real classroom, in the wild) report facial
  confusion at **AUC 0.649**. So the behavioural result here (0.747) is *above* the published facial
  benchmark for confusion — on a different corpus, so not a head-to-head, but it means the
  behavioural channel is not the weak junior partner the thesis currently assumes.

## Deployment calibration — the number that matters for an ASSISTIVE system

The platform's purpose is to *offer help* (`show_hint`, `show_alternative`, `show_breakdown` in
`nodes/pedagogical.py`), not to score well on a benchmark. That makes **precision** the operating
metric, not accuracy or recall: a hint shown to a learner who was coping is mildly redundant, while
a false positive actively interrupts someone who was fine. Recall is cheap to sacrifice because
confusion persists — a missed window is usually followed by another.

Raw Model B precision peaks at only **0.43** across all thresholds, which alone would make more than
half of all interventions unwarranted. But `edges.passes_adaptation_gate` (confidence floor +
persistence + cooldown) was built for exactly this. Simulating that gate over the real out-of-fold
predictions, with `ADAPT_MIN_CONSECUTIVE = 2` and `ADAPT_COOLDOWN_CYCLES = 3`:

| Threshold | Precision | Interventions/hour | One every |
|---|---|---|---|
| no gate at all | 0.181 | 120.0 | 30 s |
| 0.55 (the old default) | 0.391 | 5.4 | 11 min |
| 0.60 | 0.429 | 4.1 | 15 min |
| 0.65 | 0.400 | 2.5 | 24 min |
| **0.70 (adopted)** | **0.500** | **1.5** | **39 min** |

**`ADAPT_MIN_CONFIDENCE` was changed 0.55 → 0.70** (`app/agents/edges.py`), which is a **2.76×
precision improvement over intervening unconditionally**. the backend suite passes unchanged (772 tests across 80 files at time of writing).

`ADAPT_MIN_CONSECUTIVE` stays at 2. Raising it to 3 does not reliably help — precision wobbles
0.447 / 0.273 / 0.385 across thresholds, which is noise on single-digit counts, not signal.

**The reportable system claim:** *the platform offers targeted help roughly once every 40 minutes and
half of those offers are warranted, against 18% if help were offered continuously; detector error is
converted into inaction rather than into a wrong intervention.* That claim does not depend on strong
per-state accuracy, which is why it survives a detector at AUC 0.747.

Two limits, both requiring re-measurement on platform data: DUX participants used business software
rather than learning material, and the simulation treated adjacent *surviving* windows as consecutive
in time (windows with <5 events were dropped at load; the live pipeline does not drop them).

## Multi-state attempt — a clean negative

v1's density made a 4-class attempt possible (`train_dux_multistate.py`, v1 only, 1,146 windows,
36 participants): none 841, Confusion 204, Anger 45, Joy 56. Anger stands in for FRUSTRATED as a
proxy; Joy maps to nothing and must never be called engagement.

| Arm | macro-F1 | κ | none | Confusion | **Anger** | Joy |
|---|---|---|---|---|---|---|
| behavioural | 0.309 | 0.189 | 0.79 | 0.45 | **0.00** | 0.04 |
| facial | 0.319 | 0.034 | 0.67 | 0.20 | 0.04 | 0.43 |
| fused | 0.363 | 0.207 | 0.79 | 0.45 | **0.02** | 0.18 |

**Only Confusion is learnable. Anger recall is 0.00–0.02 — the frustration proxy is not detected at
all**, and Joy is barely better from behaviour (0.04). With 45 Anger and 56 Joy windows this is the
expected outcome, and it was predicted in advance: the architecture review advised against splitting
this data further, and the prediction is now measured rather than assumed.

Note on reading the accuracies: every arm scores below the 0.734 majority baseline, but all arms are
`class_weight="balanced"`, which trades majority accuracy for minority recall by design. κ 0.207 with
Confusion recall 0.45 is real learning at a deliberately shifted operating point, not failure. The
per-state recalls are what condemn Anger and Joy, not the aggregate.

**Consequence:** DUX supports a BINARY confusion detector and nothing more. The four-state taxonomy
cannot be validated on public data — it needs platform data. Report the binary result as the public
data contribution and keep the four-state head as the platform's design, evaluated separately.

### Normalisation audit — RESOLVED, and it retracts an earlier claim

`dux_normalisation_audit.py`, same folds, same classifier, only the representation varies:

| Representation | AUC | 95% CI | perm *p* |
|---|---|---|---|
| raw (no normalisation) | 0.734 | [0.701, 0.766] | 0.0005 |
| transductive (whole session) | 0.747 | [0.716, 0.778] | 0.0005 |
| **causal (expanding past only)** | **0.741** | [0.708, 0.774] | 0.0005 |

Total transductive gain **+0.013**, of which **+0.007 survives being causal** and +0.007 was
temporal leakage. So the leak was real but small, and the deployable number (0.741) is barely
distinguishable from doing nothing (0.734).

**This retracts the v0-era claim that per-learner normalisation is worth +0.079 AUC.** That figure
came from 10 participants; at 46 it collapses to +0.013, which is well inside one standard error
(the CI half-width here is ~0.033). The +0.079 was a small-sample artefact — the same lesson the
facial arm taught, in the other direction.

The right conclusion is *better* than the one it replaces: **the result does not depend on a
normalisation trick.** Raw features already reach AUC 0.734 subject-independent. The signal is in
the 16-feature extractor, not in a preprocessing choice. Keep the causal implementation (it is
free, principled, and deployable) but do not present it as a contribution or expect it to carry
weight — and do not use the transductive version for any reported number.

---

## What changed

Every previous attempt to train Model B on public data failed for one of two reasons: no corpus
carries mouse+keyboard+scroll with learning-affect labels, or the only available labels came from
another model's output (which makes the exercise distillation, not affect detection).

`emotion_manual_Confusion` in DUX v0 escapes both. It is a **human annotation**, recorded on the
same sessions as both the raw interaction events and the AFFDEX facial channels. That triple —
independent human label, behavioural channel, facial channel, same 30 seconds of the same person —
is what makes a real unimodal-vs-fused comparison possible without collecting new data.

## What is actually in the corpus

Measured directly over `v0.csv` (146,119 rows, 10 sessions, 63 MB):

| Manual channel | Non-zero rows | Usable? |
|---|---|---|
| Confusion | 4,202 (2.88%) | **yes** |
| Joy | 1,104 | marginal |
| Judgement | 977 | marginal |
| Anger | 398 | no |
| Disgust / Contempt / Sadness / Surprise | 114 / 96 / 13 / 2 | no |
| **Engagement / Fear / Neutral / Sentimentality** | **0** | **identically zero** |

`emotion_manual_Engagement` is zero across the entire file. **DUX cannot supply a human engagement
label however it is windowed.** Confusion is the only densely annotated state.

At 30 s windows: **273 windows, 46 confused (16.8%), across 10/10 participants** — every
participant has both classes, so leave-one-participant-out is well posed.

## Result

Leave-one-participant-out, pooled out-of-fold probabilities, `HistGradientBoostingClassifier`
(depth 3, `class_weight="balanced"`) over the 80 aggregate features derived from the shared
16-feature extractor. Significance is a **within-participant permutation test** (2,000 draws):
labels are shuffled inside each participant, so the null preserves each person's positive rate and
cannot be beaten by learning who the participant is — the failure mode that matters most with 10
subjects and famously identifying motor behaviour.

| Arm | AUC | 95% CI | perm *p* | confused recall |
|---|---|---|---|---|
| **behavioural (per-learner z)** | **0.678** | [0.594, 0.758] | **0.0005** | 0.261 |
| behavioural (raw) | 0.599 | [0.515, 0.685] | 0.0015 | 0.239 |
| facial, 12 AFFDEX channels (raw) | 0.610 | [0.525, 0.692] | 0.0305 | 0.239 |
| facial (per-learner z) | 0.562 | [0.467, 0.661] | 0.0175 | 0.348 |
| fused (raw) | 0.619 | [0.537, 0.708] | 0.0025 | 0.174 |
| fused (per-learner z) | 0.673 | [0.593, 0.751] | 0.0005 | 0.152 |

Majority baseline is **0.832 accuracy** by always answering "not confused" while detecting nothing.
Accuracy is therefore meaningless here and is not the headline; AUC and confused-class recall are.

### Three findings worth reporting

**1. Interaction behaviour predicts human-annotated confusion above chance.**
AUC 0.678, *p* = 0.0005, subject-independent. Weak in absolute terms, but real, on real humans,
against a human label, with the identity-learning confound explicitly controlled.

**2. Per-learner normalisation helps the behavioural channel and hurts the facial one.**
Behavioural 0.599 → 0.678; facial 0.610 → 0.562. Motor behaviour is strongly person-specific, so
deviation from a personal baseline carries more signal than the absolute value; AFFDEX already
emits roughly calibrated probabilities, so re-centring them destroys information. This is a
**design change to adopt** — the platform has each learner's own session history at serve time, so
it is deployable, and it uses no labels.

**3. Fusion gives no gain on this corpus.** −0.005 with normalisation, +0.009 without; confidence
intervals overlap heavily in both directions. Consistent with D'Mello & Kory (2015), whose
natural-data fusion gain is only 4.59% (vs 12.7% acted) — but here it is indistinguishable from
zero.

### The facial channel's own confusion output is useless here

`emotion_affectiva_Confusion` scores **AUC 0.476 against the human confusion label — worse than
chance.** The facial signal that *does* exist sits in other channels (Joy inversely, AUC 0.368;
Surprise, 0.604). A commercial classifier's label for a state is not the same thing as the state.
This is direct evidence for the thesis premise that a facial channel alone is insufficient.

## Robustness

Window length × normalisation grid, behavioural arm only (`dux_sensitivity.py`):

| Window | raw | per-learner z |
|---|---|---|
| 10 s | 0.580 * | 0.617 * |
| 20 s | 0.474 | 0.556 * |
| 30 s | 0.599 * | **0.678** * |
| 60 s | **0.703** * | 0.666 * |

\* beats the within-participant null at *p* < 0.05. **7 of 8 configurations beat the null.**

Read this as robustness, not model selection. The grid spans 0.229 AUC on 39–57 positives, so the
*existence* of the effect is well supported and its *magnitude* is not: the honest estimate is
"somewhere around 0.58–0.70". The 20 s raw cell fails outright, which the write-up should state
rather than hide. Quoting 0.703 as the result would be selecting the maximum of eight draws.

## Limitations — carry all of these

1. **One state, not four.** Binary confused-vs-not. Bored, frustrated and engaged are absent, and
   engagement is *unobtainable* from DUX (see above). This does not answer RQ1 as posed.
2. **Not a learning task.** Participants filled in business travel-expense forms. Confusion while
   using enterprise software is not confusion while studying.
3. **Tiny.** 10 participants, 46 positive windows. Evidence, not proof.
4. **The facial arm is not Model A.** DUX ships AFFDEX channel scores, not video, so the CNN-LSTM
   cannot be run on it. The fusion result is "interaction + commercial facial features", not
   "Model B + Model A".
5. **v0 only.** v0 has 10 sessions with emotional triggers *disabled*; v1 (255 MB, 36 sessions,
   triggers enabled) is not held locally and would roughly quadruple N.

## Consequences for the platform

- **Per-learner z-normalisation is optional, not a win.** ~~Biggest measured single win (+0.079
  AUC)~~ — RETRACTED, see the normalisation audit above. That figure came from 10 participants;
  at 46 the total transductive gain is +0.013, of which only +0.0067 survives being made causal
  and +0.0065 was temporal leakage. Raw features already reach AUC 0.7341 subject-independently.
  The deployable figure is 0.7408. This is the stronger conclusion: the result does not depend
  on a preprocessing choice.
- **Consider a binary needs-help head as the reportable output**, with the 4-class head retained for
  pedagogical routing. This mirrors the two-stage design already in `two_stage.py`.
- **Do not claim a fusion gain.** It is not there in the only paired data available. If the thesis
  claims one, it must come from platform data where Model A and Model B genuinely co-occur.
- **Download DUX v1** before finalising — 36 sessions with triggers enabled is the cheapest
  available increase in N.
