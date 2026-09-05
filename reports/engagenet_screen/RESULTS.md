# EngageNet facial disengagement — final result

Three-stage protocol fixed before any number existed (`training/engagenet/SEARCH_PROTOCOL.md`).
Screen on Validation, confirm on Train, report on Test **once**.

## Headline

| metric | Test | note |
|---|---|---|
| **AUC** | **0.9225** | subject-level 95% CI [0.8893, 0.9478], excludes chance |
| **accuracy** | **0.8630** | majority baseline 0.6884, margin **+17.5 points** |
| published EngageNet | 0.7360 | **+12.7 points ahead** |
| within-subject permutation *p* | 0.0005 | resolution floor of 2,000 draws |
| average precision | 0.8301 | 2.66× base rate |
| Cohen κ | 0.6877 | |
| score spread | 0.9678 | min 0.028, max 0.996 |

Fitted on Train (2,177 clips / 91 subjects), scored on Test (2,256 clips / 26 subjects).
Subject disjointness asserted in code, not assumed.

## The three stages

| stage | split | subjects | AUC | accuracy vs baseline |
|---|---|---|---|---|
| 1 screen | Validation | 11 | 0.8880 | 0.853 vs 0.786 |
| 2 confirm | Train | 91 | 0.8921 | 0.830 vs 0.687 |
| 3 report | Test | 26 | **0.9225** | **0.863 vs 0.688** |

Stable across an 8× change in subject count, and the interval tightened as subjects grew
([0.824, 0.931] → [0.862, 0.920] → [0.889, 0.948]).

## The model

`4_lean` — 20 features from **four** per-frame channels: `gaze_x`, `gaze_y`, `mouth_open`,
`motion`, each aggregated by five statistics (mean/std/min/max/trend) over 10 frames at 1 fps.
`HistGradientBoostingClassifier`, depth 3, 120 iterations, lr 0.06, balanced class weights.

* ONNX 60 KB, **0.110 ms per clip on CPU**, parity with sklearn to 2.1e-07
* ~600 bytes per serving cycle against ~3.3 MB of face crops — **no pixels leave the browser**
* max-F1 threshold 0.501, so the 0.5 default is already near-optimal (P 0.759, R 0.821)

## One premise of the plan was wrong, and one is unsupported

All eight rungs below were re-run on the FINAL extractor after the original Colab runtime was lost,
so the table is internally consistent. An earlier version of this file quoted a mixture of the two
extractors and overstated both findings; the numbers here supersede it.

| rung | feat | AUC | 95% CI | acc / base | AP x |
|---|---|---|---|---|---|
| 4_lean (gaze+mouth+motion) | 20 | **0.8887** | [0.822, 0.932] | 0.856 / 0.786 | 3.32 |
| 6_no_headpose | 35 | 0.8865 | [0.814, 0.932] | 0.858 / 0.786 | 3.36 |
| 5_lean_away | 44 | 0.8830 | [0.807, 0.930] | 0.847 / 0.786 | 3.27 |
| 3b_all_away | 74 | 0.8642 | [0.806, 0.910] | 0.825 / 0.786 | 3.16 |
| 3_all_geometry | 50 | 0.8640 | [0.810, 0.908] | 0.824 / 0.786 | 3.15 |
| 7_gaze_only | 10 | 0.7978 | [0.675, 0.884] | 0.797 / 0.786 | 2.85 |
| 2_headpose_away | 39 | 0.7392 | [0.667, 0.821] | 0.756 / 0.786 | 2.12 |
| 1_headpose | 15 | 0.7303 | [0.659, 0.809] | 0.733 / 0.786 | 1.99 |

1. **Head pose is net harmful — CONFIRMED.** Removing yaw/pitch/roll raises AUC by **+0.023**
   (0.8640 -> 0.8865), and head pose alone is the weakest rung tested. The label says "frequently
   glances away from the screen" and the plan was built on head geometry, but the discrimination
   comes from **gaze**. `mean_gaze_y` is the strongest single feature by 5x in permutation
   importance, and its AUC computed WITHIN each subject averages 0.755 - so it is behaviour, not
   camera placement.

2. **The away block has no detectable effect — NOT the negative result previously claimed.** Across
   its three pairings the deltas are +0.0088 (on head pose), +0.0002 (on all geometry) and -0.0057
   (on lean). Mixed in sign and far inside the interval widths. The earlier claim that it "failed
   three times" was an artefact of comparing across two different extractors; the honest reading is
   no effect, not a negative one.

Performance improves as channels are **removed**, 0.8640 at 50 features -> 0.8887 at 20, but almost
all of that is the head-pose removal (+0.023). Dropping the eye channels adds a further +0.002,
which is neutral rather than beneficial. Reduction stops paying at gaze alone: 0.7978 (-0.091), so
mouth openness and motion do carry complementary signal.

## Limitations, stated plainly

* **8 variants were screened on Validation**, so the winner carries a selection effect. Stages 2
  and 3 substantially answer it — a variant selected by chance on 11 subjects would not survive 91
  and then 26 disjoint ones — but the Test figure is "the winner of an 8-way search", not a single
  pre-specified model.
* **Face-detection failure correlates with the label** (*r* = −0.31; clips with no face found are
  100% low-engagement). Dropping all 255 affected clips costs only 0.01 AUC (0.9127) and still
  beats that subset's harder 0.7456 baseline by 11.4 points — so it is a real corpus property but
  not what drives the result.
* **`subject_105` scores AUC 0.299** — the model ranks that person backwards. One of 26 subjects
  below chance; median per-subject AUC 0.8848. Leave-one-subject-out AUC spans 0.9175–0.9278, so no
  single face carries the headline.
* Trained on EngageNet, which is in-the-wild webcam video but **not** this project's learners. No
  claim transfers to the deployed population without measurement.

## Provenance

Every figure traces to a committed JSON here. Nothing derived from EngageNet video is committed:
the EULA forbids redistribution in source or binary form. Cite doi 10.1145/3577190.3614164.
