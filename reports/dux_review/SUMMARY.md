# DUX confusion channel: review experiments

Generated from the JSON files in this directory by `training/behavioral/review/dux_review.py`
(platform-like loader: `training/behavioral/review/input_matched.py`). The deployed-rule gate sweep
for both channels is in `reports/gate_deployed_rule/sweep.json` (`evaluation/gate_deployed_rule.py`).

Settings for every number: DUX v0 + v1, 46 sessions, 1,419 windows of 30 s, 257 confused (base rate
0.181); leave-one-session-out; the deployed GBDT configuration (depth 3, 120 iterations, lr 0.06,
leaf 8, L2 1.0, balanced), seed 42; 95% intervals resample sessions (2,000 draws); permutation p
shuffles labels within each session (2,000 draws, floor 0.0005).

## 1. Detection by normalisation (`raw_and_normalised_arms.json`)

| Arm | AUC [95% CI] | kappa at 0.5 | Accuracy (majority 0.819) | AP / base rate | p |
|---|---|---|---|---|---|
| Interaction, raw (deployed) | 0.734 [0.700, 0.767] | 0.219 | 0.684 | 2.00 | 0.0005 |
| Interaction, causal | 0.741 [0.705, 0.776] | 0.291 | 0.739 | 2.11 | 0.0005 |
| Interaction, transductive | 0.747 [0.711, 0.781] | 0.274 | 0.730 | 1.93 | 0.0005 |
| AFFDEX facial, raw | 0.477 [0.439, 0.521] | -0.025 | 0.625 | 0.94 | 0.139 |
| AFFDEX facial, causal | 0.551 [0.510, 0.590] | 0.001 | 0.649 | 1.09 | 0.001 |
| AFFDEX facial, transductive | 0.553 [0.507, 0.600] | 0.057 | 0.677 | 1.16 | 0.0005 |
| Fusion, raw | 0.681 [0.641, 0.720] | 0.203 | 0.697 | 1.60 | 0.0005 |
| Fusion, causal | 0.723 [0.690, 0.752] | 0.267 | 0.743 | 2.02 | 0.0005 |
| Fusion, transductive | 0.735 [0.697, 0.771] | 0.268 | 0.739 | 1.92 | 0.0005 |

The transductive rows reproduce the committed 0.7473 / 0.5532 / 0.7351. The raw interval replaces the
one in `reports/dux_confusion/normalisation.json`, which resampled windows, not sessions.

## 2. Paired session-bootstrap differences (`paired_differences.json`)

| Normalisation | Interaction - fusion | Interaction - facial |
|---|---|---|
| Raw | +0.054 [+0.030, +0.077] | +0.258 [+0.211, +0.303] |
| Causal | +0.018 [+0.002, +0.036] | +0.189 [+0.137, +0.244] |
| Transductive | +0.012 [-0.008, +0.031] | +0.194 [+0.135, +0.253] |

## 3. Platform-like inputs (`input_matched.json`)

Pointer resampled to one sample per 100 ms slot (latest position), one click per physical click
(MouseButtonDown; MouseClick only without an unmatched down in the previous 1,000 ms; double clicks
dropped), and the serving window rule (scored if at least one event: 1,421 windows, same 257
positives).

| Arm | AUC [95% CI] | Max score | 99th percentile |
|---|---|---|---|
| Native inputs (reference) | 0.734 | 0.881 | 0.809 |
| Platform-like, same 1,419 windows | 0.703 [0.674, 0.732] | 0.829 | 0.784 |
| Platform-like, serving-rule windows | 0.703 [0.675, 0.731] | 0.819 | 0.775 |
| Native-trained model scored on platform-like inputs | 0.678 [0.647, 0.710] | 0.726 | 0.685 |

Native minus platform-like (same windows): +0.031 [+0.009, +0.055]. Feature shift, native to
platform-like: hover_dwell_mean max 2.10 s to 0.90 s; click_count mean 0.245 to 0.130;
mouse_velocity_std max 54.8 to 4.24. The live platform maximum was 0.633.

## 4. Gate sweeps (`gate_sweeps.json`; deployed rule also in `../gate_deployed_rule/sweep.json`)

At a floor of 0.70:

| Predictions | Offline replay: offers, precision [CI], lift | Deployed rule: offers, precision [CI], lift |
|---|---|---|
| Transductive (paper-v3 Table 5) | 18, 0.500 [0.273, 0.737], 2.76 | 51, 0.471 [0.349, 0.581], 2.60 |
| Raw (deployed model) | 22, 0.364 [0.158, 0.571], 2.01 | 64, 0.422 [0.294, 0.545], 2.33 |
| Platform-like, serving rule | 18, 0.278 [0.067, 0.500], 1.54 | 63, 0.270 [0.164, 0.382], 1.49 |

Offline replay = `evaluation/gate_calibration.simulate` (both readings must clear the floor; next
offer four cycles later). Deployed rule = `backend/app/agents/edges.py` (only the current reading must
clear the floor; the previous reading must share the state; next offer three cycles later).

## 5. Nested floor selection (`nested_floor.json`, offline replay)

Rule fixed before running: in each training fold, the lowest floor whose inner 5-fold precision is at
least twice the inner base rate with at least 10 offers, else 0.90.

| Predictions | Offers | Precision [CI] | Lift |
|---|---|---|---|
| Raw | 34 | 0.441 [0.289, 0.595] | 2.44 |
| Platform-like, same windows | 56 | 0.339 [0.213, 0.475] | 1.87 |
| Platform-like, serving rule | 61 | 0.344 [0.224, 0.462] | 1.90 |

## 6. Calibration (`calibration.json`)

Raw model: ECE 0.201, Brier 0.187, mean score 0.382 against a base rate of 0.181; windows scored 0.7
to 0.8 are 41.6% positive and 0.8 to 0.9 are 38.9% positive. Nested Platt scaling: ECE 0.016,
Brier 0.134, AUC 0.729, and no calibrated score exceeds about 0.53, so the calibrated sweep makes no
offer at any floor from 0.50. Platform-like inputs: ECE 0.205 to 0.015.

## 7. Baselines on raw interaction features (`baselines.json`)

Logistic regression 0.692 [0.656, 0.730] (GBDT +0.042 [+0.011, +0.072]); random forest 0.722
[0.694, 0.752] (GBDT +0.012 [-0.008, +0.032]); majority accuracy 0.819.

## 8. By release, raw features (`robustness_v0_v1.json`)

v0 only (10 sessions): interaction 0.599 [0.558, 0.677], facial 0.664 [0.569, 0.717].
v1 only (36 sessions): interaction 0.742 [0.701, 0.785], facial 0.479 [0.430, 0.527] (p 0.22).

## 9. Frustration (`frustration.json`, read from `reports/dux_multistate/multistate.json`)

v1 only, 5 folds, Anger (45 windows) as proxy: Anger recall 0.000 (interaction), 0.044 (facial),
0.022 (fusion).
