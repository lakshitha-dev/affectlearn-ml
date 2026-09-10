# Method search protocol - EngageNet facial disengagement

Fixed before any number existed. Recorded here because the Colab runtime holding it was lost.

## Three stages, in order

1. **SCREEN on Validation** (11 subjects, 1,071 clips). Try variants freely. Nothing here is
   reportable; the point is to choose one candidate.
2. **CONFIRM on Train** (91 subjects, 2,414-clip subject-stratified subsample). The screening
   winner only. If it does not hold on 91 independent subjects it is not real.
3. **REPORT on Test** (26 subjects, 2,257 clips, majority baseline 68.8%). Scored **exactly once**,
   after stage 2. Sealed until then.

## Five metrics, every rung, no exceptions

| metric | why |
|---|---|
| AUC + **subject-level** bootstrap 95% CI | clips of one person are not independent observations |
| **within-subject** permutation p | a model that learned identity cannot pass this |
| accuracy **beside** its majority baseline | never one without the other |
| average precision / base rate | at ~31% positives, ranking can flatter precision |
| **score spread** (max - min of P) | the previous model died on a 0.127-wide band, not on AUC |

## Pre-registered success criteria

- **Floor:** beat the Test majority baseline of 68.8%, with the subject-level CI excluding chance
  and within-subject permutation p < 0.05.
- **Target:** approximately 73.6%, the published EngageNet figure.
- **No ceiling claimed.** Nobody reaches 90% on this benchmark.
- **A negative result is permitted and must be reported as one.**
- **Not a criterion:** beating the DAiSEE number.

## Selection-effect accounting

8 variants were screened on Validation. That is recorded so the eventual Test number can be read
honestly rather than as if a single pre-specified model had been fit once.

## Threshold provenance

`AWAY_DEG` was changed from (10, 15, 20) to (3, 5, 8, 12) during screening, chosen from the
**label-free** marginal distribution of |yaw|/|pitch| over Validation, before any model was fit.
No label was consulted, so nothing leaked; Test was untouched.
