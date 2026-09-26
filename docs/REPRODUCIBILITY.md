# Reproducing the paper's system and results

This supports paper V4 ("A Gated Affective Multi-Agent System for Pedagogical Adaptation in Online Learning"). It records what the code alone does not: the deployed configuration, where the prompts live, and how each reported number is produced.

## 1. Deployed configuration against code defaults

Several code defaults would change behaviour if the deployment settings were lost. The deployed values were read on 25 Sep 2026; see `docs/paper/V3/live-config-2026-09-25.json` in the documents repository. Paths are in `affectlearn/backend/app/`.

| Setting | Code default | Deployed | Effect of the default |
|---|---|---|---|
| `ADAPT_STATES` (`agents/edges.py`) | `bored,confused,frustrated` | `bored,confused` | Would act on frustration, which no channel detects |
| `ADAPT_WITHHOLD_RATE` (`agents/edges.py`) | 0.35 | 0 | Would withhold 35% of eligible offers |
| `ADAPT_MIN_CONFIDENCE`, the global floor (`agents/edges.py`) | 0.70 | 0.50 | The global floor applies only to channels without their own floor |
| Per-channel floors | geometry 0.70, behavioural 0.70, performance 0.60 | the same (behavioural floor since 6 Sep; 0.50 global before) | none |
| `FUSION_DRIVES_DECISION` (`agents/nodes/affect_detection.py`) | 1 | 0 | Fused readings would decide |
| `AFFECT_MODEL_KIND` (`agents/affect_mapping.py`) | `engagement` | `geometry` | Every geometry window, engaged ones included, would be labelled bored |
| LLM (`core/config.py`) | `http://vllm:8080`, `affectlearn/llama-3-8b-pedagogical`, 3 s timeout | `https://api.openai.com`, `gpt-4o` (alias), 15 s | A different model and timeout |
| Behavioural model (`services/behavioral_inference.py`) | `models/behavioral_bilstm.onnx` (trained on synthetic data) | `models/behavioral_confusion_gbdt.onnx` | A different, synthetic-data model |

**Code versions.** The paper describes `affectlearn` `main` at `b6d2b58`, as deployed. The current code base is `develop` at `b4496cf`, which adds delivery-only offer accounting, the server-clock cooldown and the lesson quiz hold (PRs #138 and #139). These are not deployed.

## 2. Prompts and LLM settings

- **Pedagogical Agent prompt:** `agents/nodes/pedagogical.py` (system prompt and message builder). Parsing and overrides are in the same file; actions and ladders are in `agents/fallbacks.py`.
- **Content Adapter prompt:** `agents/nodes/content_adapter.py`. The section context and quiz masking are in `services/content_context_service.py`.
- **Video sub-agent:** `services/video_resource_agent.py` (query and choice prompts; 3 s and 2 s LLM steps; 5 s budget).
- **Settings:** temperature 0.3, 400 output tokens. The deployed model is the hosted `gpt-4o` alias; the snapshot it resolved to during the record was not logged. The offline evaluation (`backend/scripts/eval_llm_offline.py`) records the snapshot the API returns.

## 3. Detection models

**Geometry (EngageNet).**

- **Features:** `training/engagenet/rungs.py` (`4_lean`, 20 features).
- **Screening:** `SEARCH_PROTOCOL.md`. The original screening run was lost; the reported screen is a re-run made after test scoring, and the surviving earlier screen (`C:/engagenet/results/regression.json`, 4 variants) also ranks `4_lean` first.
- **Fit and export:** `training/engagenet/fit_export_lean.py`. A refit under scikit-learn 1.9 gives AUC 0.9239 against the served model's 0.9225 (per-clip differences up to 0.163); the served ONNX reproduces the committed test predictions to 1e-7.
- **Training subset:** at most 27 clips per participant, drawn in proportion to each participant's labels. All 79 capped participants satisfy n_pos = round(27 × rate). The file list is `C:/engagenet/trainsub_files.txt`; the script that drew it was not preserved.

**Interaction (DUX).** `training/behavioral/train_dux_confusion.py` and `export_dux_confusion.py` (raw features, an all-data refit of the leave-one-session-out configuration). The review runs are in `training/behavioral/review/dux_review.py`.

**GBDT configuration.** Depth 3, 120 iterations, learning rate 0.06, minimum leaf 8, L2 1.0, balanced class weights. It has been in the code since the first DUX experiments (14 Aug 2026), before any EngageNet data were scored. No hyperparameter search exists in the repository.

## 4. Where each paper number comes from

The full claim-to-source map is `docs/paper/V4/v4-sources.md` in the documents repository. The review experiments:

| Experiment | Script | Output |
|---|---|---|
| Deployed-rule gate replay (Fig. 2) | `evaluation/gate_deployed_rule.py` | `reports/gate_deployed_rule/sweep.json` |
| Gate ablation | `evaluation/gate_deployed_rule.py --ablation` | `reports/gate_deployed_rule/ablation.json` |
| Out-of-sample facial floor | `evaluation/facial_floor_oos.py` | `reports/engagenet_review/oos_floor.json` |
| EngageNet baselines and four-class | `training/engagenet/review/baselines.py` | `reports/engagenet_review/baselines.json` |
| Table 3 accuracy, balanced accuracy and κ | `training/engagenet/review/table3_metrics.py`; `training/behavioral/review/late_fusion_and_metrics.py` | `reports/*/table3_metrics.json` |
| DUX arms, platform-like input, calibration, nested floor | `training/behavioral/review/dux_review.py` | `reports/dux_review/*.json` |
| Late fusion | `training/behavioral/review/late_fusion_and_metrics.py` | `reports/dux_review/late_fusion.json` |
| DUX label threshold | `training/behavioral/review/label_threshold.py` | `reports/dux_review/label_threshold.json` |

Corpora are not redistributed. EngageNet is licensed for research use, and DUX is CC BY 4.0 (Zenodo 10.5281/zenodo.7778612). Only aggregates are committed.
