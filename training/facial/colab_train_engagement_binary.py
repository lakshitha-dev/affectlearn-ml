"""One-shot Colab runner: train a BINARY DISENGAGEMENT detector on DAiSEE and dump predictions.

WHY THIS RUN EXISTS
-------------------
The project reports that engagement is "unreachable as a property of the data". That is settled
for one binary cut and unevidenced for the other:

    ANY  {1,2,3} vs {0}    4 disengaged clips in the whole Test split  -> unmeasurable, forever
    HIGH {2,3}   vs {0,1}  85 disengaged clips                         -> marginal, never tried

The figure the claim rests on -- accuracy 0.9353 against a 0.9481 baseline, kappa 0.1029,
disengaged recall 0.0941 -- is a POST-HOC probability merge of the trained 4-level model.
`reports/facial_confusion/FINDINGS.md:11-15` calls it "pure arithmetic" from a matrix stored
inside the checkpoint. No model was ever TRAINED on the binary disengagement task, and no AUC was
ever computed on the HIGH cut.

Meanwhile the recipe that turned confusion from ~chance (AUC 0.5893) into real signal (0.6414,
kappa 0.2119) -- frozen backbone, direct binary head, balanced sampling -- was applied to
Confusion and to Frustration, and never to Engagement. This run applies it.

BOTH OUTCOMES ARE WORTH THE GPU TIME
------------------------------------
If it detects disengagement, the project gains the construct its title names. If it does not, the
negative claim stops resting on arithmetic and starts resting on a trained model under the same
recipe that succeeded on two other targets -- which is exactly the negative-control logic of
Section 4.5.2. What is NOT acceptable is leaving it untested and asserting it either way.

WHAT THIS RUN CAN AND CANNOT SETTLE
-----------------------------------
85 disengaged test clips bound the precision of any rate estimated from them: the worst-case 95%
half-width on a recall is ~1.96*sqrt(0.25/85) = +/-10.6 points. So this run can distinguish
"clearly useless" from "possibly usable". It cannot produce a tight estimate, and any number it
yields must be quoted with that interval rather than as a point.

USAGE (mirrors colab_train_confusion.py -- the bridge drops on long runs, so detach and poll)
--------------------------------------------------------------------------------------------
    !cd /content && nohup python colab_train_engagement_binary.py --stage all \
        > /content/drive/MyDrive/affectlearn-ml/logs/engagement_binary.log 2>&1 &

    !tail -40 /content/drive/MyDrive/affectlearn-ml/logs/engagement_binary.log

    --stage audit        labels only, seconds, no GPU. Records the counts that exist NOWHERE
                         in this project: engagement per-level counts for Train and Validation.
    --stage preprocess   video -> .npy. Skipped entirely if the Engagement or Confusion run
                         already cached it -- .npy clips carry no label, so the cache is shared.
    --stage train        training + ONNX export -> cnn_lstm_engagement_highcut.*
    --stage evaluate     test predictions -> .npz + metrics json
    --stage all          audit -> preprocess -> train -> evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("engagement-binary")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DRIVE = Path("/content/drive/MyDrive/affectlearn-ml")
LABELS_DIR = DRIVE / "data/DAiSEE/Labels"
LOCAL_PREPROCESSED = Path("/content/daisee_preprocessed")
PREPROCESSED_CACHE = DRIVE / "data/daisee_preprocessed"
OUT_DIR = DRIVE / "reports/facial_engagement"

TARGET = "Engagement"
BINARY_CUT = "high"          # {2,3} engaged vs {0,1} disengaged
# Below this share of minority examples on Test, no rate can be estimated to a useful precision:
# 25 examples is the point where the worst-case 95% half-width on a recall drops under 20 points.
MIN_MINORITY_TEST_N = 25


def sh(cmd: list[str], cwd: Path | None = None) -> int:
    log.info("$ %s", " ".join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd], cwd=str(cwd) if cwd else None)


def recall_half_width(n: int) -> float:
    return 1.96 * (0.25 / n) ** 0.5 if n else float("inf")


def stage_audit(out_dir: Path) -> dict:
    """Audit the ENGAGEMENT labels -- which this project has never done.

    `check_daisee_labels.py` computes all three splits, but was only ever invoked with
    `--target Confusion`, and its output went to Drive and was never committed. So the
    engagement per-level counts for Train and Validation exist nowhere: not in the repo, not in
    the thesis, not in any branch. This stage records them, which closes a gap in Chapter 4
    independently of whether the training that follows succeeds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "engagement_label_audit.json"
    rc = sh([sys.executable, HERE / "check_daisee_labels.py",
             "--labels-dir", LABELS_DIR, "--target", TARGET, "--out", out])
    if rc != 0:
        raise SystemExit("label audit failed — check the labels path before spending GPU time")

    audit = json.loads(out.read_text(encoding="utf-8"))
    for split, s in audit.get("splits", {}).items():
        log.info("%-11s n=%-6s levels=%s", split, s.get("n"), s.get("levels"))

    test = audit["splits"].get("Test")
    if not test:
        raise SystemExit("no Test split in the audit; cannot judge reportability")

    frac = test["binary_high"]["fraction"]          # share ENGAGED under the HIGH cut
    minority_n = int(round(min(frac, 1 - frac) * test["n"]))
    log.info("TEST HIGH cut: %.1f%% engaged -> %d disengaged clips", 100 * frac, minority_n)
    log.info("a recall estimated on %d examples carries +/-%.1f points (worst case, 95%%)",
             minority_n, 100 * recall_half_width(minority_n))

    # State the comparison up front, so the result cannot be misread later as a success.
    log.info("majority baseline on this cut is %.4f — the accuracy to beat is NOT 0.50",
             max(frac, 1 - frac))

    if minority_n < MIN_MINORITY_TEST_N:
        raise SystemExit(
            f"STOP. Only {minority_n} minority clips on Test. A rate estimated from that many "
            f"carries +/-{100 * recall_half_width(minority_n):.0f} points, which cannot support any "
            "claim in either direction. This is the ANY-cut situation (4 clips) that produced the "
            "AUC of 0.7942 this project disowns. Do not spend GPU time to generate a number that "
            "cannot be interpreted.")

    log.info("audit PASSED. Report AUC, kappa and disengaged RECALL with its interval — "
             "never bare accuracy, which the baseline dominates on this cut.")
    return audit


def stage_preprocess(cfg_path: Path) -> None:
    """Decode video -> .npy, reusing any existing cache.

    The .npy clips are decoded frames and carry NO label, so whatever the Engagement or
    Confusion run already cached is byte-identical to what this needs. Preprocessing is the
    expensive stage; training is not.
    """
    if LOCAL_PREPROCESSED.exists() and any(LOCAL_PREPROCESSED.rglob("*.npy")):
        log.info("preprocessed clips already on local disk — skipping")
        return
    if PREPROCESSED_CACHE.exists() and any(PREPROCESSED_CACHE.rglob("*.npy")):
        log.info("found cached preprocessing on Drive — copying to local disk")
        shutil.copytree(PREPROCESSED_CACHE, LOCAL_PREPROCESSED, dirs_exist_ok=True)
        return
    log.info("no cache — running full preprocessing (the slow stage)")
    if sh([sys.executable, HERE / "preprocess.py", "--config", cfg_path]) != 0:
        raise SystemExit("preprocessing failed")
    PREPROCESSED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(LOCAL_PREPROCESSED, PREPROCESSED_CACHE, dirs_exist_ok=True)


def write_config(base_cfg: Path) -> Path:
    """Emit a binary-engagement config beside the original rather than editing in place.

    `binary_cut: high` makes `train_cnn_lstm.py` derive num_classes=2 and name the checkpoint
    `cnn_lstm_engagement_highcut` — NOT `cnn_lstm_best`, which belongs to the deployed 4-level
    engagement model and whose 47 MB ONNX the backend serves. Overwriting it is unrecoverable.

    The recipe is the one that worked for Confusion: freeze the whole backbone (~800k trainable)
    and let balanced sampling handle the imbalance. Unfreezing layer4 gave 11.9M trainable
    parameters, and training loss fell 0.1678 -> 0.0528 while validation DECLINED — 12M
    parameters memorising 4,852 clips. With ~250 disengaged training examples here, that failure
    mode is closer, not further away.
    """
    import yaml
    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    cfg["target_affect"] = TARGET
    cfg["binary_cut"] = BINARY_CUT
    cfg.setdefault("training", {})
    cfg["training"]["balanced_sampling"] = True
    cfg["training"]["freeze_backbone_full"] = True
    cfg["training"]["freeze_backbone"] = False
    out = base_cfg.parent / "config.engagement_binary.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log.info("wrote %s (target=%s, cut=%s, frozen backbone, balanced sampling)",
             out, TARGET, BINARY_CUT)
    log.info("checkpoints -> cnn_lstm_engagement_highcut.pt/.onnx (cnn_lstm_best is untouched)")
    return out


def stage_train(cfg_path: Path) -> None:
    if sh([sys.executable, HERE / "train_cnn_lstm.py", "--config", cfg_path]) != 0:
        raise SystemExit("training failed")


def stage_evaluate(cfg_path: Path, out_dir: Path) -> None:
    """Dump per-clip test probabilities, so every later question is answerable offline.

    This is the most valuable artifact of the run. No facial per-clip probabilities were ever
    committed in this project, which is why every DAiSEE figure in Chapter 4 is prose that
    cannot be re-derived, and why the binary recompute needed Colab at all.
    """
    ckpt = Path("/content/checkpoints") / f"cnn_lstm_{TARGET.lower()}_{BINARY_CUT}cut.pt"
    if not ckpt.exists():
        alt = DRIVE / "checkpoints" / ckpt.name
        if not alt.exists():
            raise SystemExit(f"{ckpt} not found — run --stage train first")
        ckpt = alt
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"engagement_{BINARY_CUT}cut_test.npz"
    rc = sh([sys.executable, REPO / "evaluation/dump_facial_predictions.py",
             "--checkpoint", ckpt, "--config", cfg_path,
             "--split", "Test", "--target", TARGET, "--binary-cut", BINARY_CUT,
             "--out", npz])
    if rc != 0:
        raise SystemExit("prediction dump failed")
    log.info("every metric is now recomputable offline, with no GPU and no corpus:")
    log.info("  python evaluation/confusion_matrix.py --predictions %s", npz)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=("audit", "preprocess", "train", "evaluate", "all"))
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    base_cfg = Path(args.config)
    out_dir = Path(args.out_dir)

    if args.stage in ("audit", "all"):
        stage_audit(out_dir)
    cfg_path = write_config(base_cfg)
    if args.stage in ("preprocess", "all"):
        stage_preprocess(cfg_path)
    if args.stage in ("train", "all"):
        stage_train(cfg_path)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(cfg_path, out_dir)

    log.info("done. Read the disengaged RECALL and the AUC, each beside the majority baseline; "
             "quote the +/-10.6 point interval that 85 test examples imposes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
