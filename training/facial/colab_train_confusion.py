"""One-shot Colab runner: train Model A on DAiSEE CONFUSION and dump reportable test predictions.

WHY A DETACHED SCRIPT RATHER THAN NOTEBOOK CELLS
------------------------------------------------
The Colab bridge drops mid-session on long runs (recorded in this project's own notes). So this is
written to be launched once with nohup, logging to Drive, and polled — never held open in an
interactive cell:

    !cd /content && nohup python colab_train_confusion.py --stage all \
        > /content/drive/MyDrive/affectlearn-ml/logs/confusion_run.log 2>&1 &

Then poll:  !tail -30 /content/drive/MyDrive/affectlearn-ml/logs/confusion_run.log

WHAT IT DOES, AND THE TWO THINGS THAT MAKE IT CHEAP
---------------------------------------------------
1. AUDIT the labels first and REFUSE to train if a binary confusion split is not viable. A 4-hour
   GPU run that produces a meaningless 95%-accuracy model is the expensive mistake here, and it is
   entirely avoidable in 30 seconds.
2. REUSE cached preprocessing. The .npy clips are decoded video frames and carry NO label, so the
   preprocessing done for the Engagement model is byte-identical to what Confusion needs. If a
   cached copy exists on Drive this skips straight to training and saves hours. Preprocessing is
   the expensive stage, not training.
3. TRAIN with target_affect=Confusion, writing to cnn_lstm_confusion.pt (NOT cnn_lstm_best.pt —
   that name belongs to the deployed Engagement model and overwriting it is unrecoverable).
4. DUMP PER-CLIP TEST PROBABILITIES to an .npz via evaluation/predictions.py. This is the most
   valuable artifact of the whole run: every metric — 4-class, binary at any threshold, AUC, kappa,
   confusion matrix — becomes recomputable offline forever, with no GPU, no Drive, and no 17 GB
   dataset. It also closes the long-standing gap that made the Model A binary recompute need Colab
   at all.

Stages are separable so a dropped session does not restart from zero:
    --stage audit        labels only, seconds, no GPU
    --stage preprocess   video -> .npy (slow; skipped if cached)
    --stage train        training + ONNX export
    --stage evaluate     test-split predictions -> .npz + metrics
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

log = logging.getLogger("confusion-run")

DRIVE = Path("/content/drive/MyDrive/affectlearn-ml")
HERE = Path(__file__).resolve().parent

# A cached copy of the decoded clips on Drive. Preprocessing is target-independent, so this is
# reusable across Engagement / Confusion / Boredom / Frustration runs.
PREPROCESSED_CACHE = DRIVE / "data" / "daisee_preprocessed"
LOCAL_PREPROCESSED = Path("/content/daisee_preprocessed")

MIN_POSITIVE_FRACTION = 0.05     # below this, a binary collapse is not defensible at all


def sh(cmd: list[str], cwd: Path | None = None) -> int:
    log.info("$ %s", " ".join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd], cwd=str(cwd) if cwd else None)


def stage_audit(labels_dir: Path, out: Path) -> dict:
    """Refuse to spend GPU time on an indefensible target."""
    rc = sh([sys.executable, HERE / "check_daisee_labels.py",
             "--labels-dir", labels_dir, "--target", "Confusion", "--out", out])
    if rc != 0:
        raise SystemExit("label audit failed — fix the labels path before continuing")
    audit = json.loads(out.read_text(encoding="utf-8"))
    test = audit["splits"].get("Test")
    if not test:
        raise SystemExit("no Test split in the audit; cannot judge reportability")

    high, anyc = test["binary_high"]["fraction"], test["binary_any"]["fraction"]
    best_cut = "binary_high" if min(high, 1 - high) >= min(anyc, 1 - anyc) else "binary_any"
    best_frac = test[best_cut]["fraction"]
    log.info("TEST positives — HIGH cut %.1f%%, ANY cut %.1f%%; better balanced: %s",
             100 * high, 100 * anyc, best_cut)

    if best_frac < MIN_POSITIVE_FRACTION:
        raise SystemExit(
            f"STOP. The best binary cut has only {100 * best_frac:.1f}% positives on Test. A binary "
            "confusion model here would score high accuracy while detecting almost nothing — the "
            "exact failure this project already measured (accuracy 0.724->0.934, kappa "
            "0.462->0.025, minority recall 0.020). Train the 4-level ordinal target instead, or "
            "pick a different state. Do not proceed just because the accuracy will look good.")
    log.info("audit PASSED — proceeding. Report AUC and positive-class recall, not bare accuracy.")
    return audit


def stage_preprocess(cfg_path: Path) -> None:
    """Decode video -> .npy, reusing the Drive cache when present."""
    if LOCAL_PREPROCESSED.exists() and any(LOCAL_PREPROCESSED.rglob("*.npy")):
        log.info("preprocessed clips already on local disk — skipping")
        return
    if PREPROCESSED_CACHE.exists() and any(PREPROCESSED_CACHE.rglob("*.npy")):
        # Copying from Drive is far cheaper than re-decoding 17 GB of video, and training I/O must
        # come off local disk — reading .npy per batch straight from Drive is pathologically slow.
        log.info("found cached preprocessing on Drive — copying to local disk (target-independent, "
                 "so the Engagement run's clips are exactly what Confusion needs)")
        shutil.copytree(PREPROCESSED_CACHE, LOCAL_PREPROCESSED, dirs_exist_ok=True)
        return
    log.info("no cache — running full preprocessing (this is the slow stage)")
    if sh([sys.executable, HERE / "preprocess.py", "--config", cfg_path]) != 0:
        raise SystemExit("preprocessing failed")
    log.info("caching preprocessed clips to Drive so no future target has to redo this")
    PREPROCESSED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(LOCAL_PREPROCESSED, PREPROCESSED_CACHE, dirs_exist_ok=True)


def write_confusion_config(base_cfg: Path) -> Path:
    """Emit a Confusion config beside the original rather than editing it in place."""
    import yaml
    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    cfg["target_affect"] = "Confusion"
    out = base_cfg.parent / "config.confusion.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log.info("wrote %s (target_affect=Confusion; checkpoints -> cnn_lstm_confusion.pt/.onnx)", out)
    return out


def stage_train(cfg_path: Path) -> None:
    if sh([sys.executable, HERE / "train_cnn_lstm.py", "--config", cfg_path]) != 0:
        raise SystemExit("training failed")


def stage_evaluate(cfg_path: Path, out_dir: Path) -> None:
    """Dump per-clip TEST probabilities, then compute metrics from that dump.

    Order matters: the .npz is written FIRST and is the durable artifact. Metrics are derived from
    it, so a change of mind about thresholds or class merges never requires the GPU again.
    """
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parents[1] / "evaluation"))
    from dataset import DAiSEEDataset                      # noqa: E402
    from model import build_model                          # noqa: E402
    from confusion_matrix import compute, plot, summary     # noqa: E402
    from predictions import Predictions                    # noqa: E402

    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    target = cfg.get("target_affect", "Confusion")
    stem = "cnn_lstm_best" if target == "Engagement" else f"cnn_lstm_{target.lower()}"
    ckpt_path = Path(cfg["paths"]["checkpoints"]) / f"{stem}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} not found — run --stage train first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    stamped = ckpt.get("target_affect")
    if stamped and stamped != target:
        raise SystemExit(f"checkpoint is for target={stamped} but config says {target} — refusing")

    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = DAiSEEDataset(cfg["paths"]["daisee_preprocessed"],
                       str(Path(cfg["paths"]["labels_dir"]) / "TestLabels.csv"),
                       split="Test", target=target, augment=False)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                        num_workers=cfg["training"]["num_workers"])
    log.info("scoring %d TEST clips (never trained on, never selected on)", len(ds))

    probs, ys = [], []
    with torch.no_grad():
        for clips, labels in loader:
            p = torch.softmax(model(clips.to(device)), dim=1)
            probs.append(p.cpu().numpy())
            ys.append(labels.numpy())
    prob = np.concatenate(probs)
    y = np.concatenate(ys)

    out_dir.mkdir(parents=True, exist_ok=True)
    labels4 = ["very_low", "low", "high", "very_high"]
    ids = [p.stem for p in ds.clips]
    recs = Predictions(y, prob.argmax(1), labels4, y_prob=prob, ids=ids)
    recs.save(out_dir / f"{target.lower()}_test_predictions.npz")
    log.info("WROTE the durable artifact: %s — every metric is now recomputable offline, forever, "
             "with no GPU and no DAiSEE", out_dir / f"{target.lower()}_test_predictions.npz")

    m4 = compute(recs)
    print(f"\n=== {target} — 4-level, held-out TEST ===")
    print(summary(m4))
    plot(recs, out_dir / f"{target.lower()}_test_matrix_4class.png", normalize="true",
         title=f"DAiSEE {target} — 4 levels, test split")

    # Both candidate binarisations, from the SAME dump. collapse() sums probabilities rather than
    # remapping the argmax, which is the correct operation and differs from remapping.
    results = {"target": target, "n_test": int(len(y)), "four_class":
               {k: v for k, v in m4.items() if k != "matrix"}, "binary": {}}
    cuts = {
        # 'high': the direct analogue of the engagement merge — levels 2,3 count as confused.
        "high": {"not_confused": ["very_low", "low"], "confused": ["high", "very_high"]},
        # 'any': any non-zero annotation counts. Usually better balanced when the state is rare.
        "any": {"not_confused": ["very_low"], "confused": ["low", "high", "very_high"]},
    }
    for cut, mapping in cuts.items():
        b = recs.collapse(mapping)
        mb = compute(b)
        base = max((b.y_true == 0).mean(), (b.y_true == 1).mean())
        print(f"\n=== {target} — BINARY '{cut}' cut, held-out TEST ===")
        print(f"  majority baseline {base:.4f} accuracy — beat this or nothing was detected")
        print(summary(mb))
        b.save(out_dir / f"{target.lower()}_test_predictions_binary_{cut}.npz")
        plot(b, out_dir / f"{target.lower()}_test_matrix_binary_{cut}.png", normalize="true",
             title=f"DAiSEE {target} — binary ({cut} cut), test split")
        results["binary"][cut] = {"majority_baseline": float(base),
                                  **{k: v for k, v in mb.items() if k != "matrix"}}

    (out_dir / f"{target.lower()}_test_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    sel = ckpt.get("val_f1")
    if sel is not None:
        print(f"\n  selection bias check: val weighted-F1 {sel:.4f} (SELECTED on) vs test "
              f"{m4['weighted_f1']:.4f} (honest). Quote the test number.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "audit", "preprocess", "train", "evaluate"])
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--out-dir", default=str(DRIVE / "reports" / "facial_confusion"))
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout, force=True)
    base_cfg = Path(a.config)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    labels_dir = Path(yaml.safe_load(base_cfg.read_text(encoding="utf-8"))["paths"]["labels_dir"])

    if a.stage in ("all", "audit"):
        stage_audit(labels_dir, out_dir / "daisee_confusion_label_audit.json")
        if a.stage == "audit":
            return 0

    cfg_path = write_confusion_config(base_cfg)
    if a.stage in ("all", "preprocess"):
        stage_preprocess(cfg_path)
        if a.stage == "preprocess":
            return 0
    if a.stage in ("all", "train"):
        stage_train(cfg_path)
        if a.stage == "train":
            return 0
    if a.stage in ("all", "evaluate"):
        stage_evaluate(cfg_path, out_dir)

    log.info("DONE. The reportable artifact is the test-predictions .npz in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
