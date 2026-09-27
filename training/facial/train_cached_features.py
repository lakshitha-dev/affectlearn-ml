"""Train the temporal head on cached ResNet18 features, for any DAiSEE target and binary cut.

WHY CACHED FEATURES
-------------------
With the backbone frozen its output per clip never changes between epochs, so the expensive
part -- 16 ResNet18 forwards per clip -- only has to happen once. Training then touches a
260 MB feature array instead of 13.6 GB of pixels, which turns a 1-4 hour run into seconds.
That is what makes repeated seeds and subject-grouped cross-validation affordable, and with a
minority class of 234 training clips a single run is not evidence of much.

ONE HONEST DIFFERENCE FROM THE ORIGINAL PIPELINE
------------------------------------------------
`train_cnn_lstm.py` freezes the backbone with `requires_grad_(False)` but never calls `.eval()`,
and puts the whole model in `.train()` each epoch -- so the "frozen" ResNet18's BatchNorm running
statistics kept adapting to DAiSEE on every previous run. Caching requires BN in eval mode. That
is arguably more correct, but it is NOT identical, so any result from this script is only
trustworthy once the confusion positive control reproduces the published AUC of 0.6414. Run:

    python training/facial/train_cached_features.py --target Confusion --cut any ...

before believing anything this produces about engagement.

WHY SELECTION IS ON AUC
-----------------------
`train_cnn_lstm.py:274` selects the best checkpoint on validation weighted F1. On the engagement
HIGH cut the validation majority class is 88.4%, so weighted F1 is dominated by it and would
happily select a model that never predicts the minority at all. AUC is threshold-free and
prevalence-robust, and it is the metric the result is reported in.

WHY INTERVALS RESAMPLE SUBJECTS
-------------------------------
DAiSEE's 85 disengaged test clips come from 16 subjects, and one subject supplies 25 of them.
Clips of one person share a face, a camera and a session; treating them as independent gives a
95% half-width of +/-10.4 points when the honest figure, at a plausible intra-subject
correlation, is +/-16 to +/-21. So the bootstrap resamples SUBJECTS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.facial.losses import FocalLoss  # noqa: E402

CUTS = {"any": (1, 2, 3), "high": (2, 3)}
SPLITS = ("Train", "Validation", "Test")


class TemporalHead(nn.Module):
    """The half of CNNLSTMModel that is actually trained when the backbone is frozen.

    Mirrors `model.CNNLSTMModel` exactly from the LSTM onward -- same hidden size, same single
    layer, same dropout on the final hidden state, same linear head -- so a result here is
    comparable to one from the full pipeline.
    """

    def __init__(self, in_dim=512, hidden=256, layers=1, dropout=0.5, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):                       # (B, T, in_dim)
        _, (h_n, _) = self.lstm(x)
        return self.head(self.dropout(h_n[-1]))


SPLIT_CSV = {"Train": "TrainLabels.csv", "Validation": "ValidationLabels.csv",
             "Test": "TestLabels.csv"}


def levels_from_csv(labels_dir: Path, split: str, target: str, clip_ids) -> np.ndarray:
    """Attach any target's ordinal levels to the cached features, joining on clip id.

    The features are a frozen CNN's output and carry no label at all, so one cache serves every
    target -- the same reason `colab_train_confusion.py` reuses the Engagement run's .npy clips.
    Reading the level here rather than trusting whatever was stamped at extraction time means a
    cache built for one target cannot silently be scored against another's labels.
    """
    import csv
    m = {}
    with open(labels_dir / SPLIT_CSV[split], newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}   # DAiSEE has a 'Frustration ' header
            m[row["ClipID"].strip()] = int(row[target])
    missing = [c for c in clip_ids if f"{c}.avi" not in m]
    if missing:
        raise SystemExit(f"{len(missing)} cached clips have no {target} label "
                         f"(first: {missing[:3]}) -- wrong labels dir?")
    return np.array([m[f"{c}.avi"] for c in clip_ids], dtype=np.int64)


def load_split(features_dir: Path, split: str, labels_dir: Path | None = None,
               target: str = "Engagement"):
    d = np.load(features_dir / f"{split}.npz", allow_pickle=False)
    clip_id = d["clip_id"]
    level = (levels_from_csv(labels_dir, split, target, clip_id) if labels_dir is not None
             else d["level"].astype(np.int64))
    return (d["feats"].astype(np.float32), d["feats_flip"].astype(np.float32),
            level, clip_id, d["subject"])


def to_binary(levels: np.ndarray, cut: str) -> np.ndarray:
    return np.isin(levels, CUTS[cut]).astype(np.int64)


def subject_bootstrap(y, p, subjects, n_boot=2000, seed=0):
    """Resample SUBJECTS, not clips. Returns (lo, hi) for AUC, and the point estimate.

    A clip-level bootstrap would treat 85 correlated clips as 85 independent observations and
    report an interval roughly half the honest width.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects)
    idx_by_subj = {s: np.flatnonzero(subjects == s) for s in uniq}
    out = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_subj[s] for s in picked])
        if len(np.unique(y[idx])) < 2:
            continue                      # a resample with one class has no AUC
        out.append(roc_auc_score(y[idx], p[idx]))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def train_one(Xtr, ytr, Xva, yva, *, seed, epochs, lr, wd, gamma, patience, device,
              num_classes=2, dropout=0.5, label_smoothing=0.0):
    """Train the head once. Returns (state_dict of the best epoch, best val AUC)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TemporalHead(num_classes=num_classes, dropout=dropout).to(device)

    # Balanced sampling handles the imbalance, so the loss must NOT also re-weight by class --
    # the double-correction guard train_cnn_lstm.py:131-136 enforces for the same reason.
    counts = np.bincount(ytr, minlength=num_classes).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    sampler = WeightedRandomSampler(torch.as_tensor(w[ytr], dtype=torch.double), len(ytr), True)
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=32, sampler=sampler, drop_last=False)

    crit = FocalLoss(gamma=gamma, alpha=None, label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xva_t = torch.from_numpy(Xva).to(device)

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.softmax(model(Xva_t).float(), 1)[:, 1].cpu().numpy()
        auc = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else float("nan")
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return best_state, best_auc


def predict(state, X, device, num_classes=2, dropout=0.5):
    m = TemporalHead(num_classes=num_classes, dropout=dropout).to(device)
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(torch.from_numpy(X).to(device)).float(), 1)[:, 1].cpu().numpy()


def report(y, p, subjects, label, thresh=0.5):
    pred = (p >= thresh).astype(int)
    counts = np.bincount(y, minlength=2)
    baseline = counts.max() / len(y)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    lo, hi = subject_bootstrap(y, p, subjects)
    rec = float((pred[y == 1] == 1).mean()) if counts[1] else float("nan")
    prec = float((y[pred == 1] == 1).mean()) if pred.any() else float("nan")
    m = {
        "label": label, "n": int(len(y)), "positives": int(counts[1]),
        "subjects": int(len(np.unique(subjects))),
        "auc": float(auc), "auc_ci_lo": lo, "auc_ci_hi": hi,
        "accuracy": float((pred == y).mean()), "majority_baseline": float(baseline),
        "kappa": float(cohen_kappa_score(y, pred)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "positive_recall": rec, "positive_precision": prec,
        "prob_min": float(p.min()), "prob_max": float(p.max()), "prob_mean": float(p.mean()),
    }
    print(f"  {label}")
    print(f"    n={m['n']} positives={m['positives']} subjects={m['subjects']}")
    print(f"    AUC {auc:.4f}  [subject bootstrap 95% CI {lo:.4f}-{hi:.4f}]")
    print(f"    accuracy {m['accuracy']:.4f} vs majority baseline {baseline:.4f}"
          f"{'  ** BELOW BASELINE **' if m['accuracy'] < baseline else ''}")
    print(f"    kappa {m['kappa']:.4f}  positive recall {rec:.4f}  precision {prec:.4f}")
    print(f"    P(positive) spread {p.min():.3f}-{p.max():.3f} (mean {p.mean():.3f})")
    return m


def run_grouped_cv(data, cut, positive, args, device):
    """Pool all three official splits and cross-validate by SUBJECT.

    The official test split holds 85 disengaged clips from 16 subjects, one of which supplies
    28% of them -- too few people to separate a real effect from one person's face, whatever the
    model does. Pooling gives ~485 disengaged clips across ~101 subjects, and every clip gets an
    out-of-fold prediction, so the final interval is computed over five times as many subjects.

    The official splits are subject-disjoint (verified), and `GroupKFold` on the 6-digit subject
    prefix keeps it that way -- no subject ever appears in both the fitting and the scoring half
    of a fold, which is the only thing that makes this comparable to the official protocol.

    Reported as SECONDARY. It is better powered but it is not the published DAiSEE protocol, and
    the headline must remain the split everyone else reports on.
    """
    from sklearn.model_selection import GroupKFold

    feats = np.concatenate([data[s][0] for s in SPLITS])
    flips = np.concatenate([data[s][1] for s in SPLITS])
    levels = np.concatenate([data[s][2] for s in SPLITS])
    clips = np.concatenate([data[s][3] for s in SPLITS])
    subj = np.concatenate([data[s][4] for s in SPLITS])

    y = to_binary(levels, cut)
    if positive == "minority" and y.mean() > 0.5:
        y = 1 - y

    print(f"\nGROUPED CV: {len(y)} clips, {int(y.sum())} positives, "
          f"{len(np.unique(subj))} subjects")

    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(feats, y, groups=subj)):
        # Carve an inner validation set out of the TRAINING subjects for early stopping, so the
        # held-out fold is never touched during fitting.
        tr_subj = np.unique(subj[tr_idx])
        rng = np.random.default_rng(fold)
        inner_val = set(rng.choice(tr_subj, max(2, len(tr_subj) // 5), replace=False).tolist())
        va_mask = np.isin(subj[tr_idx], list(inner_val))
        fit_idx, val_idx = tr_idx[~va_mask], tr_idx[va_mask]

        Xfit = np.concatenate([feats[fit_idx], flips[fit_idx]]) if not args.no_flip else feats[fit_idx]
        yfit = np.concatenate([y[fit_idx], y[fit_idx]]) if not args.no_flip else y[fit_idx]

        if len(np.unique(y[val_idx])) < 2 or len(np.unique(y[te_idx])) < 2:
            print(f"    fold {fold}: skipped (a split holds only one class)")
            continue

        state, vauc = train_one(Xfit, yfit, feats[val_idx], y[val_idx], seed=fold,
                                epochs=args.epochs, lr=args.lr, wd=args.weight_decay,
                                gamma=args.focal_gamma, patience=args.patience, device=device,
                                dropout=args.dropout, label_smoothing=args.label_smoothing)
        oof[te_idx] = predict(state, feats[te_idx], device, dropout=args.dropout)
        print(f"    fold {fold}: {len(te_idx):5} clips, {int(y[te_idx].sum()):4} positives, "
              f"{len(np.unique(subj[te_idx])):3} subjects  inner-val AUC {vauc:.4f}  "
              f"fold AUC {roc_auc_score(y[te_idx], oof[te_idx]):.4f}")

    ok = ~np.isnan(oof)
    print("\nPOOLED OUT-OF-FOLD")
    m = report(y[ok], oof[ok], subj[ok], f"grouped 5-fold CV, {cut}-cut")
    return m, y[ok], oof[ok], subj[ok], clips[ok]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--labels-dir", default=None,
                    help="DAiSEE Labels dir. Given, the target's levels are joined from "
                         "the CSVs by clip id, so one feature cache serves every target.")
    ap.add_argument("--target", default="Engagement")
    ap.add_argument("--cut", default="high", choices=tuple(CUTS))
    ap.add_argument("--positive", default="minority",
                    help="'minority' scores the rare class as positive (disengaged for the "
                         "engagement HIGH cut); 'level' keeps the cut's own polarity.")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--no-flip", action="store_true", help="drop the mirrored copies")
    ap.add_argument("--cv", action="store_true",
                    help="also run the subject-grouped 5-fold CV over all three splits")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fdir = Path(args.features_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ldir = Path(args.labels_dir) if args.labels_dir else None
    data = {s: load_split(fdir, s, ldir, args.target) for s in SPLITS}

    def xy(split, augment):
        f, ff, lv, cid, subj = data[split]
        y = to_binary(lv, args.cut)
        # Score the RARE class as positive. On the engagement HIGH cut the cut's own polarity
        # makes "engaged" positive at 95% prevalence, and a recall of 0.99 on that says nothing.
        if args.positive == "minority" and y.mean() > 0.5:
            y = 1 - y
        if augment and not args.no_flip:
            return np.concatenate([f, ff]), np.concatenate([y, y]), np.concatenate([subj, subj]), cid
        return f, y, subj, cid

    Xtr, ytr, _, _ = xy("Train", True)
    Xva, yva, _, _ = xy("Validation", False)
    Xte, yte, ste, cte = xy("Test", False)
    print(f"target={args.target} cut={args.cut} positive=rare-class "
          f"flip={'off' if args.no_flip else 'on'} device={device}")
    print(f"  train {Xtr.shape} positives {int(ytr.sum())} | "
          f"val {Xva.shape} positives {int(yva.sum())} | "
          f"test {Xte.shape} positives {int(yte.sum())} across {len(np.unique(ste))} subjects")

    probs, val_aucs = [], []
    for s in range(args.seeds):
        state, vauc = train_one(Xtr, ytr, Xva, yva, seed=s, epochs=args.epochs, lr=args.lr,
                                wd=args.weight_decay, gamma=args.focal_gamma,
                                patience=args.patience, device=device,
                                dropout=args.dropout, label_smoothing=args.label_smoothing)
        p = predict(state, Xte, device, dropout=args.dropout)
        probs.append(p); val_aucs.append(vauc)
        print(f"  seed {s}: val AUC {vauc:.4f}  test AUC {roc_auc_score(yte, p):.4f}")

    per_seed = [roc_auc_score(yte, p) for p in probs]
    print(f"\n  across {args.seeds} seeds: test AUC mean {np.mean(per_seed):.4f} "
          f"sd {np.std(per_seed):.4f}  (val AUC mean {np.mean(val_aucs):.4f})")

    mean_p = np.mean(probs, axis=0)
    print("\n  ENSEMBLE (mean of seeds)")
    metrics = report(yte, mean_p, ste, f"{args.target} {args.cut}-cut, official test split")

    # One subject carries 28% of the disengaged test clips; the headline must not rest on it.
    big = max(np.unique(ste), key=lambda s: int(((ste == s) & (yte == 1)).sum()))
    keep = ste != big
    if len(np.unique(yte[keep])) > 1:
        print(f"\n  LEAVE-ONE-SUBJECT-OUT sensitivity (dropping {big}, "
              f"{int(((ste == big) & (yte == 1)).sum())} of {int(yte.sum())} positives)")
        metrics["without_largest_subject"] = report(
            yte[keep], mean_p[keep], ste[keep], f"without subject {big}")

    if args.cv:
        cv_m, cv_y, cv_p, cv_s, cv_c = run_grouped_cv(data, args.cut, args.positive, args, device)
        metrics["grouped_cv"] = cv_m
        np.savez_compressed(Path(args.out).with_name(Path(args.out).stem + "_cv.npz"),
                            y_true=cv_y, y_prob=cv_p, subject=cv_s, clip_id=cv_c)

    metrics.update({"target": args.target, "cut": args.cut, "seeds": args.seeds,
                    "per_seed_test_auc": per_seed, "val_auc_per_seed": val_aucs,
                    "flip_augmentation": not args.no_flip})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out.with_suffix(".npz"), y_true=yte, y_prob=mean_p,
                        subject=ste, clip_id=cte)
    print(f"\n  wrote {out} and {out.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
