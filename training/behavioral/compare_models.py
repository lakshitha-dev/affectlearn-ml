"""Compare four behavioural-model arms on IDENTICAL grouped CV folds.

    flat_bilstm      4-class Bi-LSTM over the (n_bins, N_FEATURES) sequence
    flat_gbdt        4-class gradient boosting over per-channel summary statistics
    two_stage_bilstm engaged-vs-help, then which help state
    two_stage_gbdt   the same head with boosted trees

WHY ALL FOUR. Two questions are open and neither should be settled by assumption:

  1. Does the sequence buy anything at this sample size? Botelho et al. (2017) showed an LSTM
     beating classical ML on these states, but on ~2.5M windows; at a few hundred the ordering
     commonly flips. Gradient boosting on tabular summaries is far more sample-efficient.
  2. Does conditioning help? Flat 4-class is the hardest framing and not what the loop asks.

Every arm sees the same folds, the same groups and the same items, so the comparison is paired and
McNemar applies. The GBDT is sklearn's `HistGradientBoostingClassifier` — histogram-based boosting,
the LightGBM algorithm — chosen because it adds NO dependency to a project that has to deploy on a
CPU-only B1 instance.

METRICS. Reported as mean +- sd across folds, with a noise floor, because a single split cannot
order two models that differ by less than fold-to-fold variation. Per-class recall is always shown:
an aggregate score can rise while a class the system exists to detect goes to zero.

Run:
    python compare_models.py                       # synthetic (pipeline validation)
    python compare_models.py --export ../../data/phase_a/export.json --labelling codebook
    python compare_models.py --folds 5 --seed 42 --out-dir ../../reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2] / "evaluation"))
from aggregate_features import aggregate  # noqa: E402
from cross_validation import stratified_group_folds  # noqa: E402
from dataset import windows_to_arrays  # noqa: E402
from model import build_model  # noqa: E402
from statistical_tests import mcnemar  # noqa: E402
from two_stage import LABELS, TwoStageClassifier, to_stage1  # noqa: E402

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ Bi-LSTM as an sklearn-ish arm

class TorchSequenceClassifier:
    """Minimal sklearn-style wrapper so the Bi-LSTM can go through the same harness.

    Deliberately simple: fixed epoch budget with an internal early-stopping holdout carved from
    the TRAINING fold only, so the CV test fold is never touched during fitting.
    """

    def __init__(self, n_classes: int, n_features: int, epochs: int = 60, lr: float = 5e-4,
                 hidden: int = 64, dropout: float = 0.3, patience: int = 12, seed: int = 0):
        self.n_classes, self.n_features = n_classes, n_features
        self.epochs, self.lr, self.hidden, self.dropout = epochs, lr, hidden, dropout
        self.patience, self.seed = patience, seed
        self.model = None
        self.classes_ = np.arange(n_classes)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
        cfg = {"hidden_size": self.hidden, "n_layers": 1, "dropout": self.dropout}
        self.model = build_model(cfg, self.n_features, n_classes=self.n_classes).to(_DEVICE)

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        cut = max(1, int(0.85 * len(X)))
        tr, va = idx[:cut], idx[cut:]
        if len(va) == 0:
            tr, va = idx, idx

        counts = np.bincount(y[tr], minlength=self.n_classes).astype(np.float64)
        w = counts.sum() / np.maximum(counts, 1)
        crit = nn.CrossEntropyLoss(
            weight=torch.tensor(w / w.sum() * self.n_classes, dtype=torch.float32, device=_DEVICE)
        )
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        Xtr = torch.tensor(X[tr], device=_DEVICE)
        ytr = torch.tensor(y[tr], device=_DEVICE)
        Xva = torch.tensor(X[va], device=_DEVICE)
        best, best_state, bad = -1.0, None, 0
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(Xtr), device=_DEVICE)
            for s in range(0, len(Xtr), 32):
                b = perm[s:s + 32]
                opt.zero_grad()
                crit(self.model(Xtr[b]), ytr[b]).backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                pred = self.model(Xva).argmax(1).cpu().numpy()
            score = f1_score(y[va], pred, average="macro", zero_division=0)
            if score > best:
                best, bad = score, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(np.asarray(X, dtype=np.float32), device=_DEVICE))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def _gbdt(n_classes: int, seed: int = 0):
    # Shallow and heavily regularised: a few hundred windows overfit trivially otherwise.
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=4, min_samples_leaf=10,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=25, random_state=seed,
    )


class _WeightedGBDT:
    """HistGradientBoosting has no `class_weight`; balance via sample_weight instead."""

    def __init__(self, seed: int = 0):
        self.est = _gbdt(0, seed)
        self.classes_ = None

    def fit(self, X, y):
        y = np.asarray(y)
        self.est.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        self.classes_ = self.est.classes_
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X)

    def predict(self, X):
        return self.est.predict(X)


# ------------------------------------------------------------------ the arms

def build_arms(n_features_seq: int, n_features_flat: int, seed: int):
    """Each arm: (name, needs_aggregated_input, factory)."""
    return [
        ("flat_bilstm", False,
         lambda: TorchSequenceClassifier(4, n_features_seq, seed=seed)),
        ("flat_gbdt", True,
         lambda: _WeightedGBDT(seed)),
        ("two_stage_bilstm", False,
         lambda: TwoStageClassifier(
             lambda: TorchSequenceClassifier(2, n_features_seq, seed=seed),
             lambda: TorchSequenceClassifier(3, n_features_seq, seed=seed))),
        ("two_stage_gbdt", True,
         lambda: TwoStageClassifier(lambda: _WeightedGBDT(seed), lambda: _WeightedGBDT(seed))),
    ]


def _fold_metrics(y_true, y_pred, proba) -> dict:
    n = len(LABELS)
    present = sorted(set(y_true.tolist()))
    rec = recall_score(y_true, y_pred, labels=list(range(n)), average=None, zero_division=0)
    m = {
        "accuracy": float((y_true == y_pred).mean()),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class_recall": {LABELS[i]: float(rec[i]) for i in range(n)},
        "classes_missing": [LABELS[i] for i in range(n) if i not in present],
    }
    # Stage-1 AUC is the number the deployed gate cares about: intervene or not.
    s1_true = to_stage1(y_true)
    s1_score = 1.0 - proba[:, 0]
    if len(set(s1_true.tolist())) == 2:
        m["needs_help_auc"] = float(roc_auc_score(s1_true, s1_score))
        m["needs_help_f1"] = float(
            f1_score(s1_true, (s1_score >= 0.5).astype(int), zero_division=0)
        )
    return m


def run(X, y, groups, n_splits: int = 5, seed: int = 42) -> dict:
    X = np.asarray(X)
    y = np.asarray(y)
    Xf = aggregate(X)
    masks = stratified_group_folds(groups, y, n_splits=n_splits, seed=seed)
    arms = build_arms(X.shape[-1], Xf.shape[-1], seed)

    results: dict = {"n": int(len(y)), "n_groups": int(len(set(np.asarray(groups).tolist()))),
                     "n_splits": n_splits, "seed": seed, "labels": LABELS, "arms": {}}
    # Per-fold predictions kept so arms can be compared pairwise on identical items.
    oof: dict[str, np.ndarray] = {name: np.full(len(y), -1, dtype=np.int64) for name, _, _ in arms}

    for name, needs_flat, factory in arms:
        data = Xf if needs_flat else X
        folds = []
        for f, te in enumerate(masks):
            tr = ~te
            if not tr.any() or not te.any():
                continue
            est = factory().fit(data[tr], y[tr])
            proba = est.predict_proba(data[te])
            pred = proba.argmax(axis=1)
            oof[name][te] = pred
            folds.append(_fold_metrics(y[te], pred, proba))

        agg = {}
        for key in ("accuracy", "weighted_f1", "macro_f1", "cohen_kappa",
                    "needs_help_auc", "needs_help_f1"):
            vals = [f[key] for f in folds if key in f]
            if vals:
                agg[key] = {"mean": float(np.mean(vals)),
                            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
        agg["per_class_recall"] = {
            lab: {"mean": float(np.mean([f["per_class_recall"][lab] for f in folds])),
                  "sd": float(np.std([f["per_class_recall"][lab] for f in folds], ddof=1))
                  if len(folds) > 1 else 0.0}
            for lab in LABELS
        }
        results["arms"][name] = {"folds": folds, "aggregate": agg}
        print(f"  {name:18s} wF1 {agg['weighted_f1']['mean']:.4f} +- "
              f"{agg['weighted_f1']['sd']:.4f}   kappa {agg['cohen_kappa']['mean']:.4f}")

    # Paired comparisons on the pooled out-of-fold predictions (identical items by construction).
    valid = oof["flat_bilstm"] >= 0
    results["pairwise_mcnemar"] = {}
    names = [n for n, _, _ in arms]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            results["pairwise_mcnemar"][f"{a}__vs__{b}"] = mcnemar(
                y[valid] == oof[a][valid], y[valid] == oof[b][valid]
            )

    best = max(results["arms"], key=lambda k: results["arms"][k]["aggregate"]["weighted_f1"]["mean"])
    results["best_arm_by_weighted_f1"] = best
    sds = [results["arms"][k]["aggregate"]["weighted_f1"]["sd"] for k in results["arms"]]
    results["noise_floor_weighted_f1"] = float(2 * np.mean(sds))
    return results


def format_report(r: dict) -> str:
    L = [
        f"MODEL COMPARISON — {r['n']} windows, {r['n_groups']} groups, "
        f"{r['n_splits']}-fold grouped CV (seed {r['seed']})",
        "=" * 78,
        "",
        f"  {'arm':18s} {'wF1':>15s} {'macroF1':>15s} {'kappa':>15s} {'help AUC':>15s}",
    ]
    for name, a in r["arms"].items():
        g = a["aggregate"]

        def cell(k):
            return f"{g[k]['mean']:.3f}+-{g[k]['sd']:.3f}" if k in g else "        n/a"
        star = "  *" if name == r["best_arm_by_weighted_f1"] else ""
        L.append(f"  {name:18s} {cell('weighted_f1'):>15s} {cell('macro_f1'):>15s} "
                 f"{cell('cohen_kappa'):>15s} {cell('needs_help_auc'):>15s}{star}")
    L.append("  (* = best weighted-F1)")

    L += ["", "  PER-CLASS RECALL (mean over folds)",
          f"  {'arm':18s}" + "".join(f"{lab[:11]:>13s}" for lab in r["labels"])]
    for name, a in r["arms"].items():
        row = f"  {name:18s}"
        for lab in r["labels"]:
            v = a["aggregate"]["per_class_recall"][lab]["mean"]
            row += f"{v:>13.3f}"
        L.append(row)

    nf = r["noise_floor_weighted_f1"]
    L += ["", f"  NOISE FLOOR: mean 2*sd of weighted-F1 across arms = {nf:.4f}.",
          f"  Two arms differing by less than {nf:.4f} are NOT ordered by this evaluation."]

    L += ["", "  PAIRED McNEMAR (pooled out-of-fold, identical items)"]
    for pair, m in r["pairwise_mcnemar"].items():
        a, b = pair.split("__vs__")
        sig = "" if m["p_value"] >= 0.05 else "   SIGNIFICANT"
        L.append(f"    {a:18s} vs {b:18s} delta_acc {m['accuracy_delta']:+.4f}  "
                 f"p={m['p_value']:.4f}  (B fixed {m['only_b_correct']}, broke "
                 f"{m['only_a_correct']}){sig}")

    zero = {n: [lab for lab in r["labels"]
                if a["aggregate"]["per_class_recall"][lab]["mean"] == 0.0]
            for n, a in r["arms"].items()}
    flagged = {n: z for n, z in zero.items() if z}
    if flagged:
        L += ["", "  *** classes NEVER detected by an arm: ***"]
        for n, z in flagged.items():
            L.append(f"    {n:18s} {', '.join(z)}")
    return "\n".join(L)


def _load(args):
    """Return (X, y, groups, source description)."""
    if args.export:
        export = json.loads(Path(args.export).read_text(encoding="utf-8"))
        if args.labelling == "codebook":
            from codebook_labels import load_codebook, load_codebook_windows
            book = load_codebook(args.codebook)
            res = load_codebook_windows(export, book,
                                        require_agreement=not args.no_agreement)
            windows = res["windows"]
            print(f"codebook labelling: {len(windows)} windows "
                  f"({'strict' if not args.no_agreement else 'design-only'})")
        else:
            from phase_a import load_phase_a_windows
            windows = load_phase_a_windows(
                export.get("items", export), propagate_ms=args.propagate_ms
            )
            print(f"self-report labelling: {len(windows)} windows")
        if not windows:
            raise SystemExit("no labelled windows — check the export and the labelling mode")
        X, y, pid = windows_to_arrays(windows)
        return X, y, pid, f"{args.export} ({args.labelling})"

    from synthetic_data import generate_dataset
    windows = generate_dataset(n_participants=15, seed=args.seed)
    X, y, pid = windows_to_arrays(windows)
    print("SYNTHETIC data — pipeline validation only; these numbers are not a result")
    return X, y, pid, "synthetic"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=None, help="research export JSON (omit for synthetic)")
    ap.add_argument("--labelling", choices=["codebook", "selfreport"], default="codebook")
    ap.add_argument("--codebook", default="../../data/codebook.json")
    ap.add_argument("--no-agreement", action="store_true",
                    help="codebook labelling without the self-report manipulation check")
    ap.add_argument("--propagate-ms", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    X, y, groups, source = _load(a)
    print(f"tensor {X.shape}  classes {np.bincount(y, minlength=4).tolist()}  "
          f"groups {len(set(groups.tolist()))}\n")
    print("training arms:")
    res = run(X, y, groups, n_splits=a.folds, seed=a.seed)
    res["source"] = source
    print("\n" + format_report(res))

    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model_comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        (out / "model_comparison.txt").write_text(format_report(res), encoding="utf-8")
        print(f"\nwrote {out / 'model_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
