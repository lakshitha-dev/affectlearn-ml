"""DUX confusion re-evaluation for the paper-v3 review (experiments 1 to 9).

Everything reuses the committed pipeline: the shared extractor (`feature_engineering`), the five
statistics (`aggregate_features`), the native loader (`external_datasets.load_dux_confusion`),
the GBDT (`train_dux_confusion._gbdt`: depth 3, 120 iterations, lr 0.06, leaf 8, L2 1.0,
balanced), the normalisers (`dux_sensitivity.per_participant_z` and `per_participant_z_causal`),
session-level intervals and within-session permutation tests (`dux_cluster_intervals`), and the
offline gate (`gate_calibration.simulate` and `cluster_ci`). Folds are leave-one-session-out
(LOSO) over the 46 namespaced DUX sessions, 30 s windows, threshold 1.0 on the maximum human
Confusion annotation. Seed 42 everywhere.

"Participant" in the older scripts is a DUX SESSION; sessions are not verified as distinct people.

Experiments and outputs (all under reports/dux_review/):
  1  raw_and_normalised_arms.json   raw, causal-z and transductive-z arms for interaction (80),
                                    AFFDEX facial (60) and early fusion (140)
  2  paired_differences.json        session-bootstrap AUC differences
  3  input_matched.json             interaction on platform-like inputs (see input_matched.py)
  4  gate_sweeps.json               floors 0.50..0.90 step 0.05, offline rule and deployed rule
  5  nested_floor.json              floor chosen inside each training fold by a fixed rule
  6  calibration.json               ECE, Brier, reliability; nested Platt scaling and its sweep
  7  baselines.json                 majority, logistic regression, random forest
  8  robustness_v0_v1.json          v0-only and v1-only LOSO
  9  frustration.json               Anger-proxy numbers read from reports/dux_multistate

Run from affectlearn-ml/:
    python training/behavioral/review/dux_review.py
    python training/behavioral/review/dux_review.py --only 4,5,6     # reuse cached predictions
"""

from __future__ import annotations

import os

# One OpenMP thread per process: HistGradientBoosting oversubscribes badly on this machine
# (a 12-thread fit took 14 s against 5 s single-threaded), so parallelism is across folds instead.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(ROOT / "evaluation"))

from aggregate_features import aggregate  # noqa: E402
from dux_cluster_intervals import cluster_bootstrap, permutation_p  # noqa: E402
from dux_sensitivity import per_participant_z, per_participant_z_causal  # noqa: E402
from external_datasets import load_dux_confusion  # noqa: E402
from feature_engineering import FEATURE_NAMES, extract_features  # noqa: E402
from gate_calibration import WINDOW_S, cluster_ci, simulate  # noqa: E402
from input_matched import load_dux_confusion_matched  # noqa: E402
from train_dux_confusion import _gbdt  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,  # noqa: E402
                             cohen_kappa_score, roc_auc_score)
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEED = 42
N_JOBS = int(os.environ.get("DUX_REVIEW_JOBS", "8"))
N_BOOT = 2000
N_PERM = 2000
FLOORS = [round(0.50 + 0.05 * i, 2) for i in range(9)]          # 0.50 .. 0.90
MIN_CONSECUTIVE = 2
COOLDOWN = 3
INNER_FOLDS = 5
# PRE-SPECIFIED nested floor rule (written before any nested result was seen):
#   the lowest floor in FLOORS whose inner-fold gated precision is >= 2 x the inner base rate
#   with at least 10 offers; if no floor qualifies, the highest floor.
NESTED_LIFT = 2.0
NESTED_MIN_OFFERS = 10


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------

def _matrices(windows: list[dict]):
    Xb = np.stack([extract_features(w["events"], 0) for w in windows])
    Xf = aggregate(np.stack([w["affectiva_seq"] for w in windows]))
    y = np.array([w["label"] for w in windows], dtype=np.int64)
    groups = np.array([w["participant"] for w in windows])
    order = np.array([w["window_index"] for w in windows])
    return Xb, Xf, y, groups, order


def load_all(dux: str, cache: Path) -> dict:
    """Native windows and input-matched windows, cached as npz for re-runs."""
    f = cache / "dux_review_features.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return {k: d[k] for k in d.files}
    t = time.time()
    nat = load_dux_confusion(dux, threshold=1.0)
    Xb, Xf, y, g, o = _matrices(nat)
    ok = ~np.isnan(Xf).any(axis=1)
    Xb, Xf, y, g, o = Xb[ok], Xf[ok], y[ok], g[ok], o[ok]
    print(f"  native windows {len(y)} ({time.time() - t:.0f}s)")

    t = time.time()
    mat = load_dux_confusion_matched(dux, threshold=1.0, min_events=1)
    mXb, mXf, my, mg, mo = _matrices(mat)
    mok = ~np.isnan(mXf).any(axis=1)
    mXb, my, mg, mo = mXb[mok], my[mok], mg[mok], mo[mok]
    print(f"  matched windows (serving rule) {len(my)} ({time.time() - t:.0f}s)")

    # Same-window subset: the native window set, recomputed from the matched stream.
    key = {(gg, int(oo)): i for i, (gg, oo) in enumerate(zip(mg, mo))}
    sel = np.array([key[(gg, int(oo))] for gg, oo in zip(g, o)])
    out = {
        "Xb": Xb, "Xf": Xf, "y": y, "g": g, "o": o,
        "mXb": mXb, "my": my, "mg": mg, "mo": mo, "same_idx": sel,
    }
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return out


# ----------------------------------------------------------------------------------------------
# models and statistics
# ----------------------------------------------------------------------------------------------

def _gbdt_default():
    return _gbdt(SEED)


def _fold(X, Xt, y, groups, s, make):
    te = groups == s
    tr = ~te
    if len(np.unique(y[tr])) < 2:
        return s, None
    return s, make().fit(X[tr], y[tr]).predict_proba(Xt[te])[:, 1]


def loso(X, y, groups, make=_gbdt_default, X_test=None, n_jobs=None):
    """Pooled out-of-fold P(confused). `X_test` scores held-out rows from a different matrix.

    Folds run in parallel processes; each fold is independent and deterministic, so the result
    is identical to the sequential loop in train_dux_confusion.leave_one_participant_out.
    """
    Xt = X if X_test is None else X_test
    prob = np.full(len(y), np.nan)
    res = Parallel(n_jobs=N_JOBS if n_jobs is None else n_jobs)(
        delayed(_fold)(X, Xt, y, groups, s, make) for s in np.unique(groups))
    for s, p in res:
        if p is not None:
            prob[groups == s] = p
    return prob


def _fold_causal(X, y, groups, order, s):
    te = groups == s
    tr = ~te
    Xn = per_participant_z_causal(X, groups, order, fit_mask=tr)
    return s, _gbdt(SEED).fit(Xn[tr], y[tr]).predict_proba(Xn[te])[:, 1]


def loso_causal(X, y, groups, order):
    """Causal z-scoring refitted inside every fold (cold start from training-fold statistics)."""
    prob = np.full(len(y), np.nan)
    res = Parallel(n_jobs=N_JOBS)(delayed(_fold_causal)(X, y, groups, order, s)
                                  for s in np.unique(groups))
    for s, p in res:
        prob[groups == s] = p
    return prob


def stats(y, prob, groups) -> dict:
    keep = ~np.isnan(prob)
    y, prob, groups = y[keep], prob[keep], groups[keep]
    pred = (prob >= 0.5).astype(np.int64)
    base = float(y.mean())
    lo, hi = cluster_bootstrap(y, prob, groups, seed=SEED, n=N_BOOT)
    ap = float(average_precision_score(y, prob))
    return {
        "n": int(len(y)), "n_sessions": int(len(np.unique(groups))), "positives": int(y.sum()),
        "base_rate": base,
        "auc": float(roc_auc_score(y, prob)), "auc_ci95_session": [lo, hi],
        "kappa_at_0.5": float(cohen_kappa_score(y, pred)),
        "accuracy_at_0.5": float(accuracy_score(y, pred)), "majority_accuracy": 1.0 - base,
        "average_precision": ap, "ap_over_base_rate": ap / base,
        "permutation_p_within_session": float(permutation_p(y, prob, groups, seed=SEED, n=N_PERM)),
        "score_max": float(prob.max()), "score_p99": float(np.percentile(prob, 99)),
        "score_mean": float(prob.mean()),
    }


def paired_diff(y, pa, pb, groups) -> dict:
    """AUC(a) - AUC(b) on identical windows, resampling sessions."""
    rng = np.random.default_rng(SEED)
    ids = np.unique(groups)
    idx_by = {s: np.flatnonzero(groups == s) for s in ids}
    obs = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    out = []
    for _ in range(N_BOOT):
        b = np.concatenate([idx_by[s] for s in rng.choice(ids, size=len(ids), replace=True)])
        if len(np.unique(y[b])) > 1:
            out.append(roc_auc_score(y[b], pa[b]) - roc_auc_score(y[b], pb[b]))
    out = np.array(out)
    return {"delta_auc": float(obs), "ci95_session": [float(np.percentile(out, 2.5)),
                                                      float(np.percentile(out, 97.5))],
            "share_of_draws_above_zero": float((out > 0).mean())}


def simulate_deployed(y, prob, groups, *, threshold, min_consecutive=MIN_CONSECUTIVE,
                      cooldown=COOLDOWN) -> dict:
    """The rule as written in backend/app/agents/edges.py:passes_adaptation_gate.

    Differs from gate_calibration.simulate in two places: persistence requires the last
    `min_consecutive` readings of the channel to share the current STATE (P >= 0.5 means
    confused), while only the CURRENT reading must clear the floor; and cooldown blocks while
    (k - k_last_offer) < cooldown, so with cooldown 3 an offer at k allows the next at k+3.
    Positions k count the windows present in the sequence (dropped windows are not cycles here).
    """
    fired = correct = 0
    for s in np.unique(groups):
        idx = np.flatnonzero(groups == s)
        last = None
        for k, i in enumerate(idx):
            if prob[i] < threshold or prob[i] < 0.5:
                continue
            if k + 1 < min_consecutive:
                continue
            if any(prob[idx[j]] < 0.5 for j in range(k - min_consecutive + 1, k)):
                continue
            if last is not None and k - last < cooldown:
                continue
            fired += 1
            correct += int(y[i] == 1)
            last = k
    hours = len(y) * WINDOW_S / 3600.0
    return {"threshold": threshold, "interventions": fired, "correct": correct,
            "precision": correct / fired if fired else float("nan"),
            "interventions_per_hour": fired / hours if hours else float("nan"),
            "minutes_between": hours * 60.0 / fired if fired else float("nan")}


def cluster_ci_deployed(y, prob, groups, *, threshold, n=N_BOOT) -> list[float]:
    rng = np.random.default_rng(SEED)
    ids = np.unique(groups)
    idx_by = {s: np.flatnonzero(groups == s) for s in ids}
    out = []
    for _ in range(n):
        ys, ps, gs = [], [], []
        for k, s in enumerate(rng.choice(ids, size=len(ids), replace=True)):
            i = idx_by[s]
            ys.append(y[i]); ps.append(prob[i]); gs.append(np.full(len(i), k))
        r = simulate_deployed(np.concatenate(ys), np.concatenate(ps), np.concatenate(gs),
                              threshold=threshold)
        if r["interventions"]:
            out.append(r["precision"])
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else [None, None]


def _sweep_row(y, prob, groups, thr, rule, base):
    if rule == "offline":
        r = simulate(y, prob, groups, threshold=thr, min_consecutive=MIN_CONSECUTIVE,
                     cooldown=COOLDOWN)
        ci = list(cluster_ci(y, prob, groups, threshold=thr, min_consecutive=MIN_CONSECUTIVE,
                             cooldown=COOLDOWN, seed=SEED, n=N_BOOT)) if r["interventions"] else [None, None]
    else:
        r = simulate_deployed(y, prob, groups, threshold=thr)
        ci = cluster_ci_deployed(y, prob, groups, threshold=thr) if r["interventions"] else [None, None]
    r["ci95_session"] = [None if (c is None or c != c) else float(c) for c in ci]
    r["lift"] = (r["precision"] / base) if r["interventions"] else None
    if r["precision"] != r["precision"]:
        r["precision"] = None
    return r


def sweep(y, prob, groups, rule: str = "offline") -> dict:
    base = float(y.mean())
    hours = len(y) * WINDOW_S / 3600.0
    rows = [{"threshold": None, "interventions": int(len(y)), "correct": int(y.sum()),
             "precision": base, "lift": 1.0, "interventions_per_hour": len(y) / hours,
             "ci95_session": None}]
    for r in Parallel(n_jobs=min(N_JOBS, len(FLOORS)))(
            delayed(_sweep_row)(y, prob, groups, thr, rule, base) for thr in FLOORS):
        rows.append(r)
    return {"rule": rule, "n_windows": int(len(y)), "n_sessions": int(len(np.unique(groups))),
            "hours_of_sequence": hours, "base_rate": base, "min_consecutive": MIN_CONSECUTIVE,
            "cooldown": COOLDOWN, "floors": FLOORS, "rows": rows}


def inner_oof(X, y, groups, tr_mask):
    """Out-of-fold predictions for the training sessions of one outer fold (5 grouped folds)."""
    idx = np.flatnonzero(tr_mask)
    Xi, yi, gi = X[idx], y[idx], groups[idx]
    oof = np.full(len(idx), np.nan)
    for a, b in GroupKFold(n_splits=INNER_FOLDS).split(Xi, yi, gi):
        oof[b] = _gbdt(SEED).fit(Xi[a], yi[a]).predict_proba(Xi[b])[:, 1]
    return idx, oof


def _lr():
    # L2 penalty is the default; `penalty` is deprecated in scikit-learn 1.8+.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))


def _rf():
    return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=SEED, n_jobs=1, min_samples_leaf=1)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _nested_fold(X, y, groups, outer_prob, s):
    te = groups == s
    tr = ~te
    idx, oof = inner_oof(X, y, groups, tr)
    yi, gi = y[idx], groups[idx]
    base_in = float(yi.mean())
    pick = FLOORS[-1]
    for thr in FLOORS:
        r = simulate(yi, oof, gi, threshold=thr, min_consecutive=MIN_CONSECUTIVE,
                     cooldown=COOLDOWN)
        if r["interventions"] >= NESTED_MIN_OFFERS and r["precision"] >= NESTED_LIFT * base_in:
            pick = thr
            break
    r_te = simulate(y[te], outer_prob[te], groups[te], threshold=pick,
                    min_consecutive=MIN_CONSECUTIVE, cooldown=COOLDOWN)
    # Platt: logistic regression on logit(inner OOF score), applied to the held-out session
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(_logit(oof).reshape(-1, 1), yi)
    platt = lr.predict_proba(_logit(outer_prob[te]).reshape(-1, 1))[:, 1]
    return s, pick, (r_te["interventions"], r_te["correct"]), platt


def nested(X, y, groups, outer_prob) -> dict:
    """Nested floor selection and nested Platt calibration, sharing one inner CV per fold."""
    chosen = {}
    per_session = {}
    platt_prob = np.full(len(y), np.nan)
    for s, pick, oc, platt in Parallel(n_jobs=N_JOBS)(
            delayed(_nested_fold)(X, y, groups, outer_prob, s) for s in np.unique(groups)):
        chosen[s] = pick
        per_session[s] = oc
        platt_prob[groups == s] = platt

    offers = sum(v[0] for v in per_session.values())
    correct = sum(v[1] for v in per_session.values())
    rng = np.random.default_rng(SEED)
    ids = np.array(list(per_session))
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(ids, size=len(ids), replace=True)
        o = sum(per_session[s][0] for s in pick)
        c = sum(per_session[s][1] for s in pick)
        if o:
            draws.append(c / o)
    vals, counts = np.unique(np.array(list(chosen.values())), return_counts=True)
    hours = len(y) * WINDOW_S / 3600.0
    return {
        "rule": (f"lowest floor in {FLOORS} whose inner {INNER_FOLDS}-fold grouped OOF gated "
                 f"precision >= {NESTED_LIFT} x inner base rate with >= {NESTED_MIN_OFFERS} "
                 "offers; else the highest floor"),
        "held_out_offers": int(offers), "held_out_correct": int(correct),
        "held_out_precision": (correct / offers) if offers else None,
        "held_out_precision_ci95_session": ([float(np.percentile(draws, 2.5)),
                                             float(np.percentile(draws, 97.5))] if draws else None),
        "base_rate": float(y.mean()),
        "lift": ((correct / offers) / float(y.mean())) if offers else None,
        "offers_per_hour": offers / hours,
        "chosen_floor_distribution": {str(float(v)): int(c) for v, c in zip(vals, counts)},
    }, platt_prob


def calibration(y, prob, n_bins: int = 10) -> dict:
    keep = ~np.isnan(prob)
    y, prob = y[keep], prob[keep]
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    table, ece = [], 0.0
    for k in range(n_bins):
        m = b == k
        if not m.any():
            table.append({"bin": [float(edges[k]), float(edges[k + 1])], "count": 0,
                          "mean_predicted": None, "observed_rate": None})
            continue
        mp, ob = float(prob[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(mp - ob)
        table.append({"bin": [float(edges[k]), float(edges[k + 1])], "count": int(m.sum()),
                      "mean_predicted": mp, "observed_rate": ob})
    return {"ece_10_equal_width": float(ece), "brier": float(brier_score_loss(y, prob)),
            "mean_predicted": float(prob.mean()), "base_rate": float(y.mean()),
            "reliability": table}


def feature_shift(Xb_nat: np.ndarray, Xb_mat: np.ndarray) -> dict:
    """Per-bin distribution of each of the 16 features, native against matched, same windows."""
    out = {}
    for j, name in enumerate(FEATURE_NAMES):
        a = Xb_nat[:, :, j].ravel()
        b = Xb_mat[:, :, j].ravel()
        out[name] = {
            "native": {"mean": float(a.mean()), "p99": float(np.percentile(a, 99)),
                       "max": float(a.max())},
            "matched": {"mean": float(b.mean()), "p99": float(np.percentile(b, 99)),
                        "max": float(b.max())},
        }
    return out


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------

def dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"  wrote {path}")


def save_pred(out: Path, name: str, y, prob, groups, order) -> None:
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions" / f"{name}.npz", y_true=y, y_prob=prob,
                        groups=groups, window_index=order)


def load_pred(out: Path, name: str):
    d = np.load(out / "predictions" / f"{name}.npz", allow_pickle=True)
    return d["y_true"], d["y_prob"], d["groups"], d["window_index"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dux", default=str(ROOT / "data/external/dux"))
    ap.add_argument("--out", default=str(ROOT / "reports/dux_review"))
    ap.add_argument("--cache", default=str(ROOT / "reports/dux_review/_cache"))
    ap.add_argument("--only", default="1,2,3,4,5,6,7,8,9")
    a = ap.parse_args()
    steps = {int(s) for s in a.only.split(",")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {"seed": SEED, "bootstrap_resamples": N_BOOT, "permutation_draws": N_PERM,
                "folds": "leave-one-session-out over namespaced DUX sessions",
                "window_s": 30, "label": "max human Confusion annotation in window >= 1.0",
                "classifier": "HistGradientBoosting depth 3, 120 iter, lr 0.06, leaf 8, L2 1.0, "
                              "balanced (train_dux_confusion._gbdt)",
                "interval": "session-level cluster bootstrap, percentile",
                "permutation": "labels shuffled within each session"}

    D = load_all(a.dux, Path(a.cache))
    Xb, Xf, y, g, o = D["Xb"], D["Xf"], D["y"], D["g"], D["o"]
    Ab = aggregate(Xb)
    fused = np.hstack([Ab, Xf])
    mXb, my, mg, mo, sel = D["mXb"], D["my"], D["mg"], D["mo"], D["same_idx"]
    mAb = aggregate(mXb)
    print(f"native n={len(y)} sessions={len(np.unique(g))} pos={int(y.sum())}; "
          f"matched serving-rule n={len(my)} pos={int(my.sum())}")

    # ---------------- 1: raw, causal and transductive arms ----------------
    if 1 in steps or 2 in steps:
        t = time.time()
        preds = {}
        for name, X in {"interaction": Ab, "facial": Xf, "fused": fused}.items():
            preds[f"raw_{name}"] = loso(X, y, g)
            preds[f"causal_{name}"] = loso_causal(X, y, g, o)
            Xt = per_participant_z(X, g)
            preds[f"transductive_{name}"] = loso(Xt, y, g)
        for k, p in preds.items():
            save_pred(out, k, y, p, g, o)
        print(f"  step 1 fits done ({time.time() - t:.0f}s)")
    else:
        preds = {k: load_pred(out, k)[1] for k in
                 [f"{n}_{m}" for n in ("raw", "causal", "transductive")
                  for m in ("interaction", "facial", "fused")]}

    if 1 in steps:
        res = {"settings": settings, "arms": {}}
        for k, p in preds.items():
            res["arms"][k] = stats(y, p, g)
            r = res["arms"][k]
            print(f"  {k:28s} AUC {r['auc']:.4f} [{r['auc_ci95_session'][0]:.3f}, "
                  f"{r['auc_ci95_session'][1]:.3f}] kappa {r['kappa_at_0.5']:.3f} "
                  f"acc {r['accuracy_at_0.5']:.3f} p {r['permutation_p_within_session']:.4f}")
        res["reproduces_committed_transductive"] = {
            "committed_interaction_auc": 0.7473060669582164,
            "this_run_interaction_auc": res["arms"]["transductive_interaction"]["auc"],
            "committed_facial_auc": 0.5532, "this_run_facial_auc":
                res["arms"]["transductive_facial"]["auc"],
            "committed_fused_auc": 0.7351, "this_run_fused_auc":
                res["arms"]["transductive_fused"]["auc"]}
        dump(out / "raw_and_normalised_arms.json", res)

    # ---------------- 2: paired differences ----------------
    if 2 in steps:
        res = {"settings": settings, "pairs": {}}
        for norm in ("raw", "causal", "transductive"):
            pi = preds[f"{norm}_interaction"]
            res["pairs"][f"{norm}: interaction - fused"] = paired_diff(y, pi, preds[f"{norm}_fused"], g)
            res["pairs"][f"{norm}: interaction - facial"] = paired_diff(y, pi, preds[f"{norm}_facial"], g)
        for k, v in res["pairs"].items():
            print(f"  {k:36s} {v['delta_auc']:+.4f} [{v['ci95_session'][0]:+.3f}, "
                  f"{v['ci95_session'][1]:+.3f}]")
        dump(out / "paired_differences.json", res)

    # ---------------- 3: input-matched interaction ----------------
    same_y, same_g, same_o = my[sel], mg[sel], mo[sel]
    assert np.array_equal(same_y, y) and np.array_equal(same_g, g) and np.array_equal(same_o, o)
    same_A = mAb[sel]
    if 3 in steps:
        t = time.time()
        p_same = loso(same_A, y, g)
        p_serv = loso(mAb, my, mg)
        # the native-trained model scored on platform-like inputs of its held-out session
        p_cross = loso(Ab, y, g, X_test=same_A)
        save_pred(out, "matched_same_windows_interaction", y, p_same, g, o)
        save_pred(out, "matched_serving_rule_interaction", my, p_serv, mg, mo)
        save_pred(out, "native_model_on_matched_inputs", y, p_cross, g, o)
        res = {
            "settings": settings,
            "method": {
                "pointer": "one sample per 100 ms slot with movement, latest position, ts = slot "
                           "start + 99 ms (platform aggregatorTick)",
                "clicks": "MouseButtonDown counts; MouseClick counts only with no unmatched down "
                          "in the preceding 1,000 ms; MouseDoubleClick dropped (platform mousedown)",
                "windows_same": "the native 1,419-window set (>= 5 native events), features "
                                "recomputed from the matched stream",
                "windows_serving_rule": "every window with >= 1 matched pointer, click, key or "
                                        "scroll event (backend _is_idle rule)",
                "unchanged": "keys, scroll dy, labels, AFFDEX, extractor and classifier"},
            "native_raw_interaction": stats(y, preds["raw_interaction"], g),
            "matched_same_windows": stats(y, p_same, g),
            "matched_serving_rule": stats(my, p_serv, mg),
            "native_model_scored_on_matched_inputs": stats(y, p_cross, g),
            "paired_native_minus_matched_same_windows": paired_diff(y, preds["raw_interaction"], p_same, g),
            "per_bin_feature_shift_same_windows": feature_shift(Xb, mXb[sel]),
            "live_reference": "platform live interaction readings never exceeded 0.633 "
                              "(reports/deployment/aggregates_30d.json)",
        }
        for k in ("native_raw_interaction", "matched_same_windows", "matched_serving_rule",
                  "native_model_scored_on_matched_inputs"):
            r = res[k]
            print(f"  {k:40s} n {r['n']:5d} AUC {r['auc']:.4f} [{r['auc_ci95_session'][0]:.3f}, "
                  f"{r['auc_ci95_session'][1]:.3f}] max {r['score_max']:.3f} p99 {r['score_p99']:.3f}")
        print(f"  step 3 done ({time.time() - t:.0f}s)")
        dump(out / "input_matched.json", res)

    def pred_or_load(name):
        return load_pred(out, name)

    # ---------------- 4: gate sweeps ----------------
    if 4 in steps:
        t = time.time()
        variants = {
            "raw_interaction": (y, preds["raw_interaction"], g),
            "matched_same_windows_interaction": pred_or_load("matched_same_windows_interaction")[:3],
            "matched_serving_rule_interaction": pred_or_load("matched_serving_rule_interaction")[:3],
            "transductive_interaction_reference": (y, preds["transductive_interaction"], g),
        }
        res = {"settings": settings,
               "note": "offline = evaluation/gate_calibration.simulate (both readings >= floor; "
                       "streak counts through cooldown; next offer at k+4). deployed = "
                       "edges.passes_adaptation_gate (current reading >= floor and previous "
                       "reading same state P>=0.5; next offer at k+3).",
               "sweeps": {}}
        for name, (yy, pp, gg) in variants.items():
            for rule in ("offline", "deployed"):
                res["sweeps"][f"{name}|{rule}"] = sweep(yy, pp, gg, rule)
                row70 = next(r for r in res["sweeps"][f"{name}|{rule}"]["rows"] if r["threshold"] == 0.7)
                print(f"  {name:40s} {rule:8s} @0.70 offers {row70['interventions']:4d} "
                      f"prec {row70['precision']} lift {row70['lift']}")
        print(f"  step 4 done ({time.time() - t:.0f}s)")
        dump(out / "gate_sweeps.json", res)

    # ---------------- 5 + 6: nested floor selection and calibration ----------------
    if 5 in steps or 6 in steps:
        t = time.time()
        nres, cres = {"settings": settings}, {"settings": settings}
        cases = {
            "raw_interaction": (Ab, y, g, preds["raw_interaction"]),
            "matched_same_windows_interaction": (same_A, y, g,
                                                 pred_or_load("matched_same_windows_interaction")[1]),
            "matched_serving_rule_interaction": (mAb, my, mg,
                                                 pred_or_load("matched_serving_rule_interaction")[1]),
        }
        for name, (X, yy, gg, pp) in cases.items():
            nr, platt = nested(X, yy, gg, pp)
            nres[name] = nr
            print(f"  nested {name:36s} offers {nr['held_out_offers']} prec "
                  f"{nr['held_out_precision']} CI {nr['held_out_precision_ci95_session']} "
                  f"floors {nr['chosen_floor_distribution']}")
            cres[name] = {
                "uncalibrated": calibration(yy, pp),
                "platt_nested": calibration(yy, platt),
                "platt_nested_auc": float(roc_auc_score(yy, platt)),
                "platt_nested_sweep_offline": sweep(yy, platt, gg, "offline"),
            }
            print(f"  calib  {name:36s} ECE {cres[name]['uncalibrated']['ece_10_equal_width']:.3f}"
                  f" -> {cres[name]['platt_nested']['ece_10_equal_width']:.3f}  Brier "
                  f"{cres[name]['uncalibrated']['brier']:.4f} -> {cres[name]['platt_nested']['brier']:.4f}")
        nres["rule_prespecified"] = True
        if 5 in steps:
            dump(out / "nested_floor.json", nres)
        if 6 in steps:
            dump(out / "calibration.json", cres)
        print(f"  steps 5-6 done ({time.time() - t:.0f}s)")

    # ---------------- 7: baselines ----------------
    if 7 in steps:
        t = time.time()
        lr = _lr
        rf = _rf
        p_lr = loso(Ab, y, g, make=lr)
        p_rf = loso(Ab, y, g, make=rf)
        save_pred(out, "baseline_lr_interaction", y, p_lr, g, o)
        save_pred(out, "baseline_rf_interaction", y, p_rf, g, o)
        res = {"settings": settings,
               "majority": {"accuracy": float(1 - y.mean()), "auc": 0.5},
               "logistic_regression_l2_standardised": stats(y, p_lr, g),
               "random_forest_500_balanced": stats(y, p_rf, g),
               "gbdt_reference_raw": stats(y, preds["raw_interaction"], g),
               "paired_gbdt_minus_lr": paired_diff(y, preds["raw_interaction"], p_lr, g),
               "paired_gbdt_minus_rf": paired_diff(y, preds["raw_interaction"], p_rf, g)}
        for k in ("logistic_regression_l2_standardised", "random_forest_500_balanced",
                  "gbdt_reference_raw"):
            r = res[k]
            print(f"  {k:40s} AUC {r['auc']:.4f} [{r['auc_ci95_session'][0]:.3f}, "
                  f"{r['auc_ci95_session'][1]:.3f}]")
        print(f"  step 7 done ({time.time() - t:.0f}s)")
        dump(out / "baselines.json", res)

    # ---------------- 8: v0-only and v1-only ----------------
    if 8 in steps:
        res = {"settings": settings, "subsets": {}}
        for tag in ("v0", "v1"):
            m = np.array([s.startswith(f"dux_{tag}_") for s in g])
            for name, X in {"interaction_raw": Ab, "facial_raw": Xf}.items():
                p = loso(X[m], y[m], g[m])
                res["subsets"][f"{tag}|{name}"] = stats(y[m], p, g[m])
                r = res["subsets"][f"{tag}|{name}"]
                print(f"  {tag} {name:16s} n {r['n']} sessions {r['n_sessions']} AUC {r['auc']:.4f} "
                      f"[{r['auc_ci95_session'][0]:.3f}, {r['auc_ci95_session'][1]:.3f}]")
        dump(out / "robustness_v0_v1.json", res)

    # ---------------- 9: frustration ----------------
    if 9 in steps:
        ms = json.loads((ROOT / "reports/dux_multistate/multistate.json").read_text(encoding="utf-8"))
        res = {"source": "reports/dux_multistate/multistate.json (read, not re-run)",
               "files": ms.get("files"), "n": ms.get("n"), "n_sessions": ms.get("n_participants"),
               "folds": ms.get("folds"), "class_counts": ms.get("class_counts"),
               "anger_is": ms.get("platform_mapping", {}).get("Anger")}
        for arm, r in ms.get("arms", {}).items():
            pc = r.get("per_class", {}).get("Anger")
            if pc:
                res[f"{arm}_anger"] = pc
        dump(out / "frustration.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
