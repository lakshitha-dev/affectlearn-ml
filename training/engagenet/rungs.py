"""Evaluate one rung of the method ladder, identically to every other rung.

Every rung goes through this file so the comparison between them means something. The metric set,
the resampling unit, the null and the fold assignment are fixed here and never passed in.

WHAT IS REPORTED, AND WHY EACH ONE

* AUC with a SUBJECT-level bootstrap interval. Clips of one person share a face, a camera and a
  session, so resampling clips treats correlated observations as independent and reports an
  interval far tighter than the data supports.
* A WITHIN-SUBJECT permutation null: labels are shuffled inside each subject, preserving that
  person's positive rate. A model that has learned who someone is rather than how engaged they are
  cannot beat this. It is the load-bearing test here, because several of these features (motion
  energy, head-pose range) plausibly encode identity - some people simply move more.
* Accuracy beside its majority baseline, always both. Never one alone.
* Average precision against the base rate, because at ~31% positives a ranking metric can look
  respectable where precision does not.
* The SCORE SPREAD - min, max, mean of P(disengaged). This is the deployment criterion and it is
  reported as a first-class result, not a diagnostic. The previous facial model had an acceptable
  AUC inside a 0.127-wide band centred on its own threshold, so no operating point could separate
  anything. A rung that cannot be thresholded is not a candidate however well it ranks.

AWAY FEATURES

"Frequently glances away" is a transition count. Neither a mean nor a standard deviation over yaw
expresses it: someone who stares 45 degrees off-screen for ten seconds and someone who glances away
five times can share both statistics. So away-ness is computed at fixed angle thresholds and
summarised three ways - what share of the window was spent away, how many times they looked back,
and the longest single stretch. Thresholds are FIXED rather than fitted, so nothing leaks across
folds and there is nothing to tune.

MEASURED OUTCOME, recorded here because it contradicts the argument above: the away block did not
help on EngageNet Validation. It was slightly negative in all three places it was tried (rung 2 vs
1, 3b vs 3, 5 vs 4). Kept in the file so the negative result stays reproducible rather than being
quietly dropped.
"""

import json
import pathlib
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold

LOW = {"Not-Engaged", "Barely-engaged"}
SNP = "SNP(Subject Not Present)"
LABEL_FILE = {"Train": "train_engagement_labels.xlsx",
              "Validation": "validation_engagement_labels.xlsx",
              "Test": "test_engagement_labels.csv"}
NAMES = ["yaw", "pitch", "roll", "ear_l", "ear_r", "ear_mean",
         "mouth_open", "gaze_x", "gaze_y", "motion", "face_found"]

# Chosen from the LABEL-FREE marginal distribution of |yaw|/|pitch| over all 1,071 Validation
# clips, before any model was fit, and frozen for every rung and split thereafter. The original
# (10, 15, 20) was picked blind and is wrong for this corpus: >15 deg fires in 11.7% of clips and
# >20 deg in 5.7%, so those features are constant zero for most of the data and carry nothing.
# Pooled |yaw| is med 2.7, p75 5.7, p90 10.2, p99 20.2 -- the mass is in single digits.
# No label was consulted in setting these, so nothing leaks; Test is untouched.
AWAY_DEG = (3.0, 5.0, 8.0, 12.0)

RUNGS = {
    "1_headpose":      ["yaw", "pitch", "roll"],
    "2_headpose_away": ["yaw", "pitch", "roll"],                      # + away features
    "3_all_geometry":  ["yaw", "pitch", "roll", "ear_l", "ear_r", "ear_mean",
                        "mouth_open", "gaze_x", "gaze_y", "motion"],
    "3b_all_away":     ["yaw", "pitch", "roll", "ear_l", "ear_r", "ear_mean",
                        "mouth_open", "gaze_x", "gaze_y", "motion"],  # + away, absolute + centred
}

# Rungs 4-7 came out of the channel ablation on rung 3: dropping head pose (+0.0256) and dropping
# eyes (+0.0257) each HELPED, so the honest next step was to drop both. Rung 4 won the screen.
RUNGS["4_lean"]        = ["gaze_x", "gaze_y", "mouth_open", "motion"]
RUNGS["5_lean_away"]   = ["gaze_x", "gaze_y", "mouth_open", "motion"]
RUNGS["6_no_headpose"] = ["ear_l", "ear_r", "ear_mean", "mouth_open", "gaze_x", "gaze_y", "motion"]
RUNGS["7_gaze_only"]   = ["gaze_x", "gaze_y"]

# Rungs whose matrix gets the away block appended.
AWAY_RUNGS = ("2_headpose_away", "3b_all_away", "5_lean_away")


def subject_of(chunk):
    return "_".join(str(chunk).split("_")[:2])


def five_stats(x):
    """mean/std/min/max/trend per channel - the aggregation the behavioural branch uses."""
    third = max(1, x.shape[0] // 3)
    early, late = np.nanmean(x[:third], axis=0), np.nanmean(x[-third:], axis=0)
    return np.concatenate([np.nanmean(x, axis=0), np.nanstd(x, axis=0),
                           np.nanmin(x, axis=0), np.nanmax(x, axis=0), late - early])


def _away_block(yaw, pitch):
    """Share away, look-back count, longest away run - at each fixed threshold."""
    out = []
    for deg in AWAY_DEG:
        away = (np.abs(yaw) > deg) | (np.abs(pitch) > deg)
        away = np.nan_to_num(away.astype(float)).astype(bool)
        trans = int(np.sum(~away[:-1] & away[1:]))
        cur = best = 0
        for a in away:
            cur = cur + 1 if a else 0
            best = max(best, cur)
        out += [float(away.mean()), float(trans), float(best)]
    return out


def away_features(seq, names):
    """Away-ness twice over: against the camera axis, and against the person's own resting pose.

    ABSOLUTE thresholds ask "is this head turned away from the screen". They are the literal
    reading of the label wording, but they also encode WHO the subject is: median |yaw| across the
    11 Validation subjects runs from 1.2 deg (subject_20) to 9.1 deg (subject_7), a 7.5x spread in
    resting pose alone. A classifier can score a stationary subject_7 as permanently "away" and a
    fidgeting subject_20 as permanently "attentive" without detecting anything.

    CENTRED thresholds subtract each clip's own median yaw and pitch first, so the feature measures
    departure from that person's baseline rather than from the camera. That removes the identity
    component the absolute block carries, at the cost of blurring a genuinely persistent turn-away.

    Both are included because they fail in opposite directions and the within-subject permutation
    null decides whether either is real. If the absolute block alone passes while the centred block
    does not, the honest reading is identity leakage, not detection.
    """
    yaw = seq[:, names.index("yaw")]
    pitch = seq[:, names.index("pitch")]
    out = _away_block(yaw, pitch)
    out += _away_block(yaw - np.nanmedian(yaw), pitch - np.nanmedian(pitch))
    return np.array(out)


def build_matrix(feats, rung):
    cols = RUNGS[rung]
    idx = [NAMES.index(c) for c in cols]
    rows = []
    for clip in feats:
        seq = clip[:, idx]
        v = five_stats(seq)
        if rung in AWAY_RUNGS:
            v = np.concatenate([v, away_features(clip, NAMES)])
        rows.append(v)
    X = np.asarray(rows, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feature_names(rung):
    """Column names for build_matrix, in the same order. Needed for importance reporting."""
    cols = RUNGS[rung]
    names = [f"{st}_{c}" for st in ("mean", "std", "min", "max", "trend") for c in cols]
    if rung in AWAY_RUNGS:
        for framing in ("abs", "ctr"):
            for deg in AWAY_DEG:
                names += [f"away{framing}{deg:g}_share", f"away{framing}{deg:g}_trans",
                          f"away{framing}{deg:g}_longest"]
    return names


def load(geom_npz, labels_dir, split):
    """Load one split, or several geometry shards concatenated.

    `geom_npz` may be a single path or a list of paths; sharded extraction writes one npz per
    shard so a lost runtime costs only the shard in flight rather than the whole run.
    """
    paths = [geom_npz] if isinstance(geom_npz, (str, pathlib.Path)) else list(geom_npz)
    feats_all, ids_all = [], []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        feats_all.append(d["feats"])
        ids_all += [str(c) for c in d["clip_id"]]
    feats_cat = np.concatenate(feats_all)

    lp = pathlib.Path(labels_dir) / LABEL_FILE[split]
    lab = pd.read_csv(lp) if lp.suffix == ".csv" else pd.read_excel(lp)
    lab = lab[lab["label"] != SNP]

    # Normalise BOTH sides of the join. Every EngageNet label file keys on the full filename
    # ("subject_7_as2uk9lhe2_vid_0_0.mp4"); the extractor stores Path.stem, which drops the
    # extension. Joined raw, nothing matches and this returns an EMPTY split silently, as an
    # array rather than an error - a stage would then "run" on nothing and report on no data.
    # Stripping the suffix on both sides makes the join independent of which convention the
    # extractor used.
    ids_all = [re.sub(r"\.mp4$", "", c) for c in ids_all]
    m = {re.sub(r"\.mp4$", "", str(k)): str(v)
         for k, v in zip(lab["chunk"], lab["label"])}
    keep = [i for i, c in enumerate(ids_all) if c in m]
    if not keep:
        raise SystemExit(f"{split}: no geometry clip matched a label. "
                         f"ids {ids_all[:2]} vs labels {list(m)[:2]}")
    feats = feats_cat[keep]
    y = np.array([1 if m[ids_all[i]] in LOW else 0 for i in keep], dtype=int)
    groups = np.array([subject_of(ids_all[i]) for i in keep])
    return feats, y, groups, [ids_all[i] for i in keep]


def gbdt(seed=42):
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=120, learning_rate=0.06, min_samples_leaf=8,
        l2_regularization=1.0, class_weight="balanced", random_state=seed)


def oof_predict(X, y, groups, n_splits=5, seed=42):
    prob = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = gbdt(seed).fit(X[tr], y[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]
    return prob


def cluster_ci(y, p, g, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.unique(g)
    by = {s: np.flatnonzero(g == s) for s in ids}
    out = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        b = np.concatenate([by[s] for s in pick])
        if len(np.unique(y[b])) > 1:
            out.append(roc_auc_score(y[b], p[b]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan,)*2


def perm_p(y, p, g, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(y, p)
    by = {s: np.flatnonzero(g == s) for s in np.unique(g)}
    hits = 0
    for _ in range(n):
        yp = y.copy()
        for i in by.values():
            yp[i] = rng.permutation(yp[i])
        if len(np.unique(yp)) > 1 and roc_auc_score(yp, p) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def report(name, y, p, g, n_features):
    ok = ~np.isnan(p)
    y, p, g = y[ok], p[ok], g[ok]
    pred = (p >= 0.5).astype(int)
    base = float(max(y.mean(), 1 - y.mean()))
    lo, hi = cluster_ci(y, p, g)
    ap = float(average_precision_score(y, p))
    r = {
        "rung": name, "n": int(len(y)), "n_features": int(n_features),
        "subjects": int(len(np.unique(g))), "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)), "auc_ci95_subject": [lo, hi],
        "ci_excludes_chance": bool(lo > 0.5),
        "perm_p_within_subject": perm_p(y, p, g),
        "accuracy": float((pred == y).mean()), "majority_baseline": base,
        "beats_baseline": bool((pred == y).mean() > base),
        "average_precision": ap, "ap_over_base_rate": ap / float(y.mean()),
        "kappa": float(cohen_kappa_score(y, pred)),
        "score_min": float(p.min()), "score_max": float(p.max()),
        "score_mean": float(p.mean()), "score_spread": float(p.max() - p.min()),
    }
    print(f"  {name:18} feat={r['n_features']:3} AUC={r['auc']:.4f} "
          f"[{lo:.3f},{hi:.3f}]{'*' if r['ci_excludes_chance'] else ' '} "
          f"p={r['perm_p_within_subject']:.4f} "
          f"acc={r['accuracy']:.3f}/base={base:.3f}{'+' if r['beats_baseline'] else '-'} "
          f"AP={ap:.3f}({r['ap_over_base_rate']:.2f}x) spread={r['score_spread']:.3f}")
    return r


def run(geom_npz, labels_dir, split, rung, out_dir="/content/results"):
    feats, y, g, ids = load(geom_npz, labels_dir, split)
    X = build_matrix(feats, rung)
    p = oof_predict(X, y, g)
    r = report(rung, y, p, g, X.shape[1])
    r["split"], r["stage"] = split, "screen" if split == "Validation" else "confirm"
    pathlib.Path(out_dir).mkdir(exist_ok=True, parents=True)
    pathlib.Path(f"{out_dir}/{split}_{rung}.json").write_text(json.dumps(r, indent=2))
    np.savez_compressed(f"{out_dir}/{split}_{rung}_preds.npz",
                        y_true=y, y_prob=p, groups=g, clip_id=np.array(ids))
    return r
