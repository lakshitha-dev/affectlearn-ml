"""Tests for the evaluation toolkit (numpy/sklearn/scipy only — no torch, no dataset).

Focused on the properties that would silently produce a WRONG THESIS NUMBER if they broke:
collapsing must sum probabilities rather than remap an argmax, an ablation must refuse
misaligned records, a missing class must be flagged rather than averaged away, and McNemar must
stay paired.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ablation_report as abl  # noqa: E402
import confusion_matrix as cmx  # noqa: E402
from cross_validation import grouped_cv, stratified_group_folds  # noqa: E402
from predictions import Predictions, dump_from_logits  # noqa: E402
from statistical_tests import (  # noqa: E402
    compare_predictions,
    hedges_g,
    independent_outcome,
    mcnemar,
    paired_outcome,
)

LAB4 = ["Very Low", "Low", "High", "Very High"]
BINARY = {"disengaged": ["Very Low", "Low"], "engaged": ["High", "Very High"]}


# ------------------------------------------------------------------ predictions / collapse

def test_roundtrip_npz_and_json(tmp_path):
    p = Predictions([0, 1, 2], [0, 1, 1], LAB4[:3], y_prob=np.eye(3), groups=["a", "a", "b"])
    for name in ("p.npz", "p.json"):
        back = Predictions.load(p.save(tmp_path / name))
        assert np.array_equal(back.y_true, p.y_true)
        assert np.array_equal(back.y_pred, p.y_pred)
        assert back.labels == p.labels
        assert np.allclose(back.y_prob, p.y_prob)
        assert list(back.groups) == list(p.groups)


def test_collapse_sums_probabilities_rather_than_remapping_argmax():
    """The whole point of collapsing from probs: sibling evidence must add up.

    Here the 4-class argmax is 'High' (0.35), so a naive argmax remap would call this 'engaged'.
    But disengaged holds 0.30 + 0.33 = 0.63 against engaged's 0.35 + 0.02 = 0.37, so the correct
    collapsed prediction is 'disengaged'. Mapping the argmax throws that away.
    """
    prob = np.array([[0.30, 0.33, 0.35, 0.02]])
    p = Predictions([0], [2], LAB4, y_prob=prob)
    c = p.collapse(BINARY)
    assert c.labels == ["disengaged", "engaged"]
    assert np.allclose(c.y_prob[0], [0.63, 0.37])
    assert c.y_pred[0] == 0, "must be disengaged (0.63), not engaged from the 4-class argmax"
    assert c.y_true[0] == 0


def test_collapse_without_probs_falls_back_to_mapping_labels():
    p = Predictions([0, 3], [1, 2], LAB4)
    c = p.collapse(BINARY)
    assert list(c.y_true) == [0, 1]
    assert list(c.y_pred) == [0, 1]


def test_collapse_rejects_an_incomplete_mapping():
    p = Predictions([0], [0], LAB4)
    for bad in ({"x": ["Very Low"]},                       # missing three
                {"x": LAB4, "y": ["High"]},                # High duplicated
                {"x": LAB4 + ["Nope"]}):                   # unknown label
        try:
            p.collapse(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_validate_rejects_mismatched_shapes():
    for kwargs in ({"y_pred": [0, 0]}, {"y_prob": np.eye(2)}, {"groups": ["a"]}):
        base = {"y_true": [0, 1, 2], "y_pred": [0, 1, 2], "labels": LAB4[:3]}
        base.update(kwargs)
        try:
            Predictions(**base)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_dump_from_logits_produces_a_valid_distribution():
    p = dump_from_logits(np.array([[2.0, 1.0, 0.0, -1.0]]), [0], LAB4)
    assert np.isclose(p.y_prob.sum(), 1.0)
    assert p.y_pred[0] == 0


# ------------------------------------------------------------------ confusion matrix

def test_zero_recall_class_is_flagged():
    # class 0 exists in truth but is never predicted.
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([1, 1, 1, 1, 2, 2, 3, 3])
    m = cmx.compute(Predictions(y_true, y_pred, LAB4))
    assert "Very Low" in m["zero_recall_classes"]
    assert "NEVER CORRECT" in cmx.summary(m)


def test_absent_class_flags_the_macro_f1_ceiling():
    """A class missing from the split caps macro-F1 — the number must not stand unqualified."""
    y = np.array([0, 0, 1, 1, 2, 2])          # class 3 absent entirely
    m = cmx.compute(Predictions(y, y, LAB4))
    assert m["classes_absent_from_this_split"] == ["Very High"]
    assert np.isclose(m["macro_f1_ceiling"], 0.75)
    assert "capped" in cmx.summary(m)


def test_adjacent_error_share_identifies_the_ordinal_boundary():
    # every error is High <-> Very High, i.e. adjacent
    y_true = np.array([2] * 10 + [3] * 10)
    y_pred = np.array([2] * 5 + [3] * 5 + [3] * 5 + [2] * 5)
    m = cmx.compute(Predictions(y_true, y_pred, LAB4))
    assert m["adjacent_error_share"] == 1.0
    assert m["worst_confused_pair"] == "High<->Very High"


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 2, 3])
    m = cmx.compute(Predictions(y, y, LAB4))
    assert m["accuracy"] == 1.0 and m["weighted_f1"] == 1.0 and m["cohen_kappa"] == 1.0


def test_figure_writes_png_and_pdf(tmp_path):
    y = np.array([0, 1, 2, 3, 2, 3])
    out = cmx.plot(Predictions(y, y, LAB4), tmp_path / "cm.png")
    assert out.exists() and out.stat().st_size > 0
    assert out.with_suffix(".pdf").exists()


# ------------------------------------------------------------------ statistical tests

def test_mcnemar_uses_only_discordant_pairs():
    a = np.array([True, True, False, False, True, True])
    b = np.array([True, False, True, True, True, True])   # fixed 2, broke 1
    r = mcnemar(a, b)
    assert r["only_b_correct"] == 2 and r["only_a_correct"] == 1
    assert r["n_discordant"] == 3
    assert r["method"] == "exact binomial"          # small discordant count


def test_mcnemar_identical_models_are_not_significant():
    a = np.array([True, False, True, False] * 10)
    r = mcnemar(a, a.copy())
    assert r["n_discordant"] == 0 and r["p_value"] == 1.0


def test_compare_predictions_refuses_unpaired_records():
    y = np.array([0, 1, 2, 3])
    a = Predictions(y, y, LAB4)
    b = Predictions(y[::-1], y[::-1], LAB4)        # same items, different order
    try:
        compare_predictions(a, b)
    except ValueError as e:
        assert "not the same items" in str(e)
    else:
        raise AssertionError("expected a refusal for misaligned records")


def test_paired_outcome_reports_effect_size_and_ci():
    rng = np.random.default_rng(0)
    pre = rng.normal(50, 10, 30)
    # A NOISY gain, not a constant shift: identical differences are degenerate for Wilcoxon and
    # nothing like real pre/post data.
    r = paired_outcome(pre, pre + rng.normal(5.0, 3.0, 30), "gain")
    assert 3.5 < r["mean_difference"] < 6.5
    assert r["cohens_dz"] > 0.8                     # a real, large effect
    assert r["difference_ci95"][0] > 0              # CI excludes zero
    assert r["p_value_wilcoxon"] < 0.01


def test_independent_outcome_effect_sizes_have_the_right_sign():
    rng = np.random.default_rng(1)
    high, low = rng.normal(0.8, 0.1, 20), rng.normal(0.5, 0.1, 20)
    r = independent_outcome(high, low)
    assert r["hedges_g"] > 0 and r["rank_biserial"] > 0
    assert hedges_g(low, high) < 0


def test_small_sample_note_is_emitted():
    from statistical_tests import interpret
    rng = np.random.default_rng(2)
    txt = interpret(independent_outcome(rng.normal(0, 1, 12), rng.normal(0, 1, 11)))
    assert "ESTIMATION" in txt


# ------------------------------------------------------------------ cross validation

def _codebook_groups(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(4, n, p=[0.52, 0.15, 0.21, 0.12])
    groups = np.array([f"sec-{y[i]}-{i % 10:02d}" for i in range(n)])   # single-labelled
    X = rng.normal(0, 1, (n, 8))
    X[np.arange(n), y % 8] += 1.5
    return X, y, groups


def test_folds_keep_every_class_and_never_split_a_group():
    X, y, groups = _codebook_groups()
    masks = stratified_group_folds(groups, y, n_splits=5, seed=3)
    seen = set()
    for m in masks:
        assert set(np.unique(y[m]).tolist()) == {0, 1, 2, 3}, "a fold lost a class"
        g = set(groups[m].tolist())
        assert not (g & seen), "a group appeared in two folds"
        seen |= g
    assert sum(m.sum() for m in masks) == len(y), "folds must partition the data"


def test_grouped_cv_reports_a_noise_floor():
    X, y, groups = _codebook_groups()

    def fit_predict(Xtr, ytr, Xte):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=300, class_weight="balanced").fit(Xtr, ytr).predict(Xte)

    res = grouped_cv(X, y, groups, fit_predict, n_splits=5, labels=LAB4)
    assert len(res["folds"]) == 5
    assert res["aggregate"]["weighted_f1"]["sd"] >= 0
    from cross_validation import format_report
    assert "NOISE FLOOR" in format_report(res)


# ------------------------------------------------------------------ ablation

def _arm(y, strength, seed):
    rng = np.random.default_rng(seed)
    lg = rng.normal(0, 1, (len(y), 4))
    lg[np.arange(len(y)), y] += strength
    return dump_from_logits(lg, y, LAB4)


def test_ablation_refuses_misaligned_records():
    rng = np.random.default_rng(4)
    y = rng.choice(4, 100)
    face, beh = _arm(y, 1.0, 5), _arm(y, 0.6, 6)
    other = _arm(rng.choice(4, 100), 1.4, 7)          # different y_true
    try:
        abl.build(face, beh, other)
    except ValueError as e:
        assert "y_true" in str(e) or "not aligned" in str(e)
    else:
        raise AssertionError("expected a refusal for misaligned records")


def test_ablation_identifies_the_best_unimodal_and_states_the_prior():
    rng = np.random.default_rng(8)
    y = rng.choice(4, 300)
    r = abl.build(_arm(y, 1.2, 9), _arm(y, 0.5, 10), _arm(y, 1.5, 11))
    assert r["best_unimodal"] == "face_only"          # the stronger arm
    assert r["prior"]["naturalistic_pct"] == 4.59
    text = abl.format_report(r)
    assert "PRIOR EXPECTATION" in text
    assert text.index("PRIOR EXPECTATION") < text.index("FUSION GAIN"), \
        "the prior must be printed BEFORE the result"


def test_ablation_reports_a_negative_gain_plainly():
    rng = np.random.default_rng(12)
    y = rng.choice(4, 200)
    # fusion deliberately worse than the best unimodal
    r = abl.build(_arm(y, 1.6, 13), _arm(y, 1.4, 14), _arm(y, 0.3, 15))
    assert r["fusion_gain"]["weighted_f1_absolute"] < 0
    assert "did NOT beat" in abl.format_report(r)


if __name__ == "__main__":
    import tempfile

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in fns:
        if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
    print(f"all {len(fns)} evaluation tests passed")
