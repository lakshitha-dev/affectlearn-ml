"""Tests for the two-stage head and the GBDT aggregate features (torch-free).

The property that matters most: stage-1 and stage-2 probabilities must COMPOSE into a valid
4-class distribution. If they do not, the output silently stops being a drop-in replacement for a
flat model — AUC, the adaptation gate's confidence floor, and every reported metric would be
computed on numbers that do not sum to 1.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from aggregate_features import aggregate, aggregate_feature_names, aggregate_window  # noqa: E402
from feature_engineering import FEATURE_NAMES, N_FEATURES  # noqa: E402
from two_stage import (  # noqa: E402
    ENGAGED,
    HELP_CLASSES,
    LABELS,
    TwoStageClassifier,
    to_stage1,
    to_stage2,
)


class _Const:
    """Estimator returning fixed probabilities — isolates the composition logic from any learner."""

    def __init__(self, proba, classes):
        self.proba = np.asarray(proba, dtype=float)
        self.classes_ = list(classes)

    def fit(self, X, y=None, sample_weight=None):
        return self

    def predict_proba(self, X):
        return np.tile(self.proba, (len(X), 1))


# ---------------------------------------------------------------- label mapping

def test_stage1_maps_engaged_against_everything_else():
    y = np.array([0, 1, 2, 3])
    assert list(to_stage1(y)) == [0, 1, 1, 1]


def test_stage2_indexes_the_help_classes_from_zero():
    assert list(to_stage2(np.array([1, 2, 3]))) == [0, 1, 2]


def test_label_constants_are_consistent():
    assert LABELS[ENGAGED] == "Engaged"
    assert [LABELS[i] for i in HELP_CLASSES] == ["Bored", "Confused", "Frustrated"]


# ---------------------------------------------------------------- probability composition

def test_probabilities_compose_and_sum_to_one():
    X = np.zeros((5, 3))
    clf = TwoStageClassifier(
        lambda: _Const([0.3, 0.7], [0, 1]),          # 30% engaged, 70% needs help
        lambda: _Const([0.5, 0.3, 0.2], [0, 1, 2]),  # of the help mass: bored/confused/frustrated
    )
    clf.stage1 = clf.make_stage1()
    clf.stage2 = clf.make_stage2()
    clf.stage2_classes_ = [0, 1, 2]

    p = clf.predict_proba(X)
    assert p.shape == (5, 4)
    assert np.allclose(p.sum(axis=1), 1.0)
    # P(engaged)=0.3 ; P(bored)=0.7*0.5 ; P(confused)=0.7*0.3 ; P(frustrated)=0.7*0.2
    assert np.allclose(p[0], [0.3, 0.35, 0.21, 0.14])


def test_predict_is_the_argmax_of_the_composed_distribution():
    X = np.zeros((3, 2))
    clf = TwoStageClassifier(
        lambda: _Const([0.4, 0.6], [0, 1]),
        lambda: _Const([0.1, 0.8, 0.1], [0, 1, 2]),
    )
    clf.stage1, clf.stage2, clf.stage2_classes_ = clf.make_stage1(), clf.make_stage2(), [0, 1, 2]
    # engaged 0.40 vs confused 0.6*0.8 = 0.48 -> confused wins
    assert list(clf.predict(X)) == [2, 2, 2]


def test_needs_help_probability_comes_straight_from_stage_one():
    """The gate consumes this — it must be stage 1's own output, not a summed softmax tail."""
    X = np.zeros((4, 2))
    clf = TwoStageClassifier(lambda: _Const([0.25, 0.75], [0, 1]), lambda: None)
    clf.stage1 = clf.make_stage1()
    assert np.allclose(clf.predict_needs_help_proba(X), 0.75)


# ---------------------------------------------------------------- degenerate folds

def _fit_on(y):
    """Fit with a real estimator on separable data so fold degeneracies are exercised."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.4, (len(y), 4))
    X[np.arange(len(y)), y % 4] += 4.0
    mk = lambda: LogisticRegression(max_iter=500)  # noqa: E731
    return TwoStageClassifier(mk, mk).fit(X, y), X


def test_single_help_class_in_fold_puts_all_help_mass_on_it():
    clf, X = _fit_on([0] * 12 + [2] * 12)          # only Confused among the help states
    p = clf.predict_proba(X)
    assert clf.stage2 is None                       # nothing to learn -> no degenerate model
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[:, 1], 0.0) and np.allclose(p[:, 3], 0.0)


def test_no_help_windows_at_all_still_yields_a_valid_distribution():
    clf, X = _fit_on([0] * 20)                      # engaged only
    p = clf.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_full_four_class_fit_recovers_the_labels():
    y = np.array([0] * 15 + [1] * 12 + [2] * 12 + [3] * 12)
    clf, X = _fit_on(y)
    assert (clf.predict(X) == y).mean() > 0.85
    assert np.allclose(clf.predict_proba(X).sum(axis=1), 1.0)


# ---------------------------------------------------------------- aggregate features

def test_aggregate_names_match_the_vector_width():
    names = aggregate_feature_names()
    assert len(names) == N_FEATURES * 5
    out = aggregate_window(np.zeros((30, N_FEATURES)))
    assert out.shape == (len(names),)
    # Channel-major: the first five entries all belong to the first channel.
    assert all(n.startswith(FEATURE_NAMES[0] + "_") for n in names[:5])


def test_aggregate_computes_the_expected_statistics():
    X = np.zeros((30, N_FEATURES))
    X[:, 0] = np.arange(30)                          # 0..29 on channel 0
    names = aggregate_feature_names()
    v = dict(zip(names, aggregate_window(X)))
    ch = FEATURE_NAMES[0]
    assert np.isclose(v[f"{ch}_mean"], 14.5)
    assert np.isclose(v[f"{ch}_min"], 0.0)
    assert np.isclose(v[f"{ch}_max"], 29.0)
    # trend = mean(last 10) - mean(first 10) = 24.5 - 4.5
    assert np.isclose(v[f"{ch}_trend"], 20.0)


def test_aggregate_batches_and_rejects_wrong_rank():
    assert aggregate(np.zeros((7, 30, N_FEATURES))).shape == (7, N_FEATURES * 5)
    for bad in (np.zeros((30, N_FEATURES)), np.zeros(5)):
        try:
            aggregate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for wrong rank")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"all {len(fns)} two-stage / aggregate tests passed")
