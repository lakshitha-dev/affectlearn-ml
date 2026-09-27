"""Hierarchical two-stage affect head: "does this learner need help?", then "what kind?".

WHY TWO STAGES
--------------
Flat 4-class is the hardest possible framing of this problem and it is not what the adaptation
loop needs. The loop's first decision is binary — intervene or not — and the Pedagogical
Strategist already selects WHICH intervention from the learner profile and content context. So
the model should be asked the question the system actually poses.

It is also far better conditioned on the available data. From the 426-window codebook pass:

    stage 1   engaged 220  vs  needs-help 206      51.6% / 48.4%   — near-perfectly balanced
    stage 2   confused 90 / bored 66 / frustrated 50   44% / 32% / 24%   — workable with weights

Stage 1 is balanced almost exactly, and two classes across 47 sections stratify trivially, which
dissolves the empty-class fold problem that makes flat 4-class macro-F1 uninterpretable at this
scale.

WHAT IT DOES NOT DO. The three help-states are NOT merged into one reported class. D'Mello's
affect dynamics separates them and the adaptations differ — bored raises difficulty, confused
scaffolds, frustrated triggers a break then a format switch — so collapsing them flat would break
the pedagogy Chapter 3 depends on. Stage 2 preserves the distinction; it is simply asked only when
stage 1 says it matters.

PROBABILITIES COMPOSE PROPERLY:

    P(engaged)  = P1(engaged)
    P(c)        = P1(needs_help) * P2(c)      for c in {bored, confused, frustrated}

which sums to 1 and is a valid 4-class distribution, so the output is a drop-in replacement for a
flat model's `predict_proba` — including for AUC and for the adaptation gate's confidence floor.

Works with any sklearn-style estimator pair (`fit`, `predict_proba`), so the same head wraps the
GBDT arm and, via a thin adapter, the Bi-LSTM.
"""

from __future__ import annotations

import numpy as np

# Canonical 4-class order, identical to synthetic_data.LABELS / phase_a.LABELS / the backend's
# BEHAVIORAL_CLASS_ORDER. Do not reorder.
LABELS = ["Engaged", "Bored", "Confused", "Frustrated"]
ENGAGED = 0
HELP_CLASSES = (1, 2, 3)                    # Bored, Confused, Frustrated

STAGE1_LABELS = ["Engaged", "NeedsHelp"]
STAGE2_LABELS = [LABELS[i] for i in HELP_CLASSES]


def to_stage1(y: np.ndarray) -> np.ndarray:
    """4-class -> binary: 0 engaged, 1 needs help."""
    return (np.asarray(y) != ENGAGED).astype(np.int64)


def to_stage2(y: np.ndarray) -> np.ndarray:
    """4-class -> stage-2 index (0 bored, 1 confused, 2 frustrated). Undefined for engaged."""
    return np.asarray(y) - 1


class _ConstantStage:
    """Degenerate-fold stand-in: always predicts the one class it was shown.

    A grouped CV fold can legitimately contain only engaged windows (or only help windows), and
    most sklearn estimators raise rather than fit a single class. Refusing to fit would abort the
    whole comparison over one unlucky fold, so record the constant and stay a valid probability
    source. `predict_proba` returns a (n, 1) column and `classes_` names it, which is exactly the
    contract the composition code already handles.
    """

    def __init__(self, cls: int):
        self.classes_ = [int(cls)]

    def fit(self, X, y=None, sample_weight=None):
        return self

    def predict_proba(self, X):
        return np.ones((len(X), 1), dtype=float)

    def predict(self, X):
        return np.full(len(X), self.classes_[0], dtype=np.int64)


class TwoStageClassifier:
    """Compose a binary stage-1 and a 3-class stage-2 into a 4-class classifier.

    `make_stage1` / `make_stage2` are zero-arg factories returning a fresh estimator, so every
    fold gets an untrained model and nothing leaks between folds.
    """

    def __init__(self, make_stage1, make_stage2):
        self.make_stage1 = make_stage1
        self.make_stage2 = make_stage2
        self.stage1 = None
        self.stage2 = None
        self.stage2_classes_: list[int] = []

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X), np.asarray(y)
        y1 = to_stage1(y)

        # A fold can contain only engaged, or only help, windows. Most estimators raise on a
        # single class, which would abort the whole comparison over one unlucky fold.
        if len(set(y1.tolist())) < 2:
            self.stage1 = _ConstantStage(int(y1[0]) if len(y1) else 0)
        else:
            self.stage1 = self.make_stage1()
            self.stage1.fit(X, y1)

        # Stage 2 sees ONLY the help windows — that is the whole point of conditioning.
        help_mask = y1 == 1
        self.stage2 = None
        self.stage2_classes_ = []
        if help_mask.sum() >= 2:
            y2 = to_stage2(y[help_mask])
            present = sorted(set(y2.tolist()))
            if len(present) >= 2:
                self.stage2 = self.make_stage2()
                self.stage2.fit(X[help_mask], y2)
                self.stage2_classes_ = list(getattr(self.stage2, "classes_", present))
            else:
                # Only one help class in this fold: stage 2 has nothing to learn, so record the
                # constant rather than fitting a degenerate model.
                self.stage2_classes_ = present
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X)
        p1 = self.stage1.predict_proba(X)
        # Guard against a stage-1 fold that saw only one class.
        cls1 = list(getattr(self.stage1, "classes_", [0, 1]))
        p_engaged = p1[:, cls1.index(0)] if 0 in cls1 else np.zeros(len(X))
        p_help = p1[:, cls1.index(1)] if 1 in cls1 else np.zeros(len(X))

        out = np.zeros((len(X), len(LABELS)), dtype=np.float64)
        out[:, ENGAGED] = p_engaged

        if self.stage2 is not None:
            p2 = self.stage2.predict_proba(X)
            for col, cls in enumerate(self.stage2_classes_):
                out[:, int(cls) + 1] = p_help * p2[:, col]
        elif len(self.stage2_classes_) == 1:
            out[:, int(self.stage2_classes_[0]) + 1] = p_help
        else:
            # No stage-2 information at all: spread the help mass evenly rather than inventing
            # a preference the data does not support.
            for c in HELP_CLASSES:
                out[:, c] = p_help / len(HELP_CLASSES)

        total = out.sum(axis=1, keepdims=True)
        return np.divide(out, total, out=np.zeros_like(out), where=total > 0)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ---- the metric the deployed system actually gates on -------------------------------

    def predict_needs_help_proba(self, X) -> np.ndarray:
        """Stage-1 probability of needing an intervention.

        This is the number the adaptation gate should consume: the gate's job is 'intervene or
        not', and stage 1 answers exactly that — with better calibration than the summed tail of
        a flat 4-class softmax.
        """
        p1 = self.stage1.predict_proba(np.asarray(X))
        cls1 = list(getattr(self.stage1, "classes_", [0, 1]))
        return p1[:, cls1.index(1)] if 1 in cls1 else np.zeros(len(X))
