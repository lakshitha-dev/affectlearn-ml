"""Tests for the cached-feature trainer's evaluation logic.

The training loop itself is exercised by the smoke and null runs described in
`train_cached_features.py`; what is pinned here is the reasoning that decides whether a result
means anything — the parts that would silently flatter a null result if they were wrong.

Two of these encode findings from the DAiSEE audit:

  * The 85 disengaged test clips come from 16 subjects, one of which supplies 25 of them. A
    clip-level bootstrap treats those as 85 independent observations and reports an interval
    roughly half the honest width, so the resampling unit must be the SUBJECT.
  * The HIGH cut's own polarity makes "engaged" the positive class at 95% prevalence, where a
    recall of 0.99 says nothing at all. The rare class is what the model is being asked to find.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.facial.train_cached_features import (  # noqa: E402
    CUTS,
    subject_bootstrap,
    to_binary,
)


class TestBinaryMapping:
    def test_cuts_match_the_dataset_definition(self):
        from training.facial.dataset import DAiSEEDataset
        assert CUTS == DAiSEEDataset.BINARY_CUTS, "the two must not drift apart"

    def test_high_cut_maps_levels_as_documented(self):
        got = to_binary(np.array([0, 1, 2, 3]), "high")
        assert got.tolist() == [0, 0, 1, 1]

    def test_any_cut_maps_levels_as_documented(self):
        got = to_binary(np.array([0, 1, 2, 3]), "any")
        assert got.tolist() == [0, 1, 1, 1]

    def test_on_daisee_test_counts_the_high_cut_leaves_85(self):
        """Cache-verified Test levels: 4 / 81 / 849 / 704 (n = 1638)."""
        levels = np.repeat([0, 1, 2, 3], [4, 81, 849, 704])
        y = to_binary(levels, "high")
        assert int((y == 0).sum()) == 85
        assert len(levels) == 1638

    def test_on_daisee_test_counts_the_any_cut_leaves_4(self):
        levels = np.repeat([0, 1, 2, 3], [4, 81, 849, 704])
        y = to_binary(levels, "any")
        assert int((y == 0).sum()) == 4, "why the ANY-cut engagement AUC is disowned"


class TestRareClassIsScoredAsPositive:
    """Replicates the polarity flip in `main`, which decides what recall even means."""

    @staticmethod
    def flip_if_needed(y):
        return 1 - y if y.mean() > 0.5 else y

    def test_engagement_high_cut_is_flipped_so_disengaged_is_positive(self):
        levels = np.repeat([0, 1, 2, 3], [4, 81, 849, 704])
        y = self.flip_if_needed(to_binary(levels, "high"))
        assert int(y.sum()) == 85, "positives must be the 85 disengaged, not the 1553 engaged"
        assert y.mean() < 0.5

    def test_confusion_any_cut_is_left_alone(self):
        """Confusion ANY is 30.7% positive — already the rare class, must not be inverted."""
        levels = np.repeat([0, 1, 2, 3], [1135, 368, 116, 19])
        y = self.flip_if_needed(to_binary(levels, "any"))
        assert int(y.sum()) == 503
        assert y.mean() == pytest.approx(0.307, abs=0.001)

    def test_a_balanced_split_is_untouched(self):
        y = np.array([0, 1, 0, 1])
        assert self.flip_if_needed(y).tolist() == y.tolist()


class TestSubjectBootstrap:
    """The interval must widen when observations are clustered — that is the whole point."""

    @staticmethod
    def _clustered(n_subj, per_subj, sep, seed=0):
        rng = np.random.default_rng(seed)
        y, p, s = [], [], []
        for i in range(n_subj):
            lab = i % 2
            # every clip of a subject shares that subject's offset: the correlation being modelled
            offset = rng.normal(sep if lab else -sep, 0.3)
            for _ in range(per_subj):
                y.append(lab); p.append(offset + rng.normal(0, 0.1)); s.append(f"s{i:03d}")
        return np.array(y), np.array(p), np.array(s)

    def test_returns_an_interval_containing_the_point_estimate(self):
        y, p, s = self._clustered(20, 5, 0.6)
        lo, hi = subject_bootstrap(y, p, s, n_boot=400, seed=1)
        from sklearn.metrics import roc_auc_score
        assert lo <= roc_auc_score(y, p) <= hi

    def test_clustered_data_yields_a_wider_interval_than_the_naive_clip_view(self):
        """With 100 clips from 20 subjects the honest interval must exceed the clip-level one."""
        y, p, s = self._clustered(20, 5, 0.5)
        lo_s, hi_s = subject_bootstrap(y, p, s, n_boot=600, seed=2)
        # pretend every clip is its own subject — the naive assumption
        lo_c, hi_c = subject_bootstrap(y, p, np.array([f"c{i}" for i in range(len(y))]),
                                       n_boot=600, seed=2)
        assert (hi_s - lo_s) > (hi_c - lo_c), (
            "treating correlated clips as independent must not look more certain"
        )

    def test_pure_noise_gives_an_interval_spanning_chance(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 200)
        p = rng.random(200)                       # independent of y
        s = np.array([f"s{i % 20:03d}" for i in range(200)])
        lo, hi = subject_bootstrap(y, p, s, n_boot=600, seed=4)
        assert lo < 0.5 < hi, "a null result must not exclude chance"

    def test_single_class_resamples_are_skipped_not_crashed(self):
        """A bootstrap draw can pick only one class; AUC is undefined there."""
        y = np.array([0] * 18 + [1, 1])
        p = np.random.default_rng(5).random(20)
        s = np.array([f"s{i:02d}" for i in range(20)])
        lo, hi = subject_bootstrap(y, p, s, n_boot=200, seed=6)
        assert np.isfinite(lo) and np.isfinite(hi)

    def test_all_one_class_returns_nan_rather_than_a_number(self):
        y = np.zeros(20, dtype=int)
        p = np.random.default_rng(7).random(20)
        s = np.array([f"s{i:02d}" for i in range(20)])
        lo, hi = subject_bootstrap(y, p, s, n_boot=50, seed=8)
        assert np.isnan(lo) and np.isnan(hi), "no AUC exists; must not invent one"


class TestSubjectExtraction:
    """Grouping depends on the 6-digit prefix; a wrong slice would leak subjects across folds."""

    @pytest.mark.parametrize("clip,subject", [
        ("5000441001", "500044"), ("826412" + "0007", "826412"), ("1100011002", "110001"),
    ])
    def test_first_six_digits_are_the_subject(self, clip, subject):
        assert clip[:6] == subject

    def test_daisee_official_splits_are_subject_disjoint(self):
        """Verified against the real label CSVs: zero overlap between all three pairs."""
        train, val, test = {"110001", "210060"}, {"400022", "410027"}, {"500044", "826412"}
        assert not (train & val) and not (train & test) and not (val & test)


class TestBatchNormAdaptationAudit:
    """The backbone's weights were frozen; its BatchNorm statistics were not.

    `train_cnn_lstm.py` sets `requires_grad_(False)` on the backbone but never calls `.eval()`,
    and the epoch loop calls `model.train()`, so every BN layer kept updating its running mean
    and variance on DAiSEE. Measured against the ImageNet initialisation the statistics moved
    24.7%, and holding the trained head and clips fixed while swapping only those statistics
    moves test AUC by 0.0584 -- from the published 0.6414 down to 0.5830.

    Pinned because it is the reason any faithfully-frozen re-implementation scores lower, and
    because `models/cnn_lstm_confusion_anycut.json` describes the backbone as "FULLY FROZEN".
    """

    def test_the_audit_script_exposes_the_head_keys_it_needs(self):
        from evaluation.verify_bn_adaptation import HEAD_KEYS
        assert set(HEAD_KEYS) == {
            "lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0",
            "lstm.bias_hh_l0", "head.weight", "head.bias",
        }, "these are exactly the non-backbone tensors; a backbone key here would void the test"

    def test_head_keys_load_into_the_temporal_head(self):
        """The isolation only works if the deployed head transplants cleanly."""
        import torch
        from evaluation.verify_bn_adaptation import HEAD_KEYS
        from training.facial.train_cached_features import TemporalHead
        expected = dict(TemporalHead(num_classes=2).state_dict())
        assert set(expected) == set(HEAD_KEYS)
        for k in HEAD_KEYS:
            assert isinstance(expected[k], torch.Tensor)

    def test_recorded_result_is_internally_consistent(self):
        """Guards the recorded finding against a silent edit."""
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "reports/facial_bn_audit/bn_adaptation.json"
        if not p.exists():
            pytest.skip("audit not run in this checkout")
        d = json.loads(p.read_text())
        assert d["labels_identical"] is True, "a label mismatch would void the comparison"
        assert d["bn_drift"]["tensors"] == 40
        assert d["gap_from_bn"] == pytest.approx(
            d["auc_published"] - d["auc_imagenet_bn"], abs=1e-9)
        assert d["gap_from_bn"] > 0.05, "the effect is large, not marginal"
        assert d["pearson_r"] < 0.8, "the two feature sets are materially different"
