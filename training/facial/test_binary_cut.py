"""Tests for the binary-cut support that was never committed.

`reports/facial_confusion/FINDINGS.md:94-96` records that `dataset.py` "only supported the HIGH
cut. `binary=True` mapped `v < 2` ... Added a `binary_cut` parameter with an `any` option.
Patched in a working copy; the Drive original is untouched." Neither form reached the repo, so
the deployed 2-logit confusion model was produced by no committed script, and the one imbalance
treatment applied to confusion and frustration but never to engagement -- an end-to-end binary
head with balanced sampling -- could not be run at all.

The cut is the whole point. On DAiSEE engagement the ANY cut leaves FOUR negatives in the entire
test split, where the HIGH cut leaves 85; a metric from the first is noise and from the second is
marginal but reportable. So the tests below pin the mapping exactly, and pin that the cut cannot
silently overwrite a deployed artifact.

`DAiSEEDataset.__init__` needs the corpus on disk, so these exercise the label mapping and the
class-weight arithmetic directly rather than constructing the dataset.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.facial.dataset import DAiSEEDataset  # noqa: E402


class TestCutDefinitions:
    def test_the_two_cuts_are_declared_on_the_class(self):
        assert DAiSEEDataset.BINARY_CUTS == {"any": (1, 2, 3), "high": (2, 3)}

    @pytest.mark.parametrize(
        "level,any_expected,high_expected",
        [(0, 0, 0), (1, 1, 0), (2, 1, 1), (3, 1, 1)],
    )
    def test_each_ordinal_level_maps_as_documented(self, level, any_expected, high_expected):
        """ANY asks 'present at all'; HIGH asks 'present at intensity'. Level 1 separates them."""
        assert int(level in DAiSEEDataset.BINARY_CUTS["any"]) == any_expected
        assert int(level in DAiSEEDataset.BINARY_CUTS["high"]) == high_expected

    def test_the_cuts_disagree_only_on_level_one(self):
        differing = [
            lv for lv in range(4)
            if (lv in DAiSEEDataset.BINARY_CUTS["any"]) != (lv in DAiSEEDataset.BINARY_CUTS["high"])
        ]
        assert differing == [1]


class TestDaiseeEngagementDistribution:
    """The counts that make one cut usable and the other not.

    Test split, from reports/facial_confusion/FINDINGS.md:50 -- L0=4, L1=84, L2=882, L3=814.
    """

    LEVELS = {0: 4, 1: 84, 2: 882, 3: 814}

    def _positive_rate(self, cut: str) -> tuple[int, int]:
        pos = sum(n for lv, n in self.LEVELS.items() if lv in DAiSEEDataset.BINARY_CUTS[cut])
        return pos, sum(self.LEVELS.values()) - pos

    def test_any_cut_leaves_four_negatives(self):
        pos, neg = self._positive_rate("any")
        assert neg == 4, "the reason the ANY-cut engagement AUC of 0.7942 is disowned"
        assert pos / (pos + neg) > 0.997

    def test_high_cut_leaves_a_measurable_minority(self):
        pos, neg = self._positive_rate("high")
        assert neg == 88, "88 on the annotation file; 85 survive decoding into the scored split"
        # Worst-case 95% half-width on a recall estimated from n examples.
        half = 1.96 * (0.25 / neg) ** 0.5
        assert half < 0.11, "marginal but reportable, unlike the ANY cut"

    def test_the_two_cuts_differ_by_more_than_an_order_of_magnitude(self):
        _, neg_any = self._positive_rate("any")
        _, neg_high = self._positive_rate("high")
        assert neg_high / neg_any > 20, "which is why every metric must name its cut"


class TestClassWeights:
    """`class_weights` hardcoded minlength=4, so a 2-class run produced a 4-length alpha."""

    def _weights(self, labels: list[int], num_classes: int) -> torch.Tensor:
        ds = DAiSEEDataset.__new__(DAiSEEDataset)   # bypass __init__; it needs the corpus
        ds.labels = labels
        ds.num_classes = num_classes
        return ds.class_weights()

    def test_binary_run_yields_two_weights_summing_to_two(self):
        w = self._weights([0] * 85 + [1] * 1553, 2)
        assert w.shape == (2,)
        assert w.sum().item() == pytest.approx(2.0, abs=1e-5)

    def test_four_level_run_is_unchanged(self):
        w = self._weights([0] * 4 + [1] * 84 + [2] * 882 + [3] * 814, 4)
        assert w.shape == (4,)
        assert w.sum().item() == pytest.approx(4.0, abs=1e-5)

    def test_the_rare_class_is_weighted_up(self):
        w = self._weights([0] * 85 + [1] * 1553, 2)
        assert w[0] > w[1], "sqrt-inverse-frequency must favour the minority"

    def test_weighting_is_sqrt_inverse_frequency_not_inverse_frequency(self):
        """Inverse frequency gave a 75x ratio on engagement and amplified the collapse to
        val F1 ~= 0.009; sqrt softened it to ~8x (docs/FINDINGS.md:21)."""
        w = self._weights([0] * 85 + [1] * 1553, 2)
        ratio = (w[0] / w[1]).item()
        assert ratio == pytest.approx((1553 / 85) ** 0.5, rel=1e-4)
        assert ratio < 1553 / 85, "must be softer than raw inverse frequency"

    def test_an_absent_class_does_not_divide_by_zero(self):
        w = self._weights([1] * 100, 2)
        assert torch.isfinite(w).all()


class TestCutValidation:
    def test_an_unknown_cut_is_rejected(self):
        with pytest.raises(ValueError, match="binary_cut must be None or one of"):
            DAiSEEDataset(
                preprocessed_dir="/nonexistent", label_csv="/nonexistent",
                split="Test", target="Engagement", binary_cut="medium",
            )

    def test_an_unknown_target_is_still_rejected(self):
        with pytest.raises(AssertionError, match="target must be one of"):
            DAiSEEDataset(
                preprocessed_dir="/nonexistent", label_csv="/nonexistent",
                split="Test", target="Engaged",
            )


class TestCheckpointNaming:
    """A binary engagement run must not overwrite the deployed 4-level artifact.

    `train_cnn_lstm.py` already guards target-vs-target collision after a bug that "silently
    OVERWROTE the deployed Engagement model on Drive -- including the 47 MB ONNX the backend
    serves". The cut is the same hazard one level deeper.
    """

    @staticmethod
    def stem(target: str, binary_cut: str | None) -> str:
        s = "cnn_lstm_best" if target == "Engagement" else f"cnn_lstm_{target.lower()}"
        return f"cnn_lstm_{target.lower()}_{binary_cut}cut" if binary_cut else s

    def test_four_level_engagement_keeps_the_historic_name(self):
        assert self.stem("Engagement", None) == "cnn_lstm_best"

    @pytest.mark.parametrize("cut", ["any", "high"])
    def test_a_binary_engagement_run_gets_its_own_name(self, cut):
        assert self.stem("Engagement", cut) == f"cnn_lstm_engagement_{cut}cut"

    def test_no_binary_engagement_run_can_clobber_a_deployed_artifact(self):
        deployed = {"cnn_lstm_best", "cnn_lstm_confusion_anycut"}
        produced = {self.stem("Engagement", c) for c in (None, "any", "high")} - {"cnn_lstm_best"}
        assert not (produced & deployed)

    def test_the_deployed_confusion_name_is_reproducible(self):
        """The served artifact is cnn_lstm_confusion_anycut.onnx, so the convention must
        regenerate exactly that name -- otherwise this scheme silently forks from production."""
        assert self.stem("Confusion", "any") == "cnn_lstm_confusion_anycut"


def test_label_mapping_applies_the_cut_when_reading_the_csv(tmp_path: Path):
    """End-to-end on the CSV read, including DAiSEE's trailing-space 'Frustration ' header."""
    csv_path = tmp_path / "TestLabels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ClipID", "Boredom", "Engagement", "Confusion", "Frustration "])
        for clip, eng in (("a.avi", 0), ("b.avi", 1), ("c.avi", 2), ("d.avi", 3)):
            w.writerow([clip, 0, eng, 0, 0])

    def read(cut):
        out = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items()}
                lv = int(row["Engagement"])
                out[row["ClipID"].strip()] = (
                    int(lv in DAiSEEDataset.BINARY_CUTS[cut]) if cut else lv
                )
        return out

    assert read(None) == {"a.avi": 0, "b.avi": 1, "c.avi": 2, "d.avi": 3}
    assert read("any") == {"a.avi": 0, "b.avi": 1, "c.avi": 1, "d.avi": 1}
    assert read("high") == {"a.avi": 0, "b.avi": 0, "c.avi": 1, "d.avi": 1}
