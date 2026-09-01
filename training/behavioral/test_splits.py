"""Tests for group-wise splits (torch-free — imports `splits`, not `dataset`).

The headline test is `test_stratified_keeps_every_class_in_every_fold_across_seeds`, which
quantifies the failure the stratified split exists to prevent.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from splits import (  # noqa: E402
    assert_all_classes_present,
    group_labels,
    split_by_participant,
    split_by_participant_stratified,
)

# The real codebook section counts: engaged 26, confused 9, bored 7, frustrated 5.
# Label indices follow LABELS = [Engaged, Bored, Confused, Frustrated].
SECTIONS_PER_CLASS = {0: 26, 1: 7, 2: 9, 3: 5}
WINDOWS_PER_SECTION = 8


def _codebook_like():
    """Build (pid, y) mirroring the real section/class distribution."""
    pid, y = [], []
    for label, n_sections in SECTIONS_PER_CLASS.items():
        for s in range(n_sections):
            name = f"sec-{label}-{s:02d}"
            pid += [name] * WINDOWS_PER_SECTION
            y += [label] * WINDOWS_PER_SECTION
    return np.array(pid), np.array(y, dtype=np.int64)


def _folds_missing_a_class(masks, y, n_classes=4):
    return sum(
        1
        for m in masks
        if set(range(n_classes)) - set(np.unique(y[m]).tolist() if m.any() else [])
    )


def test_group_labels_recovers_one_label_per_section():
    pid, y = _codebook_like()
    gl = group_labels(pid, y)
    assert len(gl) == sum(SECTIONS_PER_CLASS.values()) == 47
    assert gl["sec-3-00"] == 3


def test_uniform_split_frequently_loses_a_class():
    """Documents the bug: an unstratified group shuffle drops classes from folds."""
    pid, y = _codebook_like()
    bad_test, bad_any = 0, 0
    for seed in range(20):
        tr, va, te, _ = split_by_participant(pid, seed=seed)
        if set(range(4)) - set(np.unique(y[te]).tolist()):
            bad_test += 1
        if _folds_missing_a_class((tr, va, te), y):
            bad_any += 1
    # Not asserting exact counts (RNG-dependent), only that this is common rather than rare.
    assert bad_test >= 10, f"expected the uniform split to lose a test class often, got {bad_test}/20"
    assert bad_any >= 15, f"expected most seeds to have some empty-class fold, got {bad_any}/20"


def test_stratified_keeps_every_class_in_every_fold_across_seeds():
    """The fix: all four classes present in train, val AND test, for every seed 0-19."""
    pid, y = _codebook_like()
    for seed in range(20):
        tr, va, te, groups, warnings = split_by_participant_stratified(pid, y, seed=seed)
        assert not warnings, f"seed {seed}: unexpected degenerate class {warnings}"
        assert _folds_missing_a_class((tr, va, te), y) == 0, f"seed {seed} lost a class"
        assert_all_classes_present(y, (tr, va, te), n_classes=4)


def test_stratified_never_puts_a_group_in_two_folds():
    pid, y = _codebook_like()
    for seed in range(20):
        _, _, _, (tr_g, va_g, te_g), _ = split_by_participant_stratified(pid, y, seed=seed)
        assert not (set(tr_g) & set(va_g))
        assert not (set(tr_g) & set(te_g))
        assert not (set(va_g) & set(te_g))
        assert len(tr_g) + len(va_g) + len(te_g) == 47


def test_stratified_masks_partition_every_window():
    pid, y = _codebook_like()
    tr, va, te, _, _ = split_by_participant_stratified(pid, y, seed=0)
    assert (tr.astype(int) + va.astype(int) + te.astype(int) == 1).all()


def test_stratified_is_deterministic_for_a_seed():
    pid, y = _codebook_like()
    a = split_by_participant_stratified(pid, y, seed=7)[3]
    b = split_by_participant_stratified(pid, y, seed=7)[3]
    assert a == b


def test_two_group_class_warns_and_skips_val():
    pid = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4 + ["e"] * 4 + ["f"] * 4)
    y = np.array([0] * 16 + [1] * 8, dtype=np.int64)  # class 1 has only 2 groups
    tr, va, te, groups, warnings = split_by_participant_stratified(pid, y, seed=0)
    assert any("only 2 groups" in w for w in warnings)
    assert 1 in set(np.unique(y[tr]).tolist())
    assert 1 in set(np.unique(y[te]).tolist())


def test_single_group_class_warns():
    pid = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4 + ["z"] * 4)
    y = np.array([0] * 16 + [1] * 4, dtype=np.int64)  # class 1 has 1 group
    *_, warnings = split_by_participant_stratified(pid, y, seed=0)
    assert any("only 1 group" in w for w in warnings)


def test_assert_all_classes_present_raises_on_a_gap():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    full = np.ones(8, bool)
    no_class_3 = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    assert_all_classes_present(y, (full, full, full), n_classes=4)  # should not raise
    try:
        assert_all_classes_present(y, (full, full, no_class_3), n_classes=4)
    except ValueError as e:
        assert "test" in str(e) and "3" in str(e)
    else:
        raise AssertionError("expected ValueError for the missing class")


def test_stratified_respects_approximate_ratios():
    pid, y = _codebook_like()
    tr, va, te, _, _ = split_by_participant_stratified(pid, y, seed=1)
    n = len(pid)
    assert 0.55 <= tr.sum() / n <= 0.80
    assert 0.08 <= va.sum() / n <= 0.28
    assert 0.08 <= te.sum() / n <= 0.28


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"all {len(fns)} split tests passed")
