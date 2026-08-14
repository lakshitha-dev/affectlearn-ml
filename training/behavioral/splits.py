"""Group-wise train/val/test splits for the behavioural dataset.

Deliberately TORCH-FREE (pure numpy) so the split logic can be unit-tested without a GPU stack
and without importing `dataset`, which pulls in torch. `dataset` re-exports both functions, so
existing `from dataset import split_by_participant` call sites are unaffected.

A "group" is whatever `participant` holds. Two schemes are in use:
  * real/synthetic multi-subject data -> participant = learner id  (subject-independent split)
  * single-subject codebook labelling -> participant = SECTION TITLE (section-independent split;
    see `codebook_labels` module docstring)

WHY THE STRATIFIED VERSION EXISTS
---------------------------------
`split_by_participant` shuffles groups uniformly at random with no regard to their class. Under
codebook labelling each section carries exactly ONE affect, and the section counts per class are
very uneven (engaged 26, confused 9, bored 7, frustrated 5). A uniform group shuffle therefore
leaves a fold with no frustrated section most of the time — measured over seeds 0-19, a test fold
was missing at least one class in 17/20 seeds.

That is not a cosmetic problem:
  * `sklearn` macro-F1 with an absent class and `zero_division=0` is mechanically capped at
    (k-1)/k — 0.75 for k=4 — so the headline metric becomes uninterpretable.
  * An absent class in the VALIDATION fold breaks early-stopping model selection, because the
    monitored score cannot see the class it is supposed to be selecting for.
  * `data_quality.report` runs on the full set *before* the split, so it cannot catch this.

`split_by_participant_stratified` splits each class's groups independently, which guarantees
every class appears in every fold whenever it has at least three groups.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def split_by_participant(pid: np.ndarray, seed: int = 42, ratios=(0.7, 0.15, 0.15)):
    """Uniform group shuffle. Kept for backwards compatibility and for balanced group sets.

    Prefer `split_by_participant_stratified` whenever group counts per class are uneven.
    """
    rng = np.random.default_rng(seed)
    users = np.array(sorted(set(pid.tolist())))
    rng.shuffle(users)
    n = len(users)
    n_tr = max(1, int(round(ratios[0] * n)))
    n_va = max(1, int(round(ratios[1] * n)))
    tr, va, te = users[:n_tr], users[n_tr:n_tr + n_va], users[n_tr + n_va:]
    if len(te) == 0:                       # tiny-N safety: borrow one from train
        te = tr[-1:]; tr = tr[:-1]
    sel = lambda group: np.isin(pid, group)  # noqa: E731
    return sel(tr), sel(va), sel(te), (tr.tolist(), va.tolist(), te.tolist())


def group_labels(pid: np.ndarray, y: np.ndarray) -> dict:
    """`{group: label}` using the majority label within each group.

    Under codebook labelling a group is a section and is single-labelled by construction, so the
    majority is exact. The mode is used anyway so the function stays correct if a grouping ever
    mixes labels (e.g. participant-level groups on multi-subject data).
    """
    per_group: dict = {}
    for g, lab in zip(pid.tolist(), y.tolist()):
        per_group.setdefault(g, []).append(lab)
    return {g: Counter(labs).most_common(1)[0][0] for g, labs in per_group.items()}


def split_by_participant_stratified(
    pid: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    ratios=(0.7, 0.15, 0.15),
):
    """Group-wise split that keeps every class present in train/val/test.

    Same return signature as `split_by_participant` — `(train_mask, val_mask, test_mask,
    (train_groups, val_groups, test_groups))` — so it is a drop-in replacement.

    Each class's groups are shuffled and divided independently. A class with >= 3 groups always
    lands at least one group in each of the three folds. A class with 2 groups gets one in train
    and one in test (val is skipped for that class); with 1 group it goes to train only. Those
    degenerate cases are reported in `warnings` rather than silently accepted.
    """
    rng = np.random.default_rng(seed)
    per_group = group_labels(pid, y)

    train: list = []
    val: list = []
    test: list = []
    warnings: list[str] = []

    for label in sorted({v for v in per_group.values()}):
        groups = sorted([g for g, lab in per_group.items() if lab == label])
        rng.shuffle(groups)
        n = len(groups)

        if n == 1:
            train += groups
            warnings.append(f"class {label}: only 1 group — train only, absent from val/test")
            continue
        if n == 2:
            test += groups[:1]
            train += groups[1:]
            warnings.append(f"class {label}: only 2 groups — no val group")
            continue

        n_test = max(1, int(round(ratios[2] * n)))
        n_val = max(1, int(round(ratios[1] * n)))
        # Never starve train: with n >= 3 this leaves at least one group in train.
        while n_test + n_val > n - 1:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        test += groups[:n_test]
        val += groups[n_test:n_test + n_val]
        train += groups[n_test + n_val:]

    sel = lambda g: np.isin(pid, np.array(g, dtype=pid.dtype)) if g else np.zeros(len(pid), bool)  # noqa: E731
    return sel(train), sel(val), sel(test), (sorted(train), sorted(val), sorted(test)), warnings


def assert_all_classes_present(y: np.ndarray, masks, n_classes: int | None = None) -> None:
    """Raise if any fold is missing a class. Call this immediately after splitting.

    Cheap insurance against the silent failure this module exists to prevent: an absent class
    caps macro-F1 and breaks early stopping, but produces no error on its own.
    """
    expected = set(range(n_classes)) if n_classes else set(np.unique(y).tolist())
    names = ("train", "val", "test")
    missing = {}
    for name, m in zip(names, masks):
        present = set(np.unique(y[m]).tolist()) if m.any() else set()
        gap = expected - present
        if gap:
            missing[name] = sorted(gap)
    if missing:
        raise ValueError(
            "split leaves classes absent from a fold, which makes macro-F1 uninterpretable "
            f"and breaks early stopping: {missing}"
        )
