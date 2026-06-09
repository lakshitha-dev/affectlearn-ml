"""Data-quality checks for Phase A behavioral data (and synthetic stand-in).

Run before training to drop degenerate participants/sessions and surface the
class-imbalance reality (PRD line 235).
"""

import numpy as np


def per_class_counts(y: np.ndarray, labels):
    counts = np.bincount(y, minlength=len(labels))
    return {labels[i]: int(counts[i]) for i in range(len(labels))}


def degenerate_participants(y: np.ndarray, pid: np.ndarray, min_classes: int = 2):
    """Participants whose labels are all one class (probably not engaging the widget)."""
    bad = []
    for p in sorted(set(pid.tolist())):
        labs = set(y[pid == p].tolist())
        if len(labs) < min_classes:
            bad.append(p)
    return bad


def report(y: np.ndarray, pid: np.ndarray, labels):
    print("  participants:", len(set(pid.tolist())), "| windows:", len(y))
    print("  class counts:", per_class_counts(y, labels))
    bad = degenerate_participants(y, pid)
    print("  degenerate (single-class) participants:", bad or "none")
    return bad
