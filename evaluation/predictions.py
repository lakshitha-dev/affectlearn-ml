"""The prediction record — the shared input every evaluation script reads.

WHY THIS EXISTS
---------------
Before this module there was no way to compute a metric without re-running a model against its
dataset. That is why `confusion_matrix.py`, `ablation_report.py`, `cross_validation.py` and
`statistical_tests.py` sat empty: each would have needed DAiSEE (17 GB, Drive-only) plus a
checkpoint plus new inference code. It also meant a finished evaluation could not be re-examined
later — e.g. collapsing four engagement levels into a binary engaged/disengaged split needs the
per-clip predictions, and nothing in the repo persisted them.

A prediction record decouples "run the model once" from "ask questions of the answers many
times". Dump one after any evaluation and every script here works offline, on a laptop, forever.

FORMAT (`.npz`, or `.json` for small sets)
------------------------------------------
    y_true   (N,)      int class indices
    y_pred   (N,)      int class indices
    y_prob   (N, C)    float, OPTIONAL — needed for AUC and for re-collapsing classes properly
    groups   (N,)      str,   OPTIONAL — participant / section / session id, for grouped CV
    ids      (N,)      str,   OPTIONAL — clip or window id, for joining back to source data
    labels   (C,)      str    class names, index-aligned

Anything derivable (accuracy, F1, a collapsed binary view) is deliberately NOT stored.

COLLAPSING
----------
`collapse()` merges classes into groups — the operation behind both the Model A binary recompute
(VL+L vs H+VH, which deletes the H<->VH boundary shown to be information-limited) and Model B's
two-stage head (engaged vs needs-help). When `y_prob` is present the collapsed probability is the
SUM over each group's member classes, which is the correct way to get a calibrated group
probability; `y_pred` is then the argmax of those sums, NOT the mapped original argmax. Those two
differ, and the sum is right — mapping the argmax discards evidence spread across sibling classes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Predictions:
    """A set of model predictions plus whatever is needed to interrogate them."""

    def __init__(self, y_true, y_pred, labels, y_prob=None, groups=None, ids=None):
        self.y_true = np.asarray(y_true, dtype=np.int64)
        self.y_pred = np.asarray(y_pred, dtype=np.int64)
        self.labels = [str(x) for x in labels]
        self.y_prob = None if y_prob is None else np.asarray(y_prob, dtype=np.float64)
        self.groups = None if groups is None else np.asarray([str(g) for g in groups])
        self.ids = None if ids is None else np.asarray([str(i) for i in ids])
        self._validate()

    def _validate(self) -> None:
        n = len(self.y_true)
        if len(self.y_pred) != n:
            raise ValueError(f"y_true has {n} rows but y_pred has {len(self.y_pred)}")
        for name, arr in (("y_prob", self.y_prob), ("groups", self.groups), ("ids", self.ids)):
            if arr is not None and len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} rows, expected {n}")
        if self.y_prob is not None and self.y_prob.shape[1] != len(self.labels):
            raise ValueError(
                f"y_prob has {self.y_prob.shape[1]} columns but {len(self.labels)} labels"
            )
        worst = max(self.y_true.max(initial=-1), self.y_pred.max(initial=-1))
        if worst >= len(self.labels):
            raise ValueError(f"class index {worst} is out of range for {len(self.labels)} labels")

    def __len__(self) -> int:
        return len(self.y_true)

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    # ---------------------------------------------------------------- io

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            payload = {
                "y_true": self.y_true.tolist(),
                "y_pred": self.y_pred.tolist(),
                "labels": self.labels,
            }
            for name in ("y_prob", "groups", "ids"):
                arr = getattr(self, name)
                if arr is not None:
                    payload[name] = arr.tolist()
            path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        else:
            arrays = {"y_true": self.y_true, "y_pred": self.y_pred,
                      "labels": np.array(self.labels)}
            for name in ("y_prob", "groups", "ids"):
                arr = getattr(self, name)
                if arr is not None:
                    arrays[name] = arr
            np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Predictions":
        path = Path(path)
        if path.suffix == ".json":
            d = json.loads(path.read_text(encoding="utf-8"))
        else:
            with np.load(path, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
        return cls(
            y_true=d["y_true"], y_pred=d["y_pred"], labels=list(d["labels"]),
            y_prob=d.get("y_prob"), groups=d.get("groups"), ids=d.get("ids"),
        )

    # ---------------------------------------------------------------- transforms

    def collapse(self, groups_of: dict[str, list[str]]) -> "Predictions":
        """Merge classes into named groups. `groups_of` maps new label -> old label names.

        Every original label must appear exactly once across the groups, so the mapping is
        total and unambiguous. When probabilities exist the group probability is the SUM over
        its members and `y_pred` is re-argmaxed from those sums (see module docstring).
        """
        assigned = [lab for members in groups_of.values() for lab in members]
        if sorted(assigned) != sorted(self.labels):
            missing = set(self.labels) - set(assigned)
            extra = set(assigned) - set(self.labels)
            raise ValueError(
                f"collapse mapping must cover every label exactly once; missing={sorted(missing)} "
                f"unknown={sorted(extra)} duplicated={len(assigned) != len(set(assigned))}"
            )

        new_labels = list(groups_of)
        old_index = {lab: i for i, lab in enumerate(self.labels)}
        old_to_new = np.empty(len(self.labels), dtype=np.int64)
        for new_i, (_, members) in enumerate(groups_of.items()):
            for lab in members:
                old_to_new[old_index[lab]] = new_i

        y_true = old_to_new[self.y_true]
        if self.y_prob is not None:
            prob = np.zeros((len(self), len(new_labels)), dtype=np.float64)
            for new_i, (_, members) in enumerate(groups_of.items()):
                cols = [old_index[lab] for lab in members]
                prob[:, new_i] = self.y_prob[:, cols].sum(axis=1)
            y_pred = prob.argmax(axis=1)
        else:
            prob = None
            y_pred = old_to_new[self.y_pred]

        return Predictions(y_true, y_pred, new_labels, y_prob=prob,
                           groups=self.groups, ids=self.ids)

    def subset(self, mask) -> "Predictions":
        mask = np.asarray(mask, dtype=bool)
        return Predictions(
            self.y_true[mask], self.y_pred[mask], self.labels,
            y_prob=None if self.y_prob is None else self.y_prob[mask],
            groups=None if self.groups is None else self.groups[mask],
            ids=None if self.ids is None else self.ids[mask],
        )


def dump_from_logits(logits, y_true, labels, groups=None, ids=None, path=None) -> Predictions:
    """Build (and optionally save) a record from raw model logits.

    This is the helper whose absence blocked every offline evaluation: call it once at the end
    of any eval loop and the predictions become permanently re-interrogable.
    """
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    prob = e / e.sum(axis=1, keepdims=True)
    preds = Predictions(y_true, prob.argmax(axis=1), labels, y_prob=prob, groups=groups, ids=ids)
    if path:
        preds.save(path)
    return preds
