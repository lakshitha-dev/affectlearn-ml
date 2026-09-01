"""Phase A real-data loader: platform research export -> labeled feature windows.

Turns the AffectLearn research export (a list of `research_events`) into the same
window records the trainer consumes — but sourced from REAL pilot data instead of the
synthetic generator. Unlike `synthetic_data.py` (which emits raw `events` re-extracted
by `feature_engineering.extract_features`), the backend already computed the aggregate
feature window at serve time and persisted it, so each real window carries `features`
directly. This guarantees train/serve parity by construction (guide §7): the model is
trained on the exact `(n_bins, N_FEATURES)` arrays it saw live — no re-extraction, and
no raw coordinates/keys leave the server (consent: "only aggregate features").

Export shape (per `backend/app/api/routes/research.py` phase-a-dataset, camelCase):
  behavioral_affect_detected.payload.features  -> (n_bins, N_FEATURES) aggregate window
  self_report.payload.{affect, skipped, omitted, prompt_index, section_id}

Labeling (guide Step 7, default = "self-reports only"): for each self-report with a
valid affect, label the nearest PRECEDING behavioral window in the same session (the
30 s window ending at the prompt). `propagate_ms > 0` widens this to every window in
`[t - propagate_ms, t]` (guide's optional label propagation for a larger, noisier set).
Skipped/omitted/neutral/null self-reports carry no training label and are ignored.
"""

from collections import defaultdict

# Class order — identical to synthetic_data.LABELS and the backend BEHAVIORAL_CLASS_ORDER.
LABELS = ["Engaged", "Bored", "Confused", "Frustrated"]
_AFFECT_IDX = {"engaged": 0, "bored": 1, "confused": 2, "frustrated": 3}


def _get(ev, *names):
    """Read the first present key (handles camelCase API export or snake_case DB dump)."""
    for n in names:
        if n in ev and ev[n] is not None:
            return ev[n]
    return None


def load_phase_a_windows(events, propagate_ms: int = 0):
    """Export events -> list of {participant, label, label_idx, features}.

    `events` is the list of research-event dicts (the API page's `items`, or a raw dump).
    Returns windows in the same shape `dataset.windows_to_arrays` consumes (with `features`
    instead of `events`). Deterministic; empty if there is nothing joinable.
    """
    behavioral = []  # (session, ts, participant, features)
    reports = []     # (session, ts, label_idx)
    for ev in events:
        etype = _get(ev, "eventType", "event_type")
        session = _get(ev, "sessionId", "session_id")
        ts = _get(ev, "timestamp")
        payload = _get(ev, "payload") or {}
        if etype == "behavioral_affect_detected":
            feats = payload.get("features")
            if feats is None or ts is None:
                continue  # error/empty cycles carry no feature window
            behavioral.append((session, int(ts), _get(ev, "learnerId", "learner_id"), feats))
        elif etype == "self_report":
            if payload.get("skipped") or payload.get("omitted") or ts is None:
                continue  # a skip/omission is not a training label
            idx = _AFFECT_IDX.get(str(payload.get("affect") or "").lower())
            if idx is None:
                continue  # excludes neutral / null (not model classes)
            reports.append((session, int(ts), idx))

    by_session = defaultdict(list)
    for b in behavioral:
        by_session[b[0]].append(b)
    for s in by_session:
        by_session[s].sort(key=lambda b: b[1])

    # (session, window_ts) -> (participant, features, label_idx); later report wins on overlap.
    labeled: dict = {}
    for (rs, rts, ridx) in sorted(reports, key=lambda r: (str(r[0]), r[1])):
        wins = by_session.get(rs, [])
        if propagate_ms > 0:
            for (bs, bts, part, feats) in wins:
                if rts - propagate_ms <= bts <= rts:
                    labeled[(bs, bts)] = (part, feats, ridx)
        else:
            preceding = [w for w in wins if w[1] <= rts]
            if preceding:
                bs, bts, part, feats = preceding[-1]
                labeled[(bs, bts)] = (part, feats, ridx)

    return [
        dict(participant=part, label=LABELS[ridx], label_idx=ridx, features=feats)
        for (part, feats, ridx) in labeled.values()
    ]
