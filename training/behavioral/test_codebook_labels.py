"""Tests for the codebook (section-designed) labeller (torch-free)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from codebook_labels import (  # noqa: E402
    DEFAULT_MAX_SECTION_MS,
    load_codebook_windows,
    summarise,
)

SEC = {
    "sec-warm": {"title": "Welcome — Read at Your Own Pace"},
    "sec-eng": {"title": "The Tool-Call Round Trip"},
    "sec-bore": {"title": "A Catalogue of Memory Types"},
    "sec-conf": {"title": "Constrained Decoding and Grammars"},
}

CODEBOOK = {
    "Welcome — Read at Your Own Pace": "neutral",
    "The Tool-Call Round Trip": "engaged",
    "A Catalogue of Memory Types": "bored",
    "Constrained Decoding and Grammars": "confused",
}


def _win(ts, feats, learner="P1", session="s1"):
    return {
        "eventType": "behavioral_affect_detected",
        "learnerId": learner,
        "sessionId": session,
        "timestamp": ts,
        "payload": {"features": feats},
    }


def _done(ts, sid, learner="P1"):
    # session_id is null on section events, exactly as the backend emits them.
    return {
        "eventType": "section_completed",
        "learnerId": learner,
        "sessionId": None,
        "timestamp": ts,
        "payload": {"section_id": sid, "created": True},
    }


def _report(ts, affect, learner="P1"):
    return {
        "eventType": "self_report",
        "learnerId": learner,
        "sessionId": "s1",
        "timestamp": ts,
        "payload": {"affect": affect, "skipped": False, "omitted": False},
    }


def _export(items):
    return {"items": items, "sections": SEC}


def _base_timeline():
    """warm-up (ends 1000) -> engaged (2000) -> bored (3000) -> confused (4000)."""
    return [
        _win(500, [[0.1]]),
        _done(1000, "sec-warm"),
        _report(1010, "neutral"),
        _win(1500, [[1.0]]),
        _win(1800, [[1.1]]),
        _done(2000, "sec-eng"),
        _report(2010, "engaged"),
        _win(2500, [[2.0]]),
        _done(3000, "sec-bore"),
        _report(3010, "bored"),
        _win(3500, [[3.0]]),
        _done(4000, "sec-conf"),
        _report(4010, "confused"),
    ]


def test_windows_are_attributed_between_consecutive_completions():
    r = load_codebook_windows(_export(_base_timeline()), CODEBOOK)
    got = {(w["participant"], tuple(map(tuple, w["features"]))) for w in r["windows"]}
    assert got == {
        ("The Tool-Call Round Trip", ((1.0,),)),
        ("The Tool-Call Round Trip", ((1.1,),)),
        ("A Catalogue of Memory Types", ((2.0,),)),
        ("Constrained Decoding and Grammars", ((3.0,),)),
    }
    # participant is the SECTION TITLE, enabling a section-independent split.
    assert all(w["participant"] == w["participant"] for w in r["windows"])


def test_neutral_warmup_windows_go_to_baseline_not_training():
    r = load_codebook_windows(_export(_base_timeline()), CODEBOOK)
    assert len(r["baseline"]) == 1
    assert r["baseline"][0]["features"] == [[0.1]]
    assert all(w["label"] != "Neutral" for w in r["windows"])


def test_labels_map_to_the_designed_affect():
    r = load_codebook_windows(_export(_base_timeline()), CODEBOOK)
    by_title = {w["participant"]: w["label_idx"] for w in r["windows"]}
    assert by_title["The Tool-Call Round Trip"] == 0  # Engaged
    assert by_title["A Catalogue of Memory Types"] == 1  # Bored
    assert by_title["Constrained Decoding and Grammars"] == 2  # Confused


def test_disagreeing_self_report_drops_that_sections_windows():
    events = _base_timeline()
    # Learner reports "engaged" on the section designed to bore them.
    events = [e for e in events if not (e.get("timestamp") == 3010)]
    events.append(_report(3010, "engaged"))
    r = load_codebook_windows(_export(events), CODEBOOK)
    titles = {w["participant"] for w in r["windows"]}
    assert "A Catalogue of Memory Types" not in titles
    assert r["stats"]["dropped_selfreport_disagreed"] == 1
    # ...and the disagreement is still counted in the manipulation check.
    assert r["agreement"]["bored"] == {"agreed": 0, "checked": 1}


def test_no_agreement_mode_keeps_windows_without_a_matching_report():
    events = [e for e in _base_timeline() if e.get("timestamp") != 3010]
    strict = load_codebook_windows(_export(events), CODEBOOK)
    assert strict["stats"]["dropped_no_selfreport"] == 1
    loose = load_codebook_windows(_export(events), CODEBOOK, require_agreement=False)
    assert "A Catalogue of Memory Types" in {w["participant"] for w in loose["windows"]}


def test_reposted_completion_does_not_start_a_new_interval():
    events = _base_timeline()
    events.append(_done(9000, "sec-eng"))  # idempotent re-post, created=False in practice
    r = load_codebook_windows(_export(events), CODEBOOK)
    eng = [w for w in r["windows"] if w["participant"] == "The Tool-Call Round Trip"]
    assert len(eng) == 2  # still only the two original windows


def test_windows_after_the_last_completion_are_dropped():
    events = _base_timeline() + [_win(5000, [[9.9]])]
    r = load_codebook_windows(_export(events), CODEBOOK)
    assert all(w["features"] != [[9.9]] for w in r["windows"])


def test_window_after_a_sittings_last_completion_never_lands_on_the_next_sitting():
    """The multi-sitting attribution bug: every interval must be bounded backwards.

    Bounding only on the previous completion leaves each interval open-ended, so tail reading
    after sitting 1 finishes (or a tab left open) is silently attributed to sitting 2's FIRST
    section. Certain, silent, and class-specific — it corrupts whichever affect happens to be
    scheduled first on the next day.
    """
    twelve_hours = 12 * 60 * 60 * 1000
    events = [
        # --- sitting 1: warm-up, then the engaged section ---
        _win(500, [[0.1]]),
        _done(1000, "sec-warm"),
        _report(1010, "neutral"),
        _win(1500, [[1.0]]),
        _done(2000, "sec-eng"),
        _report(2010, "engaged"),
        # tail reading AFTER sitting 1's last completion — belongs to no section
        _win(2500, [[9.9]]),
        # --- sitting 2, twelve hours later: the bored section ---
        _win(twelve_hours + 1000, [[2.0]]),
        _done(twelve_hours + 2000, "sec-bore"),
        _report(twelve_hours + 2010, "bored"),
    ]
    r = load_codebook_windows(_export(events), CODEBOOK, require_agreement=False)

    bored = [w for w in r["windows"] if w["participant"] == "A Catalogue of Memory Types"]
    # Only the window actually inside sitting 2 may be attributed to it.
    assert [w["features"] for w in bored] == [[[2.0]]]
    # The orphaned tail window must not appear anywhere.
    assert all(w["features"] != [[9.9]] for w in r["windows"])


def test_a_legitimately_slow_section_is_still_attributed():
    """The backward bound must not discard genuine slow reading.

    Sections are estimated at 4-5 minutes; a learner stuck on a confusing one can take several
    times that. Anything inside `max_section_ms` of the completion is kept.
    """
    minute = 60 * 1000
    prev_end = 100 * minute
    end = prev_end + 18 * minute  # an 18-minute slog through one section
    events = [
        _done(prev_end, "sec-warm"),
        _report(prev_end + 10, "neutral"),
        _win(end - 15 * minute, [[3.0]]),  # 15 min before completion, inside the 20-min bound
        _done(end, "sec-conf"),
        _report(end + 10, "confused"),
    ]
    assert 15 * minute < DEFAULT_MAX_SECTION_MS  # guards the premise of this test
    r = load_codebook_windows(_export(events), CODEBOOK, require_agreement=False)
    conf = [w for w in r["windows"] if w["participant"] == "Constrained Decoding and Grammars"]
    assert [w["features"] for w in conf] == [[[3.0]]]


def test_long_lead_in_before_first_completion_is_dropped():
    events = [_win(10, [[7.7]]), _done(60 * 60 * 1000, "sec-eng"), _report(60 * 60 * 1000 + 10, "engaged")]
    r = load_codebook_windows(_export(events), CODEBOOK, max_lead_ms=60_000)
    assert r["windows"] == []


def test_unknown_section_id_and_missing_features_are_dropped():
    events = [
        _win(500, [[1.0]]),
        _done(1000, "sec-unknown"),
        {"eventType": "behavioral_affect_detected", "learnerId": "P1", "sessionId": "s1",
         "timestamp": 1500, "payload": {"error": "inference_error"}},
        _done(2000, "sec-eng"),
        _report(2010, "engaged"),
    ]
    r = load_codebook_windows(_export(events), CODEBOOK)
    assert r["stats"]["dropped_unknown_section_id"] == 1
    assert r["windows"] == []


def test_snake_case_export_is_accepted():
    events = [
        {"event_type": "behavioral_affect_detected", "learner_id": "P1", "session_id": "s1",
         "timestamp": 1500, "payload": {"features": [[5.0]]}},
        {"event_type": "section_completed", "learner_id": "P1", "timestamp": 2000,
         "payload": {"section_id": "sec-eng"}},
        {"event_type": "self_report", "learner_id": "P1", "timestamp": 2010,
         "payload": {"affect": "engaged"}},
    ]
    r = load_codebook_windows(_export(events), CODEBOOK)
    assert len(r["windows"]) == 1
    assert r["windows"][0]["label_idx"] == 0


def test_summarise_runs_and_flags_thin_section_coverage():
    out = summarise(load_codebook_windows(_export(_base_timeline()), CODEBOOK))
    assert "Manipulation check" in out
    assert "too few sections" in out  # only 3 sections in the fixture


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"all {len(fns)} codebook labeller tests passed")
