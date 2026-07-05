"""Tests for the Phase A export -> labeled-window loader (torch-free)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phase_a import load_phase_a_windows  # noqa: E402


def _sample_events():
    return [
        # session s1 / participant P1 — two behavioral windows, then a self-report
        {"eventType": "behavioral_affect_detected", "sessionId": "s1", "learnerId": "P1",
         "timestamp": 1000, "payload": {"features": [[1.0]]}},
        {"eventType": "behavioral_affect_detected", "sessionId": "s1", "learnerId": "P1",
         "timestamp": 2000, "payload": {"features": [[2.0]]}},
        {"eventType": "self_report", "sessionId": "s1", "timestamp": 2100,
         "payload": {"affect": "confused", "skipped": False, "omitted": False}},
        # a skip, an omission, and a neutral — none are training labels
        {"eventType": "self_report", "sessionId": "s1", "timestamp": 3000,
         "payload": {"affect": None, "skipped": True}},
        {"eventType": "self_report", "sessionId": "s1", "timestamp": 3100,
         "payload": {"affect": None, "omitted": True}},
        {"eventType": "self_report", "sessionId": "s1", "timestamp": 3200,
         "payload": {"affect": "neutral", "skipped": False}},
        # session s2 / participant P2 — snake_case dump, mixed-case affect
        {"event_type": "behavioral_affect_detected", "session_id": "s2", "learner_id": "P2",
         "timestamp": 500, "payload": {"features": [[9.0]]}},
        {"event_type": "self_report", "session_id": "s2", "timestamp": 600,
         "payload": {"affect": "Frustrated", "skipped": False}},
    ]


def test_nearest_preceding_join_labels_one_window_per_report():
    windows = load_phase_a_windows(_sample_events())
    assert len(windows) == 2
    by_pid = {w["participant"]: w for w in windows}
    # confused@2100 -> nearest preceding s1 window is the 2000 one ([[2.0]])
    assert by_pid["P1"]["label_idx"] == 2 and by_pid["P1"]["features"] == [[2.0]]
    # Frustrated@600 (case-insensitive, snake_case) -> the 500 window ([[9.0]])
    assert by_pid["P2"]["label_idx"] == 3 and by_pid["P2"]["features"] == [[9.0]]


def test_skipped_omitted_neutral_and_null_are_excluded():
    events = [
        {"eventType": "behavioral_affect_detected", "sessionId": "s1", "learnerId": "P1",
         "timestamp": 1000, "payload": {"features": [[1.0]]}},
        {"eventType": "self_report", "sessionId": "s1", "timestamp": 1100,
         "payload": {"affect": None, "skipped": True}},
        {"eventType": "self_report", "sessionId": "s1", "timestamp": 1200,
         "payload": {"affect": None, "omitted": True}},
        {"eventType": "self_report", "sessionId": "s1", "timestamp": 1300,
         "payload": {"affect": "neutral", "skipped": False}},
    ]
    assert load_phase_a_windows(events) == []


def test_label_propagation_widens_to_all_windows_in_range():
    windows = load_phase_a_windows(_sample_events(), propagate_ms=1500)
    # s1 confused@2100 now labels BOTH windows in [600, 2100]; s2 unchanged (1 window)
    s1 = [w for w in windows if w["participant"] == "P1"]
    assert len(s1) == 2 and all(w["label_idx"] == 2 for w in s1)
    assert len(windows) == 3


def test_behavioral_windows_without_features_are_ignored():
    events = [
        {"eventType": "behavioral_affect_detected", "sessionId": "s1", "learnerId": "P1",
         "timestamp": 1000, "payload": {"error": "inference_error"}},  # no features
        {"eventType": "self_report", "sessionId": "s1", "timestamp": 1100,
         "payload": {"affect": "engaged", "skipped": False}},
    ]
    assert load_phase_a_windows(events) == []


if __name__ == "__main__":
    test_nearest_preceding_join_labels_one_window_per_report()
    test_skipped_omitted_neutral_and_null_are_excluded()
    test_label_propagation_widens_to_all_windows_in_range()
    test_behavioral_windows_without_features_are_ignored()
    print("all phase_a loader tests passed")
