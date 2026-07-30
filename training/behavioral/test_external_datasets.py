"""Tests for the public-dataset adapters (torch-free; self-contained fixtures)."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np                                    # noqa: E402
from external_datasets import load_dux, load_emosurv, load_external_windows  # noqa: E402
from feature_engineering import extract_features, FEATURE_NAMES              # noqa: E402

_DUX_COLS = ["session", "timestamp", "type", "key", "x", "y", "yPosition",
             "emotion_affectiva_Neutral"]


def _write_dux(dirpath: Path):
    """A tiny DUX-format TSV: 2 sessions, mouse+key+scroll, one neutral + one emotion window."""
    rows = [_DUX_COLS]
    # session 1 — neutral (Neutral=95): spread across a 30s window, >5 events
    base = 1_000_000
    for i in range(8):
        rows.append(["1", str(base + i * 1000), "MouseMovement", "", str(800 + i), str(400 + i), "", "95"])
    rows.append(["1", str(base + 9000), "KeyPressed", "LETTER_A", "", "", "", "95"])
    rows.append(["1", str(base + 9500), "Scroll", "", "", "", "1.0", "95"])
    # session 2 — emotion (Neutral=40): includes a backspace + clicks
    for i in range(8):
        rows.append(["2", str(base + i * 1000), "MouseMovement", "", str(300 + i * 5), str(200 + i * 5), "", "40"])
    rows.append(["2", str(base + 8000), "KeyPressed", "BACK_SPACE", "", "", "", "40"])
    rows.append(["2", str(base + 8500), "MouseClick", "", "500", "500", "", "40"])
    rows.append(["2", str(base + 9000), "Scroll", "", "", "", "-1.0", "40"])
    (dirpath / "v0.csv").write_text("\n".join("\t".join(r) for r in rows))


def test_dux_adapter_produces_valid_windows():
    with tempfile.TemporaryDirectory() as d:
        _write_dux(Path(d))
        w = load_dux(d)
    assert len(w) == 2
    labels = sorted(x["proxy_label"] for x in w)
    assert labels == [0, 1]                            # one neutral, one emotion
    for x in w:
        feats = extract_features(x["events"], 0)
        assert feats.shape == (30, len(FEATURE_NAMES))  # (30, 13)
    # the emotion session had a backspace -> backspace_pct > 0 somewhere
    emo = next(x for x in w if x["proxy_label"] == 1)
    assert extract_features(emo["events"], 0)[:, FEATURE_NAMES.index("backspace_pct")].max() > 0


def test_emosurv_adapter_keyboard_only():
    with tempfile.TemporaryDirectory() as d:
        rows = ["subject,press_time,key,emotion"]
        base = 500_000
        for i in range(8):
            rows.append(f"S1,{base + i * 1000},a,Neutral")
        rows.append(f"S1,{base + 8500},backspace,Neutral")
        Path(d, "emosurv.csv").write_text("\n".join(rows))
        w = load_emosurv(d)
    assert len(w) >= 1
    feats = extract_features(w[0]["events"], 0)
    ks = feats[:, FEATURE_NAMES.index("keystroke_count")].sum()
    mouse = feats[:, FEATURE_NAMES.index("mouse_velocity_mean")].sum()
    assert ks > 0 and mouse == 0                        # keyboard populated, mouse stays zero


def test_missing_paths_return_empty():
    assert load_external_windows({"external": {"dux_path": "nope", "emosurv_path": "nope"}}) == []


if __name__ == "__main__":
    test_dux_adapter_produces_valid_windows()
    test_emosurv_adapter_keyboard_only()
    test_missing_paths_return_empty()
    print("all external_datasets tests passed")
