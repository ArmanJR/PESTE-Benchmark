"""VibeVoice structured segment scoring policy."""

from peste.adapters.vibevoice import flatten_segments


def test_flatten_segments_preserves_reported_order() -> None:
    payload = {
        "segments": [
            {"speaker_id": 0, "start_time": 0.0, "text": " سلام "},
            {"speaker_id": 1, "start_time": 1.0, "text": "دنیا"},
        ]
    }
    assert flatten_segments(payload) == "سلام دنیا"


def test_flatten_segments_handles_nested_and_empty_payloads() -> None:
    assert (
        flatten_segments({"result": {"transcription": [{"text": "الف"}, {"text": "ب"}]}}) == "الف ب"
    )
    assert flatten_segments(None) == ""
