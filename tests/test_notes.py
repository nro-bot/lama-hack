"""Parser tests. No hardware, no network — allow_llm=False throughout."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import notes
from notes import LABELS, UnparseableSong, parse_song


def _parse(text: str, **kw) -> list[str]:
    return parse_song(text, allow_llm=False, **kw)


# ---------------------------------------------------------------------------
# The core invariant: nothing but a trained label ever reaches the policy.
# Everything else in this file is a refinement of this one assertion.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "C D E",
        "C D E F G A B C",
        "hit A, B, C D F",
        "play the big red C then D",
        "G F E D C",
        "a b c d e f g",
        "C5 D E",
    ],
)
def test_output_is_always_a_trained_label(text: str) -> None:
    labels = _parse(text)
    assert labels, f"{text!r} produced no notes"
    for label in labels:
        assert label in LABELS.values(), f"{label!r} is not a trained label"


def test_simple_sequence_order_is_preserved() -> None:
    assert _parse("C D E") == [
        LABELS["C_LOW"],
        LABELS["D"],
        LABELS["E"],
    ]


def test_ascending_scale_resolves_final_c_to_the_high_bar() -> None:
    """The obvious demo input must play an actual scale, not thump one bar twice."""
    labels = _parse("C D E F G A B C")
    assert len(labels) == 8
    assert labels[0] == LABELS["C_LOW"]
    assert labels[-1] == LABELS["C_HIGH"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("big C", "C_LOW"),
        ("small C", "C_HIGH"),
        ("the big red C", "C_LOW"),
        ("small red C", "C_HIGH"),
        ("high C", "C_HIGH"),
        ("low C", "C_LOW"),
    ],
)
def test_explicit_qualifiers_win(text: str, expected: str) -> None:
    assert _parse(text) == [LABELS[expected]]


@pytest.mark.parametrize(
    "text,expected",
    [("C4", "C_LOW"), ("C5", "C_HIGH"), ("C'", "C_HIGH"), ("C^", "C_HIGH")],
)
def test_octave_markers(text: str, expected: str) -> None:
    assert _parse(text) == [LABELS[expected]]


def test_bare_c_falls_back_to_the_default() -> None:
    assert _parse("C") == [LABELS["C_LOW"]]
    assert _parse("C", default_c="C_HIGH") == [LABELS["C_HIGH"]]


def test_default_c_does_not_override_an_explicit_qualifier() -> None:
    assert _parse("big C", default_c="C_HIGH") == [LABELS["C_LOW"]]


def test_note_letters_inside_words_are_not_notes() -> None:
    """Word boundaries are what make the coverage heuristic meaningful."""
    with pytest.raises(UnparseableSong):
        _parse("please deface a cabbage")


def test_prose_routes_to_the_fallback() -> None:
    with pytest.raises(UnparseableSong):
        _parse("play twinkle twinkle little star")


def test_empty_input_routes_to_the_fallback() -> None:
    with pytest.raises(UnparseableSong):
        _parse("")


def test_rejects_an_invalid_default_c() -> None:
    with pytest.raises(ValueError):
        parse_song("C", default_c="C_MIDDLE", allow_llm=False)


def test_note_keys_covers_every_label() -> None:
    """NOTE_KEYS and LABELS must not drift apart."""
    assert set(notes.NOTE_KEYS) == set(LABELS)


# ---------------------------------------------------------------------------
# LLM fallback — mocked. Verifies the trigger condition and that keys, not raw
# strings, are what the model is allowed to return.
# ---------------------------------------------------------------------------


def test_llm_fallback_is_invoked_for_prose(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    def fake_llm(text: str, model: str = "") -> list[str]:
        called.append(text)
        return ["C_LOW", "D", "E"]

    monkeypatch.setattr(notes, "_llm_parse", fake_llm)
    labels = parse_song("play twinkle twinkle little star")

    assert called == ["play twinkle twinkle little star"]
    assert labels == [LABELS["C_LOW"], LABELS["D"], LABELS["E"]]


def test_llm_fallback_is_not_invoked_for_a_clean_note_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback costs a round-trip; it must never fire speculatively."""

    def boom(text: str, model: str = "") -> list[str]:
        raise AssertionError(f"LLM fallback should not have fired for {text!r}")

    monkeypatch.setattr(notes, "_llm_parse", boom)
    assert parse_song("C D E F G") == [
        LABELS["C_LOW"],
        LABELS["D"],
        LABELS["E"],
        LABELS["F"],
        LABELS["G"],
    ]
