"""
Note parsing — free text in, exact trained label strings out.

The MolmoAct2 fine-tune was trained on eight specific instruction strings. Any
deviation ("hit A", "note A", "Hitting A") is out-of-distribution, so this module
is the single chokepoint that guarantees only trained labels reach the policy.

Deliberately free of hardware imports: this runs on a laptop with no arm, which
is what makes tests/test_notes.py cheap.
"""
from __future__ import annotations

import os
import re

# ---------------------------------------------------------------------------
# The eight trained labels. These strings are load-bearing — they must match the
# training data byte for byte. Do not "clean up" the capitalization or the
# parenthetical colors.
#
# Two bars are red (a low C and an octave-up C) and two are blue (A and G). The
# red pair is a *parsing* ambiguity, because both are spelled "C" — see
# _disambiguate_c. The blue pair is not: A and G are distinct letters with
# distinct labels, so the parser can never confuse them. (If the model confuses
# them visually that is a data problem, not a parsing one.)
# ---------------------------------------------------------------------------

LABELS: dict[str, str] = {
    "A": "Hitting note A (blue)",
    "B": "Hitting note B (purple)",
    "C_LOW": "Hitting note C (big red)",
    "C_HIGH": "Hitting note C (small red)",
    "D": "Hitting note D (orange)",
    "E": "Hitting note E (yellow)",
    "F": "Hitting note F (green)",
    "G": "Hitting note G (blue)",
}

# Keys in scale order, low C first. Used by the ascending-context heuristic.
NOTE_KEYS = ["C_LOW", "D", "E", "F", "G", "A", "B", "C_HIGH"]

# The octave the low C sits in, for "C4" vs "C5" style input.
BASE_OCTAVE = 4

# Qualifier words that pin a bare C to one bar or the other.
_LOW_WORDS = {"big", "low", "lower", "long", "bottom", "left"}
_HIGH_WORDS = {"small", "high", "higher", "upper", "short", "top", "right"}

# Vocabulary the parser understands but does not act on: bar colors and the
# imperative framing people naturally type. Counted as *consumed* when scoring
# coverage, so "hit the big red C" reads as fully understood while "play twinkle
# twinkle little star" does not. Without this, any qualifier or verb would drag
# coverage down and send perfectly good input to the LLM.
_FILLER_WORDS = (
    _LOW_WORDS
    | _HIGH_WORDS
    | {"red", "blue", "purple", "orange", "yellow", "green"}
    | {"play", "hit", "hitting", "note", "notes", "then", "and", "the"}
)


class UnparseableSong(ValueError):
    """Raised when neither the regex nor the LLM fallback yields any notes."""


# ---------------------------------------------------------------------------
# Tokenizer
#
# Matches a note letter plus any modifiers attached to it. Modifiers may appear
# before the letter ("big C", "high C") or after ("C big", "C5", "C'"), so the
# caller scans for letters and then inspects a window of surrounding text rather
# than trying to express the whole grammar in one pattern.
# ---------------------------------------------------------------------------

# A note letter that is not part of a longer word: "C D E" matches, the "a" and
# "e" inside "please" do not. This word-boundary discipline is what makes the
# coverage ratio below a meaningful signal.
_LETTER = re.compile(r"(?<![A-Za-z])([A-Ga-g])(?![A-Za-z])([0-9]|['^])?")

_WORD = re.compile(r"[A-Za-z]+")


def _octave_key(letter: str, octave: str | None) -> str | None:
    """Resolve a C's octave marker to a label key. None if not a C or no marker."""
    if letter != "C" or not octave:
        return None
    if octave in ("'", "^"):
        return "C_HIGH"
    if octave.isdigit():
        return "C_HIGH" if int(octave) > BASE_OCTAVE else "C_LOW"
    return None


def _qualifier_near(text: str, start: int, end: int) -> str | None:
    """Look for a big/small qualifier in the words bracketing a note letter.

    Scans the two words before the letter and the two after. "big red C",
    "C (small red)" and "high C" all land. Returns "C_LOW", "C_HIGH", or None.
    """
    before = _WORD.findall(text[max(0, start - 24) : start])[-2:]
    after = _WORD.findall(text[end : end + 24])[:2]
    for word in (w.lower() for w in before + after):
        if word in _LOW_WORDS:
            return "C_LOW"
        if word in _HIGH_WORDS:
            return "C_HIGH"
    return None


def _disambiguate_c(keys: list[str | None], default_c: str) -> list[str]:
    """Resolve every unresolved bare C, in three tiers.

    keys entries are None exactly where a C could not be pinned by tier 1
    (explicit qualifier or octave marker), which ran during tokenization.
    """
    resolved: list[str] = []
    for i, key in enumerate(keys):
        if key is not None:
            resolved.append(key)
            continue

        # Tier 2 — ascending context. A bare C right after A or B is the
        # octave-completing high C. This is what makes the obvious demo input
        # "C D E F G A B C" play an actual scale rather than thumping the same
        # bar at both ends.
        prev = resolved[-1] if resolved else None
        if prev in ("A", "B"):
            resolved.append("C_HIGH")
            continue

        # Tier 3 — caller's default.
        resolved.append(default_c)
    return resolved


def _regex_parse(text: str, default_c: str) -> tuple[list[str], float]:
    """Tokenize with the regex. Returns (label keys, coverage ratio 0..1).

    Coverage is matched characters over non-whitespace characters — the signal
    for whether this was really a note list or prose that happens to contain
    stray note letters.
    """
    keys: list[str | None] = []
    matched_chars = 0

    for m in _LETTER.finditer(text):
        letter = m.group(1).upper()
        octave = m.group(2)
        matched_chars += len(m.group(0))

        if letter != "C":
            keys.append(letter)
            continue
        # Tier 1 — explicit octave marker or nearby qualifier word.
        keys.append(
            _octave_key(letter, octave) or _qualifier_near(text, m.start(), m.end())
        )

    # Credit understood vocabulary, so qualifiers and imperative framing don't
    # look like noise. Words already claimed as note letters are single chars and
    # never appear here (every filler word is 2+ chars except none).
    for word in _WORD.findall(text):
        if word.lower() in _FILLER_WORDS:
            matched_chars += len(word)

    non_ws = len(re.sub(r"\s", "", text)) or 1
    coverage = matched_chars / non_ws
    return _disambiguate_c(keys, default_c), coverage


# ---------------------------------------------------------------------------
# LLM fallback
#
# Fires only when the regex clearly did not understand the input — never
# speculatively, because it costs a round-trip and can fail mid-performance.
#
# The model returns *keys*, not label strings, constrained to an enum by
# structured outputs. That is the whole point: it is then physically incapable
# of emitting an instruction the policy was not trained on.
# ---------------------------------------------------------------------------

_FALLBACK_COVERAGE = 0.5

_LLM_SYSTEM = (
    "You convert a music request into a sequence of xylophone note keys.\n"
    f"Valid keys, use ONLY these: {list(LABELS)}\n"
    "C_LOW is the low C (the big red bar). C_HIGH is the octave-up C (the small "
    "red bar). The instrument is a one-octave diatonic xylophone: there are no "
    "sharps or flats, so transpose or drop any note that is not in the key list.\n"
    "If asked for a well-known tune, return its melody. Keep it under 40 notes."
)


def _llm_parse(text: str, model: str = "claude-opus-4-8") -> list[str]:
    """Ask Claude for a note sequence. Raises UnparseableSong if unavailable."""
    try:
        import anthropic
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise UnparseableSong(
            f"Could not parse {text!r} as notes, and the LLM fallback is "
            f"unavailable ({exc}). Install with: uv sync"
        ) from exc

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise UnparseableSong(
            f"Could not parse {text!r} as notes, and ANTHROPIC_API_KEY is unset "
            "so the LLM fallback cannot run. Pass an explicit note list instead, "
            'e.g. --song "C D E F G".'
        )

    # A Literal of the valid keys is what constrains the output. Built off
    # LABELS so the two can never drift apart.
    from typing import Literal

    class Song(BaseModel):
        notes: list[
            Literal["A", "B", "C_LOW", "C_HIGH", "D", "E", "F", "G"]
        ]

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=2000,
        system=_LLM_SYSTEM,
        output_config={"format": Song, "effort": "low"},
        messages=[{"role": "user", "content": text}],
    )
    keys = response.parsed_output.notes
    if not keys:
        raise UnparseableSong(f"The LLM returned no notes for {text!r}.")
    return list(keys)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_song(
    text: str,
    *,
    default_c: str = "C_LOW",
    allow_llm: bool = True,
) -> list[str]:
    """Turn free text into a list of exact trained label strings.

    "C D E"              -> the three labels, in order
    "C D E F G A B C"    -> the last C resolves to the small red bar
    "big C"              -> the big red bar
    "play twinkle"       -> routed to the LLM fallback

    Raises UnparseableSong if nothing usable comes out either path.
    """
    if default_c not in ("C_LOW", "C_HIGH"):
        raise ValueError(f"default_c must be C_LOW or C_HIGH, got {default_c!r}")

    keys, coverage = _regex_parse(text, default_c)

    needs_llm = not keys or coverage < _FALLBACK_COVERAGE
    if needs_llm:
        if not allow_llm:
            raise UnparseableSong(
                f"Could not parse {text!r} as a note list "
                f"(coverage {coverage:.0%}) and the LLM fallback is disabled."
            )
        keys = _llm_parse(text)

    return [LABELS[k] for k in keys]
