"""Codec round-trip against the REAL newt SDK.

server/codec.py is a copy of the SDK's private packer. A copy that drifts is
worse than no copy: it packs and unpacks perfectly against itself and then fails
only on the wire, at the one moment there's a robot arm involved. These tests
pin our copy to the real thing in both directions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

from newt._client import robot as newt_robot  # the real SDK

import codec


def _obs_frame() -> dict:
    """The frame shape embodiment.read_state() actually produces."""
    return {
        "type": "obs",
        "prompt": "Hitting note C (big red)",
        "state": np.arange(6, dtype=np.float32),
        "images": {
            cam: np.full((3, 378, 378), i, dtype=np.uint8)
            for i, cam in enumerate(("top", "side"))
        },
        "max_duration": 42.0,
    }


def test_our_unpack_reads_what_the_sdk_packs() -> None:
    """Client -> server direction: the real path an obs frame takes."""
    wire = newt_robot._pack(_obs_frame())
    decoded = codec.unpack(wire)

    assert codec.get_str(decoded, "type") == "obs"
    assert codec.get_str(decoded, "prompt") == "Hitting note C (big red)"
    np.testing.assert_array_equal(
        codec.get_key(decoded, "state"), np.arange(6, dtype=np.float32)
    )

    images = codec.get_key(decoded, "images")
    top = codec.get_key(images, "top")
    assert top.shape == (3, 378, 378)
    assert top.dtype == np.uint8
    assert top[0, 0, 0] == 0
    assert codec.get_key(images, "side")[0, 0, 0] == 1


def test_the_sdk_reads_what_we_pack() -> None:
    """Server -> client direction: the action chunk going back."""
    chunk = np.random.randn(10, 6).astype(np.float32)
    decoded = newt_robot._unpack(codec.pack({"type": "action", "chunk": chunk}))

    assert newt_robot._str_field(decoded, "type") == "action"
    np.testing.assert_array_equal(decoded["chunk"], chunk)
    assert decoded["chunk"].dtype == np.float32


def test_the_sdk_reads_our_terminal_frame() -> None:
    decoded = newt_robot._unpack(
        codec.pack({"type": "terminal", "stop_reason": "max_duration"})
    )
    assert newt_robot._str_field(decoded, "type") == "terminal"
    assert newt_robot._str_field(decoded, "stop_reason") == "max_duration"


def test_ndarray_envelope_keys_are_bytes_not_str() -> None:
    """The subtlety that breaks a from-memory reimplementation.

    _pack_array emits Python bytes keys; msgpack encodes them as `bin` and they
    come back as bytes even under raw=False. A copy that used str keys would
    pass every self-consistency test and still fail on the wire.
    """
    import msgpack

    raw = msgpack.unpackb(codec.pack(np.zeros(3, dtype=np.float32)), raw=False)
    assert b"__ndarray__" in raw
    assert "__ndarray__" not in raw


def test_our_packer_is_byte_identical_to_the_sdks() -> None:
    frame = _obs_frame()
    assert codec.pack(frame) == newt_robot._pack(frame)


def test_get_key_tolerates_both_key_flavors() -> None:
    assert codec.get_key({"a": 1}, "a") == 1
    assert codec.get_key({b"a": 1}, "a") == 1
    assert codec.get_key({}, "a", default=9) == 9
    assert codec.get_str({b"t": b"obs"}, "t") == "obs"
    assert codec.get_str({}, "missing", default="fallback") == "fallback"


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint8, np.int64])
def test_dtypes_survive_the_round_trip(dtype) -> None:
    arr = np.arange(12, dtype=dtype).reshape(3, 4)
    out = newt_robot._unpack(codec.pack(arr))
    np.testing.assert_array_equal(out, arr)
    assert out.dtype == arr.dtype


# ---------------------------------------------------------------------------
# The seam itself, tested against the real SDK.
#
# The entire design rests on one behavior of _build_obs_frame: an obs dict
# carrying its own "prompt" key overrides the argument passed to run(). If a
# future SDK version latches the prompt at connection setup instead, these tests
# fail here — at `pytest`, not at the xylophone.
# ---------------------------------------------------------------------------


def test_an_injected_prompt_overrides_the_run_argument() -> None:
    obs = {"state": np.zeros(6, dtype=np.float32), "prompt": "Hitting note E (yellow)"}
    frame = newt_robot._build_obs_frame(obs, "SHOULD NOT WIN", None)
    assert frame["prompt"] == "Hitting note E (yellow)"


def test_the_run_argument_still_wins_when_the_obs_has_no_prompt() -> None:
    frame = newt_robot._build_obs_frame({"state": np.zeros(6)}, "fallback", None)
    assert frame["prompt"] == "fallback"


def test_the_prompt_is_rebuilt_on_every_frame_not_just_the_first() -> None:
    """Confirms retasking mid-session is possible at all.

    The `first` gate inside _run_blocking_once guards max_duration and model.
    If it ever grows to guard prompt too, one session per note becomes the only
    option and sequencer.py needs rewriting.
    """
    import inspect

    source = inspect.getsource(newt_robot._build_obs_frame)
    assert "if not frame.get(\"prompt\")" in source, (
        "_build_obs_frame no longer honors an obs-supplied prompt; "
        "sequencer.py's injection seam is broken"
    )

    # And the call site must still pass prompt unconditionally.
    loop = inspect.getsource(newt_robot.Robot._run_blocking_once)
    assert "max_duration if first else None" in loop, (
        "_run_blocking_once's first-frame gating changed; verify prompt is "
        "still sent on every frame"
    )
