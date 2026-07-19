"""Smoke test against the REAL deployed Modal policy server. No robot involved.

Run this before you ever touch the arm:

    export NT_INFERENCE_URL=wss://<workspace>--xylophone-policy-policy-web.modal.run
    uv run --with pytest --with websockets python -m pytest tests/test_fake_client.py -v -s

It sends one synthetic observation and checks what comes back. That single
round-trip validates, in one shot, the three things most likely to be wrong:

  1. the codec  — does our msgpack copy round-trip against the server's?
  2. the loader — did MolmoAct2Policy.from_pretrained work, or did it silently
                  fall through to the transformers path?
  3. the wiring — is the chunk the shape and dtype the arm expects?

Skipped automatically when NT_INFERENCE_URL is unset.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import codec
from embodiment import _IMAGE_SIZE
from notes import LABELS

websockets = pytest.importorskip("websockets")

URL = os.environ.get("NT_INFERENCE_URL")

pytestmark = pytest.mark.skipif(
    not URL, reason="NT_INFERENCE_URL unset — deploy server/modal_ws.py first"
)

# Generous: a scaled-to-zero container has to boot a GPU and load the checkpoint.
COLD_START_TIMEOUT_S = 240


def _mock_obs(prompt: str) -> dict:
    """An obs frame shaped exactly like embodiment.read_state() produces.

    Zeros, not real camera data — this test asks "does the pipe work", not "does
    the policy do the right thing". The latter needs the physical rig.

    Image size comes from the embodiment rather than a literal, so this test
    exercises the size the rig will actually send. The checkpoint's config.json
    declares [3, 224, 224]; if those ever disagree the server logs a warning at
    load and the model quietly sees the wrong thing.
    """
    return {
        "type": "obs",
        "prompt": prompt,
        "state": np.zeros(6, dtype=np.float32),
        "images": {
            cam: np.zeros((3, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.uint8)
            for cam in ("top", "side")
        },
        "max_duration": 60.0,
    }


async def _one_round_trip(prompt: str) -> dict:
    async with websockets.connect(URL, open_timeout=COLD_START_TIMEOUT_S) as sock:
        await sock.send(codec.pack(_mock_obs(prompt)))
        reply = await asyncio.wait_for(sock.recv(), timeout=COLD_START_TIMEOUT_S)
        return codec.unpack(reply)


def test_server_returns_a_usable_action_chunk() -> None:
    frame = asyncio.run(_one_round_trip(LABELS["C_LOW"]))

    kind = codec.get_str(frame, "type")
    if kind == "terminal":
        pytest.fail(
            "server sent a terminal frame instead of an action: "
            f"{codec.get_str(frame, 'stop_reason')}"
        )
    assert kind == "action", f"unexpected frame type {kind!r}"

    chunk = codec.get_key(frame, "chunk")
    assert isinstance(chunk, np.ndarray), f"chunk is {type(chunk)}, not ndarray"
    assert chunk.ndim == 2, f"expected (N, 6), got {chunk.shape}"
    assert chunk.shape[1] == 6, f"expected 6 joints, got {chunk.shape[1]}"
    assert np.isfinite(chunk).all(), "chunk contains NaN or inf — the arm would fault"

    print(f"\n[smoke] chunk {chunk.shape} {chunk.dtype}")
    print(f"[smoke] joint ranges: {chunk.min(axis=0)} .. {chunk.max(axis=0)}")


def test_every_trained_label_is_accepted() -> None:
    """All eight instructions should produce a chunk, not an error frame."""
    for key, label in LABELS.items():
        frame = asyncio.run(_one_round_trip(label))
        assert codec.get_str(frame, "type") == "action", (
            f"{key} ({label!r}) did not produce an action frame: "
            f"{codec.get_str(frame, 'stop_reason')}"
        )


def test_different_notes_produce_different_actions() -> None:
    """A policy ignoring the instruction is the failure this catches.

    If the prompt weren't reaching the model — the seam broken, the label
    mangled — every note would return the same motion and the whole framework
    would be a very elaborate way to hit one bar repeatedly.

    Zeroed camera input makes this weaker than it looks (the model has no visual
    evidence of where the bars are), so treat a failure here as a signal to
    investigate on the real rig, not as proof the policy is broken.
    """
    low_c = asyncio.run(_one_round_trip(LABELS["C_LOW"]))
    note_g = asyncio.run(_one_round_trip(LABELS["G"]))

    a = codec.get_key(low_c, "chunk")
    b = codec.get_key(note_g, "chunk")
    if np.allclose(a, b):
        pytest.fail(
            "C and G produced identical action chunks — the instruction may not "
            "be reaching the model. Check the prompt key in server/modal_ws.py."
        )
