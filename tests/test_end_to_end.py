"""End-to-end: real newt.Robot -> real WebSocket -> fake policy -> fake rig.

Everything here is the production path except the two ends: the policy returns
canned chunks instead of running MolmoAct2, and the rig records instead of
moving servos. In between it is the genuine SDK run loop over a genuine
WebSocket speaking the genuine msgpack protocol.

This is what proves the central claim: one session, one run() call, and the
instruction changes underneath it as the song advances.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import codec
from notes import LABELS
from sequencer import SequenceComplete, XylophoneSequencer

websockets = pytest.importorskip("websockets")

SONG = [LABELS["C_LOW"], LABELS["D"], LABELS["E"]]


class RecordingRig:
    """Fake SO101 that records the prompts the server saw it under."""

    def __init__(self) -> None:
        self.chunks_observed = 0
        self.resets = 0
        self.executed: list[np.ndarray] = []

    def read_state(self) -> dict:
        return {
            "state": np.zeros(6, dtype=np.float32),
            "images": {
                cam: np.zeros((3, 378, 378), dtype=np.uint8)
                for cam in ("top", "side")
            },
        }

    def execute(self, chunk: np.ndarray) -> None:
        self.executed.append(np.asarray(chunk))
        self.chunks_observed += 1

    def reset_for_next_note(self) -> None:
        self.resets += 1


class FakePolicyServer:
    """Minimal stand-in for server/modal_ws.py: obs in, action chunk out.

    Records every prompt it receives, which is how we verify the retasking
    actually reached the wire rather than just the local obs dict.
    """

    def __init__(self) -> None:
        self.prompts_seen: list[str] = []
        self.port: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    async def _handler(self, sock) -> None:
        try:
            async for message in sock:
                frame = codec.unpack(message)
                if codec.get_str(frame, "type") != "obs":
                    continue
                self.prompts_seen.append(codec.get_str(frame, "prompt"))
                chunk = np.zeros((10, 6), dtype=np.float32)
                await sock.send(codec.pack({"type": "action", "chunk": chunk}))
        except Exception:
            pass

    def _serve(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def start():
            server = await websockets.serve(self._handler, "127.0.0.1", 0)
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await asyncio.Future()

        try:
            self._loop.run_until_complete(start())
        except Exception:
            self._ready.set()

    def __enter__(self) -> "FakePolicyServer":
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10) or self.port is None:
            raise RuntimeError("fake policy server failed to start")
        return self

    def __exit__(self, *exc) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)


@pytest.fixture
def _estop_clear():
    import embodiment

    embodiment._emergency_stop.clear()
    yield
    embodiment._emergency_stop.clear()


def test_one_session_plays_the_whole_song(monkeypatch, _estop_clear) -> None:
    """The central claim, proven over a real WebSocket.

    One run() call. Three notes. The server must see all three instructions,
    in order, without the client ever reconnecting.
    """
    import newt

    with FakePolicyServer() as server:
        # NT_INFERENCE_URL bypasses registry discovery entirely (robot.py:821),
        # which is exactly how the Modal deployment is reached in production.
        monkeypatch.setenv("NT_INFERENCE_URL", f"ws://127.0.0.1:{server.port}")
        monkeypatch.setenv("NT_API_KEY", "nt_dummy_for_test")

        rig = RecordingRig()
        seq = XylophoneSequencer(rig, SONG, seconds_per_note=0.3)
        robot = newt.Robot(embodiment=seq, connect_timeout=30)

        with pytest.raises(SequenceComplete):
            robot.run("", max_duration=60)

        # Every note reached the wire, in order, with no label the policy
        # was not trained on.
        assert server.prompts_seen, "the server received no observations"
        ordered = [p for i, p in enumerate(server.prompts_seen)
                   if i == 0 or p != server.prompts_seen[i - 1]]
        assert ordered == SONG
        for prompt in server.prompts_seen:
            assert prompt in LABELS.values()

        # The arm was actually driven, and each boundary re-armed the settle.
        assert rig.executed, "the rig was never given a chunk"
        assert rig.resets == 2


def test_the_empty_run_prompt_never_reaches_the_wire(monkeypatch, _estop_clear) -> None:
    """run("") is a placeholder; the sequencer must override it on every frame.

    If the injection ever silently stopped working, the server would receive
    empty prompts and the policy would get no instruction at all — a failure
    that would otherwise only show up as an arm sitting still.
    """
    import newt

    with FakePolicyServer() as server:
        monkeypatch.setenv("NT_INFERENCE_URL", f"ws://127.0.0.1:{server.port}")
        monkeypatch.setenv("NT_API_KEY", "nt_dummy_for_test")

        seq = XylophoneSequencer(RecordingRig(), SONG, seconds_per_note=0.3)
        robot = newt.Robot(embodiment=seq, connect_timeout=30)

        with pytest.raises(SequenceComplete):
            robot.run("", max_duration=60)

        assert "" not in server.prompts_seen
