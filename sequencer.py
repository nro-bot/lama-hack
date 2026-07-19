"""
XylophoneSequencer — plays a list of notes through one policy session.

The policy is single-task: it takes one instruction and strikes one bar. To play
a melody we need to retask it repeatedly without paying a WebSocket reconnect
between every note, because that gap is audible.

The seam is in the newt SDK, `_build_obs_frame` (newt/_client/robot.py:1275):

    frame = {k: v for k, v in obs.items()}
    frame["type"] = "obs"
    if not frame.get("prompt"):
        frame["prompt"] = prompt

Two facts follow. The prompt is rebuilt from the obs dict on *every* frame (the
`first` flag in _run_blocking_once guards only max_duration and model), and an
obs dict carrying its own "prompt" key wins over the argument passed to run().

So this class wraps the embodiment, calls robot.run(prompt="") exactly once for
the whole song, and injects the current note's label from read_state(). One
WebSocket, one call, prompt mutating underneath. No SDK subclassing and no
reimplementation of the run loop — this survives SDK updates as long as those
four lines keep their shape.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from embodiment import _check_emergency_stop


class SequenceComplete(Exception):
    """Raised from read_state() once the final note's window closes.

    This is the real terminator for a song. It propagates out of robot.run()
    through the same path _EmergencyStop already uses, and _run_blocking_once's
    `finally` closes the WebSocket cleanly on the way out. run()'s max_duration
    is only a backstop against this never firing.
    """


class XylophoneSequencer:
    """Wraps an SO101; injects a per-note prompt and advances on a wall clock.

    Implements newt.Embodiment (read_state / execute) by delegation, so it can
    be handed to newt.Robot(embodiment=...) in place of the bare rig.

    Termination is by fixed duration per note. There is no success signal on
    this rig — no microphone, no force sensor — so "the strike is done" is a
    wall-clock judgement, which has the side benefit of giving tempo control.
    """

    def __init__(
        self,
        inner: Any,
        labels: list[str],
        seconds_per_note: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        require_chunk: bool = True,
    ) -> None:
        """
        inner:            the SO101 rig (or any object with read_state/execute)
        labels:           exact trained label strings, in play order
        seconds_per_note: how long each note holds before advancing
        clock:            injectable for tests
        require_chunk:    if True, never advance before the policy has returned
                          at least one chunk for the current note. Guards against
                          a slow first inference eating a whole note's window and
                          the arm never being told to strike.
        """
        if not labels:
            raise ValueError("XylophoneSequencer needs at least one note to play.")
        if seconds_per_note <= 0:
            raise ValueError(f"seconds_per_note must be > 0, got {seconds_per_note}")

        self._inner = inner
        self._labels = list(labels)
        self._spn = seconds_per_note
        self._clock = clock
        self._require_chunk = require_chunk

        self._idx = 0
        self._note_t0: float | None = None
        self._chunks_at_note_start = 0
        self._announced = False

    # -----------------------------------------------------------------------
    # Embodiment protocol
    # -----------------------------------------------------------------------

    def read_state(self) -> dict:
        """Delegate to the rig, then stamp the current note's label onto the obs."""
        # The SDK calls _check_emergency_stop() inside execute() but never here,
        # so a Ctrl+H pressed while blocked on a server round-trip would go
        # unnoticed until the next chunk arrived. Checking here closes that gap.
        _check_emergency_stop()

        now = self._clock()
        if self._note_t0 is None:
            self._note_t0 = now
            self._announce()
        elif now - self._note_t0 >= self._spn and self._may_advance():
            self._advance()  # raises SequenceComplete after the last note
            self._note_t0 = now
            self._announce()

        obs = self._inner.read_state()
        obs["prompt"] = self._labels[self._idx]  # ← the seam
        return obs

    def execute(self, chunk: np.ndarray) -> None:
        self._inner.execute(chunk)

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _may_advance(self) -> bool:
        """Has this note actually been played yet?

        Without this gate, a cold-start inference slower than seconds_per_note
        would silently burn through notes while the arm sat still. It also keeps
        the prompt from flipping mid-chunk, which reads as a jerk in the motion.
        """
        if not self._require_chunk:
            return True
        observed = getattr(self._inner, "chunks_observed", 0)
        return observed > self._chunks_at_note_start

    def _advance(self) -> None:
        """Step to the next note. Raises SequenceComplete past the last one.

        Order matters: the end-of-song check comes before the settle reset, so
        finishing a song doesn't re-arm a move-to-start for a note that will
        never be played.
        """
        self._idx += 1
        if self._idx >= len(self._labels):
            raise SequenceComplete()

        self._chunks_at_note_start = getattr(self._inner, "chunks_observed", 0)
        # Re-arm the move-to-start settle: the next bar may be far from where
        # this strike ended, and nothing in the SDK resets this for us.
        reset = getattr(self._inner, "reset_for_next_note", None)
        if reset is not None:
            reset()

    def _announce(self) -> None:
        """Log the note boundary so failures are visible on video replay."""
        print(
            f"[seq] note {self._idx + 1}/{len(self._labels)}: "
            f"{self._labels[self._idx]}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    @property
    def current_label(self) -> str:
        return self._labels[self._idx]

    @property
    def notes_remaining(self) -> int:
        return len(self._labels) - self._idx

    def estimated_duration(self) -> float:
        """Wall-clock estimate for the whole song, for run()'s max_duration."""
        return len(self._labels) * self._spn
