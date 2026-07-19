"""Sequencer tests against a fake rig — no hardware, no network, fake clock."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embodiment
from notes import LABELS
from sequencer import SequenceComplete, XylophoneSequencer

SONG = [LABELS["C_LOW"], LABELS["D"], LABELS["E"]]


class FakeRig:
    """Stands in for SO101: records what it was asked to do."""

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
        self.executed.append(chunk)
        self.chunks_observed += 1

    def reset_for_next_note(self) -> None:
        self.resets += 1


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture(autouse=True)
def _clear_estop():
    """The e-stop flag is a module global; keep tests from leaking into each other."""
    embodiment._emergency_stop.clear()
    yield
    embodiment._emergency_stop.clear()


def _play(seq: XylophoneSequencer, rig: FakeRig, clock: FakeClock, steps: int):
    """Drive the read_state/execute cycle the way newt's run loop would."""
    seen = []
    for _ in range(steps):
        obs = seq.read_state()
        seen.append(obs["prompt"])
        seq.execute(np.zeros((5, 6), dtype=np.float32))
        clock.advance(1.0)
    return seen


# ---------------------------------------------------------------------------


def test_prompt_is_stamped_onto_the_obs_dict() -> None:
    """This is the seam the whole design rests on."""
    rig, clock = FakeRig(), FakeClock()
    seq = XylophoneSequencer(rig, SONG, seconds_per_note=3.0, clock=clock)

    obs = seq.read_state()
    assert obs["prompt"] == LABELS["C_LOW"]
    # And the rig's own observation survives alongside it.
    assert obs["state"].shape == (6,)
    assert set(obs["images"]) == {"top", "side"}


def test_notes_advance_in_order_on_the_wall_clock() -> None:
    rig, clock = FakeRig(), FakeClock()
    seq = XylophoneSequencer(rig, SONG, seconds_per_note=3.0, clock=clock)

    seen = _play(seq, rig, clock, steps=9)

    # 3s per note, 1s per cycle => three cycles per note, in order.
    assert seen[0:3] == [LABELS["C_LOW"]] * 3
    assert seen[3:6] == [LABELS["D"]] * 3
    assert seen[6:9] == [LABELS["E"]] * 3


def test_sequence_complete_is_raised_after_the_last_note() -> None:
    rig, clock = FakeRig(), FakeClock()
    seq = XylophoneSequencer(rig, SONG, seconds_per_note=3.0, clock=clock)

    _play(seq, rig, clock, steps=9)
    with pytest.raises(SequenceComplete):
        seq.read_state()


def test_settle_state_is_re_armed_at_every_note_boundary() -> None:
    """Without this the second bar gets no move-to-start settle."""
    rig, clock = FakeRig(), FakeClock()
    seq = XylophoneSequencer(rig, SONG, seconds_per_note=3.0, clock=clock)

    _play(seq, rig, clock, steps=9)
    assert rig.resets == 2  # two boundaries in a three-note song


def test_reset_is_wired_to_the_real_embodiment_method_name() -> None:
    """_advance() looks the reset up by name; guard against a rename drifting."""
    assert hasattr(embodiment.SO101, "reset_for_next_note")


def test_a_note_is_never_skipped_before_the_policy_has_acted() -> None:
    """A cold-start inference slower than seconds_per_note must not burn notes."""
    rig, clock = FakeRig(), FakeClock()
    seq = XylophoneSequencer(rig, SONG, seconds_per_note=1.0, clock=clock)

    # Ten read_state calls with no execute in between: the policy has returned
    # nothing, so the arm has struck nothing, so we must still be on note 1.
    for _ in range(10):
        seq.read_state()
        clock.advance(5.0)

    assert seq.current_label == LABELS["C_LOW"]
    assert seq.notes_remaining == 3


def test_require_chunk_can_be_disabled() -> None:
    rig, clock = FakeRig(), FakeClock()
    seq = XylophoneSequencer(
        rig, SONG, seconds_per_note=1.0, clock=clock, require_chunk=False
    )

    seq.read_state()
    clock.advance(5.0)
    assert seq.read_state()["prompt"] == LABELS["D"]


def test_emergency_stop_is_detected_in_read_state() -> None:
    """The SDK only checks this inside execute(); read_state closes the gap."""
    rig, clock = FakeRig(), FakeClock()
    seq = XylophoneSequencer(rig, SONG, seconds_per_note=3.0, clock=clock)

    embodiment._emergency_stop.set()
    with pytest.raises(embodiment._EmergencyStop):
        seq.read_state()


def test_execute_is_delegated_to_the_rig() -> None:
    rig, clock = FakeRig(), FakeClock()
    seq = XylophoneSequencer(rig, SONG, seconds_per_note=3.0, clock=clock)

    chunk = np.ones((5, 6), dtype=np.float32)
    seq.execute(chunk)
    assert len(rig.executed) == 1
    assert np.array_equal(rig.executed[0], chunk)


def test_estimated_duration_covers_the_song() -> None:
    seq = XylophoneSequencer(FakeRig(), SONG, seconds_per_note=2.5)
    assert seq.estimated_duration() == pytest.approx(7.5)


@pytest.mark.parametrize(
    "labels,spn", [([], 3.0), (SONG, 0.0), (SONG, -1.0)]
)
def test_rejects_nonsense_construction(labels, spn) -> None:
    with pytest.raises(ValueError):
        XylophoneSequencer(FakeRig(), labels, seconds_per_note=spn)
