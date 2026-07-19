#!/usr/bin/env python3
"""
Play a melody on the xylophone.

    uv run python3 run.py --song "C D E F G"
    uv run python3 run.py --song "play twinkle twinkle little star"   # LLM fallback
    uv run python3 run.py --dry-run --song "C D E"                    # parse only
    uv run python3 run.py --check                                     # no hardware

The policy is single-task — one instruction, one bar struck. This driver parses
your text into the exact trained labels, opens ONE inference session, and swaps
the instruction underneath it as the song advances. See sequencer.py for how the
prompt injection works.

Before the first run:
    export NT_INFERENCE_URL=wss://<workspace>--xylophone-policy-policy-web.modal.run
    export NT_API_KEY=dummy     # must be non-empty; the SDK checks it before the URL

Press Ctrl+H at any time to abort and home the arm.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import notes
from sequencer import SequenceComplete, XylophoneSequencer

# ---------------------------------------------------------------------------
# Run-policy knobs. These are properties of how we play, not of the rig, so they
# live here rather than in embodiment.py.
# ---------------------------------------------------------------------------

# Ignored when NT_INFERENCE_URL is set (which is our normal path — the fine-tune
# is self-hosted on Modal, not in New Theory's registry).
MODEL = "so101"

# How long each note holds before the sequencer moves on.
#
# Budget per bar: FIRST_CHUNK_SETTLE_S (1.5s) moving to the strike start, then
# MAX_ACTIONS_PER_CHUNK (15) actions at ACTION_INTERVAL_S (1/15s) = ~1.0s of
# strike, plus a server round-trip. That's ~2.5s before any closed-loop
# correction, so 4.0 leaves room for roughly one corrective cycle.
#
# Tune by ear — this is also the tempo control. Too low and the prompt flips
# mid-swing; too high and the arm idles between notes.
SECONDS_PER_NOTE: float = 4.0

# Extra wall-clock headroom on top of the song's nominal length. This is only a
# backstop — SequenceComplete is what actually ends the run.
DURATION_SLACK_S: float = 30.0

# Cold-start budget. A scaled-to-zero Modal container needs to boot a GPU and
# load the checkpoint; 30-90s is typical.
CONNECT_TIMEOUT_S: float = 180.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--song",
        default=None,
        help='notes to play, e.g. "C D E F G" or "play twinkle twinkle"',
    )
    parser.add_argument(
        "--seconds-per-note",
        type=float,
        default=SECONDS_PER_NOTE,
        help=f"how long each note holds (default {SECONDS_PER_NOTE})",
    )
    parser.add_argument(
        "--default-c",
        choices=["low", "high"],
        default="low",
        help="which C bar a bare, unqualified 'C' means (default low/big red)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="fail instead of falling back to Claude on unparseable text",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse the song and print the labels; touch no hardware",
    )
    parser.add_argument(
        "--check", action="store_true", help="no-hardware config + parser check"
    )
    parser.add_argument("--arm", default=None, help="arm id from nt.toml")
    parser.add_argument("--site-config", default=None, help="path to nt.toml")
    parser.add_argument(
        "--list-notes", action="store_true", help="print the trained labels and exit"
    )
    return parser.parse_args()


def _resolve_song(args: argparse.Namespace) -> list[str]:
    """Parse --song into trained labels, or exit with a usable message."""
    if not args.song:
        print(
            'error: --song is required, e.g. --song "C D E F G"\n'
            "Run --list-notes to see the eight bars this policy knows.",
            file=sys.stderr,
        )
        sys.exit(2)

    default_c = "C_LOW" if args.default_c == "low" else "C_HIGH"
    try:
        labels = notes.parse_song(
            args.song, default_c=default_c, allow_llm=not args.no_llm
        )
    except notes.UnparseableSong as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"[song] {len(labels)} notes parsed from {args.song!r}:")
    for i, label in enumerate(labels, 1):
        print(f"  {i:2d}. {label}")
    return labels


def _run_check(args: argparse.Namespace) -> None:
    """Verify config and parsing without importing lerobot or touching hardware."""
    from embodiment import _CAMERA_KEYS, _load_arm_port, _load_cameras, _load_site_config

    try:
        raw = _load_site_config(args.site_config)
    except Exception as exc:
        print(f"check failed at stage: config\n  error: {exc}", file=sys.stderr)
        sys.exit(1)
    print("[check] config: loaded")

    try:
        arm_id, port = _load_arm_port(raw, args.arm)
    except Exception as exc:
        print(f"check failed at stage: arm-selection\n  error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"[check] arm: {arm_id} (port {port})")

    cameras = _load_cameras(raw)
    missing = [c for c in _CAMERA_KEYS if c not in cameras]
    if missing:
        print(
            f"check failed at stage: camera-selection\n"
            f"  error: nt.toml missing camera(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[check] cameras: {', '.join(_CAMERA_KEYS)}")

    scale = notes.parse_song("C D E F G A B C", allow_llm=False)
    assert all(label in notes.LABELS.values() for label in scale)
    print(f"[check] parser: 8-note scale OK (last C -> {scale[-1]})")

    url = os.environ.get("NT_INFERENCE_URL")
    print(f"[check] NT_INFERENCE_URL: {url or 'UNSET — will use the newt registry'}")
    if not os.environ.get("NT_API_KEY"):
        print("[check] warning: NT_API_KEY unset; the SDK raises AuthError without it")
    print("[check] ok")


def main() -> int:
    args = _parse_args()

    if args.list_notes:
        for key, label in notes.LABELS.items():
            print(f"  {key:7s} {label}")
        return 0

    if args.check:
        _run_check(args)
        return 0

    labels = _resolve_song(args)

    if args.dry_run:
        print("[song] dry run — no hardware touched")
        return 0

    # Hardware imports deferred until we know we're actually playing, so
    # --dry-run and --check work on a laptop with no arm.
    import newt
    from embodiment import SO101, _EmergencyStop, _start_keyboard_listener

    if not os.environ.get("NT_INFERENCE_URL"):
        print(
            "[rig] warning: NT_INFERENCE_URL is unset, so the SDK will resolve "
            f"model={MODEL!r} against New Theory's registry — which does not host "
            "your fine-tune. Deploy server/modal_ws.py and export the URL.",
            file=sys.stderr,
        )

    _start_keyboard_listener()

    rig = SO101.from_config(site_config_path=args.site_config, arm_id=args.arm)
    seq = XylophoneSequencer(rig, labels, seconds_per_note=args.seconds_per_note)

    # embodiment= alone is enough: _validate_embodiment (robot.py:645) pulls
    # read_state/execute off the object. Passing all three is redundant.
    robot = newt.Robot(
        embodiment=seq, model=MODEL, connect_timeout=CONNECT_TIMEOUT_S
    )

    exit_code = 0
    try:
        # One run() for the whole song. prompt="" because the sequencer stamps
        # the real instruction onto every obs frame; max_duration is a backstop.
        robot.run("", max_duration=seq.estimated_duration() + DURATION_SLACK_S)
        print("[seq] session ended before the song finished (server terminal frame)")
    except SequenceComplete:
        print(f"[seq] song complete — {len(labels)} notes played")
    except _EmergencyStop:
        rig.emergency_home()
        return 130
    except KeyboardInterrupt:
        print("\n[rig] interrupted")
        exit_code = 130
    finally:
        rig.teardown()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
