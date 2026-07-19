"""
SO101 — hardware embodiment class for the xylophone sequencer.

VENDORED from ../../newt-starter-so101/embodiment.py. Copied rather than
imported because that repo has no package structure and keeps module-global
e-stop state; a sys.path import breaks the moment either repo moves.

Local changes from the original, all for xylophone sequencing:
  - MAX_ACTIONS_PER_CHUNK 15 -> 5  (the fine-tune emits 10-step chunks, so 15
    never truncated; 5 gives real closed-loop correction mid-strike)
  - reset_for_next_note()         (the settle-state reset the sequencer needs)

Implements newt.Embodiment (read_state / execute) so any newt.Robot can drive
this rig. Instantiate via SO101.from_config() — it reads the same
~/.config/nt/nt.toml that the starter has always used.

This file is yours. Rename the class, edit the wiring, add teardown logic —
anything with read_state() / execute() works as an embodiment.

The SO-101 is a far simpler robot than the Trossen WidowX: pure joint-space
position control over USB webcams. There is no cartesian IK, no depth cameras,
no quaternion math, no driver bypass. We drive *through* lerobot's SO101Follower
(send_action / get_observation), not around it. The whole rig is six joint
scalars in, six joint scalars out, plus two RGB frames.
"""
from __future__ import annotations

import sys

import atexit
import select
import termios
import threading
import time
import tomllib
import tty
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Hardware deps (lerobot SO101Follower + feetech servo SDK) — lazy import for a
# clean failure message. The module stays importable without lerobot;
# _import_hardware_deps() populates the globals on first hardware use. Lazy (not
# a try/except at module top) so the --check path never triggers a driver
# import. SO101.__init__ raises if the deps are missing.
#
# Pin: lerobot-nt @ 50168c2a (the same SHA the Trossen starter vendors). At this
# pin SO101Follower lives at lerobot.robots.so101_follower and is byte-identical
# to upstream lerobot v0.3.3. Do NOT use upstream lerobot >= v0.4.4 — the
# so101_follower module was refactored into a unified so_follower package and
# this import path is gone.
# ---------------------------------------------------------------------------

_LEROBOT_AVAILABLE: bool | None = None  # None = import not yet attempted
_LEROBOT_IMPORT_ERR: Exception | None = None
SO101Follower = None
SO101FollowerConfig = None
OpenCVCameraConfig = None


def _import_hardware_deps() -> None:
    """Import lerobot's SO101Follower + OpenCV camera config into module globals.

    Idempotent. Called at the top of SO101.__init__ — i.e. only on the code path
    that actually touches hardware. The --check path never calls this. Catches
    Exception (not just ImportError) defensively: lerobot's optional camera
    backends can raise non-ImportError errors when their native deps are absent.
    """
    global _LEROBOT_AVAILABLE, _LEROBOT_IMPORT_ERR
    global SO101Follower, SO101FollowerConfig, OpenCVCameraConfig

    if _LEROBOT_AVAILABLE is not None:
        return
    try:
        from lerobot.robots.so101_follower import (
            SO101Follower as _SO101Follower,
            SO101FollowerConfig as _SO101FollowerConfig,
        )
        from lerobot.cameras.opencv import OpenCVCameraConfig as _OpenCVCameraConfig
        SO101Follower = _SO101Follower
        SO101FollowerConfig = _SO101FollowerConfig
        OpenCVCameraConfig = _OpenCVCameraConfig
        _LEROBOT_AVAILABLE = True
    except Exception as _err:
        _LEROBOT_AVAILABLE = False
        _LEROBOT_IMPORT_ERR = _err


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Camera keys, in the order the live so101 serve contract names them.
# Source: brief-251 closeout + brief-258c — cameras ["top", "side"], required:[]
# (a missing camera is zero-filled server-side with a DegradationWarning, not a
# hard close). The order here is the order images are presented to the model.
_CAMERA_KEYS = ["top", "side"]

# Canonical 6-DOF joint order — load-bearing for the model's state/action
# encoder. This is lerobot's SO101Follower motor order by construction, which is
# the order the MolmoAct2-SO100_101 fine-tune was trained on (lerobot-collected
# data). __init__ asserts the connected robot's action_features match this order
# and fails loud if lerobot ever changes it. (Open item T-B, resolved here.)
# Source: lerobot-nt @ 50168c2a src/lerobot/robots/so101_follower/so101_follower.py
_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Model input image size. 224, NOT the starter's 378: our fine-tune's
# config.json declares input_features observation.images.cam0/cam1 with shape
# [3, 224, 224]. The 378 figure belongs to New Theory's hosted so101 base model,
# which we no longer talk to — we serve our own checkpoint from Modal.
#
# Two cameras at 3x224x224 uint8 is ~301 KB on the wire, comfortably under the
# websockets 1 MiB default (378 was ~857 KB, native 640x480 would be ~1.77 MB
# and the server would tear down the connection mid-send).
# Source: ArjunPrasaath/play_xylophone_100 config.json.
_IMAGE_SIZE = 224

# ACTION_INTERVAL_S: per-action cadence while streaming a chunk. 15 fps is the
# common lerobot teleop/eval cadence; the exact fps the so101 fine-tune expects
# is a smoke-tunable (see smoke.md §cadence). Tune if motion is jerky or laggy.
ACTION_INTERVAL_S: float = 1.0 / 15  # ~0.0667s

# MAX_ACTIONS_PER_CHUNK: receding-horizon truncation. We play the first N
# actions, then read a fresh obs and query again — closed-loop. 0 = full horizon
# (open-loop). We slice client-side, no protocol change.
#
# The fine-tune's config.json has chunk_size=30 / n_action_steps=30, so a full
# chunk is 30 actions ~= 2.0s of motion at 15fps. 15 plays half of that (~1.0s),
# which should cover a strike while still leaving room for a closed-loop
# correction before the next one.
#
# This is the knob most likely to need tuning. Too low and the mallet stops
# mid-swing and never reaches the bar; too high and the arm runs open-loop
# through the whole strike. If notes sound weak or missed, raise it toward 30.
MAX_ACTIONS_PER_CHUNK: int = 15

# FIRST_CHUNK_SETTLE_S: the first chunk's action 0 may be far from the arm's
# current pose. SO101Follower.send_action has no goal-time interpolation (it
# sync-writes Goal_Position; the servos travel at their own speed), so we send
# action 0 and pause to let the arm reach it before streaming the rest. (Smoke
# confirms this is long enough for a safe move-to-start.)
FIRST_CHUNK_SETTLE_S: float = 1.5

# Rest pose used by --reset and emergency_home(): motor -> normalized position.
# With use_degrees=False the arm joints are in [-100, 100] (0 = centre) and the
# gripper is in [0, 100]. All-centred + gripper 0 is a SANE DEFAULT, not a
# verified-safe pose — confirm on the physical arm during the smoke (see
# smoke.md). Do NOT trust these numerics for an unattended move. (Open item T-D.)
_REST_POSE: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}

# Default site config path.
_DEFAULT_SITE_CONFIG_PATH = Path("~/.config/nt/nt.toml")

# ---------------------------------------------------------------------------
# Site config loading — reads ~/.config/nt/nt.toml.
# Only from_config() calls these; __init__ takes explicit values and reads no
# file (the 248-addendum factory split — from_config() is the sole config
# reader). Tests assert this by patching `open` to raise and constructing
# SO101 directly.
# ---------------------------------------------------------------------------


def _load_site_config(path: Path | str | None = None) -> dict:
    """Load ~/.config/nt/nt.toml. Raises FileNotFoundError if absent."""
    resolved = Path(path or _DEFAULT_SITE_CONFIG_PATH).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Site config not found at {resolved}.\n"
            "Populate ~/.config/nt/nt.toml with the arm serial port and the "
            "top/side camera indices.\n"
            "See README.md §Troubleshooting or conf/nt.toml.example."
        )
    with open(resolved, "rb") as f:
        return tomllib.load(f)


def _load_arm_port(raw: dict, arm_id: str | None = None) -> tuple[str, str]:
    """Extract (id, port) for the selected arm from the site config TOML dict.

    arm_id: explicit arm id from --arm or the config `arm` key.
      None + single-arm config  → arms[0].
      None + multi-arm config   → refuses with a named error listing ids + ports.
      <id> found                → returns that entry's (id, port).
      <id> not found            → fails naming the unknown id + all configured ids.

    Config key: [robot_config]\narm = "<id>" is honored as implicit selection;
    the --arm flag (arm_id arg) wins if both are present. The id doubles as the
    lerobot calibration id (calibration JSON is keyed by it).
    """
    arms = raw.get("robot_config", {}).get("arms", [])
    if not arms:
        raise ValueError(
            "nt.toml has no [[robot_config.arms]] entry. "
            "Add port = '/dev/ttyACM0' (Linux) or '/dev/tty.usbmodem*' (macOS) "
            "for the SO-101 arm."
        )

    config_arm_id = raw.get("robot_config", {}).get("arm")
    effective_id = arm_id or config_arm_id

    if effective_id is not None:
        for entry in arms:
            if entry.get("id") == effective_id:
                port = entry.get("port")
                if not port:
                    raise ValueError(
                        f"nt.toml arm '{effective_id}' has no 'port'. "
                        "Set port = '/dev/ttyACM0' (Linux) or "
                        "'/dev/tty.usbmodem*' (macOS)."
                    )
                return str(effective_id), str(port)
        configured = ", ".join(e.get("id", "?") for e in arms)
        raise ValueError(
            f"--arm '{effective_id}' not found in nt.toml.\n"
            f"Configured arm ids: {configured}\n"
            "Check the `id` field in [[robot_config.arms]]."
        )

    if len(arms) > 1:
        arm_lines = "\n".join(
            f"  --arm {e.get('id', '?')}   (port: {e.get('port', '?')})"
            for e in arms
        )
        raise ValueError(
            f"nt.toml has {len(arms)} arms configured but no arm was selected.\n"
            "Run with --arm to choose one:\n"
            f"{arm_lines}"
        )

    entry = arms[0]
    port = entry.get("port")
    if not port:
        raise ValueError(
            "nt.toml [[robot_config.arms]] missing 'port'. "
            "Set port = '/dev/ttyACM0' (Linux) or '/dev/tty.usbmodem*' (macOS)."
        )
    return str(entry.get("id", "so101")), str(port)


def _load_cameras(raw: dict) -> dict[str, dict]:
    """Extract per-camera {index_or_path, width, height, fps} from site config.

    Returns a dict keyed by camera id (e.g. "top", "side"). USB webcams only —
    no serial numbers, no extrinsics, no depth (the SO-101 has none of those).
    """
    cameras: dict[str, dict] = {}
    for c in raw.get("camera_config", {}).get("cameras", []):
        cam_id = c.get("id")
        idx = c.get("index_or_path")
        if cam_id is None or idx is None:
            continue
        cameras[str(cam_id)] = {
            "index_or_path": idx,
            "width": int(c.get("width", 640)),
            "height": int(c.get("height", 480)),
            "fps": int(c.get("fps", 30)),
        }
    return cameras


# ---------------------------------------------------------------------------
# Ctrl+H emergency stop — keyboard listener + abort flag.
#
# Port of imitation_learning/src/infra/inference/client/runtime.py:540–597:
#   tty.setcbreak + select.select + sys.stdin.read(1) checks for '\x08'.
# Our robot.run() is sync, so a daemon thread flips a threading.Event the action
# path checks before each driver write. Stdlib-only — no new deps. Embodiment-
# agnostic; kept verbatim from the Trossen starter. T1: the `newt` library stays
# headless; this lives in user code.
# ---------------------------------------------------------------------------


_emergency_stop = threading.Event()

# Saved cooked-mode tty settings so we can restore the terminal on exit.
_saved_tty_settings: list | None = None
_keyboard_thread: threading.Thread | None = None


class _EmergencyStop(Exception):
    """Raised inside execute()/read_state() when Ctrl+H is detected.

    Propagates up through robot.run() to main(), which runs the safe-home
    sequence and exits with status 130 (Ctrl+C convention).
    """


def _restore_terminal() -> None:
    """Restore cooked tty settings. Idempotent. Safe to call at exit."""
    global _saved_tty_settings
    if _saved_tty_settings is None:
        return
    try:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _saved_tty_settings)
    except (termios.error, ValueError, OSError):
        pass
    _saved_tty_settings = None


def _keyboard_listener_loop() -> None:
    """Background daemon: poll stdin for Ctrl+H (\\x08); set _emergency_stop.

    Polls every 50ms. Exits cleanly when _emergency_stop is set so the main
    thread can handle the abort sequence.
    Source: imitation_learning runtime.py:577–597.
    """
    while not _emergency_stop.is_set():
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        except (ValueError, OSError):
            # stdin closed (e.g. test harness, piped invocation tear-down)
            return
        if not ready:
            continue
        try:
            key = sys.stdin.read(1)
        except (ValueError, OSError):
            return
        if key == "\x08":  # Ctrl+H
            print(
                "\r\n[rig] EMERGENCY STOP — Ctrl+H pressed; "
                "aborting inference, homing arm…",
                flush=True,
            )
            _emergency_stop.set()
            return


def _start_keyboard_listener() -> None:
    """Switch stdin to cbreak + start the listener thread. Tty-only.

    Skips silently when stdin is not a TTY (piped/SSH bash -s invocation — no
    keyboard to read from anyway).
    Source: imitation_learning runtime.py:567–570.
    """
    global _saved_tty_settings, _keyboard_thread
    if not sys.stdin.isatty():
        print(
            "[rig] stdin is not a TTY — Ctrl+H listener disabled "
            "(use Ctrl+C to abort).",
            flush=True,
        )
        return
    try:
        _saved_tty_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    except (termios.error, OSError) as exc:
        print(f"[rig] could not set cbreak; Ctrl+H disabled ({exc})", flush=True)
        _saved_tty_settings = None
        return
    atexit.register(_restore_terminal)
    _keyboard_thread = threading.Thread(
        target=_keyboard_listener_loop,
        name="ctrl-h-listener",
        daemon=True,
    )
    _keyboard_thread.start()
    print(
        "[rig] Ctrl+H emergency-stop listener armed. "
        "Press Ctrl+H during the trial to home + safe the arm.",
        flush=True,
    )


def _check_emergency_stop() -> None:
    """Raise _EmergencyStop if Ctrl+H fired. Called between driver writes."""
    if _emergency_stop.is_set():
        raise _EmergencyStop()


# ---------------------------------------------------------------------------
# SO101 — SO-101 follower + USB webcam hardware embodiment.
#
# Drives the SO-101 through lerobot's SO101Follower (send_action /
# get_observation). Pure joint-space: six motor positions in, six out, plus two
# RGB frames. Implements newt.Embodiment (read_state / execute).
# ---------------------------------------------------------------------------


class SO101:
    """SO-101 follower + USB webcam hardware embodiment for newt.Robot.

    Implements newt.Embodiment: read_state() → obs dict, execute(chunk).
    Construct via from_config() to load the serial port and camera indices from
    ~/.config/nt/nt.toml — the same file run.py has always used.

    This class is yours. Rename it, edit the wiring, subclass it — anything with
    read_state() / execute() satisfies the protocol.
    """

    @classmethod
    def from_config(
        cls,
        site_config_path: Path | str | None = None,
        max_actions_per_chunk: int = MAX_ACTIONS_PER_CHUNK,
        arm_id: str | None = None,
    ) -> "SO101":
        """Construct from ~/.config/nt/nt.toml (or a custom path).

        This is the sole code path that reads the site config file. It loads the
        arm serial port and the top/side camera indices, then calls __init__
        with explicit values. Raises FileNotFoundError if the config is absent.

        arm_id: select which [[robot_config.arms]] entry to use (by its `id`
        field). Required when the config has more than one arm. Pass the value of
        --arm from run.py; single-arm configs ignore it. The id also names the
        lerobot calibration file (~/.cache/huggingface/lerobot/.../<id>.json).
        """
        raw = _load_site_config(site_config_path)
        selected_id, port = _load_arm_port(raw, arm_id)
        cameras = _load_cameras(raw)
        return cls(
            arm_id=selected_id,
            port=port,
            cameras=cameras,
            max_actions_per_chunk=max_actions_per_chunk,
        )

    def __init__(
        self,
        arm_id: str,
        port: str,
        cameras: dict,
        max_actions_per_chunk: int = MAX_ACTIONS_PER_CHUNK,
        use_degrees: bool = False,
        max_relative_target: int | None = None,
    ) -> None:
        """Construct from explicit values — reads no config file.

        use_degrees: False (default) reports/commands joints in lerobot's
          normalized range ([-100, 100] arm, [0, 100] gripper), which is the
          dataset convention the MolmoAct2-SO100_101 fine-tune was trained on.
          Flip to True only if the smoke shows the model expects degrees.
          (Open item T-C, resolved to the dataset default here.)
        max_relative_target: None (default) sends goal positions unclamped, as
          lerobot does by default. Set a positive scalar to cap per-step motion
          for safety once the rig's motion is characterized in the smoke.
        """
        _import_hardware_deps()
        if not _LEROBOT_AVAILABLE:
            raise ImportError(
                "lerobot hardware dependencies not installed.\n"
                "Run: uv sync --extra hardware\n"
                f"Original error: {_LEROBOT_IMPORT_ERR}"
            )

        self._arm_id = arm_id
        self.max_actions_per_chunk = max_actions_per_chunk
        self.chunks_observed: int = 0  # incremented by execute(); read by trial loop
        self._first_chunk_sent = False

        # Validate all required cameras are present before touching hardware.
        for cam in _CAMERA_KEYS:
            if cam not in cameras:
                raise ValueError(
                    f"Camera '{cam}' not found in nt.toml. "
                    f"Add a [[camera_config.cameras]] entry with id = '{cam}' "
                    "and its index_or_path (e.g. 0). The so101 model expects "
                    f"both {_CAMERA_KEYS} cameras."
                )

        # Build per-camera OpenCV configs (RGB, no depth).
        cameras_config: dict = {
            cam: OpenCVCameraConfig(
                index_or_path=cameras[cam]["index_or_path"],
                fps=cameras[cam]["fps"],
                width=cameras[cam]["width"],
                height=cameras[cam]["height"],
            )
            for cam in _CAMERA_KEYS
        }

        # Build the lerobot SO101Follower and connect.
        robot_config = SO101FollowerConfig(
            port=port,
            id=arm_id,
            cameras=cameras_config,
            use_degrees=use_degrees,
            max_relative_target=max_relative_target,
        )
        self._robot = SO101Follower(robot_config)

        # connect(calibrate=True) reuses an existing calibration file (one ENTER
        # to confirm) or runs lerobot's calibration flow if none exists. The
        # SO-101 is NOT factory-calibrated — run `lerobot-calibrate` once before
        # the first run (see README §Calibration). The id above names the
        # calibration file.
        try:
            self._robot.connect()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to SO-101 '{self._arm_id}' on port {port}: "
                f"{exc}\nCheck the port (ls /dev/ttyACM* or /dev/tty.usbmodem*) "
                "and that you have run `lerobot-calibrate` (see README §Calibration)."
            ) from exc

        # Capture action features in deterministic order and assert they match
        # the canonical joint order. Fail loud if lerobot ever changes it — the
        # state/action vector order is load-bearing for the model's encoder.
        raw_action_feats = getattr(self._robot, "action_features", {})
        feat_order = [k.removesuffix(".pos") for k in raw_action_feats]
        if feat_order != _JOINT_ORDER:
            raise RuntimeError(
                "SO101Follower joint order does not match the expected "
                f"training order.\n  expected: {_JOINT_ORDER}\n  got:      "
                f"{feat_order}\nThe model's state/action encoder depends on this "
                "order. Do not proceed — re-pin lerobot or escalate."
            )

    # -----------------------------------------------------------------------
    # Sensor side
    # -----------------------------------------------------------------------

    def read_state(self) -> dict:
        """Read one observation frame from the hardware.

        Returns an obs dict the newt.Robot serializes to the wire:
          state:  (6,) float32  — joint positions in _JOINT_ORDER
          images: dict[cam_key → (3, 378, 378) uint8 CHW RGB]

        No depth/intrinsics/extrinsics: the SO-101 has none, and the so101 serve
        contract has them as required:[] (a missing camera is zero-filled
        server-side with a DegradationWarning, not a hard close).
        """
        obs_raw = self._robot.get_observation()

        # State: six joint scalars in the canonical order.
        state = np.array(
            [float(obs_raw[f"{motor}.pos"]) for motor in _JOINT_ORDER],
            dtype=np.float32,
        )

        # Images: HWC RGB uint8 → resize to 378x378 square → CHW.
        import cv2 as _cv2

        images: dict[str, np.ndarray] = {}
        for cam in _CAMERA_KEYS:
            frame = np.asarray(obs_raw[cam], dtype=np.uint8)  # (H, W, 3) RGB
            resized = _cv2.resize(
                frame, (_IMAGE_SIZE, _IMAGE_SIZE), interpolation=_cv2.INTER_AREA
            )
            images[cam] = resized.transpose(2, 0, 1)  # (3, 378, 378)

        return {"state": state, "images": images}

    # -----------------------------------------------------------------------
    # Action side
    # -----------------------------------------------------------------------

    def execute(self, chunk: np.ndarray) -> None:
        """Apply one action chunk to the robot.

        Each row is six joint targets in _JOINT_ORDER. We map each row to
        {motor.pos: value} and send it via lerobot's send_action (sync_write
        Goal_Position). Streamed one action per ACTION_INTERVAL_S tick.

        First chunk only: send action 0 and pause FIRST_CHUNK_SETTLE_S to let
        the arm reach the start pose before streaming the rest (send_action has
        no goal-time interpolation; the servos travel at their own speed).
        """
        _check_emergency_stop()

        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != len(_JOINT_ORDER):
            raise ValueError(
                f"execute() expected (N, {len(_JOINT_ORDER)}) chunk, got "
                f"{chunk.shape}. Check the so101 server output shape "
                f"(action_shape [30, {len(_JOINT_ORDER)}])."
            )

        n_steps = chunk.shape[0]
        if self.max_actions_per_chunk > 0:
            n_steps = min(n_steps, self.max_actions_per_chunk)
        if n_steps <= 0:
            self.chunks_observed += 1
            return

        # First-chunk-only: send action 0, then settle before streaming.
        start_idx = 0
        if not self._first_chunk_sent:
            print(
                f"[rig] first chunk: move-to-start, settling "
                f"{FIRST_CHUNK_SETTLE_S}s",
                flush=True,
            )
            self._robot.send_action(self._row_to_action(chunk[0]))
            time.sleep(FIRST_CHUNK_SETTLE_S)
            self._first_chunk_sent = True
            start_idx = 1
            if n_steps <= 1:
                self.chunks_observed += 1
                return

        # Stream actions [start_idx, n_steps): one per ACTION_INTERVAL_S tick.
        for t in range(start_idx, n_steps):
            _check_emergency_stop()
            t0 = time.perf_counter()
            self._robot.send_action(self._row_to_action(chunk[t]))
            rest = ACTION_INTERVAL_S - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)

        self.chunks_observed += 1

    def reset_for_next_note(self) -> None:
        """Re-arm the first-chunk move-to-start settle for the next note.

        _first_chunk_sent is per-instance and nothing in the newt SDK ever
        resets it, so without this call every note after the first would start
        streaming immediately from wherever the previous strike left the arm —
        no settle, and the first action of the new bar's chunk can be far away.
        The sequencer calls this at each note boundary.
        """
        self._first_chunk_sent = False

    @staticmethod
    def _row_to_action(row: np.ndarray) -> dict[str, float]:
        """Map a 6-vector (in _JOINT_ORDER) to a lerobot {motor.pos: val} dict."""
        return {f"{motor}.pos": float(row[i]) for i, motor in enumerate(_JOINT_ORDER)}

    def teardown(self) -> None:
        """Disconnect from hardware. Disables torque (back-drivable) per the
        SO101FollowerConfig.disable_torque_on_disconnect default."""
        try:
            self._robot.disconnect()
        except Exception as exc:
            print(f"[rig] teardown warning: {exc}", file=sys.stderr)

    def emergency_home(self) -> None:
        """Send the arm to _REST_POSE, then disconnect. Ctrl+H abort sequence.

        _REST_POSE is a SANE DEFAULT, not a verified-safe pose — see its
        definition and confirm on the physical arm during the smoke.
        """
        try:
            print(
                "[rig] EMERGENCY STOP — moving arm to rest, then disconnecting…",
                flush=True,
            )
            self._robot.send_action(
                {f"{m}.pos": v for m, v in _REST_POSE.items()}
            )
            time.sleep(FIRST_CHUNK_SETTLE_S)
        except Exception as exc:
            print(f"[rig] emergency_home: rest move failed: {exc}", file=sys.stderr)
        try:
            self._robot.disconnect()
        except Exception as exc:
            print(f"[rig] emergency_home: disconnect failed: {exc}", file=sys.stderr)
        print("[rig] EMERGENCY STOP — arm safed at rest", flush=True)
