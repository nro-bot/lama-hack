"""
v2 of annotate_episode.py -- kept as a separate file so the original stays
untouched. Only change from v1: the impact circle and the struck key's
highlight/recolor are now driven by the exact same trough_start/trough_end
window (v1 had them desynced -- the circle used the full trough-persistence
window from a later fix, but the key highlight was left on the older
single-peak-time decay animation).

Annotate one SO-100 xylophone episode with strike-detection visuals for the
Rerun Viewer: impact ripples at the 3D strike position, and a highlighted
xylophone key panel.

This does NOT rewrite the original episode .rrd (which is mostly ~80MB of
camera images). Instead it writes a small companion file
output/{episode}_annotations_v2.rrd that shares the original's
application_id and recording_id, so opening BOTH files together in the
Rerun Viewer merges them into a single recording with the new
/annotations/* entities overlaid.

Usage:
    python scripts/annotate_episode_v2.py                        # first .rrd in data/
    python scripts/annotate_episode_v2.py data/episode_04.rrd
    python scripts/annotate_episode_v2.py data/episode_04.rrd --prominence 10 --decay-frames 8

Then in the Rerun Viewer: File > Open both
  data/episode_04.rrd  and  output/episode_04_annotations_v2.rrd
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr

import rrd_utils as ru

MIN_RADIUS_DEFAULT = 0.02
MAX_RADIUS_DEFAULT = 0.04
DECAY_FRAMES_DEFAULT = 6
STRIKE_JOINT = "elbow_flex"
KEY_BOX_HALF_SIZE = (0.015, 0.05, 0.006)

# Xylophone base plate: spans the full key row (x in [-0.28, 0.28]) plus
# margin, positioned so its top surface sits right at the bottom of the key
# boxes (key bottom = PANEL_Z - KEY_BOX_HALF_SIZE[2] = -0.086) -- the keys
# read as "sitting on" the base rather than floating.
BASE_HALF_SIZE = (0.32, 0.09, 0.01)
BASE_COLOR = (120, 80, 50)  # wood-toned brown, distinct from key colors
_BASE_TOP_Z = ru.PANEL_Z - KEY_BOX_HALF_SIZE[2]
BASE_CENTER = (0.0, ru.PANEL_Y, _BASE_TOP_Z - BASE_HALF_SIZE[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", nargs="?", default=None, help="Path to episode .rrd (default: first in data/)")
    parser.add_argument("--prominence", type=float, default=15.0, help="scipy.signal.find_peaks prominence (default: 15.0)")
    parser.add_argument("--height", type=float, default=None, help="scipy.signal.find_peaks height (default: None)")
    parser.add_argument(
        "--debounce-seconds",
        type=float,
        default=0.3,
        help="Collapse peaks closer together than this (ringing from one strike) into one. 0 disables. (default: 0.3)",
    )
    parser.add_argument(
        "--debounce-strategy",
        choices=["first", "max"],
        default="first",
        help="Which peak to keep per debounce cluster: earliest ('first', default) or highest-magnitude ('max')",
    )
    parser.add_argument(
        "--normalize-against",
        type=float,
        default=None,
        help="Global max |load| to normalize intensity against (default: this episode's own max)",
    )
    parser.add_argument("--decay-frames", type=int, default=DECAY_FRAMES_DEFAULT, help="Impact/highlight decay length in frames")
    parser.add_argument("--min-radius", type=float, default=MIN_RADIUS_DEFAULT)
    parser.add_argument("--max-radius", type=float, default=MAX_RADIUS_DEFAULT)
    parser.add_argument("--output", type=Path, default=None, help="Output .rrd path (default: output/{episode}_annotations_v2.rrd)")
    parser.add_argument("--force-recalibrate", action="store_true", help="Rebuild the xylophone key layout cache")
    return parser.parse_args()


def find_default_episode() -> Path:
    candidates = sorted(ru.DATA_DIR.glob("*.rrd"))
    if not candidates:
        print(f"No .rrd files found in {ru.DATA_DIR}.")
        sys.exit(1)
    return candidates[0]


def lerp_color(c0: tuple[int, int, int], c1: tuple[int, int, int], frac: float) -> tuple[int, int, int]:
    return tuple(int(round(a + (b - a) * frac)) for a, b in zip(c0, c1))


def glow_color(base_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Brightened variant of a base color, blended 65% toward white."""
    return lerp_color(base_rgb, (255, 255, 255), 0.65)


def sanitize_key(key_id: str) -> str:
    return key_id.replace(" ", "_")


def log_strike_visuals(
    frame_times,
    trough_start,
    trough_end,
    impact_position: np.ndarray,
    key_id: str,
    key_position,
    base_rgb,
    intensity: float,
    decay_frames: int,
    min_radius: float,
    max_radius: float,
) -> None:
    """Log the impact circle AND the struck key's highlight from the exact
    same frame set, in the same loop -- v1 had these desynced (the circle
    used the full trough-persistence window from a later fix, but the key
    highlight was left on an older single-peak-time decay animation with a
    different frame count). Driving both from one shared iteration makes
    that class of bug structurally impossible: there is only one set of
    timestamps, used for both entities.

    Persist phase: circle at constant radius + key at constant "glow" color,
    across every frame within [trough_start, trough_end] (the load signal's
    actual measured below-baseline duration -- see rrd_utils.detect_strikes'
    trough_start/trough_end, from scipy.signal.peak_widths). Decay phase:
    both shrink/fade back to base together, across the next decay_frames
    timestamps after trough_end.
    """
    radius0 = min_radius + intensity * (max_radius - min_radius)
    glow = glow_color(base_rgb)
    key_entity = f"/annotations/xylophone_keys/{sanitize_key(key_id)}"

    persist_start_idx = frame_times.searchsorted(trough_start)
    persist_end_idx = frame_times.searchsorted(trough_end, side="right")
    persist_times = frame_times[persist_start_idx:persist_end_idx]
    if len(persist_times) == 0:
        persist_times = [trough_start]

    for t in persist_times:
        rr.set_time("time", timestamp=t)
        rr.log("/annotations/impact", rr.Points3D(positions=[impact_position], radii=[radius0], colors=[(*base_rgb, 255)]))
        rr.log(key_entity, rr.Boxes3D(centers=[key_position], half_sizes=[KEY_BOX_HALF_SIZE], colors=[(*glow, 255)]))

    decay_times = frame_times[persist_end_idx : persist_end_idx + decay_frames]
    fractions = ru.decay_schedule(len(decay_times))
    for t, frac in zip(decay_times, fractions):
        radius = radius0 * (1.0 - 0.85 * frac)
        alpha = int(255 * (1.0 - frac) ** 1.5)
        key_color = lerp_color(glow, base_rgb, frac)
        rr.set_time("time", timestamp=t)
        rr.log("/annotations/impact", rr.Points3D(positions=[impact_position], radii=[radius], colors=[(*base_rgb, alpha)]))
        rr.log(key_entity, rr.Boxes3D(centers=[key_position], half_sizes=[KEY_BOX_HALF_SIZE], colors=[(*key_color, 255)]))


def log_xylophone_panel(layout: dict, panel_positions: dict, start_time) -> None:
    """Log the static row of key bars once, at the episode's first frame.

    Uses the schematic (evenly-spaced, arm-clear) panel_positions for where
    to draw each box -- NOT layout[key]["position"], which is the noisy
    calibrated strike position used only for the geometric mismatch check.
    """
    rr.set_time("time", timestamp=start_time)
    for key_id, info in layout.items():
        if key_id.startswith("_") or key_id not in panel_positions:
            continue
        rgb = ru.key_color_rgb(info["color"])
        entity = f"/annotations/xylophone_keys/{sanitize_key(key_id)}"
        rr.log(
            entity,
            rr.Boxes3D(
                centers=[panel_positions[key_id]],
                half_sizes=[KEY_BOX_HALF_SIZE],
                colors=[(*rgb, 255)],
                labels=[key_id],
            ),
        )


def log_xylophone_base(start_time) -> None:
    """A flat base plate beneath the key row so the keys read as sitting on
    a physical instrument body rather than floating boxes."""
    rr.set_time("time", timestamp=start_time)
    rr.log(
        "/annotations/xylophone_base",
        rr.Boxes3D(centers=[BASE_CENTER], half_sizes=[BASE_HALF_SIZE], colors=[(*BASE_COLOR, 255)]),
    )


def main() -> None:
    args = parse_args()
    episode_path = Path(args.episode) if args.episode else find_default_episode()
    if not episode_path.exists():
        print(f"File not found: {episode_path}")
        sys.exit(1)

    print(f"Loading {episode_path} ...")
    store, entry = ru.load_store_and_entry(episode_path)

    task_text = ru.read_task_text(store)
    print(f"Task: {task_text!r}")
    parsed = ru.parse_task(task_text)
    if parsed is None:
        print("Cannot proceed without a parseable task label (key/color unknown).")
        sys.exit(1)
    task_key_id, task_color = parsed
    task_rgb = ru.key_color_rgb(task_color)
    print(f"Parsed key={task_key_id!r} color={task_color!r}")

    # --- Step 1: strike detection ---
    load_df = ru.scalar_series_df(store, "/follower/load")
    if STRIKE_JOINT not in load_df.columns:
        print(f"ERROR: {STRIKE_JOINT!r} not found in joints {list(load_df.columns)}")
        sys.exit(1)

    strikes = ru.detect_strikes(
        load_df[STRIKE_JOINT],
        height=args.height,
        prominence=args.prominence,
        debounce_seconds=args.debounce_seconds,
        debounce_strategy=args.debounce_strategy,
    )
    scores = ru.intensity_scores(strikes, args.normalize_against)
    print(
        f"\nDetected {len(strikes)} strikes on {STRIKE_JOINT} load "
        f"(prominence={args.prominence}, height={args.height}, debounce={args.debounce_seconds}s/{args.debounce_strategy}):"
    )
    for i, (row, score) in enumerate(zip(strikes.itertuples(), scores)):
        print(f"  #{i}  t={row.time}  load={row.value:.1f}  |load|={row.magnitude:.1f}  intensity={score:.2f}")

    if strikes.empty:
        print("No strikes detected -- nothing to annotate. Try lowering --prominence.")
        sys.exit(0)

    # --- Step 2 prep: end-effector (jaw) 3D trajectory ---
    fk_df = ru.fk_tip_positions(store)
    strike_positions = np.stack([ru.nearest_position(fk_df, t) for t in strikes["time"]])

    # --- Step 3 prep: calibrated xylophone key layout (cached) ---
    layout = ru.build_xylophone_layout(
        force=args.force_recalibrate,
        prominence=args.prominence,
        debounce_seconds=args.debounce_seconds,
        debounce_strategy=args.debounce_strategy,
    )
    if task_key_id not in layout:
        print(f"WARNING: no calibrated layout position for this episode's own key {task_key_id!r}; panel highlight will be skipped for it.")

    # Cross-check: does the *geometrically nearest* calibrated key (along the
    # dominant separating axis -- see rrd_utils.dominant_separating_axis)
    # match the label?
    mismatches = 0
    for i, pos in enumerate(strike_positions):
        nearest_key = ru.nearest_layout_key(layout, pos)
        if nearest_key is not None and nearest_key != task_key_id:
            mismatches += 1
            print(
                f"WARNING: strike #{i} at {pos.round(3).tolist()} is geometrically closest to "
                f"key {nearest_key!r}, not this episode's task label {task_key_id!r} -- possible mislabeled episode."
            )
    if mismatches == 0:
        print("Geometric cross-check: all strikes land closest to this episode's own task-label key. OK.")

    # --- Log everything to a companion recording sharing the original's IDs ---
    # rr.save() must be called *before* any logging (per its docstring: "Call
    # this before you log any data!") -- otherwise events are held in an
    # unbounded pre-sink buffer and large runs can silently lose early data
    # once a sink finally attaches. So: init -> save -> THEN log.
    rr.init(application_id=entry.application_id, recording_id=entry.recording_id, spawn=False)
    output_path = args.output or (ru.OUTPUT_DIR / f"{episode_path.stem}_annotations_v2.rrd")
    ru.OUTPUT_DIR.mkdir(exist_ok=True)
    rr.save(str(output_path))

    frame_times = load_df.index
    panel_positions = ru.schematic_panel_positions()
    log_xylophone_base(frame_times[0])
    log_xylophone_panel(layout, panel_positions, frame_times[0])

    # Continuous elbow_flex load as its own single-series entity: /follower/load
    # is one batched Scalars call covering all 6 joints, which a blueprint
    # TimeSeriesView can't slice down to just elbow_flex -- so we re-log it
    # here as a dedicated series the blueprint's bottom panel can target.
    for t, value in load_df[STRIKE_JOINT].items():
        rr.set_time("time", timestamp=t)
        rr.log("/annotations/elbow_flex_load", rr.Scalars([value]))

    # Impact circles render at the STRUCK KEY's schematic panel position, not
    # the raw end-effector transform: the panel is a deliberately schematic,
    # evenly-spaced layout (see rrd_utils.schematic_panel_positions) that has
    # no relationship to the arm's real coordinates, so using the raw FK
    # position here would make the circle float away from the key it
    # actually landed on. All strikes on this episode's key show up at the
    # same panel spot (offset slightly above the box top surface), which is
    # correct since they're all hits on the same physical key.
    impact_display_position = None
    if task_key_id in panel_positions:
        impact_display_position = np.array(panel_positions[task_key_id]) + np.array([0.0, 0.0, KEY_BOX_HALF_SIZE[2] + 0.003])
    else:
        print(f"WARNING: {task_key_id!r} has no panel position; impact circles will be skipped.")

    for score, row in zip(scores, strikes.itertuples()):
        if impact_display_position is not None and task_key_id in panel_positions:
            log_strike_visuals(
                frame_times,
                row.trough_start,
                row.trough_end,
                impact_display_position,
                task_key_id,
                panel_positions[task_key_id],
                task_rgb,
                score,
                args.decay_frames,
                args.min_radius,
                args.max_radius,
            )

        # Sparse marker at the strike time, rendered as points (not a line) for the blueprint's timeseries panel.
        rr.set_time("time", timestamp=row.time)
        rr.log(
            "/annotations/strike_markers",
            rr.Scalars([row.value]),
            rr.SeriesPoints(markers=["Cross"], marker_sizes=[8], colors=[task_rgb]),
        )

    print(f"\nSaved {output_path}")
    print(f"application_id={entry.application_id!r} recording_id={entry.recording_id!r} (matches original -- open both files together in the Viewer)")


if __name__ == "__main__":
    main()
