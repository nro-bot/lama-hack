"""
v2 of compare_takes.py -- kept as a separate file so the original stays
untouched (paired with make_blueprint_v2.py / used alongside
annotate_episode_v2.py). Changes from v1, across three rounds of fixes:

Cross-take comparison: compare several episodes of the SAME xylophone
key/task via small multiples, not one shared 3D scene (an earlier
shared-scene version was abandoned after repeated fix attempts -- tangled
trajectories, floating labels, missing camera panels).

  1. One independent MINI 3D panel per take: its own full copy of the
     schematic key-row panel (same one annotate_episode.py uses, static --
     always visible, a fixed reference) plus a single impact circle at the
     struck key. Every take's geometry lives under its own /take_N/...
     entity subtree with NO paths shared across takes -- there is no
     shared coordinate space for anything to tangle in.
     The circle is colored with that TAKE's palette color (not the key's
     color) at reduced alpha (~67%, so overlapping takes visibly blend)
     and carries a "Take N" label. It is TIMELINE-GATED, not static: only
     visible across its own [trough_start, trough_end] window plus a short
     decay tail, on the same elapsed-since-first-strike scale used for
     camera playback sync -- see log_impact_circle_on_timeline, which
     mirrors annotate_episode_v2.log_strike_visuals' persist+decay
     algorithm (same idea, logged with duration= instead of timestamp=
     since this recording's shared timeline is relative, not absolute).
     A static circle looked frozen/broken next to camera feeds that DO
     play across the timeline, which is why this isn't static=True.
  2. A native rr.BarChart panel: ONE entity (/bar_chart/intensity_comparison)
     carrying the full array of per-take intensity values plus a per-bar
     color array, logged once -- not one BarChart entity per take (that
     produced 5 separate one-bar charts instead of one 5-bar chart, which
     is what the SDK's "should only be 1D" warning was flagging).
  3. The legend and metrics table stay static (fixed reference info, not
     part of the "does something happen when I scrub" experience). Camera
     rows: only the SIDE camera (cam1 -- see SIDE_CAMERA) is used, 5 panels
     total instead of 10 for cam0+cam1; camera presence is checked
     directly against each SOURCE episode's schema
     (rrd_utils.episode_camera_entities, cheap manifest-only read) *before*
     any copying is attempted, and reported per take -- a take missing
     camera frames in its source file gets no camera panel in the
     blueprint at all (no black placeholder). Every row-like blueprint
     layout uses rrb.Grid, never rrb.Horizontal (see make_blueprint_v2.py's
     build_compare_takes_blueprint docstring for why).

Usage:
    python scripts/compare_takes_v2.py "big C"
    python scripts/compare_takes_v2.py "Hitting note D (orange)" --max-takes 4
    python scripts/compare_takes_v2.py "big C" --episodes data/episode_04.rrd data/episode_05.rrd
    python scripts/compare_takes_v2.py "big C" --no-cameras   # skip copying the side camera (much faster, no video)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr

import rrd_utils as ru
from make_blueprint_v2 import build_compare_takes_blueprint, verify_blueprint_containers

STRIKE_JOINT = "elbow_flex"
COMPARE_APP_ID = "so-100-compare-takes"

# Per-TAKE palette (not per-key!) -- up to 5 takes get visually distinct
# colors so repeated demonstrations of the same key don't get confused
# with the xylophone's own key-color scheme.
TAKE_PALETTE = [
    ("teal", (0, 150, 136)),
    ("magenta", (216, 27, 96)),
    ("yellow", (253, 216, 53)),
    ("orange", (255, 111, 0)),
    ("purple", (142, 36, 170)),
]

KEY_BOX_HALF_SIZE = (0.015, 0.05, 0.006)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_or_key", help='Task text (e.g. "Hitting note big C (red)") or bare key (e.g. "big C")')
    parser.add_argument("--max-takes", type=int, default=5, help="Max number of episodes to compare (default: 5)")
    parser.add_argument("--episodes", nargs="+", type=Path, default=None, help="Explicit episode paths instead of auto-searching data/")
    parser.add_argument("--prominence", type=float, default=15.0)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument(
        "--debounce-seconds",
        type=float,
        default=0.3,
        help="Collapse peaks closer together than this (ringing from one strike) into one. 0 disables. (default: 0.3)",
    )
    parser.add_argument("--debounce-strategy", choices=["first", "max"], default="first")
    parser.add_argument(
        "--width-rel-height",
        type=float,
        default=0.5,
        help="Trough width measured at this fraction of prominence (default: 0.5, tighter than rrd_utils' 1.0 default -- see log_take's docstring)",
    )
    parser.add_argument("--no-cameras", action="store_true", help="Skip copying the side camera's frames (much faster, no video in the grid)")
    parser.add_argument("--output", type=Path, default=None, help="Output .rrd path (default: output/compare_{key}_takes_v2.rrd)")
    parser.add_argument("--blueprint-output", type=Path, default=None, help="Output .rbl path (default: output/blueprint_compare_{key}_v2.rbl)")
    return parser.parse_args()


def find_matching_episodes(task_or_key: str, max_takes: int, data_dir: Path = ru.DATA_DIR) -> list[Path]:
    matches: list[Path] = []
    for path in sorted(data_dir.glob("*.rrd")):
        if len(matches) >= max_takes:
            break
        task_text = ru.read_task_text_fast(path)
        if not task_text:
            continue
        parsed = ru.parse_task(task_text)
        key_id = parsed[0] if parsed else None
        if task_text == task_or_key or key_id == task_or_key:
            matches.append(path)
    return matches


SIDE_CAMERA = "cam1"  # confirmed by decoding a sample frame from each: cam0 is
# a top-down view straight over the xylophone/mallet, cam1 is an eye-level
# angled view of the arm and xylophone from the side. Only cam1 is used --
# 5 camera panels total (one per take) instead of 10 (cam0+cam1 x 5 takes).


def copy_camera(entity_root: str, store, camera_name: str, zero_t) -> None:
    """Copy raw EncodedImage bytes for one camera into the new recording,
    timed as elapsed seconds relative to zero_t. No re-encoding -- just
    moving already-compressed JPEG bytes. Only called after
    rrd_utils.episode_camera_entities() has already confirmed this camera
    exists in the source file, so an empty result here would indicate a
    genuine inconsistency, not an expected case."""
    df = store.reader(index="time", contents=f"/camera/{camera_name}").to_pandas()
    if df.empty:
        print(f"  ANOMALY: source pre-check said /camera/{camera_name} exists, but the query returned no rows.")
        return
    blob_col = f"/camera/{camera_name}:EncodedImage:blob"
    media_col = f"/camera/{camera_name}:EncodedImage:media_type"
    sync_t = (df["time"] - zero_t).dt.total_seconds().to_numpy()
    for t_sync, blob, media_type in zip(sync_t, df[blob_col], df[media_col]):
        rr.set_time("time", duration=float(t_sync))
        rr.log(f"{entity_root}/camera/{camera_name}", rr.EncodedImage(contents=bytes(blob[0]), media_type=str(media_type[0])))


def log_take(
    take_idx: int,
    path: Path,
    prominence: float,
    height: float | None,
    debounce_seconds: float,
    debounce_strategy: str,
    has_side_camera: bool,
    width_rel_height: float = 0.5,
) -> dict:
    """Load one episode, detect its single strike, and copy the side camera
    (if present in the source). Returns a stats dict; the mini key panel /
    impact circle / bar chart are logged separately, once global intensity
    normalization is known (see log_mini_panels).

    width_rel_height=0.5 (scipy's own default), NOT rrd_utils.detect_strikes'
    default of 1.0: checked empirically across the 5 test episodes and found
    at rel_height=1.0 some troughs' measured duration balloons to 3-6+
    seconds (30-56% of the whole episode) whenever the load signal's
    baseline is noisy on the way back up -- a timeline-gated circle
    "persisting" for that long looks just as frozen as the static version
    this fix replaces. 0.5 keeps every test episode's window under ~2s.
    This override is local to compare_takes_v2.py -- rrd_utils.py's shared
    default (used by annotate_episode.py/v2 unmodified) is untouched.
    """
    color_name, color = TAKE_PALETTE[(take_idx - 1) % len(TAKE_PALETTE)]
    entity_root = f"/take_{take_idx}"

    store, _entry = ru.load_store_and_entry(path)
    task_text = ru.read_task_text(store)
    parsed = ru.parse_task(task_text)

    load_df = ru.scalar_series_df(store, "/follower/load")
    strikes = ru.detect_strikes(
        load_df[STRIKE_JOINT],
        height=height,
        prominence=prominence,
        debounce_seconds=debounce_seconds,
        debounce_strategy=debounce_strategy,
        width_rel_height=width_rel_height,
    )
    zero_t = strikes["time"].iloc[0] if len(strikes) else load_df.index[0]

    if has_side_camera:
        copy_camera(entity_root, store, SIDE_CAMERA, zero_t)

    # Same sync convention as camera playback (elapsed seconds since this
    # take's own first strike): the trough window and this take's own frame
    # times, both converted onto that scale, are what let the impact circle
    # be timeline-gated (see log_impact_circle_on_timeline) instead of
    # static, in step with the camera row below it.
    trough_start_sync = trough_end_sync = None
    frame_times_sync = None
    if len(strikes):
        row = strikes.iloc[0]
        trough_start_sync = (row["trough_start"] - zero_t).total_seconds()
        trough_end_sync = (row["trough_end"] - zero_t).total_seconds()
        frame_times_sync = (load_df.index - zero_t).total_seconds().to_numpy()

    return {
        "entity_root": entity_root,
        "path": path,
        "task": task_text,
        "key_id": parsed[0] if parsed else None,
        "n_strikes": len(strikes),
        "magnitude": float(strikes["magnitude"].iloc[0]) if len(strikes) else float("nan"),
        "color_name": color_name,
        "color": color,
        "has_side_camera": has_side_camera,
        "trough_start_sync": trough_start_sync,
        "trough_end_sync": trough_end_sync,
        "frame_times_sync": frame_times_sync,
    }


IMPACT_CIRCLE_ALPHA = 170  # ~67% opacity (60-70% requested), so overlapping
# takes' circles visibly blend instead of one fully occluding the others.
MIN_CIRCLE_RADIUS = 0.008
MAX_CIRCLE_RADIUS = 0.028
DECAY_FRAMES = 6


def log_impact_circle_on_timeline(
    entity: str,
    frame_times_sync: np.ndarray,
    trough_start_sync: float,
    trough_end_sync: float,
    position,
    color: tuple[int, int, int],
    intensity: float,
    label: str,
) -> None:
    """Timeline-gated version of the impact circle: persist at constant
    radius across every frame within this take's own [trough_start_sync,
    trough_end_sync] (on the SAME elapsed-since-first-strike scale already
    used for camera sync), then shrink/fade over DECAY_FRAMES more --
    mirroring annotate_episode_v2.log_strike_visuals' persist+decay
    algorithm, just logged with duration= (the comparison view's shared
    timeline) instead of timestamp= (single-episode's absolute timeline).
    Before the persist window, nothing is logged for this entity yet, so
    the Viewer shows it as absent -- there is no need for an explicit
    "invisible" state; latest-at semantics give it for free."""
    radius0 = MIN_CIRCLE_RADIUS + intensity * (MAX_CIRCLE_RADIUS - MIN_CIRCLE_RADIUS)

    persist_start_idx = int(np.searchsorted(frame_times_sync, trough_start_sync))
    persist_end_idx = int(np.searchsorted(frame_times_sync, trough_end_sync, side="right"))
    persist_times = frame_times_sync[persist_start_idx:persist_end_idx]
    if len(persist_times) == 0:
        persist_times = [trough_start_sync]

    for t in persist_times:
        rr.set_time("time", duration=float(t))
        rr.log(entity, rr.Points3D(positions=[position], radii=[radius0], colors=[(*color, IMPACT_CIRCLE_ALPHA)], labels=[label]))

    decay_times = frame_times_sync[persist_end_idx : persist_end_idx + DECAY_FRAMES]
    fractions = ru.decay_schedule(len(decay_times))
    for t, frac in zip(decay_times, fractions):
        radius = radius0 * (1.0 - 0.85 * frac)
        alpha = int(IMPACT_CIRCLE_ALPHA * (1.0 - frac) ** 1.5)
        rr.set_time("time", duration=float(t))
        rr.log(entity, rr.Points3D(positions=[position], radii=[radius], colors=[(*color, alpha)], labels=[label]))


def log_mini_panels(stats: list[dict], layout: dict, panel_positions: dict) -> list[dict]:
    """For each take: its OWN full copy of the 8-key schematic panel
    (static -- always visible, a fixed reference) plus its own single
    impact circle (timeline-gated -- see log_impact_circle_on_timeline),
    under /take_N/xylophone_keys/** and /take_N/impact_circle. The circle
    is colored with that TAKE's palette color (never the key's color --
    key identity is already shown by the key boxes' own colors) at
    reduced alpha, and carries a "Take N" text label so which independent
    panel belongs to which take is visible at a glance, not just in the
    panel's title bar. Sets each stat's "mean_intensity", normalized
    against the strongest strike across ALL compared takes (each episode
    has only one strike, so a per-episode max would trivially be 1.00 for
    everyone).

    Returns a list of {"take", "key_id", "color_name", "circle_color",
    "circle_alpha", "position", "trough_start_sync", "trough_end_sync"}
    for the caller to print/verify.
    """
    magnitudes = [s["magnitude"] for s in stats if s["magnitude"] == s["magnitude"]]
    global_max = max(magnitudes) if magnitudes else 1.0

    logged: list[dict] = []
    for i, s in enumerate(stats, start=1):
        root = s["entity_root"]

        for key_id, position in panel_positions.items():
            color = ru.key_color_rgb(layout.get(key_id, {}).get("color", "")) if key_id in layout else (120, 120, 120)
            rr.log(
                f"{root}/xylophone_keys/{key_id.replace(' ', '_')}",
                rr.Boxes3D(centers=[position], half_sizes=[KEY_BOX_HALF_SIZE], colors=[(*color, 255)], labels=[key_id]),
                static=True,
            )

        key_id = s.get("key_id")
        intensity = 0.0
        if key_id in panel_positions and s["magnitude"] == s["magnitude"] and s["frame_times_sync"] is not None:
            intensity = s["magnitude"] / global_max if global_max > 0 else 0.0
            log_impact_circle_on_timeline(
                f"{root}/impact_circle",
                s["frame_times_sync"],
                s["trough_start_sync"],
                s["trough_end_sync"],
                panel_positions[key_id],
                s["color"],
                intensity,
                f"Take {i}",
            )
            logged.append(
                {
                    "take": i,
                    "key_id": key_id,
                    "color_name": s["color_name"],
                    "circle_color": s["color"],
                    "circle_alpha": IMPACT_CIRCLE_ALPHA,
                    "position": panel_positions[key_id],
                    "trough_start_sync": s["trough_start_sync"],
                    "trough_end_sync": s["trough_end_sync"],
                }
            )
        elif key_id not in panel_positions:
            print(f"  WARNING: take {i} has no recognized key ({s['task']!r}); skipping its impact circle and bar.")
        s["mean_intensity"] = intensity

    return logged


def log_bar_chart(stats: list[dict]) -> list[float]:
    """ONE rr.BarChart entity carrying the full array of per-take
    intensities (previously this logged 5 separate single-value BarChart
    entities -- one bar chart each -- which is what the SDK's "should only
    be 1D" warning was flagging; a single archetype instance with a 5-value
    array is the correct way to get one chart with 5 bars). BarChart has no
    per-bar label field, so bar-index-to-take mapping is spelled out in the
    legend panel instead; abscissa=[0..n-1] still gives bars distinct x
    positions in Take order, and color is a per-bar array matching the
    take palette.

    Returns the logged values array (for the caller to print/verify).
    """
    values = [s["mean_intensity"] for s in stats]
    colors = [s["color"] for s in stats]
    rr.log(
        "/bar_chart/intensity_comparison",
        rr.BarChart(values, color=colors, abscissa=list(range(len(values)))),
        static=True,
    )
    return values


def log_legend(stats: list[dict]) -> None:
    lines = ["# Take legend", ""]
    for i, s in enumerate(stats, start=1):
        lines.append(f"- **Take {i}** = {s['color_name']}  (`{s['path'].name}`)")
    lines.append("")
    lines.append(
        "**Bar chart key:** bars are in Take order left to right -- "
        "bar 1 = Take 1, bar 2 = Take 2, etc. (BarChart has no per-bar text label, "
        "so the mapping is: " + ", ".join(f"bar {i} = Take {i} ({s['color_name']})" for i, s in enumerate(stats, start=1)) + ")"
    )
    rr.log("/legend", rr.TextDocument("\n".join(lines), media_type="text/markdown"), static=True)


def _fmt(value: float, suffix: str = "") -> str:
    return "n/a" if value != value else f"{value:.2f}{suffix}"  # value != value <=> NaN


def log_metrics(stats: list[dict]) -> None:
    counts = [s["n_strikes"] for s in stats]

    lines = ["# Consistency metrics", "", "| Take | Strikes | Intensity | Side camera |", "|---|---|---|---|"]
    for i, s in enumerate(stats, start=1):
        cam_str = "cam1" if s["has_side_camera"] else "none"
        lines.append(f"| {i} ({s['color_name']}) | {s['n_strikes']} | {_fmt(s['mean_intensity'])} | {cam_str} |")
    lines.append("")
    lines.append(f"**Strike-count stdev across takes:** {np.std(counts):.2f}")
    lines.append(
        "No strike-interval comparison: each episode is a single-strike \"hit this key once\" "
        "demonstration (detect_strikes keeps only the first real trough per episode)."
    )

    rr.log("/metrics", rr.TextDocument("\n".join(lines), media_type="text/markdown"), static=True)


def verify_panel_independence(stats: list[dict]) -> None:
    """Print an explicit check that every take's entity subtree is fully
    disjoint from every other's (required so a mini panel's Spatial3DView,
    scoped to just that take's origin, can never show another take's
    geometry)."""
    print("\n(a) Mini 3D panel independence check:")
    roots = [s["entity_root"] for s in stats]
    ok = len(roots) == len(set(roots))
    for i, root in enumerate(roots, start=1):
        others = [r for j, r in enumerate(roots) if j != i - 1]
        print(f"  take_{i}: {root}/xylophone_keys/**, {root}/impact_circle  (disjoint from: {others})")
    print(
        "  CONFIRMED: all take entity roots are unique -> no shared paths -> no cross-contamination possible."
        if ok
        else "  ERROR: duplicate entity roots detected!"
    )


def main() -> None:
    args = parse_args()

    episodes = args.episodes if args.episodes else find_matching_episodes(args.task_or_key, args.max_takes)
    if not episodes:
        print(f"No episodes found matching {args.task_or_key!r}.")
        sys.exit(1)
    if len(episodes) > 5:
        print(f"NOTE: {len(episodes)} episodes matched; the take-color palette only has 5 distinct colors, they will repeat.")

    print(f"Comparing {len(episodes)} take(s) for {args.task_or_key!r}:")
    for p in episodes:
        print(f"  {p}")

    # (c) Camera presence checked directly against each SOURCE file's
    # schema, before any copying is attempted -- not inferred from a copy
    # step's success/failure. Only the side camera (cam1) is checked/used --
    # see SIDE_CAMERA's comment for how cam0 vs cam1 was identified.
    print(f"\n(c) Camera data check (source files, side camera = {SIDE_CAMERA!r}):")
    camera_presence = {}
    for i, path in enumerate(episodes, start=1):
        if args.no_cameras:
            has_side = False
        else:
            has_cam0, has_cam1 = ru.episode_camera_entities(path)
            has_side = has_cam1 if SIDE_CAMERA == "cam1" else has_cam0
        camera_presence[i] = has_side
        print(f"  take_{i} ({path.name}): {SIDE_CAMERA}={'present' if has_side else 'MISSING'}")

    safe_name = args.task_or_key.replace(" ", "_").replace("(", "").replace(")", "")

    # rr.save() must be called *before* any logging (its docstring: "Call
    # this before you log any data!") -- otherwise events sit in an
    # unbounded pre-sink buffer, and a run this large (up to 5 takes x
    # ~90MB of camera frames each) can silently lose early-logged data
    # once a sink finally attaches at the end. init -> save -> log.
    rr.init(application_id=COMPARE_APP_ID, recording_id=f"compare-{args.task_or_key}", spawn=False)
    output_path = args.output or (ru.OUTPUT_DIR / f"compare_{safe_name}_takes_v2.rrd")
    ru.OUTPUT_DIR.mkdir(exist_ok=True)
    rr.save(str(output_path))

    stats = []
    for i, path in enumerate(episodes, start=1):
        print(f"\nTake {i}: {path.name}")
        s = log_take(
            i,
            path,
            args.prominence,
            args.height,
            debounce_seconds=args.debounce_seconds,
            debounce_strategy=args.debounce_strategy,
            has_side_camera=camera_presence[i],
            width_rel_height=args.width_rel_height,
        )
        stats.append(s)
        print(f"  task={s['task']!r}  color={s['color_name']}  strikes={s['n_strikes']}  magnitude={_fmt(s['magnitude'])}")

    panel_positions = ru.schematic_panel_positions()
    layout = ru.build_xylophone_layout()  # cached; just used here for key -> color
    circle_report = log_mini_panels(stats, layout, panel_positions)  # sets s["mean_intensity"] per take

    verify_panel_independence(stats)

    print(
        "\nTimeline note: mini-panel impact circles are now timeline-gated (NOT "
        "static) -- each is logged only across its own [trough_start_sync, "
        "trough_end_sync] window plus a short decay tail, on the exact same "
        "elapsed-since-first-strike scale already used for the camera row's "
        "playback sync below. Scrubbing from before t=0 to after t=0 should show "
        "each circle go invisible -> visible/colored -> faded out, roughly in step "
        "with its own camera reaching the strike moment (each take's true trough "
        "timing relative to its own first-strike t=0 varies slightly, so the exact "
        "on/off instants differ per take -- that's expected, not a bug)."
    )

    print("\nComparison-view circle check -- per-take color (not key color), reduced alpha, take label, timeline window:")
    for row in circle_report:
        print(
            f"  Take {row['take']}: key={row['key_id']!r}  color={row['color_name']}={row['circle_color']}  "
            f"alpha={row['circle_alpha']} ({row['circle_alpha'] / 255:.0%})  label='Take {row['take']}'  "
            f"position={[round(float(x), 3) for x in row['position']]}  "
            f"window=[{row['trough_start_sync']:.3f}s, {row['trough_end_sync']:.3f}s] (relative to this take's own t=0)"
        )

    bar_values = log_bar_chart(stats)
    print("\nBar chart data -- ONE entity, ONE array (not 5 separate single-value charts):")
    print(f"  /bar_chart/intensity_comparison  values={[round(v, 3) for v in bar_values]}")
    print(f"  colors (by bar index, Take order): {[s['color_name'] for s in stats]}")

    log_legend(stats)
    log_metrics(stats)

    print("\n--- Consistency summary ---")
    counts = [s["n_strikes"] for s in stats]
    print(f"Strike counts per take: {counts}  (stdev={np.std(counts):.2f})")
    print(f"Intensities per take: {[round(s['mean_intensity'], 2) for s in stats]}")

    print(f"\nSaved {output_path}")

    camera_flags = [s["has_side_camera"] for s in stats]
    blueprint_path = args.blueprint_output or (ru.OUTPUT_DIR / f"blueprint_compare_{safe_name}_v2.rbl")
    build_compare_takes_blueprint(
        n_takes=len(stats),
        take_labels=[f"Take {i} ({s['color_name']})" for i, s in enumerate(stats, start=1)],
        camera_flags=camera_flags,
    ).save(COMPARE_APP_ID, str(blueprint_path))
    print(f"Saved {blueprint_path} (application_id={COMPARE_APP_ID!r}, {len(stats)} take row(s))")

    print("\n(c) Camera panel row container type -- verified against the SAVED blueprint file, not just construction code:")
    verify_blueprint_containers(str(blueprint_path))


if __name__ == "__main__":
    main()
