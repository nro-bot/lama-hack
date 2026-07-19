"""
Reads all episodes in data/, copies their cameras and strike annotations, 
and stitches them sequentially into a single continuous Rerun timeline.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rerun as rr

import rrd_utils as ru
from compare_takes_v2 import copy_camera, SIDE_CAMERA

# Import only constants and math helpers, NO logging functions
from annotate_episode_v2 import (
    MIN_RADIUS_DEFAULT, MAX_RADIUS_DEFAULT, DECAY_FRAMES_DEFAULT, 
    KEY_BOX_HALF_SIZE, BASE_CENTER, BASE_HALF_SIZE, BASE_COLOR,
    glow_color, lerp_color, sanitize_key
)

def log_strike_visuals_sync(frame_times, trough_start, trough_end, impact_position, key_id, key_position, base_rgb, intensity):
    """A duration-synced version of the impact animation."""
    radius0 = MIN_RADIUS_DEFAULT + intensity * (MAX_RADIUS_DEFAULT - MIN_RADIUS_DEFAULT)
    glow = glow_color(base_rgb)
    key_entity = f"/annotations/xylophone_keys/{sanitize_key(key_id)}"

    persist_start_idx = frame_times.searchsorted(trough_start)
    persist_end_idx = frame_times.searchsorted(trough_end, side="right")
    persist_times = frame_times[persist_start_idx:persist_end_idx]
    if len(persist_times) == 0:
        persist_times = [trough_start]

    for t in persist_times:
        rr.set_time("time", duration=float(t))
        rr.log("/annotations/impact", rr.Points3D(positions=[impact_position], radii=[radius0], colors=[(*base_rgb, 255)]))
        rr.log(key_entity, rr.Boxes3D(centers=[key_position], half_sizes=[KEY_BOX_HALF_SIZE], colors=[(*glow, 255)]))

    decay_times = frame_times[persist_end_idx : persist_end_idx + DECAY_FRAMES_DEFAULT]
    fractions = ru.decay_schedule(len(decay_times))
    for t, frac in zip(decay_times, fractions):
        radius = radius0 * (1.0 - 0.85 * frac)
        alpha = int(255 * (1.0 - frac) ** 1.5)
        key_color = lerp_color(glow, base_rgb, frac)
        rr.set_time("time", duration=float(t))
        rr.log("/annotations/impact", rr.Points3D(positions=[impact_position], radii=[radius], colors=[(*base_rgb, alpha)]))
        rr.log(key_entity, rr.Boxes3D(centers=[key_position], half_sizes=[KEY_BOX_HALF_SIZE], colors=[(*key_color, 255)]))

def main():
    episodes = sorted(ru.DATA_DIR.glob("*.rrd"))
    if not episodes:
        print("No episodes found.")
        sys.exit(1)

    print(f"Stitching {len(episodes)} episodes into a continuous stream...")

    rr.init(application_id="so-100-stream", recording_id="continuous_highlight_reel", spawn=False)
    output_path = ru.OUTPUT_DIR / "continuous_stream.rrd"
    ru.OUTPUT_DIR.mkdir(exist_ok=True)
    rr.save(str(output_path))

    layout = ru.build_xylophone_layout()
    panel_positions = ru.schematic_panel_positions()

    # Log base and panel once using static=True so they exist permanently on the timeline
    rr.log("/annotations/xylophone_base", rr.Boxes3D(centers=[BASE_CENTER], half_sizes=[BASE_HALF_SIZE], colors=[(*BASE_COLOR, 255)]), static=True)
    for key_id, info in layout.items():
        if key_id.startswith("_") or key_id not in panel_positions:
            continue
        rgb = ru.key_color_rgb(info["color"])
        entity = f"/annotations/xylophone_keys/{sanitize_key(key_id)}"
        rr.log(entity, rr.Boxes3D(centers=[panel_positions[key_id]], half_sizes=[KEY_BOX_HALF_SIZE], colors=[(*rgb, 255)], labels=[key_id]), static=True)

    global_offset_seconds = 0.0

    for i, path in enumerate(episodes, start=1):
        print(f"\nProcessing [{i}/{len(episodes)}]: {path.name}")
        store, _ = ru.load_store_and_entry(path)
        
        task_text = ru.read_task_text(store)
        parsed = ru.parse_task(task_text)
        task_key_id = parsed[0] if parsed else None
        task_rgb = ru.key_color_rgb(parsed[1]) if parsed else (150, 150, 150)

        load_df = ru.scalar_series_df(store, "/follower/load")
        if "elbow_flex" not in load_df.columns or load_df.empty:
            continue

        zero_t = load_df.index[0]
        
        has_cam0, has_cam1 = ru.episode_camera_entities(path)
        if has_cam1:
            copy_camera("", store, SIDE_CAMERA, zero_t - pd.Timedelta(seconds=global_offset_seconds))

# Look for narrow, deep troughs
        strikes = ru.detect_strikes(
            load_df["elbow_flex"],
            prominence=10.0,              # Extremely low to guarantee the small key hit passes
            max_width_samples=4.0,       # Narrow enough to reject the initial slow motor drop
            debounce_seconds=1,        # INCREASED: Forces the key and table hit into the same cluster
            debounce_strategy="second",   # Keeps the chronological first hit (the key), erases the table
            width_rel_height=1         
        )

        scores = ru.intensity_scores(strikes)
        for t_abs, value in load_df["elbow_flex"].items():
            t_sync = (t_abs - zero_t).total_seconds() + global_offset_seconds
            rr.set_time("time", duration=float(t_sync))
            rr.log("/annotations/elbow_flex_load", rr.Scalars([value]))

        impact_display_position = None
        if task_key_id in panel_positions:
            impact_display_position = np.array(panel_positions[task_key_id]) + np.array([0.0, 0.0, 0.012])

        for score, row in zip(scores, strikes.itertuples()):
            t_start = (row.trough_start - zero_t).total_seconds() + global_offset_seconds
            t_end = (row.trough_end - zero_t).total_seconds() + global_offset_seconds
            t_strike = (row.time - zero_t).total_seconds() + global_offset_seconds

            synth_frames = np.linspace(t_start, t_end + 0.5, num=30)

            if impact_display_position is not None:
                log_strike_visuals_sync(
                    synth_frames, t_start, t_end, impact_display_position, task_key_id,
                    panel_positions[task_key_id], task_rgb, score
                )

            rr.set_time("time", duration=float(t_strike))
            rr.log(
                "/annotations/strike_markers",
                rr.Scalars([row.value]),
                rr.SeriesPoints(markers=["Cross"], marker_sizes=[8], colors=[task_rgb]),
            )

        episode_duration = (load_df.index[-1] - zero_t).total_seconds()
        global_offset_seconds += episode_duration + 1.0

    print(f"\nDone! Saved continuous stream to {output_path}")

if __name__ == "__main__":
    main()