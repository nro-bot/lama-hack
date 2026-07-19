"""
Shared helpers for the SO-100 xylophone strike-detection / annotation scripts.

Used by annotate_episode.py, compare_takes.py, and make_blueprint.py.
Kept separate from explore_episode.py / dump_all_tasks.py so those two remain
fully standalone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rerun.experimental import RrdReader
from scipy.signal import find_peaks, peak_widths
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

# --- Loading ---------------------------------------------------------------


def load_store_and_entry(path: Path):
    """Load a single-episode .rrd's first recording into a fully materialized store.

    Returns (store, entry) where entry.application_id / entry.recording_id
    identify the original recording (needed to write a companion .rrd that
    the Rerun Viewer will merge with the original).
    """
    reader = RrdReader(path)
    recordings = reader.recordings()
    if not recordings:
        raise ValueError(f"No recording stores found in {path}")
    entry = recordings[0]
    store = reader.store(store=entry).stream().collect()
    return store, entry


def scalar_series_df(store, entity_path: str, index: str = "time") -> pd.DataFrame:
    """Query a Scalars+SeriesLines entity into a tidy per-joint DataFrame."""
    df = store.reader(index=index, contents=entity_path).to_pandas()

    scalar_cols = [c for c in df.columns if c.endswith(":Scalars:scalars")]
    if not scalar_cols:
        raise ValueError(f"No Scalars component found at {entity_path}")
    scalar_col = scalar_cols[0]

    names_cols = [c for c in df.columns if c.endswith(":SeriesLines:names")]
    names = None
    if names_cols:
        non_null_names = df[names_cols[0]].dropna()
        if len(non_null_names):
            names = list(non_null_names.iloc[0])

    values = df[scalar_col]
    n_joints = len(values.iloc[0])
    if names is None:
        names = [f"joint_{i}" for i in range(n_joints)]

    out = pd.DataFrame(values.tolist(), columns=names)
    out[index] = df[index].values
    return out.set_index(index).sort_index()


def read_task_text(store) -> str | None:
    if "/task" not in store.schema().entity_paths():
        return None
    df = store.reader(index=None, contents="/task").to_pandas()
    text_cols = [c for c in df.columns if c.endswith(":TextDocument:text")]
    if not text_cols or df.empty:
        return None
    value = df[text_cols[0]].iloc[0]
    return value[0] if hasattr(value, "__len__") and not isinstance(value, str) else value


def read_task_text_fast(path: Path) -> str | None:
    """Read only /task from a file without materializing camera/joint chunks.

    Much cheaper than load_store_and_entry() when scanning many episodes just
    to check the task label (used by the xylophone-layout calibration scan).
    """
    reader = RrdReader(path)
    recordings = reader.recordings()
    if not recordings:
        return None
    entry = recordings[0]
    store = reader.store(store=entry).stream().filter(content="/task").collect()
    return read_task_text(store)


def episode_camera_entities(path: Path) -> tuple[bool, bool]:
    """Cheaply check whether a SOURCE episode has /camera/cam0 and
    /camera/cam1 entities at all, without materializing any chunk data
    (LazyStore.schema() reads only the manifest). Meant to be checked
    BEFORE attempting any camera copy, so a missing-camera take is a
    verified fact about the source file, not inferred from a copy failure.

    Returns (has_cam0, has_cam1).
    """
    reader = RrdReader(path)
    recordings = reader.recordings()
    if not recordings:
        return False, False
    lazy_store = reader.store(store=recordings[0])  # no .stream().collect() -- schema only
    entities = set(lazy_store.schema().entity_paths())
    return "/camera/cam0" in entities, "/camera/cam1" in entities


def elapsed_seconds(index: pd.Index) -> pd.Series:
    """Convert a timedelta/datetime index to elapsed seconds from the first sample."""
    delta = index - index[0]
    return pd.Series(delta.total_seconds(), index=index)


# --- Task label parsing ------------------------------------------------------

TASK_PATTERN = re.compile(r"^Hitting note (?:(big|small) )?([A-G]) \(([a-z ]+)\)$")

COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (214, 39, 40),
    "orange": (255, 127, 14),
    "yellow": (219, 190, 30),
    "green": (44, 160, 44),
    "blue": (31, 119, 180),
    "dark blue": (23, 60, 120),
    "purple": (148, 103, 189),
}

# Canonical left-to-right ordering for the xylophone bar panel (one octave).
KEY_ORDER = ["small C", "D", "E", "F", "G", "A", "B", "big C"]


def parse_task(task_text: str | None) -> tuple[str, str] | None:
    """Parse a /task string into (key_id, color_name), e.g. ('big C', 'red').

    Prints a warning and returns None if task_text is missing or doesn't
    match the expected "Hitting note [big|small] <LETTER> (<color>)" pattern
    -- this is how we catch label format variants early.
    """
    if not task_text:
        print("WARNING: episode has no /task text")
        return None
    match = TASK_PATTERN.match(task_text.strip())
    if not match:
        print(f"WARNING: task string does not match expected pattern: {task_text!r}")
        return None
    size, letter, color = match.groups()
    key_id = f"{size} {letter}" if size else letter
    if color not in COLOR_RGB:
        print(f"WARNING: unrecognized color {color!r} in task string: {task_text!r}")
    return key_id, color


def key_color_rgb(color_name: str) -> tuple[int, int, int]:
    return COLOR_RGB.get(color_name, (150, 150, 150))


# --- Xylophone panel: schematic (visual) layout, distinct from the noisy
# calibrated positions used for the geometric mismatch cross-check --------

# The calibrated per-key strike positions (build_xylophone_layout) are too
# close together and too noisy to place non-overlapping boxes directly (see
# that function's docstring) -- confirmed by inspecting the follower jaw's
# actual position across several episodes/keys: z stays in [0.0, 0.31] for
# every link in every episode checked (base is exactly z=0, arm only ever
# extends upward), while the panel needs to sit clear of that. So the panel
# uses a separate, deliberately schematic, evenly-spaced layout offset well
# below the arm's reach volume; it is NOT meant to reflect the xylophone's
# true physical position.
PANEL_Z = -0.08  # below the base (arm never goes below z=0)
PANEL_Y = -0.15  # roughly centered under the arm's forward reach
PANEL_X_STEP = 0.08  # >> 2 * box half-width (0.015), so boxes can't touch


def schematic_panel_positions(
    key_order: list[str] = KEY_ORDER, z: float = PANEL_Z, y: float = PANEL_Y, x_step: float = PANEL_X_STEP
) -> dict[str, list[float]]:
    """Evenly-spaced, non-overlapping key positions for the visual panel,
    offset below the arm's workspace. Centered on x=0."""
    n = len(key_order)
    center = (n - 1) / 2
    return {key: [(i - center) * x_step, y, z] for i, key in enumerate(key_order)}


def comparison_ring_position(
    key_position, take_idx: int, n_takes: int, ring_radius: float = 0.02, z_lift: float = 0.012
) -> list[float]:
    """Position for one take's comparison circle: arranged in a small ring
    above a key's panel box (take_idx is 1-based), so up to 5 takes at the
    SAME key are each visible instead of fully overlapping at one point."""
    angle = 2 * np.pi * (take_idx - 1) / max(n_takes, 1)
    x, y, z = key_position
    return [x + ring_radius * np.cos(angle), y + ring_radius * np.sin(angle), z + z_lift]


# --- Forward kinematics: base -> jaw ------------------------------------------

FK_ENTITY = "/follower/joint_transforms"
FK_BASE_FRAME = "follower/base"
FK_TIP_FRAME = "follower/jaw"


def fk_tip_positions(
    store,
    entity_path: str = FK_ENTITY,
    base_frame: str = FK_BASE_FRAME,
    tip_frame: str = FK_TIP_FRAME,
) -> pd.DataFrame:
    """Compute tip_frame's 3D position in base_frame at each timestamp.

    /follower/joint_transforms logs one (child_frame, parent_frame,
    quaternion, translation) row per link per frame -- 6 links/frame in this
    dataset (base->shoulder->upper_arm->lower_arm->wrist->gripper->jaw), all
    sharing the same `time` value. Querying with index="time" silently
    collapses same-timestamp rows via latest-at semantics (keeping only 1 of
    6); index="log_time" (unique per row) avoids that, and `time` survives
    as a plain column we group by to reassemble each frame's full chain.

    Returns a DataFrame indexed by `time` with columns x, y, z, sorted by time.
    Frames where the chain doesn't fully reach base_frame are skipped.
    """
    df = store.reader(index="log_time", contents=entity_path).to_pandas()

    child_col = f"{entity_path}:Transform3D:child_frame"
    parent_col = f"{entity_path}:Transform3D:parent_frame"
    quat_col = f"{entity_path}:Transform3D:quaternion"
    trans_col = f"{entity_path}:Transform3D:translation"

    df["child_frame"] = df[child_col].apply(lambda a: a[0])
    df["parent_frame"] = df[parent_col].apply(lambda a: a[0])
    df["quat"] = df[quat_col].apply(lambda a: np.asarray(a[0], dtype=float))
    df["trans"] = df[trans_col].apply(lambda a: np.asarray(a[0], dtype=float))

    times: list = []
    positions: list = []
    for t, group in df.groupby("time", sort=True):
        links = {
            c: (p, tr, q)
            for c, p, tr, q in zip(
                group["child_frame"], group["parent_frame"], group["trans"], group["quat"]
            )
        }

        chain: list = []
        frame = tip_frame
        while frame != base_frame and frame in links:
            parent, trans, quat = links[frame]
            chain.append((trans, quat))
            frame = parent
        if frame != base_frame:
            continue  # chain didn't reach base_frame at this timestamp; skip

        transform = np.eye(4)
        for trans, quat in reversed(chain):
            link = np.eye(4)
            link[:3, :3] = Rotation.from_quat(quat).as_matrix()
            link[:3, 3] = trans
            transform = transform @ link
        positions.append(transform[:3, 3])
        times.append(t)

    return pd.DataFrame(
        positions, columns=["x", "y", "z"], index=pd.Index(times, name="time")
    ).sort_index()


def nearest_position(fk_df: pd.DataFrame, t) -> np.ndarray:
    """The fk_df row's (x, y, z) closest in time to t."""
    idx = fk_df.index.get_indexer([t], method="nearest")[0]
    return fk_df.iloc[idx][["x", "y", "z"]].to_numpy(dtype=float)


# --- Strike detection ----------------------------------------------------------


def _cluster_by_gap(times_ns: np.ndarray, debounce_seconds: float) -> list[list[int]]:
    """Group indices (already time-sorted) into clusters, starting a new
    cluster whenever the gap since the PREVIOUS peak exceeds debounce_seconds.
    This is a simple chain/burst clustering: peaks that ring/echo in quick
    succession (each close to the last) all fall into one cluster even if the
    cluster's first and last member are individually more than
    debounce_seconds apart."""
    clusters = [[0]]
    for i in range(1, len(times_ns)):
        gap_s = (times_ns[i] - times_ns[i - 1]) / 1e9
        if gap_s < debounce_seconds:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def _frac_index_to_time(index: pd.DatetimeIndex, frac_idx: np.ndarray) -> pd.DatetimeIndex:
    """Convert scipy peak_widths' fractional sample positions (e.g. 12.3 =
    30% of the way from sample 12 to sample 13) into interpolated
    timestamps, by linear interpolation on the underlying int64 ns values."""
    time_ns = index.to_numpy().astype("int64").astype(float)
    i0 = np.clip(np.floor(frac_idx).astype(int), 0, len(time_ns) - 1)
    i1 = np.clip(i0 + 1, 0, len(time_ns) - 1)
    frac = frac_idx - i0
    interpolated_ns = (time_ns[i0] * (1 - frac) + time_ns[i1] * frac).astype("int64")
    return pd.to_datetime(interpolated_ns)


def detect_strikes(
    load: pd.Series,
    *,
    height: float | None = None,
    prominence: float = 15.0,
    debounce_seconds: float = 0.3,
    debounce_strategy: str = "first",
    max_strikes: int | None = 1,
    width_rel_height: float = 1.0,
    max_width_samples: float | None = None,  # NEW PARAMETER
) -> pd.DataFrame:
    
    values = load.to_numpy()
    trough_depth = -values  # Reverted: peaks of this = troughs of the signed load

    # Construct arguments for find_peaks
    kwargs = {"height": height, "prominence": prominence}
    if max_width_samples is not None:
        kwargs["width"] = (None, max_width_samples) # Filters out peaks wider than this

    peak_idx, props = find_peaks(trough_depth, **kwargs)

    if len(peak_idx) > 0:
        _widths, _width_heights, left_ips, right_ips = peak_widths(
            trough_depth,
            peak_idx,
            rel_height=width_rel_height,
            prominence_data=(props["prominences"], props["left_bases"], props["right_bases"]),
        )
        trough_start = _frac_index_to_time(load.index, left_ips)
        trough_end = _frac_index_to_time(load.index, right_ips)
    else:
        trough_start = pd.DatetimeIndex([])
        trough_end = pd.DatetimeIndex([])

    strikes = pd.DataFrame(
        {
            "time": load.index[peak_idx],
            "value": values[peak_idx],
            "magnitude": trough_depth[peak_idx], # Reverted back to trough_depth
            "prominence": props["prominences"],
            "trough_start": trough_start,
            "trough_end": trough_end,
        }
    ).sort_values("time").reset_index(drop=True)

    if debounce_seconds > 0 and len(strikes) > 1:
        times_ns = strikes["time"].to_numpy().astype("int64")
        clusters = _cluster_by_gap(times_ns, debounce_seconds)

        if debounce_strategy == "max":
            magnitudes = strikes["magnitude"].to_numpy()
            keep = [cluster[int(np.argmax(magnitudes[cluster]))] for cluster in clusters]
        else:
            keep = [cluster[0] for cluster in clusters]

        strikes = strikes.iloc[keep].reset_index(drop=True)

    if max_strikes is not None:
        strikes = strikes.iloc[:max_strikes].reset_index(drop=True)

    return strikes

    return strikes.iloc[keep].reset_index(drop=True)


def intensity_scores(strikes: pd.DataFrame, normalize_against: float | None = None) -> np.ndarray:
    """0-1 intensity per strike from magnitude, normalized against either a
    supplied global reference max or (default) this episode's own max."""
    if strikes.empty:
        return np.array([])
    ref_max = normalize_against if normalize_against is not None else strikes["magnitude"].max()
    if ref_max <= 0:
        return np.zeros(len(strikes))
    return np.clip(strikes["magnitude"].to_numpy() / ref_max, 0.0, 1.0)


# --- Xylophone key-panel calibration --------------------------------------------


def find_representative_episodes(data_dir: Path = DATA_DIR, n_per_key: int = 6) -> dict[str, list[Path]]:
    """Scan data/*.rrd for up to n_per_key episodes matching each KEY_ORDER key.

    Uses the fast /task-only reader, so this only pays the cost of a full
    episode load for the (small) set of representative episodes it selects,
    not for every file scanned.

    A single episode's strike positions turn out to be too noisy (stdev
    comparable to the physical spacing between keys) to calibrate a key's
    position reliably, so calibration averages over several episodes per key.
    """
    found: dict[str, list[Path]] = {key: [] for key in KEY_ORDER}
    for path in sorted(data_dir.glob("*.rrd")):
        if all(len(v) >= n_per_key for v in found.values()):
            break
        task_text = read_task_text_fast(path)
        parsed = parse_task(task_text) if task_text else None
        if parsed is None:
            continue
        key_id, _color = parsed
        if key_id in found and len(found[key_id]) < n_per_key:
            found[key_id].append(path)
    return found


def build_xylophone_layout(
    cache_path: Path | None = None,
    force: bool = False,
    prominence: float = 15.0,
    n_episodes_per_key: int = 6,
    debounce_seconds: float = 0.3,
    debounce_strategy: str = "first",
) -> dict[str, dict]:
    """Calibrate each xylophone key's 3D bar position by averaging strike
    positions across several representative episodes of that key. Cached to
    output/xylophone_layout.json.

    NOTE on reliability: a single episode's strikes land within a ~3-5cm
    stdev per axis (the mallet's approach angle/depth varies strike to
    strike), which is comparable to or larger than the physical distance
    between neighboring keys. Averaging over n_episodes_per_key episodes
    (each contributing several strikes) reduces this via the usual sqrt(N)
    averaging, but with only ~6 episodes worth of strikes per key, some
    residual position noise/overlap between adjacent keys should be expected
    -- this is diagnosed by build_xylophone_layout's printed inter-key vs
    intra-key spread, not swept under the rug.

    Returns {key_id: {"color": str, "position": [x, y, z], "n_strikes": int}}.
    """
    cache_path = cache_path or (OUTPUT_DIR / "xylophone_layout.json")
    if cache_path.exists() and not force:
        with open(cache_path) as f:
            return json.load(f)

    print(f"Calibrating xylophone key layout from up to {n_episodes_per_key} episodes/key (one-time; cached afterward)...")
    representative = find_representative_episodes(n_per_key=n_episodes_per_key)
    missing = [key for key, paths in representative.items() if not paths]
    if missing:
        print(f"WARNING: no example episodes found for keys: {missing}")

    layout: dict[str, dict] = {}
    all_strike_positions: dict[str, np.ndarray] = {}
    for key_id in KEY_ORDER:
        paths = representative.get(key_id, [])
        if not paths:
            continue
        color = None
        positions_for_key: list[np.ndarray] = []
        for path in paths:
            store, _entry = load_store_and_entry(path)
            task_text = read_task_text(store)
            parsed = parse_task(task_text)
            color = parsed[1] if parsed else color

            load_df = scalar_series_df(store, "/follower/load")
            if "elbow_flex" not in load_df.columns:
                print(f"  WARNING: 'elbow_flex' not in joints {list(load_df.columns)} ({path.name}); skipping this episode")
                continue

            strikes = detect_strikes(
                load_df["elbow_flex"],
                prominence=prominence,
                debounce_seconds=debounce_seconds,
                debounce_strategy=debounce_strategy,
            )
            if strikes.empty:
                continue

            fk_df = fk_tip_positions(store)
            positions_for_key.append(np.stack([nearest_position(fk_df, t) for t in strikes["time"]]))

        if not positions_for_key:
            print(f"  WARNING: no strikes found across {len(paths)} calibration episode(s) for {key_id}")
            continue

        combined = np.concatenate(positions_for_key, axis=0)
        all_strike_positions[key_id] = combined
        print(
            f"  {key_id!r} <- {len(paths)} episode(s), {len(combined)} strikes, "
            f"stdev={combined.std(axis=0).round(3).tolist()}"
        )
        layout[key_id] = {"color": color or "gray", "position": combined.mean(axis=0).tolist(), "n_strikes": len(combined)}

    _report_layout_separation(layout, all_strike_positions)
    layout["_dominant_axis"] = dominant_separating_axis(layout)
    axis_name = "xyz"[layout["_dominant_axis"]]
    print(
        f"Dominant separating axis: {axis_name!r} (strike approach on the other two axes is noisier than "
        "key-to-key spacing, so nearest-key matching below only compares along this axis)"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"Saved calibrated layout to {cache_path}")
    return layout


def dominant_separating_axis(layout: dict[str, dict]) -> int:
    """Axis (0=x, 1=y, 2=z) along which the calibrated key centroids are
    most spread out. Strike-position noise from mallet approach angle/depth
    is concentrated on the other two axes (see build_xylophone_layout's
    docstring), so nearest-key matching should only compare along this one."""
    centroids = np.array([info["position"] for key, info in layout.items() if not key.startswith("_")])
    return int(np.argmax(centroids.std(axis=0)))


def nearest_layout_key(layout: dict[str, dict], position: np.ndarray) -> str | None:
    """The calibrated key whose position is closest to `position`, comparing
    only along the dominant separating axis (see dominant_separating_axis)."""
    keys = [k for k in layout if not k.startswith("_")]
    if not keys:
        return None
    axis = layout.get("_dominant_axis", dominant_separating_axis(layout))
    best_key, best_dist = None, float("inf")
    for key_id in keys:
        dist = abs(layout[key_id]["position"][axis] - position[axis])
        if dist < best_dist:
            best_key, best_dist = key_id, dist
    return best_key


def _report_layout_separation(layout: dict[str, dict], all_strike_positions: dict[str, np.ndarray]) -> None:
    """Print inter-key distance vs intra-key strike spread, so it's clear
    whether the calibrated layout can actually discriminate between keys."""
    keys = list(layout.keys())
    if len(keys) < 2:
        return
    inter_dists = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = np.linalg.norm(np.array(layout[keys[i]]["position"]) - np.array(layout[keys[j]]["position"]))
            inter_dists.append(d)
    intra_stdevs = [np.linalg.norm(pos.std(axis=0)) for pos in all_strike_positions.values()]
    print(
        f"Layout diagnostics: inter-key center distance min={min(inter_dists):.3f} "
        f"mean={np.mean(inter_dists):.3f}  |  per-episode intra-key strike stdev mean={np.mean(intra_stdevs):.3f} "
        "(units match /follower/joint_transforms translation, likely meters)"
    )
    if min(inter_dists) < np.mean(intra_stdevs):
        print(
            "WARNING: smallest inter-key distance is below typical strike-position noise -- "
            "nearest-key geometric matching will be unreliable for closely-spaced keys."
        )


# --- Decay animation (impact ripple / key highlight fade) ----------------------


def decay_schedule(n_frames: int) -> list[float]:
    """Fractions in [0, 1] (0 = strike moment, 1 = end of decay) for n_frames steps."""
    if n_frames <= 1:
        return [0.0]
    return [i / (n_frames - 1) for i in range(n_frames)]
