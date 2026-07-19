"""
Explore a single SO-100 xylophone teleop episode (.rrd).

Loads /follower/{position,goal,load,current} as tidy per-joint pandas
DataFrames indexed on the `time` timeline, prints the /task label and basic
stats, and saves diagnostic plots to output/.

Usage:
    python scripts/explore_episode.py                       # first .rrd in data/
    python scripts/explore_episode.py data/episode_42.rrd
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rerun.experimental import RrdReader

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"


def find_default_episode() -> Path:
    candidates = sorted(DATA_DIR.glob("*.rrd"))
    if not candidates:
        print(f"No .rrd files found in {DATA_DIR}.")
        sys.exit(1)
    return candidates[0]


def load_store(path: Path):
    reader = RrdReader(path)
    recordings = reader.recordings()
    if not recordings:
        print(f"No recording stores found in {path}.")
        sys.exit(1)
    # SO-100 episode files are one episode per file, so the first recording is it.
    return reader.store(store=recordings[0]).stream().collect()


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
    schema_entities = store.schema().entity_paths()
    if "/task" not in schema_entities:
        return None
    df = store.reader(index=None, contents="/task").to_pandas()
    text_cols = [c for c in df.columns if c.endswith(":TextDocument:text")]
    if not text_cols or df.empty:
        return None
    value = df[text_cols[0]].iloc[0]
    return value[0] if hasattr(value, "__len__") and not isinstance(value, str) else value


def elapsed_seconds(index: pd.Index) -> pd.Series:
    """Convert a timedelta/datetime index to elapsed seconds from the first sample."""
    delta = index - index[0]
    return pd.Series(delta.total_seconds(), index=index)


def plot_load_and_current(load: pd.DataFrame, current: pd.DataFrame, out_path: Path) -> None:
    # Zip by joint order rather than by column name/row count: SeriesLines
    # names and sample counts aren't guaranteed to match exactly across entities.
    n_joints = min(load.shape[1], current.shape[1])
    t_load = elapsed_seconds(load.index)
    t_current = elapsed_seconds(current.index)

    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2.2 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        joint = load.columns[i]
        # load and current live on very different scales (e.g. load ~O(10),
        # current ~O(1000)); a shared axis flattens load to an invisible
        # line, so give each its own y-axis.
        ax.plot(t_load.values, load.iloc[:, i].values, color="tab:red", label="load")
        ax.set_ylabel(joint, rotation=0, ha="right", va="center", fontsize=9)
        ax.tick_params(axis="y", labelcolor="tab:red")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(t_current.values, current.iloc[:, i].values, color="tab:blue", alpha=0.7, label="current")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

    handles = [
        plt.Line2D([], [], color="tab:red", label="load"),
        plt.Line2D([], [], color="tab:blue", label="current"),
    ]
    axes[0].legend(handles=handles, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Follower load & current per joint (spikes = candidate mallet strikes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tracking_error(position: pd.DataFrame, goal: pd.DataFrame, out_path: Path) -> None:
    # Zip by joint order: /follower/goal's SeriesLines names carry a " goal"
    # suffix and its row count can differ slightly from /follower/position.
    n_joints = min(position.shape[1], goal.shape[1])
    t_position = elapsed_seconds(position.index)
    t_goal = elapsed_seconds(goal.index)

    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2.2 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        joint = position.columns[i]
        ax.plot(t_position.values, position.iloc[:, i].values, label="position", color="tab:green")
        ax.plot(
            t_goal.values, goal.iloc[:, i].values, label="goal", color="tab:orange", linestyle="--"
        )
        ax.set_ylabel(joint, rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Follower position vs goal per joint (tracking error)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_default_episode()
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Loading {path} ...")
    store = load_store(path)

    task_text = read_task_text(store)
    print(f"\nTask: {task_text!r}")

    load = scalar_series_df(store, "/follower/load")
    current = scalar_series_df(store, "/follower/current")
    position = scalar_series_df(store, "/follower/position")
    goal = scalar_series_df(store, "/follower/goal")

    n_frames = len(load)
    duration_s = elapsed_seconds(load.index).iloc[-1] if n_frames else 0.0
    variances = load.var().sort_values(ascending=False)

    print(f"Frames: {n_frames}")
    print(f"Duration: {duration_s:.2f} s")
    print("\nLoad variance per joint (highest = best strike-detection candidate):")
    print(variances.to_string())
    print(f"\nHighest-variance joint: {variances.index[0]}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = path.stem

    load_current_path = OUTPUT_DIR / f"{stem}_load_current.png"
    plot_load_and_current(load, current, load_current_path)
    print(f"\nSaved {load_current_path}")

    tracking_path = OUTPUT_DIR / f"{stem}_tracking_error.png"
    plot_tracking_error(position, goal, tracking_path)
    print(f"Saved {tracking_path}")


if __name__ == "__main__":
    main()
