"""
Loop over every .rrd file in data/ and print a table of
episode filename -> /task TextDocument text.

Usage:
    python scripts/dump_all_tasks.py
"""

from pathlib import Path

from rerun.experimental import RrdReader

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def read_task(path: Path) -> str | None:
    reader = RrdReader(path)
    recordings = reader.recordings()
    if not recordings:
        return None

    store = reader.store(store=recordings[0]).stream().collect()
    if "/task" not in store.schema().entity_paths():
        return None

    df = store.reader(index=None, contents="/task").to_pandas()
    text_cols = [c for c in df.columns if c.endswith(":TextDocument:text")]
    if not text_cols or df.empty:
        return None

    value = df[text_cols[0]].iloc[0]
    return value[0] if hasattr(value, "__len__") and not isinstance(value, str) else value


def main() -> None:
    paths = sorted(DATA_DIR.glob("*.rrd"))
    if not paths:
        print(f"No .rrd files found in {DATA_DIR}.")
        return

    rows: list[tuple[str, str]] = []
    for path in paths:
        task = read_task(path)
        rows.append((path.name, task if task is not None else "<no /task found>"))

    name_width = max(len(name) for name, _ in rows)
    for name, task in rows:
        print(f"{name.ljust(name_width)}  ->  {task}")

    print(f"\n{len(rows)} episodes, {len({t for _, t in rows})} distinct task strings.")


if __name__ == "__main__":
    main()
