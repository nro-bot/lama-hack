"""
Inspect Rerun .rrd recording(s) and print their entity tree, components, and
schema so we can confirm exact entity paths before writing analysis code.

Usage:
    python scripts/inspect_data.py                  # scans data/*.rrd
    python scripts/inspect_data.py path/to/file.rrd  # inspects one or more files

Handles both:
  - a single .rrd containing multiple recordings/episodes (each is inspected)
  - multiple per-episode .rrd files dropped into data/
"""

import sys
from pathlib import Path

from rerun.experimental import RrdReader

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def inspect_store(reader: RrdReader, entry) -> None:
    print(
        f"\n  Recording: application_id={entry.application_id!r} "
        f"recording_id={entry.recording_id!r}"
    )

    store = reader.store(store=entry)
    schema = store.schema()

    print(f"  Chunks: {len(store)}")

    print("\n  -- Entity paths --")
    entity_paths = schema.entity_paths()
    if not entity_paths:
        print("    (none found)")
    for entity_path in sorted(entity_paths):
        print(f"    {entity_path}")

    print("\n  -- Entity paths & components --")
    by_entity: dict[str, list[str]] = {}
    for column in schema.component_columns():
        by_entity.setdefault(column.entity_path, []).append(column.component)

    for entity_path in sorted(by_entity):
        print(f"    {entity_path}")
        for component in sorted(set(by_entity[entity_path])):
            print(f"        - {component}")

    print("\n  -- Archetypes logged --")
    for archetype in schema.archetypes():
        print(f"    {archetype}")

    print("\n  -- Index / timeline columns --")
    for column in schema.index_columns():
        print(f"    {column.name}")


def inspect_rrd(path: Path) -> None:
    print(f"\n{'=' * 80}")
    print(f"File: {path}")
    print(f"{'=' * 80}")

    reader = RrdReader(path)
    recordings = reader.recordings()

    if not recordings:
        print("  (no recording stores found in this file)")
        return

    print(f"Found {len(recordings)} recording(s) in this file.")
    for entry in recordings:
        inspect_store(reader, entry)


def find_rrd_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.rrd"))


def main() -> None:
    args = sys.argv[1:]

    if args:
        paths = [Path(arg) for arg in args]
    else:
        paths = find_rrd_files()

    if not paths:
        print(f"No .rrd file given and none found in {DATA_DIR}.")
        print("Drop one or more .rrd files into data/, or pass a path as an argument:")
        print("  python scripts/inspect_data.py path/to/recording.rrd")
        sys.exit(1)

    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"File not found: {p}")
        sys.exit(1)

    print(f"Found {len(paths)} .rrd file(s) to inspect.")
    for path in paths:
        inspect_rrd(path)


if __name__ == "__main__":
    main()
