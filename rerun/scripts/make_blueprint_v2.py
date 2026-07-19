"""
v2 of make_blueprint.py -- kept as a separate file so the original stays
untouched; used by annotate_episode_v2.py / compare_takes_v2.py. Changes
from v1: comparison-view impact circles are take-colored with reduced
alpha (not key-colored/opaque) and each mini panel is labeled with its
take number, not just the key letter.

Build Rerun Viewer blueprints for the xylophone episode demo.

Two variants:
  - single: cam0 + cam1 top row, a large 3D view (arm + impact circles +
    xylophone key panel) in the middle, and the elbow_flex load timeseries
    with strike markers at the bottom. Matches annotate_episode.py's output
    (application_id="so-100" -- shared with the original episode files).
  - compare: small multiples, not one shared 3D scene (a shared-scene
    version was tried and abandoned after repeated fix attempts -- tangled
    trajectories, floating "Take N" labels, missing camera panels). Layout,
    top to bottom: legend + metrics, a native BarChartView (ONE BarChart
    entity carrying all takes' intensities, not one entity per take), a row
    of N independent mini 3D panels (one per take, each scoped via
    Spatial3DView(origin=f"/take_{i}", contents="$origin/**") so a panel
    can only ever show that take's own geometry), then a camera panel row
    -- side camera (cam1) only, included only for takes whose SOURCE
    episode was confirmed (rrd_utils.episode_camera_entities) to actually
    have it, never a black placeholder for a stream with no data. Matches
    compare_takes.py's output (application_id="so-100-compare-takes").
    compare_takes.py builds and saves this variant itself (it knows the
    actual take count/labels/camera presence for the run); running this
    script standalone produces a generic N-take preview instead (with
    cameras, arbitrarily).

Two different explicit camera framings are used (auto-fit-to-scene is not
used anywhere in either blueprint, so framing doesn't jump around as
entity counts/marker sizes change):
  - scene_eye_controls(): frames the follower arm's full swing together
    with the single-episode xylophone panel (x:[-0.02,0.09] y:[-0.31,0.04]
    z:[0,0.31] for the arm, x:[-0.3,0.3] y=-0.15 z=-0.08 for the panel,
    measured/derived across several episodes).
  - panel_eye_controls(): frames ONLY a schematic key-row panel + one
    impact circle (no arm) -- used for compare's per-take mini panels, a
    much smaller and differently-shaped scene than scene_eye_controls was
    tuned for.

Usage:
    python scripts/make_blueprint.py single           # -> output/blueprint_single_episode.rbl
    python scripts/make_blueprint.py compare           # -> output/blueprint_compare_takes.rbl (generic 5-take preview)
    python scripts/make_blueprint.py compare --n-takes 3
    python scripts/make_blueprint.py both               # (default)

Then in the Rerun Viewer: open the blueprint .rbl alongside the matching
data file(s) (e.g. data/episode_04.rrd + output/episode_04_annotations.rrd
for "single", or output/compare_*_takes.rrd for "compare").
"""

import argparse

import rerun.blueprint as rrb

import rrd_utils as ru

SINGLE_APP_ID = "so-100"
COMPARE_APP_ID = "so-100-compare-takes"

# See module docstring for how these were derived.
SCENE_EYE_POSITION = (0.55, 0.5, 0.55)
SCENE_LOOK_TARGET = (0.0, -0.13, 0.11)
SCENE_EYE_UP = (0.0, 0.0, 1.0)

PANEL_EYE_POSITION = (0.0, 0.15, 0.35)
PANEL_LOOK_TARGET = (0.0, -0.15, -0.073)
PANEL_EYE_UP = (0.0, 0.0, 1.0)


def scene_eye_controls() -> rrb.EyeControls3D:
    return rrb.EyeControls3D(kind="Orbital", position=SCENE_EYE_POSITION, look_target=SCENE_LOOK_TARGET, eye_up=SCENE_EYE_UP)


def panel_eye_controls() -> rrb.EyeControls3D:
    return rrb.EyeControls3D(kind="Orbital", position=PANEL_EYE_POSITION, look_target=PANEL_LOOK_TARGET, eye_up=PANEL_EYE_UP)


def build_single_episode_blueprint() -> rrb.Blueprint:
    # Grid, not Horizontal: a Container that is a sibling of a bare View (not
    # wrapped in any container) within the same parent Vertical/Horizontal
    # gets mis-serialized as Tabs by the SDK (reproduced and confirmed by
    # reading saved .rbl files back -- see verify_blueprint_containers and
    # build_compare_takes_blueprint's docstring). arm_view/load_view below are
    # bare Views sitting directly in the root Vertical, so `cameras` -- if it
    # were a Horizontal -- would trigger exactly that. Grid does not have
    # this problem in any reproduction tried.
    cameras = rrb.Grid(
        rrb.Spatial2DView(origin="/camera/cam0", name="cam0"),
        rrb.Spatial2DView(origin="/camera/cam1", name="cam1"),
        grid_columns=2,
    )
    arm_view = rrb.Spatial3DView(
        origin="/",
        contents=[
            "+ /follower/**",
            "+ /annotations/impact",
            "+ /annotations/xylophone_keys/**",
        ],
        name="Arm + xylophone",
        eye_controls=scene_eye_controls(),
    )
    load_view = rrb.TimeSeriesView(
        origin="/annotations",
        contents=[
            "+ /annotations/elbow_flex_load",
            "+ /annotations/strike_markers",
        ],
        name="elbow_flex load (strikes marked)",
    )
    return rrb.Blueprint(
        rrb.Vertical(cameras, arm_view, load_view, row_shares=[1, 3, 1]),
        collapse_panels=True,
    )


def build_compare_takes_blueprint(
    n_takes: int = 5,
    take_labels: list[str] | None = None,
    camera_flags: list[bool] | None = None,
    verbose: bool = True,
) -> rrb.Blueprint:
    """Small multiples, top to bottom: legend + metrics, a native bar chart
    (intensity per take), a row of N independent mini 3D panels (one per
    take), then a camera panel row (side camera only) for takes with
    confirmed camera data.

    Each mini panel is a Spatial3DView with origin=f"/take_{i}" and
    contents="$origin/**" -- scoped to exactly that take's own entity
    subtree, so it is structurally impossible for one panel to render
    another take's geometry, regardless of what else is logged in the
    recording.

    Every row-like layout here uses rrb.Grid, never rrb.Horizontal --
    isolated, reproduced, and confirmed by reading SAVED .rbl files back
    (verify_blueprint_containers): a Container that sits as a *sibling of a
    bare View* (a View not wrapped in any container) within the same parent
    Vertical/Horizontal gets its ContainerBlueprint:container_kind silently
    stored as Tabs (2) instead of Horizontal (1), regardless of what the
    Python construction code says. This blueprint's root Vertical has
    exactly that shape (`bar_chart` is a bare BarChartView sitting between
    `top` and `mini_panel_row`), so both of those had to become Grid to
    come out correctly typed -- Grid never showed this problem in any
    reproduction. (An earlier, narrower theory -- that nesting a Container
    inside a Grid *cell* was the trigger -- turned out to be incomplete:
    Grid cells containing only bare Views, as used here, are unaffected
    either way.)

    camera_flags is a per-take [has_side_camera, ...] list (required from
    compare_takes.py, which checks the source files directly via
    rrd_utils.episode_camera_entities before calling this; defaults to no
    cameras for any take if omitted, e.g. the standalone generic preview).
    """
    if take_labels is None:
        take_labels = [f"Take {i}" for i in range(1, n_takes + 1)]

    # Grid everywhere below, never Horizontal: a Container sibling to a bare
    # View (not wrapped in a container) within the same parent gets
    # mis-serialized as Tabs (reproduced and confirmed against saved .rbl
    # files -- see this function's docstring). bar_chart is a bare
    # BarChartView sitting directly in the root Vertical alongside `top` and
    # `mini_panel_row`, which is exactly the trigger condition, so both of
    # those must be Grid (not Horizontal) to come out correctly typed.
    top = rrb.Grid(
        rrb.TextDocumentView(origin="/legend", name="Legend"),
        rrb.TextDocumentView(origin="/metrics", name="Metrics"),
        grid_columns=2,
    )

    bar_chart = rrb.BarChartView(origin="/bar_chart", contents="/bar_chart/**", name="Intensity comparison")

    mini_panels = []
    for i in range(1, n_takes + 1):
        label = take_labels[i - 1] if i - 1 < len(take_labels) else f"Take {i}"
        mini_panels.append(
            rrb.Spatial3DView(
                origin=f"/take_{i}",
                contents="$origin/**",
                name=label,
                eye_controls=panel_eye_controls(),
            )
        )
    mini_panel_row = rrb.Grid(*mini_panels, grid_columns=n_takes)

    camera_views = []
    for i in range(1, n_takes + 1):
        label = take_labels[i - 1] if i - 1 < len(take_labels) else f"Take {i}"
        has_camera = camera_flags[i - 1] if camera_flags and i - 1 < len(camera_flags) else False
        if has_camera:
            camera_views.append(rrb.Spatial2DView(origin=f"/take_{i}/camera/cam1", name=f"{label} (side cam)"))

    sections = [top, bar_chart, mini_panel_row]
    row_shares = [1, 2, 3]
    if camera_views:
        sections.append(rrb.Grid(*camera_views, grid_columns=len(camera_views)))
        row_shares.append(2)

    if verbose:
        print("Blueprint container types (as constructed -- see verify_blueprint_containers for what's actually stored):")
        print("  root                   -> Vertical")
        print("  legend+metrics section -> Grid (grid_columns=2)")
        print("  bar chart section      -> BarChartView (single view)")
        print(f"  mini panel row         -> Grid (grid_columns={n_takes}, {n_takes} independent Spatial3DView(s), each origin-scoped to its own /take_N)")
        if camera_views:
            print(f"  camera row section     -> Grid (grid_columns={len(camera_views)}, flat Spatial2DView cells)")
        else:
            print("  camera row section     -> omitted (no take had confirmed camera data)")

    return rrb.Blueprint(rrb.Vertical(*sections, row_shares=row_shares), collapse_panels=True)


CONTAINER_KIND_NAMES = {1: "Horizontal", 2: "Tabs", 3: "Vertical", 4: "Grid"}


def verify_blueprint_containers(path: str) -> None:
    """Read back a SAVED .rbl file and print the container_kind Rerun
    actually stored for every container -- not what the construction code
    intended. This is the only way to catch the nested-container-in-Grid
    quirk described in build_compare_takes_blueprint's docstring, since
    that bug doesn't show up by inspecting the Python object graph."""
    from rerun.experimental import RrdReader

    reader = RrdReader(path)
    blueprints = reader.blueprints()
    if not blueprints:
        print(f"  Could not verify {path}: no blueprint store found.")
        return
    store = reader.store(store=blueprints[0]).stream().collect()
    schema = store.schema()
    container_paths = sorted({p for p in schema.entity_paths() if p.startswith("/container/")})

    print(f"Confirmed container types actually stored in {path}:")
    any_tabs = False
    for p in container_paths:
        df = store.reader(index="blueprint", contents=p).to_pandas()
        kind_col = f"{p}:ContainerBlueprint:container_kind"
        if kind_col not in df.columns:
            continue
        kind_val = int(df[kind_col].iloc[-1][0])
        kind_name = CONTAINER_KIND_NAMES.get(kind_val, f"UNKNOWN({kind_val})")
        is_grid_with_cameras = f"{p}:ContainerBlueprint:grid_columns" in df.columns
        marker = " <- camera row" if is_grid_with_cameras else ""
        print(f"  {p}: {kind_name}{marker}")
        if kind_name == "Tabs":
            any_tabs = True
    if any_tabs:
        print("  WARNING: at least one container was stored as Tabs.")
    else:
        print("  No Tabs containers found -- confirmed Grid/Horizontal/Vertical throughout.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", nargs="?", choices=["single", "compare", "both"], default="both")
    parser.add_argument("--n-takes", type=int, default=5, help="(compare variant only) number of take rows in the generic preview")
    args = parser.parse_args()

    ru.OUTPUT_DIR.mkdir(exist_ok=True)

    if args.variant in ("single", "both"):
        path = ru.OUTPUT_DIR / "blueprint_single_episode_v2.rbl"
        build_single_episode_blueprint().save(SINGLE_APP_ID, str(path))
        print(f"Saved {path} (application_id={SINGLE_APP_ID!r})")

    if args.variant in ("compare", "both"):
        path = ru.OUTPUT_DIR / "blueprint_compare_takes_v2.rbl"
        camera_flags = [True] * args.n_takes
        build_compare_takes_blueprint(n_takes=args.n_takes, camera_flags=camera_flags).save(COMPARE_APP_ID, str(path))
        print(f"Saved {path} (application_id={COMPARE_APP_ID!r}, generic {args.n_takes}-take preview)")
        verify_blueprint_containers(str(path))


if __name__ == "__main__":
    main()
