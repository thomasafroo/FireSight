# Grid & labels

This is the step that turns "a pile of fire detection points" and "a weather grid" into "a table a
classifier can be trained on." It's arguably the most consequential design decision in the whole
pipeline — get the grid or the labels wrong and every downstream model is learning from noise, no
matter how good the features are.

## Why a grid at all

FIRMS gives point detections (exact lat/lon). A classifier needs fixed, comparable units of space to
assign features and labels to — you can't have a variable-shaped "row" per unique location, because
there'd be almost no repeat structure to learn from (satellite pixel centers rarely land on the
exact same coordinate twice). A **grid cell** is an artificial, fixed-size unit of area that:

- Aggregates nearby detections into the same unit, so "5 detections scattered across 200m of the
same hillside" becomes one clear signal rather than 5 near-duplicate rows.
- Gives every day a fixed, enumerable set of "did it burn here" rows — including cells where nothing
happened, which is the whole point (see below).
- Matches naturally to the ERA5-Land weather grid, which is *also* a grid, just coarser.

## How cell math works (`features/grid.py`)

A grid cell is defined by a size in **kilometers** (`cell_size_km`, default 5.0) but grid math needs
to happen in **degrees** of lat/lon, since that's the coordinate system the data comes in.
Converting between them is `cell_size_degrees()`:

```
lat_size = cell_size_km / 111.0
lon_size = cell_size_km / (111.0 * cos(radians(reference_latitude)))
```

`111.0` is roughly km per degree of **latitude** everywhere on Earth — lines of latitude are evenly
spaced. Longitude is different: lines of longitude converge toward the poles, so a degree of
longitude covers *less* ground the further you are from the equator. The `cos(latitude)` term
corrects for that — at Kamloops' ~50.6°N, a degree of longitude is only about 63% as wide as a
degree of latitude, so `lon_size` comes out noticeably larger than `lat_size` for the same km
target.

**The single-reference-latitude simplification:** the correction uses one fixed `reference_latitude`
(50.6°, the module default) for the whole bounding box, rather than correcting per-row based on each
point's own latitude. Cells near the north edge of the bbox (51.5°) are therefore very slightly
narrower in real km than cells near the south edge (49.8°) than the target `cell_size_km` — small
enough to ignore over a ~2° latitude span, but it's a real approximation that would need fixing
(per-row correction, or projecting into a proper equal-area CRS) if the grid ever expands to cover
all of BC, which spans nearly 15° of latitude.

**Two functions, two jobs, one shared scheme:**

- `assign_cell_ids(df)` — given a DataFrame of points (e.g. fire detections), computes which cell
each point falls into: `row = floor(lat / lat_size)`, `col = floor(lon / lon_size)`,
`cell_id = f"{row}_{col}"`. Only produces cell_ids for cells that actually appear in the input data.
- `build_grid_cells(bbox)` — enumerates **every** cell inside a bounding box, whether or not it ever
appears in the fire data, with each cell's centroid coordinates. This is the piece that makes the
"cells with no fire" rows possible at all.

Both use the exact same `row`/`col` floor-division scheme against the same reference latitude, which
is why a `cell_id` produced by one lines up exactly with the same `cell_id` from the other — that
consistency is load-bearing, not incidental (see `tests/test_grid.py`,
`tests/test_labels.py::test_build_grid_cells_covers_bbox_corners`).

## Why the label table needs a full cross-product, not just fire rows

If the label table only contained rows where a fire was detected, every row would have
`ignited == 1` — there'd be nothing to learn "this *didn't* burn" from, which is most of what a
classifier actually needs to see to be useful (a model that's only ever seen positives can't produce
a meaningful probability). The label table has to include the **vast majority of (cell, day)
combinations where nothing happened**, so the true (heavily imbalanced) class distribution is what
the model trains and gets evaluated against.

`features/labels.py::build_label_scaffold` builds this as a **cross join** (`grid_cells x dates`,
via `pd.merge(..., how="cross")`): every cell paired with every date in the range, `ignited`
defaulting to 0, then set to 1 wherever a filtered detection's `(cell_id, date)` matches. For the
current Kamloops bbox at 5km resolution and 2012-2024, that's 1,443 cells × 4,748 days ≈ 6.9M rows —
see [glossary.md](glossary.md#cross-join) if "cross join" is a new term.

## Filtering to real fires first

Before any of the above, `features/labels.py::filter_real_fires` drops every FIRMS row where
`type != 0` — active volcanoes, static land sources (flares, industrial heat sources), and offshore
detections. These aren't wildfires and would corrupt the label if included: they'd mark a cell as
`ignited == 1` for a phenomenon that has nothing to do with vegetation burning risk, teaching the
model a false association between weather and a target that weather doesn't actually predict for
that row.

## The tradeoff in choosing `cell_size_km = 5.0`

Not (yet) tuned, just a reasonable starting point, but worth naming the tradeoff explicitly since it
affects everything downstream:

- **Smaller cells** (e.g. 1km) → more precise "where," but even sparser positive labels per cell
(already at 0.16% positive at 5km) and a much larger row count for the same time range, which is
more compute for the same information (ERA5-Land's own resolution is ~9km — went much below that and
multiple fire-grid cells would just be duplicating the same weather features from a shared nearest
ERA5 point anyway).
- **Larger cells** → denser positive signal per cell, faster to train on, but a "yes" prediction
becomes less actionable (a 20km-wide risk cell is much harder to dispatch a crew to than a 5km one).

5km sits close to ERA5-Land's own native resolution, which is a reasonable anchor: going
meaningfully finer buys little, since the weather signal (the dominant feature source) can't
distinguish sub-9km detail anyway.
