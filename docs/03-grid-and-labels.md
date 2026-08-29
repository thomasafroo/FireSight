# Grid & labels

This is the step that turns "a pile of fire detection points" and "a weather grid" into "a table a
classifier can be trained on." It's arguably the most consequential design decision in the whole
pipeline, get the grid or the labels wrong and every downstream model is learning from noise, no
matter how good the features are.

## Why a grid at all

FIRMS gives point detections (exact lat/lon). A classifier needs fixed, comparable units of space to
assign features and labels to, you can't have a variable-shaped "row" per unique location, because
there'd be almost no repeat structure to learn from (satellite pixel centers rarely land on the
exact same coordinate twice). A **grid cell** is an artificial, fixed-size unit of area that:

- Aggregates nearby detections into the same unit, so "5 detections scattered across 200m of the
same hillside" becomes one clear signal rather than 5 near-duplicate rows.
- Gives every day a fixed, enumerable set of "did it burn here" rows, including cells where nothing
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

`111.0` is roughly km per degree of **latitude** everywhere on Earth, lines of latitude are evenly
spaced. Longitude is different: lines of longitude converge toward the poles, so a degree of
longitude covers *less* ground the further you are from the equator. The `cos(latitude)` term
corrects for that, at Kamloops' ~50.6°N, a degree of longitude is only about 63% as wide as a
degree of latitude, so `lon_size` comes out noticeably larger than `lat_size` for the same km
target.

**The single-reference-latitude simplification:** the correction uses one fixed `reference_latitude`
(50.6°, the module default) for the whole bounding box, rather than correcting per-row based on each
point's own latitude. Cells near the north edge of the bbox (51.5°) therefore come out very
slightly narrower in real km than the target `cell_size_km`, and cells near the south edge (49.8°)
very slightly wider, small enough to ignore over a ~2° latitude span, but it's a real
approximation that would need fixing (per-row correction, or projecting into a proper equal-area
CRS) if the grid ever expands to cover
all of BC, which spans nearly 12° of latitude.

**Two functions, two jobs, one shared scheme:**

- `assign_cell_ids(df)`, given a DataFrame of points (e.g. fire detections), computes which cell
each point falls into: `row = floor(lat / lat_size)`, `col = floor(lon / lon_size)`,
`cell_id = f"{row}_{col}"`. Only produces cell_ids for cells that actually appear in the input data.
- `build_grid_cells(bbox)`, enumerates **every** cell inside a bounding box, whether or not it ever
appears in the fire data, with each cell's centroid coordinates. This is the piece that makes the
"cells with no fire" rows possible at all.

Both use the exact same `row`/`col` floor-division scheme against the same reference latitude, which
is why a `cell_id` produced by one lines up exactly with the same `cell_id` from the other, that
consistency is load-bearing, not incidental (see `tests/test_grid.py`,
`tests/test_labels.py::test_build_grid_cells_covers_bbox_corners`).

## Why the label table needs a full cross-product, not just fire rows

If the label table only contained rows where a fire was detected, every row would have
`ignited == 1`, there'd be nothing to learn "this *didn't* burn" from, which is most of what a
classifier actually needs to see to be useful (a model that's only ever seen positives can't produce
a meaningful probability). The label table has to include the **vast majority of (cell, day)
combinations where nothing happened**, so the true (heavily imbalanced) class distribution is what
the model trains and gets evaluated against.

`features/labels.py::build_label_scaffold` builds this as a **cross join** (`grid_cells x dates`,
via `pd.merge(..., how="cross")`): every cell paired with every date in the range, `ignited`
defaulting to 0, then set to 1 wherever a filtered detection's `(cell_id, date)` matches. For the
current Kamloops bbox at 5km resolution and 2012-2024, that's 1,443 cells × 4,749 days ≈ 6.9M rows,
see [glossary.md](glossary.md#cross-join) if "cross join" is a new term.

## Filtering to real fires first

Before any of the above, `features/labels.py::filter_real_fires` drops every FIRMS row where
`type != 0`, active volcanoes, static land sources (flares, industrial heat sources), and offshore
detections. These aren't wildfires and would corrupt the label if included: they'd mark a cell as
`ignited == 1` for a phenomenon that has nothing to do with vegetation burning risk, teaching the
model a false association between weather and a target that weather doesn't actually predict for
that row.

## The multi-day-ahead label: `ignited_next_Nd` (2026-08-21)

[Problem framing](01-problem-framing.md#what-were-actually-predicting) names the natural extension
past same-day prediction: "will a fire be detected in cell *C* on day *D* ... or within the next *N*
days?" `features/labels.py::add_forward_ignition_label(df, n_days)` builds exactly that as a second
label column (`ignited_next_3d` for `n_days=3`, the value `pipeline/build_dataset.py::MULTI_DAY_WINDOW`
currently uses) alongside `ignited`, not in place of it, a pure label transform, not a new feature:
every row still only ever uses conditions known as of day *D*, this just widens what counts as a hit
*on* that row. `n_days=1` reproduces `ignited` exactly, so it's a strict generalization of the
existing label, not a parallel definition that happens to agree at the edges.

**Mechanically, a forward rolling max, computed the "reverse, roll, un-reverse" way, since `pandas`
has no native support for a forward-looking window:** sort by `(cell_id, date)` (so each cell's rows are
contiguous), reverse the whole frame (which reverses each cell's block too), take a *backward*-looking
`rolling(n_days, min_periods=n_days).max()` in that reversed order, which is a forward-looking window
in real time, then let pandas' index-aligned assignment un-reverse it back onto the original frame.
Needs the same dense (every cell x every date) panel `add_neighbor_fire_features`
([Feature engineering](05-feature-engineering.md)) already requires, for the same reason: a forward
window spanning a date gap would silently mean something different than intended. Rows in the
trailing `n_days - 1` days of a cell's history (no full forward window available yet) get `NaN`, the
same "don't guess, drop it" precedent `add_days_since_rain` established, left for callers to drop
explicitly rather than folded into `drop_incomplete_history`, since unlike `ENGINEERED_COLUMNS` this
label isn't unconditionally required by every consumer of the dataset.

**Why the temporal train/val/test split doesn't need special trimming at its boundaries for this
label, verified rather than assumed.** A forward-looking label naturally raises a real concern: could
a training row near `TRAIN_END` (2023-01-01) have its label determined by a date that actually falls
in the val period, letting val-period ground truth leak backward into a training label? In general,
yes, but this project's fire-season filtering (`baseline.py::filter_fire_season`, May 1 - Oct 15)
already puts a ~2.5-month buffer between the latest fire-season date in any year (Oct 15) and the next
split boundary (Jan 1), so even `n_days=3`'s forward window (Oct 15 -> Oct 17) never gets close to
crossing into the next year, let alone into the next split. Same reasoning applies at the very end of
the whole dataset (`END_DATE = "2024-12-31"`): the only rows actually left with a `NaN` label are deep
in December, already outside the fire-season filter every training/eval path applies. No trimming code
was added because none is needed *given the existing fire-season scope*, worth re-checking if this
project's date range or season window ever changes.

**Modeling result and serving decision, see [Modeling &
evaluation](06-modeling-and-evaluation.md#testing-the-multi-day-ahead-label-2026-08-21) for the full
numbers.** The served RandomForest's exact hyperparameters, unretuned, clear the dummy floor by a wide
margin on this new target, but fall meaningfully short of the same-day model's own test-set numbers, a
real, expected gap (a genuinely harder target, same information), not a bug. A follow-up hyperparameter
retune made val scores better but test scores *worse*, the same single-fold-overfitting shape this
project has hit before, so it was discarded in favor of the original unretuned params.

**Served as a second, parallel model, not a replacement for same-day prediction.**
`training/export_model.py::export_multi_day_model` fits the same `BEST_RANDOM_FOREST_PARAMS` against
`ignited_next_3d` and saves a second `ModelBundle` to `data/processed/model_3day.joblib`, an
independent artifact `api/main.py` loads optionally at startup (its absence degrades one endpoint,
`GET /predict/live/multi-day`, rather than blocking the whole API the way a missing same-day bundle
does). No calibrator is attached yet: the pooled calibration methodology
[Serving](07-serving.md#calibration-ignition_probability-vs-calibrated_probability) uses depends on
`evaluation/backtest.py::run_rolling_origin_backtest`, which is hardcoded to the same-day label, a
real, accepted gap for this first served version, not a silent omission, since `ModelBundle` already
treats `calibrator=None` as "unavailable" rather than a false zero.

## The tradeoff in choosing `cell_size_km = 5.0`

Not (yet) tuned, just a reasonable starting point, but worth naming the tradeoff explicitly since it
affects everything downstream:

- **Smaller cells** (e.g. 1km) → more precise "where," but even sparser positive labels per cell
(already at 0.16% positive at 5km) and a much larger row count for the same time range, which is
more compute for the same information (ERA5-Land's own resolution is ~9km, go much below that and
multiple fire-grid cells would just be duplicating the same weather features from a shared nearest
ERA5 point anyway).
- **Larger cells** → denser positive signal per cell, faster to train on, but a "yes" prediction
becomes less actionable (a 20km-wide risk cell is much harder to dispatch a crew to than a 5km one).

5km sits close to ERA5-Land's own native resolution, which is a reasonable anchor: going
meaningfully finer buys little, since the weather signal (the dominant feature source) can't
distinguish sub-9km detail anyway.
