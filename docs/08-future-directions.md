# Future directions & open questions

This page exists so a real decision doesn't get re-litigated, and a real piece of research doesn't
get redone, just because it isn't sitting in `FEATURE_COLUMNS` or `src/firesight/` yet. Everything
below is either a deliberate scope boundary (decided, with a reason) or a genuinely open question
(identified, with what it would take to answer it), not a to-do list of things this project failed
to get to.

## Scaling to all of BC

**Status: not planned.** FireSight is a personal project scoped permanently to the Kamloops Fire
Centre (see [Problem framing](01-problem-framing.md) and the README's scope note), not a startup MVP
working toward province-wide coverage. The analysis below is kept for completeness, in case the
scope ever changes, or as a reference for what the real tradeoffs would be, not because BC-wide
expansion is on a roadmap. Three concrete things would need to happen before it would even be safe to
attempt, not just "run the same pipeline on a bigger bbox":

1. **`features/grid.py`'s single-reference-latitude approximation needs correcting first.** It
   converts degrees to km using one fixed reference latitude (50.6°) for the whole bounding box,
   which is a fine approximation across Kamloops Fire Centre's narrow latitude range but distorts
   meaningfully stretched across all of BC's roughly 12 degrees of latitude (coast to Rockies to
   the far north). This is a correctness fix, not an optimization, left wrong, it would silently
   misshape the grid rather than error out.
2. **Real volume growth.** The current dataset is ~6.7M rows over 1,443 cells for one fire centre.
   Scaling to all of BC is a "at least 50M rows" project (per direct estimate, not a guess pulled from
   nowhere), which also means re-checking whether the Windows `RandomizedSearchCV`/`n_jobs=-1` memory
   issue already hit once at the current size (see [Modeling &
   evaluation](06-modeling-and-evaluation.md#widening-the-search-randomizedsearchcv--predefinedsplit))
   gets materially worse.
3. **The fuel-type cache is Kamloops-specific** (`data/raw/fuel_type/kamloops_fuel_type.parquet`) and
   would need re-extraction from BC's Provincial Fuel Type Layer for a larger area.

### The harder question underneath all three: one model, or several?

Kamloops Fire Centre is one fire-climate regime (semi-arid interior). Coastal BC and the north are
genuinely different fire behavior, fuel types, and seasonality, not more rows of the same pattern.
Pooling everything into one RandomForest risks diluting the interior-specific signal the current model
actually leans on (soil moisture, 7-day temp/humidity trends). The candidate partition, if this gets
picked up, should be **BC Wildfire Service's existing Fire Centre boundaries**, not an invented
clustering, they already track real climatic and operational differences, and using them keeps
results interpretable against real prohibition/resourcing decisions instead of a boundary nobody
outside this project would recognize.

"How many models" isn't a decision to make up front, it's an empirical comparison, the same way
RF-vs-XGBoost and the fuel-type ablation were:

1. **One global model, no region signal**, the naive pooling case, expected to dilute Kamloops'
   signal.
2. **One global model with region as a categorical feature**, nearly free to test (one extra
   column), lets the RandomForest itself decide how much to split on region without committing to
   separate training runs.
3. **A fully separate model per Fire Centre**, only justified if it actually beats option 2 on
   held-out performance, since splitting an already-rare positive class (0.16% base rate at Kamloops
   alone) six ways risks starving each region's model of positives, the same failure mode already
   documented in the [sparse-fire-year calibration
   finding](06-modeling-and-evaluation.md#does-pooled-leave-one-year-out-validated-calibration-actually-help).

Whichever wins, it needs the same validation discipline already established here: a per-region
rolling-origin backtest (not one aggregate number), a monthly breakdown, and its own calibration
check, Kamloops' results are not assumed to transfer.

## Closing the winter/shoulder-season blind spot

**Status: structural, one real candidate fix identified and put on hold.** The served model misses
nearly all winter fires (see [known
limitation](06-modeling-and-evaluation.md#known-limitation-a-wintershoulder-season-blind-spot)) because
they're disproportionately human-caused, and every feature in `FEATURE_COLUMNS` is weather- or
fuel-derived. Two attempts to fix this with static features (calendar signals, road/place proximity)
both failed to move it.

Lightning-strike data is the one category with a real theoretical case for helping, since it's an
actual ignition-source signal, not another static proxy. Researched in
[`research/lightning-data.md`](../research/lightning-data.md): three real sources checked (ECCC
gridded lightning, the commercial CLDN archive, NOAA GOES satellite lightning), each with a load-bearing
catch, ECCC only starts 2023, CLDN's pricing was never actually verified, and the free GOES option
means a ~870GB raw backfill (or ~58GB subsampled, matching the precedent `ingest_era5.py` already set
for ERA5's own hourly-to-6-hourly tradeoff). **Put on hold at the point of deciding how to handle that
volume**, a real time/bandwidth/storage commitment, not a default to make unilaterally. The research
note has concrete next steps if this gets revisited.

## Calibrating the multi-day-ahead model

**Status: identified, not started.** `data/processed/model_3day.joblib` (serving
`/predict/live/multi-day`) has no calibrator, unlike the same-day model. Fitting one properly means
parameterizing `evaluation/backtest.py`'s rolling-origin loop by label column (it's currently hardcoded
to same-day `ignited`) and redoing the same pooled, leave-one-year-out-validated approach the same-day
model went through, not a quick single-split isotonic fit, which this project already found unreliable
for exactly this kind of rare-event data (see
[Calibration](06-modeling-and-evaluation.md#calibration-is-ignition_probability-a-real-probability)).
Secondary-endpoint polish, not urgent.

## Feature/data columns already tried (index, not a repeat)

Full reasoning for each lives in [Modeling & evaluation](06-modeling-and-evaluation.md); this is just
the map so a future session doesn't re-propose something already tested.

| Feature / source | Verdict | Why |
| --- | --- | --- |
| Fuel type (BC's Provincial FBP layer) | **Promoted, served** | Clean, real win alone, see [ablation](06-modeling-and-evaluation.md#closing-the-feature-category-gap-fwi-terrain-and-fuel-type-2026-08-21) |
| Spatial lag (`neighbor_fire_count_*d`) | **Promoted, served** | Real signal, implemented from the [future-directions proposal](06-modeling-and-evaluation.md#1-spatial-lag-features-neighbor-cells-recent-fire-history) |
| CAPE / convective precip | Tried, not promoted | Joined into the dataset but left out of `FEATURE_COLUMNS`, a re-tune showed no measured test-set benefit, see [feature addition](06-modeling-and-evaluation.md#adding-capeconvective_precip_mm-to-feature_columns) |
| Canadian FWI System (FFMC/DMC/DC/ISI/BUI) | Not promoted alone | Neutral alone, diluted fuel type's signal when combined, see the ablation link above |
| Terrain (elevation/slope/aspect) | Not promoted alone | Same ablation, same neutral-alone/diluting-combined result |
| Calendar features (day-of-year, season flag) | Tried, reverted | Didn't move the winter blind spot; hurt top-10% capture |
| Road/place proximity | Tried, reverted | Same failure mode as calendar features, static per-cell signal can't explain day-to-day variation |
| Lightning strikes | Researched, on hold | See the section above |

**One caveat worth stating plainly:** FWI and terrain being "not promoted" is a verdict for *this*
model on *ignition* prediction at Kamloops' scale, not a claim the raw physical data is useless. Both
are literally two of the three inputs the Canadian FBP fire-*spread* system needs (see the note on
fire-spread modeling below), a different problem might make different use of exactly the same
columns already sitting in this project's data.

## Model families tried beyond RandomForest/XGBoost

A 1D-CNN sequence model, an attention-pooling variant of it, and Venn-Abers per-prediction uncertainty
intervals were all built and evaluated against the same temporal split and metrics as everything else
here, see [`research/neural-networks.md`](../research/neural-networks.md) and [Testing the
sequence-modeling hypothesis](06-modeling-and-evaluation.md#testing-the-sequence-modeling-hypothesis).
None beat the tuned RandomForest; none are served. Not an open question at this point, listed here so
it isn't re-asked: tree ensembles have won every real comparison run against this data so far.

## Explicitly out of scope, not "not done yet"

Two things that come up naturally as "what's next" but aren't extensions of what FireSight solves,
listed here so they don't get proposed as if they were:

- **Fire-spread modeling** (predicting how far/fast an already-burning fire grows) is a different
  problem from ignition risk: a spatiotemporal simulation over burned-area perimeters, not a
  per-(cell, day) classifier, and it needs fire-perimeter time series data this project doesn't
  collect today (FIRMS gives point detections, not growing shapes). See [Problem
  framing](01-problem-framing.md#what-were-actually-predicting).
- **Authentication and production/multi-tenant deployment hardening** are not planned. FireSight stays
  a self-hosted project, run by following the README's setup steps, by explicit decision.
