# Lightning strike data for FireSight: a feasibility writeup

Research notes, not a decision — written after a strategy discussion (2026-08-21) about where
FireSight should invest next. The winter/shoulder-season blind spot
([docs/06](../docs/06-modeling-and-evaluation.md#known-limitation-a-wintershoulder-season-blind-spot))
is a real, unfixed ceiling: winter fires are more often human-caused than weather-driven, and every
feature currently in `FEATURE_COLUMNS` is weather- or fuel-derived. Two feature-engineering attempts
to work around it (calendar features, road/place proximity) already failed to move it — real
ignition-source data is the one category with a genuine theoretical case for closing it, since it's
the one thing those two attempts *weren't*. Lightning is the most obviously available candidate:
naturally-caused fires are lightning-driven often enough that a per-cell, per-day strike signal could
plausibly separate "no ignition source present" from "conditions were right but nothing struck."

This was scoped as *feasibility only* — no code was written, no data was fetched beyond a handful of
live checks against the real services below. Status: **on hold**, not pursued further this session.

## Three candidate sources, each with a different, load-bearing limitation

| Source | Coverage | Cost | The catch |
| --- | --- | --- | --- |
| ECCC gridded lightning density (MSC Datamart / WMS, 2.5km grid) | **2023-01-30 → present only** | Free | Doesn't reach the 2012-2022 training window at all — only useful as a live (`/predict/live`-style) feature, not for training |
| CLDN raw strike archive (the "20 years of CLDN data" dataset, 1998-2023) | Matches the training window exactly | **Likely commercial** — the network is Vaisala-operated | Never checked directly against a real price/access page — the one source here that wasn't `verified live against the actual service` before writing this, unlike this project's usual discipline (see `features/fuel_type.py`'s module docstring for that standard) |
| NOAA GOES-17/18 GLM satellite lightning (AWS S3, `noaa-goes17`/`noaa-goes18` buckets) | **Feb 2019 → present** (GOES-17 was GOES-West until GOES-18 took over Jan 2023) | Free, verified live — anonymous HTTPS `ListObjectsV2` against the bucket works with no AWS account | Zero coverage for 2012-2018, cutting the training window from 11 years to ~4; detection efficiency also drops toward the edge of the satellite's field of view, and Kamloops (49.8-51.5°N) is within GLM's <55°N cutoff but not centered in it |

GOES GLM was the one actually verified live: an anonymous `GET` against
`https://noaa-goes18.s3.amazonaws.com/?list-type=2&prefix=GLM-L2-LCFA/2024/200/18/` returned a real
object listing with no credentials needed — confirming the free-access claim rather than assuming it
from NOAA's own marketing copy, the same standard every other data source in this project was held to.

## The GLM volume problem

NOAA's own *aggregated* GLM product (1-minute/5-minute gridded flash-extent-density, 8km at nadir)
would have been the tractable option, but it's distributed via the Satellite Broadcast Network / LDM
starting **March 2023** — no earlier historical archive was found, so it doesn't help here any more
than the ECCC gridded product does.

That leaves the raw **L2 LCFA** event files as the only historically-complete free option — and they're
one file per 20-second scan window, ~200KB each, with no server-side spatial filtering (S3 is a flat
object store; every file has to be downloaded whole and filtered client-side to the Kamloops bbox).
Working the actual numbers for a fire-season-only backfill (May 1 - Oct 15, 2019-2024, ~1,000 days):

- 4,320 files/day × ~1,000 days ≈ **4.3 million files**
- ~200KB/file × 4.3M files ≈ **~870GB of raw downloads**

For comparison, this project's largest existing backfill (ERA5-Land, 2012-2024) is 156 monthly files.
This is roughly 2-3 orders of magnitude more requests and data volume than anything else in
FireSight's pipeline.

**A direct precedent exists for the obvious mitigation.** `pipeline/ingest_era5.py` already accepts
6-hourly instead of hourly ERA5 sampling specifically "to keep the historical backfill a reasonable
size" (see [Data sources](../docs/02-data-sources.md#era5-land--weather-the-feature-source)).
The same move applies here: subsampling GLM files (e.g., one every 5 minutes instead of every 20
seconds) would cut volume ~15x to roughly 58GB/280K files — since the actual feature this project
would build is a coarse daily "did lightning occur near this cell" signal (matching the granularity
of `neighbor_fire_count_*d` and `days_since_rain`), not a precise per-strike count, undercounting
individual strikes within a storm is an acceptable loss the way undercounting exact rainfall timing
already is elsewhere in this pipeline.

## Status: on hold

Put on hold at the point of deciding *how* to handle the backfill volume (full-resolution ~870GB,
subsampled ~58GB, or a narrower 1-2-season pilot first) — a real time/bandwidth/storage commitment
that's the user's call, not something to default into. No code exists for this yet.

**If revisited, the concrete next steps are:**

1. Verify the CLDN commercial pricing directly (never done here) — if genuinely cheap for a project
   this size, it sidesteps the entire 2019-cutoff and volume problem at once and may be worth it.
2. If sticking with free GOES GLM: pick a backfill strategy (subsampled is the recommended default,
   matching ERA5's own precedent) and run a narrow pilot (e.g., just August 2023, the one month with
   the most known lightning-driven activity in this project's existing error analysis) to confirm the
   L2 LCFA event data actually contains non-trivial strike counts inside the Kamloops FC bbox before
   committing to the full multi-year pull — the same "prove the source is real before building the
   pipeline around it" discipline `features/fuel_type.py` and `features/topography.py` both followed.
3. Whatever the source, the training window truncation (2012-2018 losing lightning coverage under the
   free path) is itself a real confound to name honestly in any results write-up, the same way this
   project already names confounds for other partial feature additions
   (see [Modeling & evaluation](../docs/06-modeling-and-evaluation.md#closing-the-feature-category-gap-fwi-terrain-and-fuel-type-2026-08-21)'s
   `max_features`/thin-class confounds as the template).
