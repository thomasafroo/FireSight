# Data sources

Four independent datasets feed this project: FIRMS and ERA5-Land (fetched by
`pipeline/ingest_firms.py` and `pipeline/ingest_era5.py`, the original two, and still the only ones
pulled by a dedicated `pipeline/ingest_*.py` script since both need a multi-year historical backfill)
plus BC's Provincial Fuel Type Layer and Open-Meteo's Elevation API (fetched directly by
`features/fuel_type.py` and `features/topography.py` — no separate ingest step, since both are
static per-cell lookups fetched once and cached, not a time-varying backfill). They come from
different agencies, on different grids, at different frequencies — none of that is a coincidence to
work around later, it's the normal shape of a real geospatial ML problem.

## FIRMS — fire *detections* (the label source)

**FIRMS** (Fire Information for Resource Management System) is NASA's service for distributing
near-real-time and historical fire detections from two satellite instrument families:

- **MODIS** — on the Terra/Aqua satellites, detections since ~2000.
- **VIIRS** — on Suomi NPP/NOAA-20, detections since ~2012, higher spatial resolution (~375m vs
MODIS's ~1km) than MODIS. This project uses `VIIRS_SNPP_SP` (`ingest_firms.py:DEFAULT_SOURCE`).

Important: FIRMS gives you **detections**, not fires. A detection is a satellite instrument flagging
a pixel as anomalously hot at the moment it passed overhead. That distinction matters for two
reasons covered more in [Grid & labels](03-grid-and-labels.md):

1. A single real fire produces many detections (satellite passes twice daily per instrument, over
multiple days) — the label scaffold collapses these to "did *any* detection land in this cell on
this day," not "how many detections."
2. Not every detection is vegetation burning. FIRMS' `type` field flags: `0` = presumed vegetation
fire, `1` = active volcano, `2` = other static land source (e.g. gas flares), `3` = offshore (e.g.
ships, oil platform flares). Only `type == 0` is a wildfire ignition — filtering the other three out
is the first step of the pipeline (`features/labels.py::filter_real_fires`).

**`_SP` vs `_NRT` suffix:** FIRMS offers both a Standard Processing archive (`_SP`,
quality-controlled, used here for historical training data) and Near-Real-Time (`_NRT`, available
within hours, lower quality assurance, intended for the eventual live risk map — not the training
data).

**Access:** free, requires a `MAP_KEY` from firms.modaps.eosdis.nasa.gov, rate-limited to 5000
transactions per 10-minute window. `ingest_firms.py` chunks a multi-year backfill into 5-day windows
(the API's max per-request range) with retries and checkpointing, since a multi-year pull is
hundreds of sequential requests and a single blip shouldn't lose the whole run.

## ERA5-Land — weather (the feature source)

**ERA5-Land** is a **reanalysis** dataset from the ECMWF (European Centre for Medium-Range Weather
Forecasts), distributed via the Copernicus Climate Data Store (CDS). "Reanalysis" means it's not a
forecast and not raw observations — it's a physics model run *backward* over historical time,
constrained by whatever real observations exist (satellites, weather stations, radiosondes) to
produce a physically consistent, gap-free grid of estimated conditions everywhere, including places
with no weather station. That gap-free property is exactly what makes it usable as ML training
features: FIRMS points are irregular and sparse, but ERA5-Land gives a value for every grid point,
every timestep, with no missing data to explain away.

**Resolution:** ~9km (0.1° lat/lon) grid — coarser than the 5km fire grid this project uses, which
is why [the weather join](04-weather-join.md) needs a nearest-neighbor step rather than a direct row
match.

**Variables pulled** (`ingest_era5.py::ERA5_LAND_VARIABLES`), and why each one is plausibly
fire-relevant:

| Variable | What it is | Why it matters for fire risk |
|---|---|---|
| `2m_temperature` (`t2m`) | Air temp at 2m height | Heat dries fuel, raises ignition probability |
| `2m_dewpoint_temperature` (`d2m`) | Dewpoint at 2m | Combined with `t2m`, determines relative humidity — dry air dries fuel faster |
| `total_precipitation` (`tp`) | Accumulated rainfall | Recent rain suppresses ignition; its absence over time (drought) raises it |
| `10m_u_component_of_wind` (`u10`) | East-west wind at 10m | Wind speed/direction drives fire spread rate once ignited, and can fan ignition |
| `10m_v_component_of_wind` (`v10`) | North-south wind at 10m | Paired with `u10` to get wind speed/direction (see [Feature engineering](05-feature-engineering.md)) |
| `volumetric_soil_water_layer_1` (`swvl1`) | Topsoil moisture | Proxy for live/dead fuel moisture — drier soil correlates with drier vegetation |

**Time resolution:** requested at 6-hour steps (00/06/12/18 UTC) rather than the full hourly
product, to keep the historical backfill (156 monthly files, 2012-2024) a reasonable size. This is
aggregated to daily values before use — see [Weather join](04-weather-join.md) for exactly how,
since it's not as simple as averaging all four variables the same way.

**Access:** free Copernicus CDS account + API key in `~/.cdsapirc`, plus accepting ERA5-Land's terms
on its CDS page before the first request. Requests are queued server-side and synchronous — a single
month can take seconds to several minutes depending on load, unlike FIRMS' near-instant responses.
`ingest_era5.py::fetch_archive` fetches one file per month, which makes the backfill naturally
resumable (already-fetched months are skipped) and keeps a single failure from losing prior
progress.

## BC Provincial Fuel Type Layer — fuel type (a feature source)

`WHSE_LAND_AND_NATURAL_RESOURCE.PROT_FUEL_TYPE_SP`, from the BC Data Catalogue, is BC Wildfire
Service's own per-forest-stand FBP fuel-type classification (C-1..C-7 conifer, D-1/2 deciduous,
M-1..M-4 mixedwood, S-1..S-3 slash, O-1a/b grass, N non-fuel, W water) — the categorical input the
Canadian FBP System is itself built around, and a signal genuinely independent of weather: two
adjacent cells with identical temperature/humidity/wind can still carry very different real ignition
risk if one is grassland and the other is wet deciduous forest.

Served as a WFS polygon layer, not a small pre-clipped file (the province-wide download is a ~4GB
File Geodatabase, needing `fiona`/GDAL to read — a dependency this project deliberately doesn't have,
see [Grid & labels](03-grid-and-labels.md) for the same tradeoff already made for the fire grid
itself). `features/fuel_type.py` instead queries the WFS endpoint per grid-cell centroid with a small
bounding box, since a raw point-`INTERSECTS` filter against this layer's BC Albers geometry doesn't
work via CQL on this service — checked directly, not assumed. **Access:** free, no API key. Fetched
once per cell and cached under `data/raw/fuel_type/` — fuel type doesn't change day to day, so
there's no historical-vs-live split the way weather has.

## Open-Meteo Elevation API — terrain (a feature source)

Copernicus DEM 2021, 90m resolution, served by [Open-Meteo's Elevation
API](https://open-meteo.com/en/docs/elevation-api) — the same provider `features/live_weather.py`
already depends on for `/predict/live` (see [Serving](07-serving.md)), reused here rather than adding
a new vendor. `features/topography.py` fetches one elevation value per grid-cell centroid (not a
raster download, avoiding the GDAL/rasterio dependency noted above) and derives slope/aspect from
each cell's 8 neighbor elevations. **Access:** free, no API key, but rate-limited per-coordinate
(~600 coordinates/minute on the free tier, confirmed by hitting a 429) — `features/topography.py`
paces its batched requests accordingly. Fetched once and cached under `data/raw/topography/`, for the
same reason fuel type is: terrain doesn't change day to day.

## Why these sources, and why this bbox

All four sources target the same **Kamloops Fire Centre bounding box** (`-121.5,49.8,-119.0,51.5`,
west/south/east/north) — a deliberately small region rather than all of BC. Per the README's stated
status, this is intentional scoping: prove the full pipeline (ingest → grid → label → join → model →
serve) works end-to-end on a small, fast-to-iterate region before paying the cost (compute, storage,
download time) of scaling to the whole province.

FIRMS and ERA5-Land were the original two — the label source and the weather feature source a
minimum-viable version of this problem needs. Fuel type and terrain were added later to close a
feature-category gap (see [Modeling &
evaluation](06-modeling-and-evaluation.md#closing-the-feature-category-gap-fwi-terrain-and-fuel-type-2026-08-21)
for the motivating competitive-landscape review and the results); they're static per-cell lookups
rather than a second time-varying feed, which is why they don't get their own `pipeline/ingest_*.py`
script the way FIRMS/ERA5-Land do.
