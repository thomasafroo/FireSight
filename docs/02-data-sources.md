# Data sources

Two independent datasets feed this project, fetched by `pipeline/ingest_firms.py` and
`pipeline/ingest_era5.py` respectively. They come from different agencies, on different grids, at
different frequencies — none of that is a coincidence to work around later, it's the normal shape of
a real geospatial ML problem.

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

## Why these two, and why this bbox

Both scripts target the same **Kamloops Fire Centre bounding box** (`-121.5,49.8,-119.0,51.5`,
west/south/east/north) — a deliberately small region rather than all of BC. Per the README's stated
status, this is intentional scoping: prove the full pipeline (ingest → grid → label → join → model →
serve) works end-to-end on a small, fast-to-iterate region before paying the cost (compute, storage,
download time) of scaling to the whole province.
