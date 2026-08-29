"""Download historical weather data from ERA5-Land via the Copernicus CDS API.

Requires a free account and API key at https://cds.climate.copernicus.eu/
Configure ~/.cdsapirc per their docs, and accept the ERA5-Land dataset's
terms on its CDS page before your first request.

Verified against https://forum.ecmwf.int (2026-08-11): the API param is
`data_format`, not the older `format`; NetCDF requests can silently come
back as a ZIP unless `download_format: unarchived` is also set, this
script sets that but unzips as a fallback anyway.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import cdsapi

MONTHS = [f"{m:02d}" for m in range(1, 13)]

# Variables relevant to fire risk: temperature, dewpoint, precipitation,
# wind components, soil moisture. Deliberately kept wider than
# baseline.py::FEATURE_COLUMNS, which dropped d2m/u10/v10 on 2026-08-15:
# engineering.py's add_relative_humidity/add_wind_features still derive
# relative_humidity and wind_speed from them, so they stay in the raw pull.
ERA5_LAND_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "volumetric_soil_water_layer_1",
]

# north, west, south, east, matches the FIRMS bbox in ingest_firms.py
BC_KAMLOOPS_AREA = [51.5, -121.5, 49.8, -119.0]


def fetch_month(year: str, month: str, out_path: Path, client: cdsapi.Client | None = None) -> Path:
    """Fetch one month of ERA5-Land variables for the Kamloops bbox.

    CDS requests are synchronous and queued server-side, a single month
    can take anywhere from under a minute to several minutes depending on
    load, unlike the near-instant FIRMS requests.
    """
    client = client or cdsapi.Client()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": ERA5_LAND_VARIABLES,
            "year": year,
            "month": month,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
            "area": BC_KAMLOOPS_AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(out_path),
    )

    if zipfile.is_zipfile(out_path):
        with zipfile.ZipFile(out_path) as zf:
            names = zf.namelist()
            zf.extractall(out_path.parent)
        out_path.unlink()
        extracted = out_path.parent / names[0]
        extracted.rename(out_path)

    return out_path


def fetch_archive(
    start_year: int,
    end_year: int,
    out_dir: Path,
    max_attempts: int = 3,
) -> list[str]:
    """Fetch every month in [start_year, end_year] into one file per month.

    Each month is its own file, so this is naturally resumable, already
    downloaded months are skipped rather than re-fetched. Returns the list
    of "{year}-{month}" strings that failed after retries, if any; the run
    keeps going past a failed month rather than aborting the whole backfill.
    """
    client = cdsapi.Client()
    failed: list[str] = []

    for year in range(start_year, end_year + 1):
        for month in MONTHS:
            label = f"{year}-{month}"
            out_path = out_dir / f"kamloops_{label}.nc"
            if out_path.exists():
                print(f"[{label}] already have it, skipping", flush=True)
                continue

            for attempt in range(1, max_attempts + 1):
                try:
                    fetch_month(str(year), month, out_path, client)
                    print(f"[{label}] done -> {out_path}", flush=True)
                    break
                except Exception as e:  # noqa: BLE001, must keep going past any single month's failure
                    if attempt == max_attempts:
                        print(f"[{label}] FAILED after {max_attempts} attempts: {e}", flush=True)
                        failed.append(label)
                    else:
                        print(f"[{label}] attempt {attempt}/{max_attempts} failed: {e}", flush=True)

    return failed


if __name__ == "__main__":
    failed_months = fetch_archive(2012, 2024, Path("data/raw/era5"))
    if failed_months:
        print(f"Done with {len(failed_months)} failures: {failed_months}", flush=True)
    else:
        print("Done: all months fetched successfully", flush=True)
