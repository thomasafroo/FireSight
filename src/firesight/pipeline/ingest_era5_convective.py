"""Download historical convective/instability weather data from full ERA5 (not ERA5-Land)
via the Copernicus CDS API.

Distinct from `ingest_era5.py`'s ERA5-Land pull: this hits `reanalysis-era5-single-levels` for
`cape` (convective available potential energy) and `cp` (convective precipitation), a
storm/lightning-potential proxy, added per docs/06's winter-blind-spot follow-up (a real
lightning-detection feed doesn't have long enough public history to cover this project's training
years; see docs/04-weather-join.md for the full reasoning).

Fetched at hourly (not 6-hourly) resolution, unlike ERA5-Land: verified directly against a real CDS
response (not assumed) that `cp` here is NOT a running accumulation since 00 UTC the way ERA5-Land's
`tp` is, each hourly sample is its own independent, already-deaccumulated value (see
features/convective.py), so 6-hourly sampling would silently capture only 1 of every 6 hours' rain.
`cape` is fetched at the same hourly cadence so its daily max isn't undersampled either, at no extra
request cost since it's the same call.

Requesting an instantaneous variable (cape) and an accumulated one (cp) together also makes CDS
return two separate NetCDF files inside one zip (split by GRIB stepType), unlike ERA5-Land's single
combined-file response, this module merges them back into one file per month so callers see the
same "one NetCDF per month" shape `ingest_era5.py` already produces.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import cdsapi
import xarray as xr

MONTHS = [f"{m:02d}" for m in range(1, 13)]

CONVECTIVE_VARIABLES = [
    "convective_available_potential_energy",
    "convective_precipitation",
]

# north, west, south, east, matches the FIRMS/ERA5-Land bbox in ingest_firms.py/ingest_era5.py.
BC_KAMLOOPS_AREA = [51.5, -121.5, 49.8, -119.0]


def fetch_month(year: str, month: str, out_path: Path, client: cdsapi.Client | None = None) -> Path:
    """Fetch one month of hourly CAPE/convective-precipitation for the Kamloops bbox."""
    client = client or cdsapi.Client()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": CONVECTIVE_VARIABLES,
            "year": year,
            "month": month,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": BC_KAMLOOPS_AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(out_path),
    )

    if zipfile.is_zipfile(out_path):
        _merge_zipped_response(out_path)

    return out_path


def _merge_zipped_response(out_path: Path) -> None:
    """Unzip CDS's split instant/accum files and merge them back into one NetCDF at out_path."""
    with zipfile.ZipFile(out_path) as zf:
        names = zf.namelist()
        zf.extractall(out_path.parent)
    out_path.unlink()

    extracted = [out_path.parent / name for name in names]
    if len(extracted) == 1:
        extracted[0].rename(out_path)
        return

    datasets = [xr.open_dataset(p) for p in extracted]
    merged = xr.merge(datasets, compat="override")
    merged.to_netcdf(out_path)
    for ds in datasets:
        ds.close()
    for p in extracted:
        p.unlink()


def fetch_archive(
    start_year: int,
    end_year: int,
    out_dir: Path,
    max_attempts: int = 3,
) -> list[str]:
    """Fetch every month in [start_year, end_year] into one file per month, resumable."""
    client = cdsapi.Client()
    failed: list[str] = []

    for year in range(start_year, end_year + 1):
        for month in MONTHS:
            label = f"{year}-{month}"
            out_path = out_dir / f"kamloops_convective_{label}.nc"
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
    failed_months = fetch_archive(2012, 2024, Path("data/raw/era5_convective"))
    if failed_months:
        print(f"Done with {len(failed_months)} failures: {failed_months}", flush=True)
    else:
        print("Done: all months fetched successfully", flush=True)
