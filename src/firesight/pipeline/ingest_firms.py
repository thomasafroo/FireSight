"""Download historical fire detection points from NASA FIRMS.

Get a MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/map_key/ and set
it as FIRMS_MAP_KEY in your .env file. Free, but rate-limited to 5000
transactions per 10-minute window.

Verified against https://firms.modaps.eosdis.nasa.gov/api/area/ (2026-08-10).
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

load_dotenv()

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAX_DAY_RANGE = 5  # hard limit enforced by the API

# west,south,east,north — start with a small bounding box (Kamloops Fire Centre)
# rather than all of BC, to keep the first pipeline run fast.
BC_KAMLOOPS_BBOX = "-121.5,49.8,-119.0,51.5"

# _SP (Standard Processing) sources are the quality-assured, complete archive —
# use these for historical training data. _NRT sources are for near-real-time
# use later (the live risk map), not backfill.
# MODIS_SP goes back to ~2000, VIIRS_SNPP_SP to ~2012 — confirm exact coverage
# via https://firms.modaps.eosdis.nasa.gov/api/data_availability/ once you
# have a MAP_KEY.
DEFAULT_SOURCE = "VIIRS_SNPP_SP"


def _session() -> requests.Session:
    """A session that retries transient connection/DNS/5xx failures.

    A multi-year backfill is hundreds of sequential requests; without this,
    one blip in the middle kills the whole run (as happened on the first try).
    """
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_window(
    bbox: str,
    source: str,
    start_date: str,
    day_range: int,
    map_key: str | None = None,
    session: requests.Session | None = None,
    max_attempts: int = 4,
) -> pd.DataFrame:
    """Fetch a single window (max 5 days) of fire detections for a bounding box.

    Retries on any request failure, including HTTP 400 — observed FIRMS
    returning a spurious 400 for a request that succeeded moments later on
    retry, so this doesn't limit itself to 5xx/connection errors.
    """
    if day_range > MAX_DAY_RANGE:
        raise ValueError(f"day_range must be <= {MAX_DAY_RANGE}, got {day_range}")
    map_key = map_key or os.environ["FIRMS_MAP_KEY"]
    url = f"{FIRMS_BASE_URL}/{map_key}/{source}/{bbox}/{day_range}/{start_date}"

    for attempt in range(1, max_attempts + 1):
        try:
            response = (session or requests).get(url, timeout=60)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
        except requests.exceptions.RequestException:
            if attempt == max_attempts:
                raise
            wait = 2**attempt
            print(f"  request failed (attempt {attempt}/{max_attempts}), retrying in {wait}s", flush=True)
            time.sleep(wait)


def fetch_archive(
    bbox: str,
    source: str,
    start_date: str,
    end_date: str,
    map_key: str | None = None,
    pause_seconds: float = 0.2,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 20,
) -> pd.DataFrame:
    """Fetch a multi-year historical range by chunking into 5-day windows.

    pause_seconds throttles requests to stay well under the 5000/10min limit.
    If checkpoint_path is given: partial results are written to disk every
    checkpoint_every chunks (and on failure) so a crash doesn't lose progress,
    and an existing checkpoint is resumed from rather than re-fetched.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    prior_chunks: list[pd.DataFrame] = []

    if checkpoint_path and checkpoint_path.exists():
        existing = pd.read_parquet(checkpoint_path)
        prior_chunks.append(existing)
        resume_start = date.fromisoformat(existing["acq_date"].max()) + timedelta(days=1)
        start = max(start, resume_start)
        print(f"Resuming from checkpoint ({len(existing)} rows) at {start.isoformat()}", flush=True)

    total_chunks = (end - start).days // MAX_DAY_RANGE + 1
    chunks: list[pd.DataFrame] = list(prior_chunks)
    session = _session()

    cursor = start
    chunk_num = 0
    try:
        while cursor <= end:
            chunk_num += 1
            chunk = fetch_window(bbox, source, cursor.isoformat(), MAX_DAY_RANGE, map_key, session)
            chunks.append(chunk)
            print(f"[{chunk_num}/{total_chunks}] {cursor.isoformat()}: {len(chunk)} detections", flush=True)

            if checkpoint_path and chunk_num % checkpoint_every == 0:
                save_raw(pd.concat(chunks, ignore_index=True), checkpoint_path)
                print(f"  checkpointed {sum(len(c) for c in chunks)} rows -> {checkpoint_path}", flush=True)

            cursor += timedelta(days=MAX_DAY_RANGE)
            time.sleep(pause_seconds)
    except Exception:
        if checkpoint_path and chunks:
            save_raw(pd.concat(chunks, ignore_index=True), checkpoint_path)
            print(
                f"Failed after {chunk_num}/{total_chunks} chunks — saved partial result to {checkpoint_path}",
                flush=True,
            )
        raise

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def save_raw(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    out_path = Path("data/raw/firms/kamloops_2018-2024.parquet")
    df = fetch_archive(
        BC_KAMLOOPS_BBOX,
        DEFAULT_SOURCE,
        "2018-01-01",
        "2024-12-31",
        checkpoint_path=out_path,
    )
    save_raw(df, out_path)
    print(f"Done: {len(df)} total detections -> {out_path}", flush=True)
