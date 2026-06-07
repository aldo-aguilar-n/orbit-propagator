"""
Title: main.py
Authors: Aldo Aguilar
Date: 2026-06-06
Description: Entry point for the orbit-propagator app. This app reads a
CSV of UTC timestamps and a NORAD catalog ID, then computes GCRF/GCRS,
ITRF/ITRS position/velocity coordinates, as well as geodetic coordinates
(lat/lon/height) at each timestamp using the SGP4 propagator with the
closest-epoch TLE for the NORAD ID. The TLEs are fetched from
Space-Track GP_History or, if credentials are not available, the latest
CelesTrak GP/TLE. The output is a CSV with the original timestamps plus
the computed coordinates and velocities. 

Inputs:
    - CSV containing a UTC ISO-8601 timestamp
      (e.g., 2026-05-31T13:29:00Z) column named "timestamp"
    - NORAD catalog ID

Outputs:
    - A CSV containing the original UTC timestamp column plus GCRF/GCRS,
      ITRF/ITRS position/velocity coordinates, as well as geodetic
      coordinates (lat/lon/height) at each timestamp.

Notes:
    - SGP4 propagates TLEs into the TEME frame. This app treats TEME
      only as the intermediate propagation frame. After propagation,
      Astropy transforms the TEME state into GCRS for the Earth-centered
      celestial/inertial-like output state, and into ITRS for the
      Earth-fixed terrestrial output state. In the output files, GCRS is
      used as the GCRF/ECI-equivalent frame, and ITRS is used as the
      ITRF/ECEF-equivalent frame. Geodetic coordinates are computed from
      the ITRS/ITRF state.
    - Historical closest-epoch TLE selection requires Space-Track 
      credentials. Create a .env file in the project directory
      containing SPACE_TRACK_IDENTITY and SPACE_TRACK_PASSWORD.
      
Usage:
    poetry run orbit-propagator --norad 68635 --input timestamps.csv
    poetry run orbit-propagator --norad 68635 --input timestamps.csv --output out.csv
    poetry run orbit-propagator --norad 25544 --input timestamps.csv --use-celestrack
    poetry run orbit-propagator --norad 25544 --input timestamps.csv --tle-buffer-days 14
    poetry run python -m orbit_propagator --norad 68635 --input timestamps.csv --output out.csv
"""

# System imports
from __future__ import annotations
import argparse
import bisect
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

# Third-party imports
import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values
from astropy import units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    GCRS,
    ITRS,
    TEME,
)
from astropy.time import Time
from sgp4.api import Satrec
from tqdm import tqdm

# Global constants
SPACE_TRACK_BASE_URL = "https://www.space-track.org"
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
PROPAGATION_CHUNK_SIZE = 5000

@dataclass(frozen=True)
class TLERecord:
    """
    A TLE and its reference epoch.
    """

    epoch: datetime
    line1: str
    line2: str
    source: str

    @property
    def satrec(self) -> Satrec:
        """
        Return an SGP4 satellite record for this TLE.
        """
        return Satrec.twoline2rv(self.line1, self.line2)

def parse_utc_timestamp(value: object) -> datetime:
    """
    Parse an ISO-8601 UTC timestamp into a timezone-aware datetime.

    Accepted examples:
        2026-05-31T13:29:00Z
        2026-05-31T13:29:00+00:00

    Invalid dates, such as 2026-05-32T13:29:00Z, raise ValueError.
    """
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid UTC timestamp {value!r}: {exc}") from exc

    if dt.tzinfo is None:
        raise ValueError(f"Timestamp {value!r} is missing a timezone; "
                         f"expected UTC/Z")

    return dt.astimezone(timezone.utc)

def tle_epoch_from_line1(line1: str) -> datetime:
    """
    Parse a TLE line-1 epoch into a timezone-aware UTC datetime.

    TLE epoch format is YYDDD.DDDDDDDD. The NORAD convention maps years
    57-99 to 1957-1999 and 00-56 to 2000-2056.
    """
    epoch_text = line1[18:32].strip()
    year_2 = int(epoch_text[:2])
    day_of_year = float(epoch_text[2:])

    year = 1900 + year_2 if year_2 >= 57 else 2000 + year_2
    jan_1 = datetime(year, 1, 1, tzinfo=timezone.utc)
    return jan_1 + timedelta(days=day_of_year - 1.0)

def fetch_historical_tles_spacetrack(norad_id: int,
                                     start: datetime,
                                     stop: datetime,
                                     identity: str,
                                     password: str) -> list[TLERecord]:
    """
    Fetch historical GP/TLE records from Space-Track GP_History.

    The query window should already include any desired buffer around
    the CSV timestamp range. Space-Track requires an account and imposes
    API terms/rate limits; avoid repeatedly polling the same large date
    windows.
    """
    session = requests.Session()

    login = session.post(
        f"{SPACE_TRACK_BASE_URL}/ajaxauth/login",
        data={"identity": identity, "password": password},
        timeout=30,
    )
    login.raise_for_status()

    # Space-Track EPOCH range syntax uses YYYY-MM-DD--YYYY-MM-DD. Use
    # dates rather than times so the buffer fully covers the requested
    # days.
    start_date = start.strftime("%Y-%m-%d")
    stop_date = stop.strftime("%Y-%m-%d")

    query_url = (
        f"{SPACE_TRACK_BASE_URL}/basicspacedata/query"
        f"/class/gp_history"
        f"/NORAD_CAT_ID/{norad_id}"
        f"/EPOCH/{start_date}--{stop_date}"
        f"/orderby/EPOCH asc"
        f"/format/tle"
    )

    response = session.get(query_url, timeout=120)
    response.raise_for_status()

    lines = [line.strip() for line in response.text.splitlines() if line.strip()]
    if len(lines) % 2 != 0:
        raise RuntimeError(
            "Space-Track returned an odd number of TLE lines; response "
            "may be malformed."
        )

    records: list[TLERecord] = []
    for line1, line2 in zip(lines[0::2], lines[1::2]):
        if not (line1.startswith("1 ") and line2.startswith("2 ")):
            raise RuntimeError(f"Malformed TLE pair:\n{line1}\n{line2}")
        records.append(
            TLERecord(
                epoch=tle_epoch_from_line1(line1),
                line1=line1,
                line2=line2,
                source="space-track-gp-history",
            )
        )

    return deduplicate_and_sort_tles(records)

def fetch_current_tle_celestrak(norad_id: int) -> list[TLERecord]:
    """
    Fetch the latest public GP/TLE from CelesTrak.

    This is a fallback only. It cannot satisfy historical closest-epoch
    propagation unless every CSV timestamp is near the current TLE epoch.
    """
    response = requests.get(
        CELESTRAK_GP_URL,
        params={"CATNR": str(norad_id), "FORMAT": "TLE"},
        timeout=30,
    )
    response.raise_for_status()

    lines = [line.strip() for line in response.text.splitlines() if line.strip()]
    tle_lines = [line for line in lines if line.startswith(("1 ", "2 "))]
    if len(tle_lines) < 2:
        raise RuntimeError(f"No TLE returned by CelesTrak for NORAD ID "
                           f"{norad_id}")

    line1, line2 = tle_lines[0], tle_lines[1]
    return [
        TLERecord(
            epoch=tle_epoch_from_line1(line1),
            line1=line1,
            line2=line2,
            source="celestrak-current-gp",
        )
    ]

def deduplicate_and_sort_tles(records: Iterable[TLERecord]) -> list[TLERecord]:
    """
    Sort TLEs by epoch and remove exact duplicate line pairs.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[TLERecord] = []

    for record in sorted(records, key=lambda item: item.epoch):
        key = (record.line1, record.line2)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)

    return unique

def closest_tle_index(target_time: datetime, epochs: list[datetime]) -> int:
    """
    Return the index of the TLE whose epoch is closest to target_time.
    """
    if not epochs:
        raise ValueError("No TLE records are available")

    idx = bisect.bisect_left(epochs, target_time)

    candidates: list[int] = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(epochs):
        candidates.append(idx)

    return min(candidates, key=lambda index: abs(epochs[index] - target_time))

def propagate_teme_batch(sat: Satrec,
                         when_utc: list[datetime]) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate one TLE to multiple UTC timestamps using SGP4.

    Returns:
        r_teme_km: TEME position vectors [km]
        v_teme_km_s: TEME velocity vectors [km/s]
    """
    obstime = Time(when_utc)
    error_codes, position_km, velocity_km_s = sat.sgp4_array(obstime.jd1, obstime.jd2)

    if np.any(error_codes != 0):
        bad_index = int(np.flatnonzero(error_codes != 0)[0])
        raise RuntimeError(
            f"SGP4 failed with error code {error_codes[bad_index]} "
            f"at {when_utc[bad_index]}"
        )

    return np.array(position_km, dtype=float), np.array(velocity_km_s, dtype=float)

def teme_to_gcrs_and_itrs_batch(r_teme_km: np.ndarray,
                                v_teme_km_s: np.ndarray,
                                when_utc: list[datetime]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform TEME position/velocity arrays to GCRS/GCRF and ITRS/ITRF.

    Astropy names the geocentric celestial frame ``GCRS``. Its axes are
    aligned with GCRF for this application, so the output columns are
    named ``gcrf_*``. Astropy names the Earth-fixed terrestrial frame
    ``ITRS``. This is the frame used here for the requested ``itrf_*``
    output columns.

    Returns:
        r_gcrs_km: GCRS/GCRF position vectors [km]
        v_gcrs_km_s: GCRS/GCRF velocity vectors [km/s]
        r_itrs_km: ITRS/ITRF position vectors [km]
        v_itrs_km_s: ITRS/ITRF velocity vectors [km/s]
        lat_deg: Geodetic latitude values [deg]
        lon_deg: Geodetic longitude values [deg]
        height_km: Geodetic height values [km]
    """
    obstime = Time(when_utc)

    rep = CartesianRepresentation(
        r_teme_km[:, 0] * u.km,
        r_teme_km[:, 1] * u.km,
        r_teme_km[:, 2] * u.km,
        differentials=CartesianDifferential(
            v_teme_km_s[:, 0] * u.km / u.s,
            v_teme_km_s[:, 1] * u.km / u.s,
            v_teme_km_s[:, 2] * u.km / u.s,
        ),
    )

    teme = TEME(rep, obstime=obstime)
    gcrs = teme.transform_to(GCRS(obstime=obstime))
    itrs = teme.transform_to(ITRS(obstime=obstime))

    r_gcrs = gcrs.cartesian.xyz.to_value(u.km).T
    v_gcrs = gcrs.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s).T
    r_itrs = itrs.cartesian.xyz.to_value(u.km).T
    v_itrs = itrs.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s).T

    earth_location = itrs.earth_location
    lat_deg = earth_location.lat.to_value(u.deg)
    lon_deg = earth_location.lon.to_value(u.deg)
    height_km = earth_location.height.to_value(u.km)

    return (
        np.array(r_gcrs, dtype=float),
        np.array(v_gcrs, dtype=float),
        np.array(r_itrs, dtype=float),
        np.array(v_itrs, dtype=float),
        np.array(lat_deg, dtype=float),
        np.array(lon_deg, dtype=float),
        np.array(height_km, dtype=float),
    )

def default_output_path(input_csv: Path) -> Path:
    """
    Return the default output CSV path when --output is not supplied.
    """
    return input_csv.with_name(f"{input_csv.stem}_states.csv")

def load_spacetrack_credentials() -> tuple[str | None, str | None]:
    """
    Load Space-Track credentials from environment variables or a local .env file.
    """
    env_path = Path.cwd() / ".env"
    values = dotenv_values(env_path) if env_path.exists() else {}

    identity = (
        os.environ.get("SPACE_TRACK_IDENTITY")
        or values.get("SPACE_TRACK_IDENTITY")
    )
    password = (
        os.environ.get("SPACE_TRACK_PASSWORD")
        or values.get("SPACE_TRACK_PASSWORD")
    )

    return identity, password

def compute_states(input_csv: Path,
                   output_csv: Path,
                   norad_id: int,
                   tle_buffer_days: float,
                   allow_celestrak_fallback: bool) -> None:
    """
    Read input CSV, fetch TLEs, compute states, and write output CSV.
    """
    df = pd.read_csv(input_csv)
    if "timestamp" not in df.columns:
        raise ValueError('Input CSV must contain a column named "timestamp"')
    if df.empty:
        raise ValueError("Input CSV contains no rows")
    print(f"Read {len(df)} rows from input CSV.")

    timestamps = [parse_utc_timestamp(value) for value in df["timestamp"]]
    start = min(timestamps) - timedelta(days=tle_buffer_days)
    stop = max(timestamps) + timedelta(days=tle_buffer_days)

    identity, password = load_spacetrack_credentials()

    if identity and password:
        print(f"Space-Track: Welcome {identity}! Fetching TLEs for NORAD ID "
              f"{norad_id} from {start.date()} to {stop.date()}...")
        
        tles = fetch_historical_tles_spacetrack(norad_id, start, stop, identity, password)
        
        print(f"Fetched {len(tles)} TLE(s) from Space-Track spanning "
              f"{tles[0].epoch} to {tles[-1].epoch}")
        
    elif allow_celestrak_fallback:
        print("Space-Track credentials not found. Falling back to the "
              "latest CelesTrak TLE. This may not be close to the "
              "requested timestamps.")
        
        tles = fetch_current_tle_celestrak(norad_id)

        print(f"Fetched 1 TLE from CelesTrak with epoch {tles[0].epoch}")
        
    else:
        raise RuntimeError(
            "Historical closest-epoch TLE selection requires Space-Track "
            "credentials. Create a .env file in the project directory "
            "with SPACE_TRACK_IDENTITY and SPACE_TRACK_PASSWORD, or pass "
            "--use-celestrak to use only the latest CelesTrak TLE."
        )

    if not tles:
        raise RuntimeError(
            f"No TLEs found for NORAD ID {norad_id} from {start.date()} "
            f"to {stop.date()}"
        )

    if allow_celestrak_fallback and len(tles) == 1 and tles[0].source == "celestrak-current-gp":
        max_tle_age_days = max(
            abs(tles[0].epoch - timestamp).total_seconds() / 86400.0
            for timestamp in timestamps
        )

        if max_tle_age_days > 7.0:
            print(
                f"WARNING: CelesTrak fallback TLE is up to "
                f"{max_tle_age_days:.1f} days away from requested timestamps.",
                file=sys.stderr,
            )

    tle_epochs = [tle.epoch for tle in tles]
    satrecs = [tle.satrec for tle in tles]
    tle_indices = np.array(
        [closest_tle_index(timestamp, tle_epochs) for timestamp in timestamps],
        dtype=int,
    )
    timestamp_values = df["timestamp"].tolist()
    n_rows = len(timestamps)

    tle_epoch_out = [""] * n_rows
    tle_age_days_out = np.empty(n_rows, dtype=float)
    lat_out = np.empty(n_rows, dtype=float)
    lon_out = np.empty(n_rows, dtype=float)
    height_out = np.empty(n_rows, dtype=float)
    gcrf_out = np.empty((n_rows, 6), dtype=float)
    itrf_out = np.empty((n_rows, 6), dtype=float)

    print("Propagating states for each timestamp...")
    with tqdm(total=n_rows, desc="Propagating", unit="row") as progress_bar:
        for tle_idx in np.unique(tle_indices):
            tle = tles[int(tle_idx)]
            sat = satrecs[int(tle_idx)]
            matching_indices = np.flatnonzero(tle_indices == tle_idx)

            for start_idx in range(0, len(matching_indices), PROPAGATION_CHUNK_SIZE):
                chunk_indices = matching_indices[start_idx:start_idx + PROPAGATION_CHUNK_SIZE]
                chunk_times = [timestamps[int(index)] for index in chunk_indices]

                r_teme_km, v_teme_km_s = propagate_teme_batch(sat, chunk_times)
                (
                    r_gcrf_km,
                    v_gcrf_km_s,
                    r_itrf_km,
                    v_itrf_km_s,
                    lat_deg,
                    lon_deg,
                    height_km,
                ) = teme_to_gcrs_and_itrs_batch(
                    r_teme_km,
                    v_teme_km_s,
                    chunk_times,
                )

                gcrf_out[chunk_indices, 0:3] = r_gcrf_km
                gcrf_out[chunk_indices, 3:6] = v_gcrf_km_s
                itrf_out[chunk_indices, 0:3] = r_itrf_km
                itrf_out[chunk_indices, 3:6] = v_itrf_km_s
                lat_out[chunk_indices] = lat_deg
                lon_out[chunk_indices] = lon_deg
                height_out[chunk_indices] = height_km

                for index in chunk_indices:
                    row_idx = int(index)
                    tle_epoch_out[row_idx] = tle.epoch.isoformat()
                    tle_age_days_out[row_idx] = (
                        abs(tle.epoch - timestamps[row_idx]).total_seconds() / 86400.0
                    )

                progress_bar.update(len(chunk_indices))

    columns = [
        "timestamp",
        "tle_epoch",
        "tle_age_days",
        "lat_deg",
        "lon_deg",
        "height_km",
        "gcrf_x_km",
        "gcrf_y_km",
        "gcrf_z_km",
        "gcrf_vx_km_s",
        "gcrf_vy_km_s",
        "gcrf_vz_km_s",
        "itrf_x_km",
        "itrf_y_km",
        "itrf_z_km",
        "itrf_vx_km_s",
        "itrf_vy_km_s",
        "itrf_vz_km_s",
    ]
    output_df = pd.DataFrame(
        {
            "timestamp": timestamp_values,
            "tle_epoch": tle_epoch_out,
            "tle_age_days": tle_age_days_out,
            "lat_deg": lat_out,
            "lon_deg": lon_out,
            "height_km": height_out,
            "gcrf_x_km": gcrf_out[:, 0],
            "gcrf_y_km": gcrf_out[:, 1],
            "gcrf_z_km": gcrf_out[:, 2],
            "gcrf_vx_km_s": gcrf_out[:, 3],
            "gcrf_vy_km_s": gcrf_out[:, 4],
            "gcrf_vz_km_s": gcrf_out[:, 5],
            "itrf_x_km": itrf_out[:, 0],
            "itrf_y_km": itrf_out[:, 1],
            "itrf_z_km": itrf_out[:, 2],
            "itrf_vx_km_s": itrf_out[:, 3],
            "itrf_vy_km_s": itrf_out[:, 4],
            "itrf_vz_km_s": itrf_out[:, 5],
        },
        columns=columns,
    )
    output_df.to_csv(output_csv, index=False)

    print(f"Wrote {n_rows} propagated rows to {output_csv}")

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute GCRF and ITRF satellite coordinates at CSV "
            "timestamps using the closest TLE epoch for a NORAD ID."
        )
    )
    parser.add_argument(
        "--norad", 
        type=int, 
        required=True, 
        help="NORAD catalog ID"
    )
    parser.add_argument(
        "--input",
        dest="input_csv",
        type=Path,
        required=True,
        help="Input CSV with a timestamp column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path. Defaults to "
            "<input_stem>_states.csv next to the input file."
        ),
    )
    parser.add_argument(
        "--tle-buffer-days",
        type=float,
        default=7.0,
        help=(
            "Days of TLE history buffer on each side of the timestamp "
            "range"
        ),
    )
    parser.add_argument(
        "--use-celestrack",
        action="store_true",
        help=(
            "Use the latest CelesTrak TLE when Space-Track credentials "
            "are not set. This is not historical closest-epoch selection."
        ),
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        compute_states(
            input_csv=args.input_csv,
            output_csv=(
                args.output if args.output is not None else default_output_path(args.input_csv)
            ),
            norad_id=args.norad,
            tle_buffer_days=args.tle_buffer_days,
            allow_celestrak_fallback=args.use_celestrack,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# ----------------------------------------------------------------------