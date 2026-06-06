# Orbit Propagator

Poetry-managed command-line app that reads a CSV with UTC timestamps, fetches historical TLEs for a NORAD catalog ID, propagates the satellite with SGP4, and writes GCRF/GCRS and ITRF/ITRS state vectors to a CSV.

SGP4 natively propagates TLEs in TEME. This app uses TEME only as an intermediate frame and then uses Astropy to transform the state into:

- `gcrf_*`: Astropy `GCRS`, used here as the GCRF-aligned geocentric celestial output frame
- `itrf_*`: Astropy `ITRS`, used here as the ITRF/Earth-fixed output frame

Positions are in km and velocities are in km/s.

## Project layout

```text
orbit-propagator/
├── pyproject.toml
├── README.md
├── .env.example
└── orbit_propagator/
    ├── __init__.py
    ├── __main__.py
    └── main.py
```

## Install with Poetry

From the project directory:

```bash
poetry install
```

## Create the `.env` file

Historical closest-epoch TLE selection requires Space-Track GP_History access. Create a file named `.env` in the project directory:

```bash
cp .env.example .env
```

Then edit `.env` so it contains your Space-Track credentials:

```text
SPACE_TRACK_IDENTITY=your@email.com
SPACE_TRACK_PASSWORD=your_space_track_password
```

The app intentionally reads credentials from the project-local `.env` file.

## Input CSV

The input CSV must contain a column named `timestamp`. Timestamps must be valid UTC ISO-8601 timestamps (e.g., `2026-05-32T13:29:00Z`).

Example:

```csv
timestamp
2026-05-31T13:29:00Z
2026-05-31T13:34:00Z
```

## Run

Main usage:

```bash
poetry run orbit-propagator --norad 25544 --input timestamps.csv --output out.csv
```

The `--output` argument is optional. If omitted, the app writes `<input_stem>_states.csv` next to the input file:

```bash
poetry run orbit-propagator --norad 25544 --input timestamps.csv
```

Example with another satellite:

```bash
poetry run orbit-propagator --norad 68635 --input my_times.csv --output satellite_states.csv
```

You can also run the package module directly:

```bash
poetry run python -m orbit_propagator --norad 25544 --input timestamps.csv --output out.csv
```

## Output columns

The output CSV contains exactly these columns:

```text
timestamp
gcrf_x_km
gcrf_y_km
gcrf_z_km
gcrf_vx_km_s
gcrf_vy_km_s
gcrf_vz_km_s
itrf_x_km
itrf_y_km
itrf_z_km
itrf_vx_km_s
itrf_vy_km_s
itrf_vz_km_s
```

The `timestamp` values are preserved from the original input column.

## CelesTrak fallback

You can run without a `.env` file by explicitly allowing the latest CelesTrak TLE fallback:

```bash
poetry run orbit-propagator --norad 25544 --input timestamps.csv --output out.csv --use-celestrack
```

This does **not** provide historical closest-epoch TLE selection; it only uses the latest public TLE. Use Space-Track credentials in `.env` for best historical propagation fidelity.

## Useful option

By default, the app fetches TLEs over the timestamp range plus a 7-day buffer on each side. You can change this with:

```bash
poetry run orbit-propagator --norad 25544 --input timestamps.csv --output out.csv --tle-buffer-days 14
```
