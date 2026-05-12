#!/usr/bin/env python3
"""
export_weather_pixels_to_csv.py
===============================

Purpose
-------
Export weather rasters to a machine-learning friendly CSV file where:
    one row = one pixel at one timestamp

Input
-----
This script expects the output directory produced by
`download_weather_rasters.py`, containing:
- temperature_2m.tif
- relative_humidity_2m.tif
- wind_speed_10m.tif
- metadata.json

Output
------
A CSV file with columns such as:
- timestamp
- pixel_row
- pixel_col
- x_utm
- y_utm
- lon
- lat
- temperature_2m
- relative_humidity_2m
- wind_speed_10m

Why this format is useful for ML
--------------------------------
- it is tabular and easy to join with other per-pixel features
- it supports feature selection later in the ML pipeline
- it preserves spatial coordinates and time explicitly

Important note
--------------
The output can become very large:
    rows = width * height * number_of_hours

Use `--row-step` and `--col-step` to downsample if needed.

Example
-------
python weather_raster_solution/export_weather_pixels_to_csv.py \
  --input-dir weather_raster_solution/output \
  --output weather_raster_solution/output/weather_pixels.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform


DEFAULT_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export weather rasters to a per-pixel per-timestamp CSV."
    )
    parser.add_argument(
        "--input-dir",
        default="weather_raster_solution/output",
        help="Directory containing weather GeoTIFF rasters",
    )
    parser.add_argument(
        "--output",
        default="weather_raster_solution/output/weather_pixels.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--variables",
        default=",".join(DEFAULT_VARIABLES),
        help="Comma-separated weather variable names",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional single timestamp to export, e.g. 2024-07-04T12:00",
    )
    parser.add_argument(
        "--row-step",
        type=int,
        default=1,
        help="Export every Nth row. Default: 1",
    )
    parser.add_argument(
        "--col-step",
        type=int,
        default=1,
        help="Export every Nth column. Default: 1",
    )
    parser.add_argument(
        "--drop-all-nan",
        action="store_true",
        help="Skip rows where all weather variables are NaN",
    )
    return parser.parse_args()


def load_rasters(input_dir: Path, variables: list[str]) -> dict[str, rasterio.io.DatasetReader]:
    datasets: dict[str, rasterio.io.DatasetReader] = {}
    for variable in variables:
        path = input_dir / f"{variable}.tif"
        if not path.exists():
            raise SystemExit(f"Missing raster: {path}")
        datasets[variable] = rasterio.open(path)
    return datasets


def validate_alignment(datasets: dict[str, rasterio.io.DatasetReader]) -> rasterio.io.DatasetReader:
    items = list(datasets.items())
    reference = items[0][1]
    for variable, src in items[1:]:
        if src.crs != reference.crs:
            raise SystemExit(f"CRS mismatch for {variable}")
        if src.transform != reference.transform:
            raise SystemExit(f"Transform mismatch for {variable}")
        if src.width != reference.width or src.height != reference.height:
            raise SystemExit(f"Shape mismatch for {variable}")
        if src.count != reference.count:
            raise SystemExit(f"Band count mismatch for {variable}")
    return reference


def resolve_band_index(reference: rasterio.io.DatasetReader, timestamp: str | None) -> list[int]:
    if timestamp is None:
        return list(range(1, reference.count + 1))

    normalized = timestamp.strip().replace("Z", "")
    descriptions = list(reference.descriptions)
    if normalized not in descriptions:
        raise SystemExit(f"Timestamp '{normalized}' not found in raster bands.")
    return [descriptions.index(normalized) + 1]


def build_coordinate_grids(reference: rasterio.io.DatasetReader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(reference.height)
    cols = np.arange(reference.width)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")

    x_grid = reference.transform.c + (col_grid + 0.5) * reference.transform.a
    y_grid = reference.transform.f + (row_grid + 0.5) * reference.transform.e

    lon, lat = transform(
        reference.crs,
        "EPSG:4326",
        x_grid.ravel().tolist(),
        y_grid.ravel().tolist(),
    )
    lon_grid = np.asarray(lon, dtype=np.float64).reshape(reference.height, reference.width)
    lat_grid = np.asarray(lat, dtype=np.float64).reshape(reference.height, reference.width)
    return row_grid, col_grid, x_grid, y_grid, lon_grid, lat_grid


def main() -> int:
    args = parse_args()
    if args.row_step <= 0 or args.col_step <= 0:
        raise SystemExit("row-step and col-step must be positive.")

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variables = [item.strip() for item in args.variables.split(",") if item.strip()]
    if not variables:
        raise SystemExit("At least one variable must be provided.")

    datasets = load_rasters(input_dir, variables)
    try:
        reference = validate_alignment(datasets)
        band_indices = resolve_band_index(reference, args.timestamp)
        row_grid, col_grid, x_grid, y_grid, lon_grid, lat_grid = build_coordinate_grids(reference)

        row_indices = np.arange(0, reference.height, args.row_step)
        col_indices = np.arange(0, reference.width, args.col_step)

        fieldnames = [
            "timestamp",
            "pixel_row",
            "pixel_col",
            "x_utm",
            "y_utm",
            "lon",
            "lat",
        ] + variables

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for band_index in band_indices:
                timestamp_value = reference.descriptions[band_index - 1]
                band_arrays = {
                    variable: datasets[variable].read(band_index)
                    for variable in variables
                }

                for row in row_indices:
                    for col in col_indices:
                        values = {
                            variable: float(band_arrays[variable][row, col])
                            if np.isfinite(band_arrays[variable][row, col])
                            else None
                            for variable in variables
                        }
                        if args.drop_all_nan and all(values[v] is None for v in variables):
                            continue

                        writer.writerow(
                            {
                                "timestamp": timestamp_value,
                                "pixel_row": int(row_grid[row, col]),
                                "pixel_col": int(col_grid[row, col]),
                                "x_utm": float(x_grid[row, col]),
                                "y_utm": float(y_grid[row, col]),
                                "lon": float(lon_grid[row, col]),
                                "lat": float(lat_grid[row, col]),
                                **values,
                            }
                        )

        print(f"Saved {output_path}")
        print(f"Exported timestamps: {len(band_indices)}")
        print(f"Exported grid: {len(row_indices)} rows x {len(col_indices)} cols")
        return 0
    finally:
        for src in datasets.values():
            src.close()


if __name__ == "__main__":
    raise SystemExit(main())
