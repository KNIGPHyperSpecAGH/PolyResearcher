#!/usr/bin/env python3
"""
download_weather_rasters.py
===========================

Purpose
-------
Create hourly weather rasters aligned to the same grid logic as the satellite
downloader in this repository.

How it works
------------
1. You provide a bounding box in WGS84:
   [lon_min, lat_min, lon_max, lat_max]
2. The script builds a target raster grid in EPSG:32634 using the same
   `bbox -> UTM bounds -> raster transform` logic as `image_downloading.py`.
3. Open-Meteo weather data is sampled for a coarse grid of points inside the
   bounding box.
4. Each hourly weather field is interpolated onto the target raster grid using
   inverse distance weighting (IDW).
5. One multi-band GeoTIFF is saved per weather variable.

What the resolution means
-------------------------
- `--resolution` is the output raster pixel size in meters.
  Default: 100 m
  This matches the satellite grid logic used in `image_downloading.py`.
- `--sample-spacing` is the spacing between weather sample points in meters.
  Default: 5000 m
  This is the effective weather sampling density before interpolation.

Important note
--------------
The final weather rasters are aligned to the satellite grid, but they do not
contain true 100 m weather observations. They contain interpolated weather
fields derived from much coarser point weather data.

Default variables
-----------------
- temperature_2m       [°C]
- relative_humidity_2m [%]
- wind_speed_10m       [m/s]

Example
-------
python weather_raster_solution/download_weather_rasters.py \
  --bbox 19.70,49.95,20.20,50.15 \
  --start 2024-07-04T00:00 \
  --end 2024-07-05T23:00 \
  --output-dir weather_raster_solution/output
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds as make_transform
from rasterio.warp import transform, transform_bounds


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WGS84 = CRS.from_epsg(4326)
TARGET_CRS = CRS.from_epsg(32634)
DEFAULT_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Open-Meteo weather rasters aligned to the satellite grid."
    )
    parser.add_argument(
        "--bbox",
        required=True,
        help="Bounding box as lon_min,lat_min,lon_max,lat_max",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start datetime in ISO format, e.g. 2024-07-04T00:00",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End datetime in ISO format, e.g. 2024-07-05T23:00",
    )
    parser.add_argument(
        "--hourly",
        default=",".join(DEFAULT_HOURLY),
        help="Comma-separated hourly variables",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=100.0,
        help="Output raster resolution in meters. Default: 100",
    )
    parser.add_argument(
        "--sample-spacing",
        type=float,
        default=5000.0,
        help="Weather sampling point spacing in meters. Default: 5000",
    )
    parser.add_argument(
        "--idw-neighbors",
        type=int,
        default=4,
        help="Number of nearest weather samples used by IDW. Default: 4",
    )
    parser.add_argument(
        "--idw-power",
        type=float,
        default=2.0,
        help="IDW power parameter. Default: 2.0",
    )
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="Open-Meteo timezone parameter. Default: UTC",
    )
    parser.add_argument(
        "--temperature-unit",
        default="celsius",
        choices=["celsius", "fahrenheit"],
        help="Temperature unit",
    )
    parser.add_argument(
        "--wind-speed-unit",
        default="ms",
        choices=["kmh", "ms", "mph", "kn"],
        help="Wind speed unit",
    )
    parser.add_argument(
        "--precipitation-unit",
        default="mm",
        choices=["mm", "inch"],
        help="Precipitation unit",
    )
    parser.add_argument(
        "--output-dir",
        default="weather_raster_solution/output",
        help="Output directory for GeoTIFF rasters and metadata",
    )
    return parser.parse_args()


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 4:
        raise SystemExit("Bounding box must have 4 comma-separated values.")

    lon_min, lat_min, lon_max, lat_max = (float(item) for item in parts)
    if lon_min >= lon_max or lat_min >= lat_max:
        raise SystemExit("Bounding box must satisfy lon_min < lon_max and lat_min < lat_max.")

    return lon_min, lat_min, lon_max, lat_max


def parse_iso_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def hourly_range(start: datetime, end: datetime) -> list[datetime]:
    current = start
    values: list[datetime] = []
    while current <= end:
        values.append(current)
        current += timedelta(hours=1)
    return values


def fetch_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    full_url = f"{url}?{urlencode(params, doseq=True)}"
    try:
        with urlopen(full_url) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}\nResponse: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc}") from exc


def split_range(
    start: datetime,
    end: datetime,
) -> tuple[tuple[date, date] | None, tuple[datetime, datetime] | None]:
    today_utc = datetime.now(UTC).date()
    historical_end = min(end.date(), today_utc - timedelta(days=1))
    forecast_start = max(start, datetime.combine(today_utc, time.min, tzinfo=UTC))

    historical_range = None
    if start.date() <= historical_end:
        historical_range = (start.date(), historical_end)

    forecast_range = None
    if forecast_start <= end:
        forecast_range = (forecast_start, end)

    return historical_range, forecast_range


def create_target_grid(
    bbox: tuple[float, float, float, float],
    resolution: float,
) -> dict[str, Any]:
    utm_bounds = transform_bounds(WGS84, TARGET_CRS, *bbox)
    width = max(1, int((utm_bounds[2] - utm_bounds[0]) / resolution))
    height = max(1, int((utm_bounds[3] - utm_bounds[1]) / resolution))
    transform_affine = make_transform(*utm_bounds, width, height)

    col_centers = transform_affine.c + (np.arange(width) + 0.5) * transform_affine.a
    row_centers = transform_affine.f + (np.arange(height) + 0.5) * transform_affine.e
    grid_x, grid_y = np.meshgrid(col_centers, row_centers)

    return {
        "bounds_utm": utm_bounds,
        "width": width,
        "height": height,
        "transform": transform_affine,
        "grid_x": grid_x,
        "grid_y": grid_y,
    }


def create_sample_points(
    bbox: tuple[float, float, float, float],
    sample_spacing: float,
) -> dict[str, np.ndarray]:
    utm_bounds = transform_bounds(WGS84, TARGET_CRS, *bbox)
    xmin, ymin, xmax, ymax = utm_bounds

    xs = np.arange(xmin + sample_spacing / 2.0, xmax, sample_spacing)
    ys = np.arange(ymin + sample_spacing / 2.0, ymax, sample_spacing)

    if xs.size == 0:
        xs = np.array([(xmin + xmax) / 2.0])
    if ys.size == 0:
        ys = np.array([(ymin + ymax) / 2.0])

    sample_x, sample_y = np.meshgrid(xs, ys)
    sample_x_flat = sample_x.ravel()
    sample_y_flat = sample_y.ravel()
    sample_lon, sample_lat = transform(
        TARGET_CRS,
        WGS84,
        sample_x_flat.tolist(),
        sample_y_flat.tolist(),
    )

    return {
        "x": sample_x_flat.astype(np.float64),
        "y": sample_y_flat.astype(np.float64),
        "lon": np.asarray(sample_lon, dtype=np.float64),
        "lat": np.asarray(sample_lat, dtype=np.float64),
    }


def extract_hourly_values(
    payload: dict[str, Any],
    hourly_variables: list[str],
) -> dict[str, dict[str, float | None]]:
    hourly = payload.get("hourly")
    if not hourly:
        raise RuntimeError("API response does not contain 'hourly' data.")

    timestamps = hourly.get("time", [])
    result: dict[str, dict[str, float | None]] = {}
    for index, timestamp in enumerate(timestamps):
        row: dict[str, float | None] = {}
        for variable in hourly_variables:
            series = hourly.get(variable, [])
            row[variable] = series[index] if index < len(series) else None
        result[timestamp] = row
    return result


def fetch_openmeteo_point_series(
    *,
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    hourly_variables: list[str],
    timezone_name: str,
    temperature_unit: str,
    wind_speed_unit: str,
    precipitation_unit: str,
) -> tuple[dict[str, dict[str, float | None]], dict[str, str]]:
    historical_range, forecast_range = split_range(start, end)
    merged: dict[str, dict[str, float | None]] = {}
    units: dict[str, str] = {}

    if historical_range:
        payload = fetch_json(
            ARCHIVE_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": historical_range[0].isoformat(),
                "end_date": historical_range[1].isoformat(),
                "hourly": hourly_variables,
                "timezone": timezone_name,
                "temperature_unit": temperature_unit,
                "wind_speed_unit": wind_speed_unit,
                "precipitation_unit": precipitation_unit,
            },
        )
        merged.update(extract_hourly_values(payload, hourly_variables))
        units = payload.get("hourly_units", {})

    if forecast_range:
        payload = fetch_json(
            FORECAST_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "start_hour": forecast_range[0].strftime("%Y-%m-%dT%H:%M"),
                "end_hour": forecast_range[1].strftime("%Y-%m-%dT%H:%M"),
                "hourly": hourly_variables,
                "timezone": timezone_name,
                "temperature_unit": temperature_unit,
                "wind_speed_unit": wind_speed_unit,
                "precipitation_unit": precipitation_unit,
            },
        )
        merged.update(extract_hourly_values(payload, hourly_variables))
        if not units:
            units = payload.get("hourly_units", {})

    return merged, units


def collect_sample_cube(
    sample_points: dict[str, np.ndarray],
    start: datetime,
    end: datetime,
    hourly_variables: list[str],
    timezone_name: str,
    temperature_unit: str,
    wind_speed_unit: str,
    precipitation_unit: str,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, str]]:
    timestamps = [dt.strftime("%Y-%m-%dT%H:%M") for dt in hourly_range(start, end)]
    point_count = sample_points["lat"].shape[0]
    cube = {
        variable: np.full((len(timestamps), point_count), np.nan, dtype=np.float32)
        for variable in hourly_variables
    }
    units: dict[str, str] = {}

    for point_idx, (lat, lon) in enumerate(zip(sample_points["lat"], sample_points["lon"])):
        series, point_units = fetch_openmeteo_point_series(
            lat=float(lat),
            lon=float(lon),
            start=start,
            end=end,
            hourly_variables=hourly_variables,
            timezone_name=timezone_name,
            temperature_unit=temperature_unit,
            wind_speed_unit=wind_speed_unit,
            precipitation_unit=precipitation_unit,
        )
        if not units:
            units = point_units
        for time_idx, timestamp in enumerate(timestamps):
            values = series.get(timestamp)
            if not values:
                continue
            for variable in hourly_variables:
                value = values.get(variable)
                cube[variable][time_idx, point_idx] = np.nan if value is None else float(value)

    return timestamps, cube, units


def precompute_idw_neighbors(
    target_x: np.ndarray,
    target_y: np.ndarray,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    neighbor_count: int,
    power: float,
) -> tuple[np.ndarray, np.ndarray]:
    target_x_flat = target_x.ravel()
    target_y_flat = target_y.ravel()
    neighbor_count = max(1, min(neighbor_count, sample_x.shape[0]))

    dx = target_x_flat[:, None] - sample_x[None, :]
    dy = target_y_flat[:, None] - sample_y[None, :]
    distances = np.sqrt(dx * dx + dy * dy)

    nearest_idx = np.argpartition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
    nearest_dist = np.take_along_axis(distances, nearest_idx, axis=1)

    weights = np.zeros_like(nearest_dist, dtype=np.float64)
    zero_mask = nearest_dist == 0
    zero_rows = np.any(zero_mask, axis=1)

    if np.any(~zero_rows):
        nonzero_dist = nearest_dist[~zero_rows]
        nonzero_weights = 1.0 / np.power(nonzero_dist, power)
        nonzero_weights /= nonzero_weights.sum(axis=1, keepdims=True)
        weights[~zero_rows] = nonzero_weights

    if np.any(zero_rows):
        first_zero = np.argmax(zero_mask[zero_rows], axis=1)
        weights[zero_rows] = 0.0
        weights[zero_rows, first_zero] = 1.0

    return nearest_idx, weights.astype(np.float32)


def interpolate_time_slice(
    sample_values: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_weights: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    sampled = sample_values[neighbor_idx]
    valid = np.isfinite(sampled)
    weights = np.where(valid, neighbor_weights, 0.0)
    weight_sums = weights.sum(axis=1, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.where(weight_sums > 0, weights / weight_sums, 0.0)

    raster_flat = np.sum(np.where(valid, sampled, 0.0) * normalized, axis=1)
    raster_flat = np.where(weight_sums[:, 0] > 0, raster_flat, np.nan)
    return raster_flat.reshape(height, width).astype(np.float32)


def save_variable_raster(
    output_path: Path,
    variable: str,
    timestamps: list[str],
    cube: np.ndarray,
    target_grid: dict[str, Any],
    bbox: tuple[float, float, float, float],
    sample_spacing: float,
    unit: str | None,
) -> None:
    profile = {
        "driver": "GTiff",
        "height": target_grid["height"],
        "width": target_grid["width"],
        "count": cube.shape[0],
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": target_grid["transform"],
        "nodata": np.nan,
        "compress": "deflate",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cube)
        dst.update_tags(
            variable=variable,
            unit=unit or "",
            bbox=",".join(str(v) for v in bbox),
            sample_spacing_m=sample_spacing,
        )
        for band_idx, timestamp in enumerate(timestamps, start=1):
            dst.set_band_description(band_idx, timestamp)
            dst.update_tags(band_idx, timestamp=timestamp)


def save_metadata(
    output_dir: Path,
    *,
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    resolution: float,
    sample_spacing: float,
    timestamp_count: int,
    sample_point_count: int,
    target_grid: dict[str, Any],
    units: dict[str, str],
    variables: list[str],
) -> None:
    metadata = {
        "bbox_wgs84": {
            "lon_min": bbox[0],
            "lat_min": bbox[1],
            "lon_max": bbox[2],
            "lat_max": bbox[3],
        },
        "time_range_utc": {
            "start": start.strftime("%Y-%m-%dT%H:%M"),
            "end": end.strftime("%Y-%m-%dT%H:%M"),
            "hours": timestamp_count,
        },
        "output_grid": {
            "crs": "EPSG:32634",
            "resolution_m": resolution,
            "width": target_grid["width"],
            "height": target_grid["height"],
            "bounds_utm": [float(v) for v in target_grid["bounds_utm"]],
        },
        "weather_sampling": {
            "sample_spacing_m": sample_spacing,
            "sample_point_count": sample_point_count,
        },
        "variables": {
            variable: {
                "unit": units.get(variable, ""),
                "path": f"{variable}.tif",
            }
            for variable in variables
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    bbox = parse_bbox(args.bbox)
    start = parse_iso_datetime(args.start)
    end = parse_iso_datetime(args.end)

    if start > end:
        raise SystemExit("Start must be earlier than or equal to end.")
    if args.sample_spacing <= 0 or args.resolution <= 0:
        raise SystemExit("resolution and sample-spacing must be positive.")

    hourly_variables = [item.strip() for item in args.hourly.split(",") if item.strip()]
    if not hourly_variables:
        raise SystemExit("At least one hourly variable is required.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_grid = create_target_grid(bbox, args.resolution)
    sample_points = create_sample_points(bbox, args.sample_spacing)
    timestamps, sample_cube, units = collect_sample_cube(
        sample_points=sample_points,
        start=start,
        end=end,
        hourly_variables=hourly_variables,
        timezone_name=args.timezone,
        temperature_unit=args.temperature_unit,
        wind_speed_unit=args.wind_speed_unit,
        precipitation_unit=args.precipitation_unit,
    )

    neighbor_idx, neighbor_weights = precompute_idw_neighbors(
        target_grid["grid_x"],
        target_grid["grid_y"],
        sample_points["x"],
        sample_points["y"],
        args.idw_neighbors,
        args.idw_power,
    )

    for variable in hourly_variables:
        stacks: list[np.ndarray] = []
        for time_idx in range(len(timestamps)):
            raster_slice = interpolate_time_slice(
                sample_cube[variable][time_idx],
                neighbor_idx,
                neighbor_weights,
                target_grid["height"],
                target_grid["width"],
            )
            stacks.append(raster_slice)

        cube = np.stack(stacks, axis=0)
        save_variable_raster(
            output_path=output_dir / f"{variable}.tif",
            variable=variable,
            timestamps=timestamps,
            cube=cube,
            target_grid=target_grid,
            bbox=bbox,
            sample_spacing=args.sample_spacing,
            unit=units.get(variable),
        )
        print(f"Saved {output_dir / f'{variable}.tif'}")

    save_metadata(
        output_dir,
        bbox=bbox,
        start=start,
        end=end,
        resolution=args.resolution,
        sample_spacing=args.sample_spacing,
        timestamp_count=len(timestamps),
        sample_point_count=int(sample_points["x"].shape[0]),
        target_grid=target_grid,
        units=units,
        variables=hourly_variables,
    )
    print(f"Saved {output_dir / 'metadata.json'}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
