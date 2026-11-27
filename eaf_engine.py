"""Shared catalog, model, grid, and export logic for the Streamlit dashboard.

The formulas here mirror the static aftershock website so the Streamlit view
and the downloadable website produce the same prototype estimates.
"""

from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

BASE_DIR = Path(__file__).resolve().parent
ASSETS = BASE_DIR / "assets"

WINDOWS = {
    "24h": {"hours": 24, "days": 1, "short": "24 h"},
    "7d": {"hours": 168, "days": 7, "short": "7 days"},
    "30d": {"hours": 720, "days": 30, "short": "30 days"},
}

MAP = {
    "x_min": -60.0,
    "x_max": 60.0,
    "y_min": -37.5,
    "y_max": 37.5,
    "cell": 3.0,
    "cols": 40,
    "rows": 25,
}

FEEDS = {
    "live-day": (
        "Live · past day",
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.csv",
        "USGS live feed",
    ),
    "live-week": (
        "Live · past week",
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.csv",
        "USGS live feed",
    ),
    "live-month": (
        "Live · past month",
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv",
        "USGS live feed",
    ),
    "bundled": (
        "Bundled snapshot · past week",
        str(ASSETS / "usgs-all-week.csv"),
        "Bundled USGS snapshot",
    ),
}

REGIONS = {
    "catalog": {"name": "USGS catalog event", "angle": 0.0, "faultLength": 60.0, "seed": 991},
    "japan": {"name": "Japan Trench", "angle": 17.0, "faultLength": 92.0, "seed": 1847},
    "chile": {"name": "Chile Margin", "angle": -27.0, "faultLength": 78.0, "seed": 2711},
    "california": {"name": "California", "angle": -7.0, "faultLength": 54.0, "seed": 3499},
}

PRESETS = {
    "japan": {"magnitude": 6.4, "depth": 28, "elapsed": 6, "window": "7d"},
    "chile": {"magnitude": 7.2, "depth": 35, "elapsed": 12, "window": "7d"},
    "california": {"magnitude": 5.8, "depth": 12, "elapsed": 2, "window": "24h"},
}

SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
BANDS = [
    (5.0, "0–5 km"),
    (10.0, "5–10 km"),
    (20.0, "10–20 km"),
    (40.0, "20–40 km"),
    (float("inf"), "40+ km"),
]

OMORI_C = 0.05
OMORI_P = 1.1
OMORI_NORM = None  # set below
MAX_CATALOG_ROWS = 20000


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def omori_integral(start_days: float, end_days: float) -> float:
    exponent = 1.0 - OMORI_P
    return ((start_days + OMORI_C) ** exponent - (end_days + OMORI_C) ** exponent) / exponent


OMORI_NORM = omori_integral(0.0, 30.0)


def js_number(value: Any) -> float:
    """Approximate JavaScript Number(): blank becomes 0, invalid becomes nan."""
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return float("nan")


def finite(value: float) -> bool:
    return math.isfinite(value)


def iso_utc(time_ms: float) -> str:
    return datetime.fromtimestamp(time_ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_usgs_csv(text: str, source_label: str, source_url: str, kind: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse a USGS-format CSV into normalized event dictionaries."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not reader.fieldnames:
        raise ValueError("The CSV has no header row.")

    headers = {name.strip().lower(): name for name in reader.fieldnames if name}
    required = {"time", "latitude", "longitude", "depth", "mag"}
    if not required.issubset(headers):
        raise ValueError("missing USGS columns: time, latitude, longitude, depth, mag")

    rows: List[Dict[str, Any]] = []
    for index, raw in enumerate(reader, start=1):
        def field(name: str) -> Any:
            key = headers.get(name)
            return raw.get(key, "") if key else ""

        try:
            time = pd_timestamp_ms(str(field("time")))
        except Exception:
            time = float("nan")
        lat = js_number(field("latitude"))
        lon = js_number(field("longitude"))
        depth = js_number(field("depth"))
        mag = js_number(field("mag"))

        if not all(finite(v) for v in (time, lat, lon, depth, mag)):
            continue
        if math.isnan(time):
            continue

        event_type = str(field("type") or "earthquake").strip().lower() or "earthquake"
        if event_type != "earthquake":
            continue

        rows.append(
            {
                "id": str(field("id") or f"row-{index}"),
                "time": time,
                "lat": lat,
                "lon": lon,
                "depth": depth,
                "mag": mag,
                "mag_type": str(field("magtype") or "").strip(),
                "place": str(field("place") or "").strip(),
                "type": event_type,
                "status": str(field("status") or "").strip(),
                "net": str(field("net") or "").strip(),
                "url": str(field("url") or "").strip(),
            }
        )

    rows.sort(key=lambda r: (-r["time"], str(r["id"])))
    rows = rows[:MAX_CATALOG_ROWS]
    meta = {
        "kind": kind,
        "label": source_label,
        "url": source_url,
        "count": len(rows),
        "latestTime": rows[0]["time"] if rows else None,
    }
    return rows, meta


def pd_timestamp_ms(value: str) -> float:
    """Parse an ISO time to epoch milliseconds without importing pandas here."""
    from datetime import datetime

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() * 1000.0


@lru_cache(maxsize=8)
def load_catalog(key: str) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, Any]]:
    """Load a bundled or live feed; live feeds fall back to the bundled CSV."""
    label, url, kind = FEEDS[key]
    fallback = False
    used_url = url
    used_label = label
    used_kind = kind

    if key.startswith("live-"):
        try:
            response = requests.get(url, timeout=14, headers={"Cache-Control": "no-cache"})
            response.raise_for_status()
            text = response.text
        except Exception:
            fallback = True
            used_label, path_url, used_kind = FEEDS["bundled"]
            used_url = path_url + " (fallback)"
            text = Path(path_url).read_text(encoding="utf-8-sig")
    else:
        text = Path(url).read_text(encoding="utf-8-sig")

    rows, meta = parse_usgs_csv(text, used_label, used_url, used_kind)
    meta["fallback"] = fallback
    return tuple(rows), meta


@lru_cache(maxsize=1)
def load_model() -> Optional[Dict[str, Any]]:
    path = ASSETS / "ml-model.json"
    if not path.exists():
        return None
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(model.get("trees"), list) or not isinstance(model.get("features"), list):
        return None
    return model


def mainshock_options(rows: Sequence[Dict[str, Any]], limit: int = 15) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: (-r["mag"], -r["time"]))
    return ordered[:limit]


def find_mainshock(rows: Sequence[Dict[str, Any]], mainshock_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if mainshock_id:
        for row in rows:
            if row["id"] == mainshock_id:
                return row
    return rows[0] if rows else None


def infer_trend(events: Sequence[Dict[str, Any]], main: Dict[str, Any]) -> Dict[str, float]:
    if len(events) < 3:
        angle = 12.0 if main["mag"] >= 7 else -6.0
        fault_length = clamp(40.0 + (main["mag"] - 5.0) * 10.0, 40.0, 140.0)
        return {"angle": angle, "faultLength": fault_length, "strength": 0.0}

    sample = events[:80]
    mean_x = sum(e["x"] for e in sample) / len(sample)
    mean_y = sum(e["y"] for e in sample) / len(sample)
    cov_xx = sum((e["x"] - mean_x) ** 2 for e in sample) / len(sample)
    cov_yy = sum((e["y"] - mean_y) ** 2 for e in sample) / len(sample)
    cov_xy = sum((e["x"] - mean_x) * (e["y"] - mean_y) for e in sample) / len(sample)

    angle = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy) * 180.0 / math.pi
    half = (cov_xx + cov_yy) / 2.0
    rad = math.sqrt(((cov_xx - cov_yy) / 2.0) ** 2 + cov_xy**2)
    eigen_max = half + rad
    eigen_min = max(0.0, half - rad)
    fault_length = clamp(math.sqrt(max(0.0, eigen_max)) * 4.0, 40.0, 150.0)
    strength = (eigen_max - eigen_min) / (eigen_max + eigen_min) if eigen_max + eigen_min > 0 else 0.0
    return {"angle": angle, "faultLength": fault_length, "strength": strength}


def compute_analysis(
    rows: Sequence[Dict[str, Any]],
    mainshock_id: Optional[str],
    latest_time_ms: Optional[float],
) -> Optional[Dict[str, Any]]:
    main = find_mainshock(rows, mainshock_id)
    if main is None:
        return None

    reference_time = latest_time_ms if latest_time_ms is not None else main["time"]
    available_hours = clamp((reference_time - main["time"]) / 3600000.0, 0.0, 720.0)
    radius_km = clamp(60.0 + (main["mag"] - 5.0) * 45.0, 80.0, 220.0)
    lon_scale = 111.32 * max(0.05, math.cos(math.radians(main["lat"])))
    lat_scale = 111.32

    observed: List[Dict[str, Any]] = []
    for row in rows:
        if row["id"] == main["id"] or row["type"] != "earthquake":
            continue
        if row["time"] <= main["time"]:
            continue
        hours = (row["time"] - main["time"]) / 3600000.0
        if hours > available_hours or hours > 720.0:
            continue
        x = (row["lon"] - main["lon"]) * lon_scale
        y = (row["lat"] - main["lat"]) * lat_scale
        distance = math.hypot(x, y)
        if distance > radius_km:
            continue
        observed.append({"row": row, "x": x, "y": y, "distance": distance, "hours": hours})

    observed.sort(key=lambda e: e["hours"])
    return {
        "main": main,
        "referenceTime": reference_time,
        "availableHours": available_hours,
        "radiusKm": radius_km,
        "observed": observed,
        "observedM3": sum(1 for e in observed if e["row"]["mag"] >= 3.0),
        "trend": infer_trend(observed, main),
    }


def observed_through(analysis: Optional[Dict[str, Any]], elapsed: float) -> List[Dict[str, Any]]:
    if not analysis:
        return []
    return [e for e in analysis["observed"] if e["hours"] <= elapsed + 0.001]


def productivity_scale_at(
    analysis: Optional[Dict[str, Any]],
    elapsed: float,
    magnitude: float,
    depth: float,
) -> float:
    early = observed_through(analysis, elapsed)
    early_m3 = sum(1 for e in early if e["row"]["mag"] >= 3.0)
    if not analysis or elapsed <= 0 or early_m3 <= 0:
        return 1.0
    magnitude_factor = 10 ** (0.8 * (magnitude - 5.0))
    depth_factor = 1.0 + depth / 500.0
    baseline_early = 12.0 * magnitude_factor * depth_factor * (omori_integral(0.0, elapsed / 24.0) / OMORI_NORM)
    if baseline_early <= 0:
        return 1.0
    return clamp(early_m3 / max(0.25, baseline_early), 0.2, 5.0)


def build_features(
    analysis: Optional[Dict[str, Any]],
    magnitude: float,
    depth: float,
    elapsed: float,
    window_key: str,
) -> Dict[str, float]:
    window_hours = WINDOWS[window_key]["hours"]
    main = analysis["main"] if analysis else {"mag": magnitude, "depth": depth}
    radius = analysis["radiusKm"] if analysis else clamp(60.0 + (magnitude - 5.0) * 45.0, 80.0, 220.0)
    depth = clamp(depth, 0.0, 700.0)
    elapsed = clamp(elapsed, 0.0, 720.0)

    early = observed_through(analysis, elapsed)
    early_m3 = sum(1 for e in early if e["row"]["mag"] >= 3.0)
    early_all = len(early)
    early_max_mag = max((e["row"]["mag"] for e in early), default=main["mag"])
    trend = infer_trend(early, main) if analysis else {"angle": 0.0, "faultLength": 60.0, "strength": 0.0}

    magnitude_factor = 10 ** (0.8 * (magnitude - 5.0))
    depth_factor = 1.0 + depth / 500.0
    omori_fraction = omori_integral(elapsed / 24.0, (elapsed + window_hours) / 24.0) / OMORI_NORM
    baseline_expected = 3.0 + 12.0 * magnitude_factor * depth_factor * omori_fraction
    baseline_early = (
        12.0 * magnitude_factor * depth_factor * (omori_integral(0.0, elapsed / 24.0) / OMORI_NORM)
        if elapsed > 0
        else 0.0
    )
    productivity = productivity_scale_at(analysis, elapsed, magnitude, depth)
    baseline_prediction = baseline_expected * productivity
    theta = math.radians(trend["angle"])

    return {
        "magnitude": magnitude,
        "depthKm": depth,
        "startHours": elapsed,
        "windowHours": window_hours,
        "radiusKm": radius,
        "log10MagnitudeFactor": math.log10(magnitude_factor),
        "depthFactor": depth_factor,
        "omoriFraction": omori_fraction,
        "log1pEarlyM3": math.log1p(early_m3),
        "log1pEarlyAll": math.log1p(early_all),
        "earlyM3PerHour": early_m3 / max(elapsed, 1.0),
        "earlyMaxMagRelative": max(0.0, early_max_mag - magnitude),
        "productivityScale": productivity,
        "faultLengthScaled": trend["faultLength"] / 100.0,
        "trendCos": math.cos(theta),
        "trendSin": math.sin(theta),
        "trendStrength": trend["strength"],
        "log1pBaselineExpected": math.log1p(baseline_expected),
        "log1pBaselinePrediction": math.log1p(baseline_prediction),
    }


def predict_ml(model: Optional[Dict[str, Any]], features: Dict[str, float]) -> Optional[float]:
    if not model:
        return None
    trees = model.get("trees") or []
    if not trees:
        return None
    feature_names = model.get("features") or list(features.keys())
    feature_values = [features.get(name, 0.0) for name in feature_names]

    total = 0.0
    for tree in trees:
        node = 0
        guard = 0
        left = tree.get("l") or []
        right = tree.get("r") or []
        feat = tree.get("f") or []
        thresh = tree.get("t") or []
        values = tree.get("v") or []
        while node < len(left) and left[node] != -1 and guard < 256:
            idx = feat[node] if node < len(feat) else -1
            value = feature_values[idx] if 0 <= idx < len(feature_values) else float("nan")
            if not finite(value):
                break
            node = left[node] if value <= thresh[node] else right[node]
            guard += 1
        total += values[node] if node < len(values) else 0.0

    mean = total / len(trees)
    return clamp(math.expm1(clamp(mean, 0.0, 20.0)), 0.02, 600.0)


def predict_baseline(
    analysis: Optional[Dict[str, Any]],
    magnitude: float,
    depth: float,
    elapsed: float,
    window_key: str,
) -> float:
    window_hours = WINDOWS[window_key]["hours"]
    magnitude_factor = 10 ** (0.8 * (magnitude - 5.0))
    depth_factor = 1.0 + depth / 500.0
    temporal = omori_integral(elapsed / 24.0, (elapsed + window_hours) / 24.0) / OMORI_NORM
    productivity = productivity_scale_at(analysis, elapsed, magnitude, depth)
    baseline_expected = 3.0 + 12.0 * magnitude_factor * temporal * depth_factor
    return clamp(baseline_expected * productivity, 0.02, 600.0)


def hash2(ix: int, iy: int, seed: int) -> float:
    value = (ix * 374761393 + iy * 668265263 + seed * 1274126177) & 0xFFFFFFFF
    value = ((value ^ (value >> 13)) * 1274126177) & 0xFFFFFFFF
    value ^= value >> 16
    return (value & 0xFFFFFFFF) / 4294967295.0


def value_noise(x: float, y: float, seed: int) -> float:
    x0, y0 = math.floor(x), math.floor(y)
    tx, ty = x - x0, y - y0
    sx = tx * tx * (3.0 - 2.0 * tx)
    sy = ty * ty * (3.0 - 2.0 * ty)
    n00 = hash2(x0, y0, seed)
    n10 = hash2(x0 + 1, y0, seed)
    n01 = hash2(x0, y0 + 1, seed)
    n11 = hash2(x0 + 1, y0 + 1, seed)
    a = n00 + (n10 - n00) * sx
    b = n01 + (n11 - n01) * sx
    return a + (b - a) * sy


def cell_centers() -> List[Tuple[float, float]]:
    centers = []
    for row in range(MAP["rows"]):
        y = MAP["y_max"] - (row + 0.5) * MAP["cell"]
        for col in range(MAP["cols"]):
            x = MAP["x_min"] + (col + 0.5) * MAP["cell"]
            centers.append((x, y))
    return centers


def region_for(region_key: str, analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    region = dict(REGIONS.get(region_key, REGIONS["catalog"]))
    if region_key == "catalog" and analysis:
        region["angle"] = analysis["trend"]["angle"]
        region["faultLength"] = analysis["trend"]["faultLength"]
    return region


def build_grid(
    expected_total: float,
    analysis: Optional[Dict[str, Any]],
    magnitude: float,
    depth: float,
    elapsed: float,
    region_key: str,
) -> Dict[str, Any]:
    centers = cell_centers()
    region = region_for(region_key, analysis)
    theta = math.radians(region["angle"])
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    sigma = (6.5 + (magnitude - 5.0) * 3.0) * (0.9 + depth / 220.0)
    ridge_width = max(1.4, 2.0 + (magnitude - 5.0) * 1.0)
    fault_half = (region["faultLength"] / 2.0) * (0.55 + (magnitude - 5.0) * 0.10)
    seed = int(region["seed"])

    prior = []
    for x, y in centers:
        along = x * cos_t + y * sin_t
        across = -x * sin_t + y * cos_t
        core = math.exp(-(along**2 / (2 * (sigma * 1.5) ** 2) + across**2 / (2 * (sigma * 0.6) ** 2)))
        ridge = math.exp(-(across**2 / (2 * ridge_width**2)) + -(along**2 / (2 * (fault_half * 1.15) ** 2)))
        noise_a = value_noise(x * 0.075, y * 0.075, seed) - 0.5
        noise_b = value_noise(x * 0.21, y * 0.21, seed + 19) - 0.5
        texture = 0.65 * noise_a + 0.35 * noise_b
        prior.append(max(0.00001, 0.92 * core + 0.78 * ridge + 0.06 + texture * 0.075))

    prior_sum = sum(prior) or 1.0
    prior_weights = [value / prior_sum for value in prior]

    early = observed_through(analysis, elapsed)
    observed_kernel = [0.0] * len(centers)
    observed_sum = 0.0
    if len(early) >= 2:
        for event in sorted(early, key=lambda e: -e["row"]["mag"])[:80]:
            mag = event["row"]["mag"]
            sigma_km = clamp(4.0 + max(0.0, 4.0 - mag) * 1.2 + (2.0 if mag >= 5 else 0.0), 4.0, 12.0)
            strength = 0.5 + max(0.0, mag) / 10.0
            radius_cells = math.ceil((sigma_km * 3) / MAP["cell"])
            ex, ey = event["x"], event["y"]
            for idx, (x, y) in enumerate(centers):
                if abs(x - ex) > radius_cells * MAP["cell"] or abs(y - ey) > radius_cells * MAP["cell"]:
                    continue
                d2 = (x - ex) ** 2 + (y - ey) ** 2
                contribution = strength * math.exp(-d2 / (2 * sigma_km**2))
                observed_kernel[idx] += contribution
                observed_sum += contribution

    use_observed = observed_sum > 0 and len(early) >= 3
    if use_observed:
        weights = [0.62 * prior_weights[i] + 0.38 * (observed_kernel[i] / observed_sum) for i in range(len(centers))]
        spatial_source = "observed + prior"
    else:
        weights = prior_weights
        spatial_source = "prior"

    probability = [min(0.995, 1.0 - math.exp(-expected_total * weight)) for weight in weights]
    high_cells = sum(1 for p in probability if p >= 0.5)

    peak_index = max(range(len(weights)), key=lambda i: weights[i])
    peak_x, peak_y = centers[peak_index]
    peak_distance = math.hypot(peak_x, peak_y)

    zone_acc: Dict[Tuple[int, int], Dict[str, float]] = {}
    for (x, y), weight in zip(centers, weights):
        distance = math.hypot(x, y)
        bearing = math.atan2(x, y) * 180.0 / math.pi
        normalized = (bearing + 360.0) % 360.0
        sector = int(((normalized + 22.5) % 360.0) / 45.0)
        band = next(i for i, (max_km, _) in enumerate(BANDS) if distance < max_km)
        key = (sector, band)
        entry = zone_acc.setdefault(key, {"mass": 0.0, "count": 0})
        entry["mass"] += weight
        entry["count"] += 1

    zones = []
    for (sector, band), entry in zone_acc.items():
        p = 1.0 - math.exp(-expected_total * entry["mass"])
        zones.append(
            {
                "zone": f"{SECTORS[sector]} · {BANDS[band][1]}",
                "probability": p,
                "expected": expected_total * entry["mass"],
                "area_km2": entry["count"] * 9.0,
                "level": "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low",
            }
        )
    zones.sort(key=lambda z: (-z["probability"], -z["expected"]))

    return {
        "centers": centers,
        "weights": weights,
        "probabilities": probability,
        "region": region,
        "spatialSource": spatial_source,
        "peak": {"x": peak_x, "y": peak_y, "distanceKm": peak_distance, "probability": probability[peak_index]},
        "highAreaKm2": high_cells * 9.0,
        "zones": zones,
    }


def daily_rates(
    expected_total: float,
    analysis: Optional[Dict[str, Any]],
    magnitude: float,
    depth: float,
    elapsed: float,
    window_key: str,
    engine: str,
) -> Tuple[List[float], int]:
    magnitude_factor = 10 ** (0.8 * (magnitude - 5.0))
    depth_factor = 1.0 + depth / 500.0
    scale = 1.0 if engine == "ml" else productivity_scale_at(analysis, elapsed, magnitude, depth)
    productivity = 12.0 * magnitude_factor * depth_factor * scale

    raw = []
    for day in range(1, 31):
        start = (elapsed + (day - 1) * 24.0) / 24.0
        end = (elapsed + day * 24.0) / 24.0
        raw.append(productivity * omori_integral(start, end) / OMORI_NORM)

    window_days = WINDOWS[window_key]["days"]
    raw_sum = sum(raw[:window_days])
    rate_scale = expected_total / raw_sum if raw_sum > 0 else 1.0
    return [value * rate_scale for value in raw], window_days


def csv_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    if any(ch in text for ch in (",", '"', "\n", "\r")):
        return '"' + text.replace('"', '""') + '"'
    return text


def bom_csv(lines: Iterable[Sequence[Any]]) -> str:
    out = io.StringIO(newline="")
    for line in lines:
        out.write(",".join(csv_escape(value) for value in line) + "\r\n")
    return "\ufeff" + out.getvalue()


def export_grid_csv(
    grid: Dict[str, Any],
    scenario: Dict[str, Any],
    catalog_meta: Optional[Dict[str, Any]],
    analysis: Optional[Dict[str, Any]],
    engine: str,
    model: Optional[Dict[str, Any]],
) -> str:
    model_validation = (
        model.get("validation", {}).get("note", "Random Forest v1.0 prototype")
        if model and engine == "ml"
        else "Transparent baseline; no trained model used."
    )
    meta = catalog_meta or {}
    main = analysis["main"] if analysis else {}
    lines: List[List[Any]] = [
        ["Aftershock Forecast Studio - USGS CSV prototype"],
        ["scenario", REGIONS[scenario["region"]]["name"]],
        ["magnitude", f"{scenario['magnitude']:.1f}"],
        ["depth_km", str(int(scenario["depth"]))],
        ["hours_since_mainshock", str(int(scenario["elapsed"]))],
        ["forecast_window", WINDOWS[scenario["window"]]["short"]],
        ["model", "Random Forest v1.0 | ETAS-inspired baseline v0.2"],
        ["model_validation", model_validation],
        ["catalog_source", f"{meta.get('kind', 'none')} · {meta.get('label', '')}"],
        ["catalog_url", meta.get("url", "")],
        ["catalog_events", str(meta.get("count", 0))],
        ["mainshock_id", main.get("id", "")],
        ["mainshock_time", iso_utc(main["time"]) if main else ""],
        ["mainshock_place", main.get("place", "")],
        ["spatial_weights", grid["spatialSource"]],
        ["note", "Model-derived demonstration output. Not validated and not for operational use."],
        [],
        ["east_km", "north_km", "probability", "expected_events", "distance_km", "bearing_deg"],
    ]

    expected_total = scenario["expectedTotal"]
    for idx, (x, y) in enumerate(grid["centers"]):
        bearing = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
        lines.append(
            [
                f"{x:.2f}",
                f"{y:.2f}",
                f"{grid['probabilities'][idx]:.6f}",
                f"{expected_total * grid['weights'][idx]:.4f}",
                f"{math.hypot(x, y):.2f}",
                f"{bearing:.2f}",
            ]
        )
    return bom_csv(lines)


def export_observed_csv(
    analysis: Optional[Dict[str, Any]],
    catalog_meta: Optional[Dict[str, Any]],
    window_key: str,
) -> str:
    if not analysis:
        return bom_csv([["No catalog loaded."]])
    main = analysis["main"]
    meta = catalog_meta or {}
    lines: List[List[Any]] = [
        ["USGS Earthquake Catalog - observed rows used by the prototype"],
        ["source", f"{meta.get('kind', '')} · {meta.get('label', '')}"],
        ["catalog_url", meta.get("url", "")],
        ["mainshock_id", main.get("id", "")],
        ["mainshock_mag", f"{main['mag']:.1f}"],
        ["search_radius_km", str(round(analysis["radiusKm"]))],
        ["available_hours", str(round(analysis["availableHours"]))],
        ["observed_m3_count", str(analysis["observedM3"])],
        ["note", "Observed USGS rows. Not model output."],
        [],
        [
            "time_utc",
            "latitude",
            "longitude",
            "depth_km",
            "magnitude",
            "mag_type",
            "place",
            "event_id",
            "distance_km",
            "hours_after_mainshock",
            "usgs_url",
        ],
    ]
    for event in analysis["observed"]:
        row = event["row"]
        lines.append(
            [
                iso_utc(row["time"]),
                f"{row['lat']:.5f}",
                f"{row['lon']:.5f}",
                f"{row['depth']:.2f}",
                f"{row['mag']:.2f}",
                row["mag_type"],
                row["place"],
                row["id"],
                f"{event['distance']:.2f}",
                f"{event['hours']:.2f}",
                row["url"],
            ]
        )
    return bom_csv(lines)


def export_needs_csv(needs: Dict[str, bool], notes: str) -> str:
    lines: List[List[Any]] = [
        ["Earthquake Aftershock Forecasting local needs checklist"],
        ["saved_at", iso_utc(datetime.now(tz=timezone.utc).timestamp() * 1000.0)],
        ["note", "Local-only export. Not sent to any agency. Do not include sensitive personal information."],
        [],
        ["need", "checked"],
    ]
    for need, checked in needs.items():
        lines.append([need, "yes" if checked else "no"])
    lines.append([])
    lines.append(["notes", notes])
    return bom_csv(lines)
