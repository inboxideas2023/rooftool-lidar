"""Read USGS 3DEP LiDAR points straight from the public Entwine Point Tile (EPT) archive.

No PDAL needed: we walk the EPT octree, download only the .laz tiles that touch a small
box around the house, and decode them with laspy + lazrs.
"""
from __future__ import annotations

import io
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import httpx
import laspy
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

RESOURCES_URL = os.getenv(
    "LIDAR_INDEX_URL",
    "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson",
)
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/rooftool-cache")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))

_client = httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True)
_index_lock = threading.Lock()
_index: tuple[STRtree, list[dict], list] | None = None


@dataclass
class LidarPoints:
    xyz: np.ndarray            # (N, 3) in the UTM zone of the query, meters
    classification: np.ndarray # (N,) ASPRS class codes
    dataset: str
    year: int | None
    utm_crs: CRS


# ---------------------------------------------------------------- dataset index
def _load_index():
    global _index
    with _index_lock:
        if _index is not None:
            return _index
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, "resources.geojson")
        if not os.path.exists(path):
            r = _client.get(RESOURCES_URL)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        with open(path) as f:
            data = json.load(f)
        geoms, props = [], []
        for feat in data["features"]:
            geoms.append(shape(feat["geometry"]))
            props.append(feat["properties"])
        _index = (STRtree(geoms), props, geoms)
        return _index


def _year(name: str) -> int | None:
    years = re.findall(r"(?:19|20)\d\d", name)
    return int(years[-1]) if years else None


def datasets_at(lat: float, lon: float) -> list[dict]:
    """All 3DEP datasets covering the point, newest first."""
    tree, props, geoms = _load_index()
    p = Point(lon, lat)
    hits = [props[i] | {"year": _year(props[i]["name"])}
            for i in tree.query(p) if geoms[i].contains(p)]
    return sorted(hits, key=lambda d: d["year"] or 0, reverse=True)


# ---------------------------------------------------------------- EPT reading
@lru_cache(maxsize=256)
def _get_json(url: str) -> dict:
    r = _client.get(url)
    r.raise_for_status()
    return r.json()


def _get_bytes(url: str) -> bytes:
    r = _client.get(url)
    r.raise_for_status()
    return r.content


def utm_crs_for(lat: float, lon: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def _ept_crs(meta: dict) -> CRS:
    srs = meta.get("srs", {})
    if srs.get("authority") and srs.get("horizontal"):
        return CRS.from_string(f"{srs['authority']}:{srs['horizontal']}")
    return CRS.from_wkt(srs["wkt"])


def _child_bounds(b, dx, dy, dz):
    xmin, ymin, zmin, xmax, ymax, zmax = b
    mx, my, mz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    return (
        mx if dx else xmin, my if dy else ymin, mz if dz else zmin,
        xmax if dx else mx, ymax if dy else my, zmax if dz else mz,
    )


def _nodes_in_box(base: str, meta: dict, box) -> list[str]:
    """Keys of every octree node with points that intersects the XY box."""
    qxmin, qymin, qxmax, qymax = box
    hier: dict[str, int] = dict(_get_json(f"{base}/ept-hierarchy/0-0-0-0.json"))
    keys: list[str] = []
    stack = [((0, 0, 0, 0), tuple(meta["bounds"]))]
    while stack:
        (d, x, y, z), b = stack.pop()
        if b[3] < qxmin or b[0] > qxmax or b[4] < qymin or b[1] > qymax:
            continue
        key = f"{d}-{x}-{y}-{z}"
        if key not in hier:
            continue
        if hier[key] == -1:  # subtree lives in its own file
            hier.update(_get_json(f"{base}/ept-hierarchy/{key}.json"))
        if hier[key] > 0:
            keys.append(key)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    stack.append(((d + 1, 2 * x + dx, 2 * y + dy, 2 * z + dz),
                                  _child_bounds(b, dx, dy, dz)))
    return keys


def _read_tile(url: str, box) -> tuple[np.ndarray, np.ndarray]:
    las = laspy.read(io.BytesIO(_get_bytes(url)))
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    m = (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])
    cls = np.asarray(las.classification).astype(np.uint8)[m]
    try:
        nret = np.asarray(las.number_of_returns)[m]
        # Unlabelled points from a pulse that came back more than once went through
        # something (leaves). Roofs give a single return. Label those as vegetation.
        cls = np.where((cls <= 1) & (nret > 1), 5, cls).astype(np.uint8)
    except Exception:
        pass
    return np.column_stack([x[m], y[m], z[m]]), cls


def fetch_points(lat: float, lon: float, radius_m: float = 35.0,
                 dataset: dict | None = None) -> LidarPoints:
    """Download all LiDAR points within radius_m (square box) of lat/lon."""
    candidates = [dataset] if dataset else datasets_at(lat, lon)
    if not candidates:
        raise LookupError("No USGS 3DEP LiDAR coverage at this location.")

    utm = utm_crs_for(lat, lon)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    cx, cy = to_utm.transform(lon, lat)

    last_err = None
    for ds in candidates:
        try:
            meta_url = ds["url"]
            base = meta_url.rsplit("/", 1)[0]
            meta = _get_json(meta_url)
            ept = _ept_crs(meta)
            utm_to_ept = Transformer.from_crs(utm, ept, always_xy=True)
            ept_to_utm = Transformer.from_crs(ept, utm, always_xy=True)

            r = radius_m * 1.05
            corners = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
            ex, ey = utm_to_ept.transform(*zip(*corners))
            box = (min(ex), min(ey), max(ex), max(ey))

            keys = _nodes_in_box(base, meta, box)
            urls = [f"{base}/ept-data/{k}.laz" for k in keys]
            with ThreadPoolExecutor(max_workers=8) as pool:
                parts = list(pool.map(lambda u: _read_tile(u, box), urls))
            if not parts:
                continue
            xyz = np.vstack([p[0] for p in parts])
            cls = np.concatenate([p[1] for p in parts])
            if len(xyz) < 50:
                continue
            ux, uy = ept_to_utm.transform(xyz[:, 0], xyz[:, 1])
            xyz = np.column_stack([ux, uy, xyz[:, 2]])
            keep = (np.abs(xyz[:, 0] - cx) <= radius_m) & (np.abs(xyz[:, 1] - cy) <= radius_m)
            return LidarPoints(xyz[keep], cls[keep].astype(np.uint8), ds["name"], ds.get("year"), utm)
        except Exception as e:  # try the next (older) dataset
            last_err = e
    raise LookupError(f"LiDAR found for this area but no usable points ({last_err}).")
