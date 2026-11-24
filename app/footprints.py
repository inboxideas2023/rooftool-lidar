"""Free, recent building outlines used to cross-check the LiDAR scan.

Sources (both free, no API key):
  * Overture Maps buildings - merges OpenStreetMap, Microsoft and Esri building outlines,
    refreshed monthly. Usually much newer than the LiDAR survey.
  * OpenStreetMap via the Overpass API - fallback if Overture is unreachable.

Google Maps imagery is NOT analysed: Google's terms forbid deriving data from it. The app
shows the outlines on its Google satellite map so the roofer can eyeball the result.
"""
from __future__ import annotations

import logging
import os
import time

import httpx
from shapely import wkb
from shapely.geometry import MultiPolygon, Polygon, shape

log = logging.getLogger(__name__)

SOURCES = [s.strip() for s in os.getenv("FOOTPRINT_SOURCES", "overture,osm").split(",") if s.strip()]
OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
TIMEOUT = float(os.getenv("FOOTPRINT_TIMEOUT", "25"))

_cache: dict[tuple, tuple[float, list[Polygon], str]] = {}
CACHE_SECONDS = 7 * 24 * 3600


def _polys(geom) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def _overture(bbox) -> list[Polygon]:
    from overturemaps import core  # imported lazily: heavy (pyarrow)

    reader = core.record_batch_reader("building", bbox=bbox,
                                      connect_timeout=10, request_timeout=TIMEOUT)
    out: list[Polygon] = []
    if reader is None:
        return out
    for batch in reader:
        for g in batch.column("geometry").to_pylist():
            if g:
                out.extend(_polys(wkb.loads(g)))
    return out


def _osm(bbox) -> list[Polygon]:
    w, s, e, n = bbox
    q = (f"[out:json][timeout:{int(TIMEOUT)}];"
         f"(way[\"building\"]({s},{w},{n},{e});relation[\"building\"]({s},{w},{n},{e}););out geom;")
    r = httpx.post(OVERPASS_URL, data={"data": q}, timeout=TIMEOUT + 5,
                   headers={"User-Agent": "RoofToolPro/1.0 (rooftoolpro.com)"})
    r.raise_for_status()
    out: list[Polygon] = []
    for el in r.json().get("elements", []):
        if el["type"] == "way" and len(el.get("geometry", [])) >= 4:
            out.append(Polygon([(p["lon"], p["lat"]) for p in el["geometry"]]))
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") == "outer" and len(m.get("geometry", [])) >= 4:
                    out.append(Polygon([(p["lon"], p["lat"]) for p in m["geometry"]]))
    return out


PROVIDERS = {"overture": _overture, "osm": _osm}


def buildings_in_bbox(bbox: tuple[float, float, float, float]) -> tuple[list[Polygon], str | None]:
    """Building outlines (lon/lat polygons) in (west, south, east, north). Returns (polys, source)."""
    key = tuple(round(v, 5) for v in bbox)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1], hit[2]
    for name in SOURCES:
        fn = PROVIDERS.get(name)
        if fn is None:
            continue
        try:
            polys = [p for p in fn(bbox) if p.is_valid or p.buffer(0).is_valid]
            polys = [p if p.is_valid else p.buffer(0) for p in polys]
            if not polys and name != SOURCES[-1]:
                continue                      # empty here; another source may know more
            _cache[key] = (time.time(), polys, name)
            return polys, name
        except Exception as e:  # try the next source
            log.warning("footprint source %s failed: %s", name, e)
    return [], None
