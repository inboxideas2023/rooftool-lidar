"""RoofTool Pro LiDAR service.

Endpoints
  GET  /health
  POST /measure          address or lat/lon  -> facets, pitch, area, edge lengths (auto, from LiDAR)
  POST /measure-property address or lat/lon  -> every structure on the property, measured from
                         LiDAR and cross-checked against current building maps
  POST /structure-at-pin lat/lon of a pin the roofer dropped -> that one structure, measured
  POST /measure-traced   facet polygons the roofer drew -> pitch + true area per facet
"""
from __future__ import annotations

import os
import time
from collections import Counter, OrderedDict

import httpx
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from pyproj import Transformer
from shapely.geometry import Polygon

from .ept import fetch_points, utm_crs_for
from .footprints import buildings_in_bbox
from .property import assign_ownership, find_structures, measure_structures, structure_json
from .roof import FT_PER_M, SQFT_PER_M2, NoRoofFound, measure_roof, plane_for_polygon

API_KEY = os.getenv("SERVICE_API_KEY", "")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

app = FastAPI(title="RoofTool Pro LiDAR service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


def require_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Bad or missing X-API-Key header.")


# ---------------------------------------------------------------- tiny result cache
_cache: OrderedDict[str, dict] = OrderedDict()
CACHE_MAX = 500


def _cache_get(k):
    if k in _cache:
        _cache.move_to_end(k)
        return _cache[k]


def _cache_put(k, v):
    _cache[k] = v
    if len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)


# ---------------------------------------------------------------- helpers
def geocode(address: str) -> tuple[float, float]:
    """Free US Census geocoder (no key, no cost)."""
    r = httpx.get(
        "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
        params={"address": address, "benchmark": "Public_AR_Current", "format": "json"},
        timeout=20,
    )
    r.raise_for_status()
    matches = r.json()["result"]["addressMatches"]
    if not matches:
        raise HTTPException(404, "Address not found.")
    c = matches[0]["coordinates"]
    return float(c["y"]), float(c["x"])


def pitch_label(p12: float) -> str:
    return f"{round(p12)}/12"


class Location(BaseModel):
    address: str | None = None
    lat: float | None = None
    lon: float | None = None

    @model_validator(mode="after")
    def need_one(self):
        if not self.address and (self.lat is None or self.lon is None):
            raise ValueError("Send an address, or lat and lon.")
        return self

    def resolve(self) -> tuple[float, float]:
        # an address wins over lat/lon left at 0 (the /docs example form fills in zeros)
        if self.address and (self.lat in (None, 0) or self.lon in (None, 0)):
            return geocode(self.address)
        return self.lat, self.lon


class TracedFacet(BaseModel):
    points: list[list[float]] = Field(..., description="[[lat, lon], ...] corners of one roof face")
    pitch_over_12: float | None = Field(None, description="Roofer's pitch; used if LiDAR is missing")


class TracedRequest(BaseModel):
    facets: list[TracedFacet]
    waste_pct: float = 10.0


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/measure", dependencies=[Depends(require_key)])
def measure(loc: Location, waste_pct: float = 10.0):
    t0 = time.time()
    lat, lon = loc.resolve()
    key = f"{lat:.6f},{lon:.6f},{waste_pct}"
    if (hit := _cache_get(key)) is not None:
        return hit | {"cached": True}

    try:
        lp = fetch_points(lat, lon)
    except LookupError as e:
        raise HTTPException(404, str(e))
    to_utm = Transformer.from_crs("EPSG:4326", lp.utm_crs, always_xy=True)
    to_ll = Transformer.from_crs(lp.utm_crs, "EPSG:4326", always_xy=True)
    tx, ty = to_utm.transform(lon, lat)

    try:
        res = measure_roof(lp.xyz, lp.classification, (tx, ty))
    except NoRoofFound as e:
        raise HTTPException(422, str(e))

    facets_out, total_sqft = [], 0.0
    pitch_area: Counter = Counter()
    for f in res.facets:
        sloped = f.plan_m2 / max(np.cos(f.slope_rad), 0.2) * SQFT_PER_M2
        total_sqft += sloped
        pitch_area[pitch_label(f.pitch_over_12)] += sloped
        ring = []
        if f.polygon is not None:
            xs, ys = f.polygon.exterior.xy
            lons, lats = to_ll.transform(list(xs), list(ys))
            ring = [[round(a, 7), round(b, 7)] for a, b in zip(lats, lons)]
        facets_out.append({
            "id": f.id,
            "pitch": pitch_label(f.pitch_over_12),
            "pitch_over_12": round(f.pitch_over_12, 1),
            "pitch_degrees": round(np.degrees(f.slope_rad), 1),
            "faces_azimuth_deg": None if f.azimuth_deg is None else round(f.azimuth_deg),
            "plan_sqft": round(f.plan_m2 * SQFT_PER_M2),
            "roof_sqft": round(sloped),
            "polygon_latlon": ring,
        })

    result = {
        "location": {"lat": lat, "lon": lon},
        "lidar": {"dataset": lp.dataset, "year": lp.year,
                  "roof_points": res.point_count, "points_per_m2": round(res.density_per_m2, 1)},
        "totals": {
            "roof_sqft": round(total_sqft),
            "squares": round(total_sqft / 100, 1),
            "squares_with_waste": round(total_sqft * (1 + waste_pct / 100) / 100, 1),
            "waste_pct": waste_pct,
            "predominant_pitch": pitch_area.most_common(1)[0][0] if pitch_area else None,
            "facet_count": len(facets_out),
            "footprint_sqft": round(res.footprint.area * SQFT_PER_M2),
        },
        "edges_ft": {k: round(v * FT_PER_M) for k, v in res.edges_m.items()},
        "facets": facets_out,
        "warnings": res.warnings + (
            [f"LiDAR was flown in {lp.year}; check for additions or re-roofs since then."]
            if lp.year and lp.year < time.gmtime().tm_year - 5 else []),
        "seconds": round(time.time() - t0, 1),
        "cached": False,
    }
    _cache_put(key, result)
    return result


@app.post("/measure-property", dependencies=[Depends(require_key)])
def measure_property(loc: Location, waste_pct: float = 10.0):
    """Every structure on the property: house, garage, ADU, sheds, patio covers.

    LiDAR structures are measured automatically. Structures on current building maps that the
    LiDAR scan doesn't have come back as `needs_trace` with their outline, for the roofer to
    add a pitch (then call /measure-traced, or use roof_sqft_by_pitch directly).
    """
    t0 = time.time()
    lat, lon = loc.resolve()
    key = f"P{lat:.6f},{lon:.6f},{waste_pct}"
    if (hit := _cache_get(key)) is not None:
        return hit | {"cached": True}

    radius = 45.0
    lp = None
    try:
        lp = fetch_points(lat, lon, radius_m=radius + 20)
        utm = lp.utm_crs
    except LookupError:
        utm = utm_crs_for(lat, lon)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    to_ll = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    ax, ay = to_utm.transform(lon, lat)

    # current building outlines for the cross-check (free)
    r = radius + 20
    lons, lats = to_ll.transform([ax - r, ax + r], [ay - r, ay + r])
    polys_ll, fp_source = buildings_in_bbox((min(lons), min(lats), max(lons), max(lats)))
    outlines = None
    if fp_source is not None:
        import shapely
        outlines = [shapely.transform(p, lambda x, y: to_utm.transform(x, y), interleaved=False)
                    for p in polys_ll]

    if lp is None:
        xyz = np.empty((0, 3))
        cls = np.empty(0, dtype=np.uint8)
    else:
        xyz, cls = lp.xyz, lp.classification

    structures, warnings = find_structures(xyz, cls, (ax, ay), outlines, radius_m=radius)
    if lp is None:
        warnings.insert(0, "No USGS LiDAR covers this address. Structures come from building "
                           "maps only; enter a pitch for each.")
    if not structures:
        raise HTTPException(404, "No structures found near this address.")
    main_id = assign_ownership(structures, (ax, ay))
    if main_id is None:
        warnings.append("Couldn't tell which building is at this address. Pick the structures "
                        "to include on the map.")
    measure_structures(structures)

    items = [structure_json(s, to_ll, waste_pct) for s in structures]
    items.sort(key=lambda d: (not d["included"], d["role"] != "main", d["id"]))
    inc = [d for d in items if d["included"]]
    measured = [d for d in inc if d["source"] == "lidar" and "roof_sqft" in d]
    pending = [d for d in inc if d["source"] == "needs_trace"]
    total = sum(d["roof_sqft"] for d in measured)
    edges: Counter = Counter()
    for d in measured:
        edges.update(d["edges_ft"])

    if lp is not None and lp.year and lp.year < time.gmtime().tm_year - 5 and fp_source is None:
        warnings.append(f"LiDAR was flown in {lp.year}; check for additions or new buildings.")

    result = {
        "location": {"lat": lat, "lon": lon},
        "main_structure_id": main_id,
        "totals_included": {
            "roof_sqft_measured": round(total),
            "squares_measured": round(total / 100, 1),
            "squares_with_waste": round(total * (1 + waste_pct / 100) / 100, 1),
            "waste_pct": waste_pct,
            "edges_ft": dict(edges),
            "structures_measured": len(measured),
            "structures_needing_pitch": len(pending),
            "complete": not pending,
        },
        "structures": items,
        "sources": {
            "lidar": None if lp is None else {"dataset": lp.dataset, "year": lp.year},
            "building_maps": fp_source,
        },
        "warnings": warnings,
        "seconds": round(time.time() - t0, 1),
        "cached": False,
    }
    _cache_put(key, result)
    return result


class Pin(BaseModel):
    lat: float
    lon: float
    waste_pct: float = 10.0


@app.post("/structure-at-pin", dependencies=[Depends(require_key)])
def structure_at_pin(pin: Pin):
    """The roofer dropped a pin on a building the property search left out (or missed).

    Returns that structure (plus any addition attached to it), measured from LiDAR, or as
    `needs_trace` with its outline if it's only on the building maps. 404 if nothing is there,
    in which case the app should let the roofer trace it.
    """
    lat, lon = pin.lat, pin.lon
    key = f"PIN{lat:.6f},{lon:.6f},{pin.waste_pct}"
    if (hit := _cache_get(key)) is not None:
        return hit
    lp = None
    try:
        lp = fetch_points(lat, lon, radius_m=35)
        utm = lp.utm_crs
    except LookupError:
        utm = utm_crs_for(lat, lon)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    to_ll = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    px, py = to_utm.transform(lon, lat)

    lons, lats = to_ll.transform([px - 35, px + 35], [py - 35, py + 35])
    polys_ll, fp_source = buildings_in_bbox((min(lons), min(lats), max(lons), max(lats)))
    outlines = None
    if fp_source is not None:
        import shapely
        outlines = [shapely.transform(p, lambda x, y: to_utm.transform(x, y), interleaved=False)
                    for p in polys_ll]
    xyz = lp.xyz if lp is not None else np.empty((0, 3))
    cls = lp.classification if lp is not None else np.empty(0, dtype=np.uint8)

    from shapely.geometry import Point
    here = Point(px, py)
    structures, warnings = find_structures(xyz, cls, (px, py), outlines, radius_m=30)
    whole = [s for s in structures if s.parent is None]
    hit_s = min(whole, key=lambda s: s.outline.distance(here), default=None)
    if hit_s is None or hit_s.outline.distance(here) > 3.0:
        raise HTTPException(404, "No building found at this pin. Trace it on the map instead.")
    picked = [hit_s] + [s for s in structures if s.parent == hit_s.id]
    for s in picked:
        s.role = "added_by_user" if s is hit_s else "addition_to_added_by_user"
        s.included = True
    measure_structures(picked)
    result = {
        "structures": [structure_json(s, to_ll, pin.waste_pct) for s in picked],
        "sources": {"lidar": None if lp is None else {"dataset": lp.dataset, "year": lp.year},
                    "building_maps": fp_source},
        "warnings": warnings,
    }
    _cache_put(key, result)
    return result


@app.post("/debug", dependencies=[Depends(require_key)])
def debug(loc: Location):
    """What the LiDAR looks like at an address: counts, heights, structures found."""
    from .roof import building_clusters, segment_planes
    lat, lon = loc.resolve()
    try:
        lp = fetch_points(lat, lon, radius_m=35)
    except LookupError as e:
        return {"lat": lat, "lon": lon, "error": str(e)}
    to_utm = Transformer.from_crs("EPSG:4326", lp.utm_crs, always_xy=True)
    tx, ty = to_utm.transform(lon, lat)
    xyz, cls = lp.xyz, lp.classification
    clusters, classified = building_clusters(xyz, cls)
    out = []
    for c in clusters:
        d = float(np.hypot(c[:, 0] - tx, c[:, 1] - ty).min())
        out.append({"points": len(c), "distance_m": round(d, 1),
                    "z_min": round(float(c[:, 2].min()), 2), "z_max": round(float(c[:, 2].max()), 2),
                    "planes_found_strict": len(segment_planes(c)) if len(c) >= 12 else 0,
                    "planes_found_loose": len(segment_planes(c, 0.4, 35.0)) if len(c) >= 12 else 0})
    out.sort(key=lambda o: o["distance_m"])
    area = float(np.ptp(xyz[:, 0]) * np.ptp(xyz[:, 1])) if len(xyz) else 0
    return {
        "lat": lat, "lon": lon, "dataset": lp.dataset, "year": lp.year,
        "points": int(len(xyz)), "points_per_m2": round(len(xyz) / max(area, 1), 2),
        "classes": {int(k): int(v) for k, v in zip(*np.unique(cls, return_counts=True))},
        "z_range": [round(float(xyz[:, 2].min()), 2), round(float(xyz[:, 2].max()), 2)] if len(xyz) else None,
        "building_class_used": classified,
        "structures_nearest_first": out[:8],
    }


@app.post("/debug-points", dependencies=[Depends(require_key)])
def debug_points(loc: Location):
    """Download the raw LiDAR points around an address as a .laz file (to send for checking)."""
    import io
    import laspy
    from fastapi.responses import Response
    lat, lon = loc.resolve()
    try:
        lp = fetch_points(lat, lon, radius_m=35)
    except LookupError as e:
        raise HTTPException(404, str(e))
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.scales = [0.01, 0.01, 0.01]
    hdr.offsets = lp.xyz.min(axis=0)
    hdr.add_crs(lp.utm_crs)
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = lp.xyz[:, 0], lp.xyz[:, 1], lp.xyz[:, 2]
    las.classification = lp.classification
    buf = io.BytesIO()
    las.write(buf, do_compress=True)
    name = f"points_{lat:.5f}_{lon:.5f}.laz"
    return Response(buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/measure-traced", dependencies=[Depends(require_key)])
def measure_traced(req: TracedRequest):
    """The roofer traces each face on the map; LiDAR supplies the pitch when it can."""
    if not req.facets:
        raise HTTPException(400, "No facets sent.")
    all_pts = [p for f in req.facets for p in f.points]
    lat = float(np.mean([p[0] for p in all_pts]))
    lon = float(np.mean([p[1] for p in all_pts]))

    lp = None
    try:
        lp = fetch_points(lat, lon, radius_m=40)
        utm = lp.utm_crs
    except LookupError:
        utm = utm_crs_for(lat, lon)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)

    out, total = [], 0.0
    for i, tf in enumerate(req.facets):
        if len(tf.points) < 3:
            raise HTTPException(400, f"Facet {i} needs at least 3 corners.")
        xs, ys = to_utm.transform([p[1] for p in tf.points], [p[0] for p in tf.points])
        poly = Polygon(zip(xs, ys))
        if not poly.is_valid:
            poly = poly.buffer(0)
        plan_m2 = poly.area
        source, p12 = "roofer", tf.pitch_over_12
        if lp is not None:
            fit = plane_for_polygon(lp.xyz, lp.classification, poly)
            if fit is not None:
                source, p12 = "lidar", fit.pitch_over_12
        if p12 is None:
            raise HTTPException(422, f"Facet {i}: no LiDAR here, please enter a pitch.")
        factor = float(np.sqrt(1 + (p12 / 12) ** 2))
        sloped = plan_m2 * factor * SQFT_PER_M2
        total += sloped
        out.append({"id": i, "pitch": pitch_label(p12), "pitch_over_12": round(p12, 1),
                    "pitch_source": source, "plan_sqft": round(plan_m2 * SQFT_PER_M2),
                    "roof_sqft": round(sloped)})
    return {
        "facets": out,
        "totals": {"roof_sqft": round(total), "squares": round(total / 100, 1),
                   "squares_with_waste": round(total * (1 + req.waste_pct / 100) / 100, 1),
                   "waste_pct": req.waste_pct},
        "lidar": None if lp is None else {"dataset": lp.dataset, "year": lp.year},
    }
