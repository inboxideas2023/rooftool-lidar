"""Measure every structure on a property and cross-check the LiDAR against newer building maps.

Without lot lines, "on the property" is decided like this:
  1. The main house is the structure the address point sits on (or the nearest one).
  2. Other big buildings (1,000+ sq ft) are treated as neighbours' houses.
  3. Each smaller structure (garage, ADU, shed, patio cover) belongs to whichever house it is
     closest to. Those closest to the main house, within 100 ft of it, are included.
Every structure in range is returned with an `included` flag, so the roofer can tick or untick
anything the rules got wrong.

Cross-check (all free): building outlines from Overture/OpenStreetMap are usually newer than
the LiDAR. An outline with no LiDAR roof under it is a structure built after the scan; an
outline that sticks out past the LiDAR roof is an addition. Those come back as
`needs_trace` with the outline ready, so the roofer only enters the pitch.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .roof import (FT_PER_M, SQFT_PER_M2, NoRoofFound, RoofResult, building_clusters,
                   cluster_outline, measure_cluster)

PRIMARY_MIN_M2 = 93.0          # ~1,000 sq ft: treated as a house
ACCESSORY_MAX_DIST_M = 30.5    # 100 ft from the main house
MAIN_MAX_DIST_M = 30.0         # address point must be within ~100 ft of a structure
MIN_PIECE_M2 = 10.0            # ignore smaller mismatches (~110 sq ft)
MIN_PIECE_WIDTH_M = 2.5        # ignore thin slivers from outlines being slightly offset


@dataclass
class Structure:
    id: int
    outline: Polygon                        # UTM meters
    source: str                             # "lidar" or "needs_trace"
    status: str                             # see STATUS_TEXT
    points: np.ndarray | None = None
    result: RoofResult | None = None
    role: str = ""
    included: bool = False
    parent: int | None = None               # for additions: the building they attach to
    notes: list[str] = field(default_factory=list)


STATUS_TEXT = {
    "confirmed": "In the LiDAR scan and on current building maps.",
    "lidar_only": "In the LiDAR scan but not on current building maps. It may have been torn "
                  "down since the scan; check the satellite map.",
    "not_checked": "In the LiDAR scan. No current building map was available to cross-check.",
    "new_since_scan": "On current building maps but not in the LiDAR scan (likely built after "
                      "it). Enter the pitch, or trace the faces.",
    "addition_since_scan": "Part of this building is not in the LiDAR scan (likely an addition). "
                           "Enter the pitch for the added part.",
}


def _pieces(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]


def _real_piece(p: Polygon) -> bool:
    return p.area >= MIN_PIECE_M2 and not p.buffer(-MIN_PIECE_WIDTH_M / 2).is_empty


def find_structures(xyz, cls, address_xy, outlines: list[Polygon] | None,
                    radius_m: float = 45.0) -> tuple[list[Structure], list[str]]:
    """Build the list of structures near the address, cross-checked against outlines."""
    warnings: list[str] = []
    here = Point(address_xy)
    clusters, classified = building_clusters(xyz, cls)
    structures: list[Structure] = []
    for pts in clusters:
        ol = cluster_outline(pts)
        if ol.distance(here) <= radius_m:
            structures.append(Structure(len(structures), ol, "lidar", "not_checked", points=pts))
    if not classified:
        warnings.append("This LiDAR survey doesn't label buildings; trees next to a roof may "
                        "be mistaken for structures.")

    if outlines is None:
        warnings.append("Couldn't reach the building maps, so new structures could not be "
                        "checked. Compare the outlines with the satellite map.")
        return structures, warnings

    outlines = [o for o in outlines if o.distance(here) <= radius_m and o.area >= MIN_PIECE_M2]
    lidar_union = unary_union([s.outline for s in structures]) if structures else Polygon()
    lidar_grown = lidar_union.buffer(1.0)

    # LiDAR structures: are they still on the current maps?
    if outlines:
        all_outlines = unary_union(outlines)
        for s in structures:
            overlap = s.outline.intersection(all_outlines.buffer(1.0)).area / s.outline.area
            s.status = "confirmed" if overlap >= 0.2 else "lidar_only"

    # Outlines: anything (or any part) the LiDAR doesn't have?
    lidar_structs = list(structures)
    for o in outlines:
        cover = o.intersection(lidar_union).area / o.area
        gap = o.difference(lidar_union)
        # judge size on the gap shrunk away from the LiDAR edge (outlines are often offset a
        # little), then measure the full gap around each real piece
        cores = [p for p in _pieces(o.difference(lidar_grown)) if _real_piece(p)]
        missing = []
        for core in cores:
            full = gap.intersection(core.buffer(1.5, join_style=2))
            missing.append(max(_pieces(full) or [core], key=lambda g: g.area))
        for piece in missing:
            if cover < 0.15:
                status, parent = "new_since_scan", None
            else:
                status = "addition_since_scan"
                parent = min(lidar_structs, key=lambda s: s.outline.distance(piece)).id
            structures.append(Structure(len(structures), piece, "needs_trace", status, parent=parent))
    return structures, warnings


def assign_ownership(structures: list[Structure], address_xy) -> int | None:
    """Mark which structures belong to the property. Returns the main structure's id."""
    here = Point(address_xy)
    whole = [s for s in structures if s.parent is None]
    if not whole:
        return None
    main = min(whole, key=lambda s: (s.outline.distance(here), -s.outline.area))
    if main.outline.distance(here) > MAIN_MAX_DIST_M:
        for s in structures:
            s.role, s.included = "other", False
        return None
    main.role, main.included = "main", True
    houses = [s for s in whole if s is not main and s.outline.area >= PRIMARY_MIN_M2]
    for s in houses:
        s.role, s.included = "neighbor", False
    anchors = [main] + houses
    for s in whole:
        if s.role:
            continue
        nearest = min(anchors, key=lambda a: a.outline.distance(s.outline))
        if nearest is main and main.outline.distance(s.outline) <= ACCESSORY_MAX_DIST_M:
            s.role, s.included = "accessory", True
        else:
            s.role, s.included = "neighbor_accessory", False
    by_id = {s.id: s for s in structures}
    for s in structures:
        if s.parent is not None:
            p = by_id[s.parent]
            s.role, s.included = f"addition_to_{p.role}", p.included
    return main.id


def measure_structures(structures: list[Structure]):
    for s in structures:
        if s.source == "lidar" and s.points is not None:
            try:
                s.result = measure_cluster(s.points)
            except NoRoofFound as e:
                s.notes.append(str(e))


# ------------------------------------------------------------------ JSON output
PITCHES = list(range(0, 13))


def _ring(poly: Polygon, to_ll) -> list[list[float]]:
    xs, ys = poly.exterior.xy
    lons, lats = to_ll.transform(list(xs), list(ys))
    return [[round(a, 7), round(b, 7)] for a, b in zip(lats, lons)]


def structure_json(s: Structure, to_ll, waste_pct: float) -> dict:
    out = {
        "id": s.id,
        "role": s.role,
        "included": s.included,
        "source": s.source,
        "status": s.status,
        "status_text": STATUS_TEXT[s.status],
        "plan_sqft": round(s.outline.area * SQFT_PER_M2),
        "outline_latlon": _ring(s.outline, to_ll),
        "notes": s.notes,
    }
    if s.parent is not None:
        out["addition_to"] = s.parent
    if s.source == "needs_trace":
        # Same pitch on every face -> roof area = plan area x pitch factor, so the roofer
        # only has to pick the pitch. Complex roofs: trace faces via /measure-traced.
        out["roof_sqft_by_pitch"] = {
            f"{p}/12": round(s.outline.area * SQFT_PER_M2 * np.sqrt(1 + (p / 12) ** 2))
            for p in PITCHES}
        return out
    r = s.result
    if r is None:
        return out
    pitch_area: Counter = Counter()
    facets, total = [], 0.0
    for f in r.facets:
        sloped = f.plan_m2 / max(np.cos(f.slope_rad), 0.2) * SQFT_PER_M2
        total += sloped
        label = f"{round(f.pitch_over_12)}/12"
        pitch_area[label] += sloped
        facets.append({
            "id": f.id, "pitch": label, "pitch_over_12": round(f.pitch_over_12, 1),
            "pitch_degrees": round(np.degrees(f.slope_rad), 1),
            "faces_azimuth_deg": None if f.azimuth_deg is None else round(f.azimuth_deg),
            "plan_sqft": round(f.plan_m2 * SQFT_PER_M2), "roof_sqft": round(sloped),
            "polygon_latlon": _ring(f.polygon, to_ll) if f.polygon is not None else [],
        })
    out.update({
        "roof_sqft": round(total),
        "squares": round(total / 100, 1),
        "squares_with_waste": round(total * (1 + waste_pct / 100) / 100, 1),
        "predominant_pitch": pitch_area.most_common(1)[0][0] if pitch_area else None,
        "edges_ft": {k: round(v * FT_PER_M) for k, v in r.edges_m.items()},
        "facets": facets,
        "lidar_points_per_m2": round(r.density_per_m2, 1),
    })
    out["notes"] = out["notes"] + r.warnings
    return out
