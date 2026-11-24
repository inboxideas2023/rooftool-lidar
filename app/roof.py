"""Turn a LiDAR point cloud into roof measurements: facets, pitch, area and edge lengths.

All math is in meters in a local UTM projection; results are converted to feet at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN

SQFT_PER_M2 = 10.7639
FT_PER_M = 3.28084
CELL = 0.5                       # raster cell size for areas (m)
OFFSET_A, OFFSET_B = 0.60, -0.20 # outline push-out vs point spacing (calibrated)

GROUND, BUILDING, WATER = 2, 6, 9
NOISE = {7, 18}


class NoRoofFound(Exception):
    pass


@dataclass
class Facet:
    id: int
    normal: np.ndarray           # unit normal, pointing up
    d: float                     # plane: n . p + d = 0
    points: np.ndarray           # inlier points (M, 3)
    polygon: Polygon | None = None
    plan_m2: float = 0.0

    @property
    def slope_rad(self) -> float:
        return float(np.arccos(np.clip(abs(self.normal[2]), 0, 1)))

    @property
    def pitch_over_12(self) -> float:
        return 12.0 * float(np.tan(self.slope_rad))

    @property
    def azimuth_deg(self) -> float | None:
        """Compass direction the facet faces (downslope), 0 = north."""
        nx, ny = self.normal[0], self.normal[1]
        if np.hypot(nx, ny) < 1e-3:
            return None
        return float(np.degrees(np.arctan2(nx, ny)) % 360)

    def z_at(self, x, y):
        n = self.normal
        return -(n[0] * x + n[1] * y + self.d) / n[2]


@dataclass
class RoofResult:
    facets: list[Facet]
    footprint: Polygon
    edges_m: dict[str, float] = field(default_factory=dict)
    point_count: int = 0
    density_per_m2: float = 0.0
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ geometry utils
def _fit_plane(p: np.ndarray):
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    return n, -float(n @ c)


def _normals(xyz: np.ndarray, k: int = 10):
    tree = cKDTree(xyz)
    k = min(k, len(xyz))
    _, idx = tree.query(xyz, k=k)
    nb = xyz[idx]                                   # (N, k, 3)
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / k
    w, v = np.linalg.eigh(cov)                      # ascending eigenvalues
    n = v[:, :, 0]
    n[n[:, 2] < 0] *= -1
    curvature = w[:, 0] / np.maximum(w.sum(axis=1), 1e-9)
    return n, curvature


# ------------------------------------------------------------------ step 1: isolate the house
def building_points(xyz, cls):
    """All points that belong to buildings (not ground, trees or noise).

    Returns (points, classified) where classified says whether the survey labelled buildings.
    """
    if len(xyz) < 20:
        return np.empty((0, 3)), True
    has_bldg_class = np.count_nonzero(cls == BUILDING) > 30
    if has_bldg_class:
        return xyz[cls == BUILDING], True
    ground = xyz[cls == GROUND]
    if len(ground) < 10:
        ground = xyz[xyz[:, 2] <= np.percentile(xyz[:, 2], 5)]
    _, gi = cKDTree(ground[:, :2]).query(xyz[:, :2], k=min(8, len(ground)))
    hag = xyz[:, 2] - np.median(ground[gi, 2], axis=1)
    # unclassified data: anything 2 m+ above ground, then drop trees by roughness
    cand = xyz[(hag > 2.0) & ~np.isin(cls, [GROUND, WATER, *NOISE])]
    if len(cand) >= 20:
        _, curv = _normals(cand)
        cand = cand[curv < 0.02]
    return cand, False


def building_clusters(xyz, cls, min_area_m2: float = 8.0):
    """Split building points into separate structures (house, garage, shed ...)."""
    cand, classified = building_points(xyz, cls)
    if len(cand) < 20:
        return [], classified
    labels = DBSCAN(eps=1.6, min_samples=4).fit_predict(cand[:, :2])
    out = []
    for lab in set(labels) - {-1}:
        pts = cand[labels == lab]
        cells = np.unique(np.floor(pts[:, :2] / 1.0).astype(int), axis=0)
        if len(cells) >= min_area_m2:                  # ~1 m cells
            out.append(pts)
    return out, classified


def isolate_building(xyz, cls, target_xy, search_m: float = 15.0):
    clusters, classified = building_clusters(xyz, cls)
    if not clusters:
        raise NoRoofFound("No building points found near this location.")
    dists = [np.hypot(c[:, 0] - target_xy[0], c[:, 1] - target_xy[1]).min() for c in clusters]
    k = int(np.argmin(dists))
    if dists[k] > search_m:
        raise NoRoofFound("No building within 15 m of this location.")
    return clusters[k], 0.0, classified


# ------------------------------------------------------------------ step 2: find roof planes
def segment_planes(pts, dist_tol=0.15, angle_tol_deg=15.0, min_pts=12, max_planes=40, seed=7):
    rng = np.random.default_rng(seed)
    normals, _ = _normals(pts)
    cos_tol = np.cos(np.radians(angle_tol_deg))
    remaining = np.arange(len(pts))
    facets: list[Facet] = []
    density = len(pts) / max(MultiPoint(pts[:, :2]).convex_hull.area, 1.0)

    while len(remaining) >= min_pts and len(facets) < max_planes:
        P, N = pts[remaining], normals[remaining]
        best = None
        for _ in range(400):
            i = rng.choice(len(P), 3, replace=False)
            n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]])
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n = n / norm
            if n[2] < 0:
                n = -n
            if n[2] < 0.25:          # steeper than ~75 deg: wall, not roof
                continue
            dist = np.abs((P - P[i[0]]) @ n)
            inl = (dist < dist_tol) & (np.abs(N @ n) > cos_tol)
            c = int(inl.sum())
            if best is None or c > best[0]:
                best = (c, inl)
        if best is None or best[0] < min_pts:
            break
        inl = best[1]
        n, d = _fit_plane(P[inl])                         # refine
        dist = np.abs(P @ n + d)
        inl = (dist < dist_tol) & (np.abs(N @ n) > cos_tol)
        # same plane can appear on separate parts of the roof: split by connectivity
        sel = np.flatnonzero(inl)
        if len(sel) < min_pts:
            break
        comp = DBSCAN(eps=1.5, min_samples=3).fit_predict(P[sel, :2])
        used = np.zeros(len(P), bool)
        for c in set(comp) - {-1}:
            members = sel[comp == c]
            used[members] = True
            if len(members) < min_pts:
                continue
            # a thin strip is points along a ridge or wall top, not a real facet
            xy = P[members, :2] - P[members, :2].mean(axis=0)
            minor_std = float(np.sqrt(np.linalg.eigvalsh(np.cov(xy.T))[0]))
            if minor_std < 0.35:
                continue
            hull = MultiPoint(P[members, :2]).convex_hull
            if len(members) / max(hull.area, 1e-6) < 0.3 * density:
                continue                                   # scattered points, not a surface
            nn, dd = _fit_plane(P[members])
            facets.append(Facet(len(facets), nn, dd, P[members]))
        if not used.any():
            break
        remaining = remaining[~used]
    return facets


# ------------------------------------------------------------------ step 3: areas from a raster
@dataclass
class Grid:
    labels: np.ndarray           # (ny, nx) facet id per cell, -1 = not roof
    x0: float
    y0: float

    def center(self, iy, ix):
        return self.x0 + (ix + 0.5) * CELL, self.y0 + (iy + 0.5) * CELL

    def facet_at(self, x, y) -> int:
        ix, iy = int((x - self.x0) // CELL), int((y - self.y0) // CELL)
        if 0 <= iy < self.labels.shape[0] and 0 <= ix < self.labels.shape[1]:
            return int(self.labels[iy, ix])
        return -1


def _regularize(poly: Polygon, min_edge: float = 1.0) -> Polygon:
    """Square up a stair-stepped outline: keep the long edges, meet them at clean corners."""
    pts = np.array(poly.exterior.coords)[:-1]
    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    long = [e for e in edges if np.hypot(*(e[1] - e[0])) >= min_edge]
    if len(long) < 3:
        return poly
    out = []
    for k, (a1, b1) in enumerate(long):
        a2, b2 = long[(k + 1) % len(long)]
        d1, d2 = (b1 - a1), (b2 - a2)
        d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) > np.sin(np.radians(20)):
            t = ((a2 - a1)[0] * d2[1] - (a2 - a1)[1] * d2[0]) / cross
            v = a1 + t * d1
            if np.hypot(*(v - b1)) < 2.0 and np.hypot(*(v - a2)) < 2.0:
                out.append(v)
                continue
        out.extend([b1, a2])
    try:
        reg = Polygon(out)
        if reg.is_valid and abs(reg.area - poly.area) < 0.08 * poly.area:
            return reg
    except ValueError:
        pass
    return poly


def rasterize(roof_pts, facets: list[Facet]):
    """Assign every roof cell to a facet, build the outline, and set facet plan areas."""
    xy = roof_pts[:, :2]
    x0, y0 = xy.min(axis=0) - 3 * CELL
    nx = int(np.ceil((xy[:, 0].max() - x0) / CELL)) + 4
    ny = int(np.ceil((xy[:, 1].max() - y0) / CELL)) + 4
    occ = np.zeros((ny, nx), bool)
    occ[((xy[:, 1] - y0) / CELL).astype(int), ((xy[:, 0] - x0) / CELL).astype(int)] = True
    occ = ndimage.binary_closing(occ, iterations=2)
    occ = ndimage.binary_fill_holes(occ)
    lab, n = ndimage.label(occ)
    if n > 1:  # keep the biggest connected piece
        sizes = ndimage.sum(occ, lab, range(1, n + 1))
        occ = lab == (1 + int(np.argmax(sizes)))

    cy, cx = np.nonzero(occ)
    centers = np.column_stack([x0 + (cx + 0.5) * CELL, y0 + (cy + 0.5) * CELL])
    lab_pts = np.vstack([f.points[:, :2] for f in facets])
    lab_ids = np.concatenate([np.full(len(f.points), f.id) for f in facets])
    _, nearest = cKDTree(lab_pts).query(centers)
    cell_facet = lab_ids[nearest]
    labels = np.full((ny, nx), -1, int)
    labels[cy, cx] = cell_facet

    cells = [box(c[0] - CELL / 2, c[1] - CELL / 2, c[0] + CELL / 2, c[1] + CELL / 2) for c in centers]
    raw = unary_union(cells)
    if raw.geom_type == "MultiPolygon":
        raw = max(raw.geoms, key=lambda g: g.area)
    raw = Polygon(raw.exterior)
    # LiDAR's outermost hits sit a little inside the true roof edge; push the outline out
    # by an amount that depends on point spacing (calibrated on synthetic roofs).
    spacing = float(np.sqrt(raw.area / len(roof_pts)))
    smooth = raw.buffer(0.75, join_style=2).buffer(-0.75, join_style=2)   # close edge notches
    smooth = _regularize(smooth.simplify(CELL * 0.7))
    offset = OFFSET_A * spacing + OFFSET_B
    footprint = smooth.buffer(offset, join_style=2) if offset > 0 else smooth
    if footprint.geom_type == "MultiPolygon":
        footprint = max(footprint.geoms, key=lambda g: g.area)
    scale = footprint.area / max(len(centers) * CELL * CELL, 1e-9)

    for f in facets:
        m = cell_facet == f.id
        f.plan_m2 = float(m.sum()) * CELL * CELL * scale
        if m.any():
            poly = unary_union([cells[k] for k in np.flatnonzero(m)])
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            f.polygon = Polygon(poly.exterior).simplify(CELL * 0.7)
    return footprint, Grid(labels, float(x0), float(y0))


def merge_coplanar(facets: list[Facet], angle_deg=5.0, dist_tol=0.2, touch_m=1.0):
    """Join neighbouring pieces that are really the same plane."""
    changed = True
    while changed:
        changed = False
        for i in range(len(facets)):
            for j in range(i + 1, len(facets)):
                a, b = facets[i], facets[j]
                if np.degrees(np.arccos(np.clip(a.normal @ b.normal, -1, 1))) > angle_deg:
                    continue
                if np.abs(b.points @ a.normal + a.d).mean() > dist_tol:
                    continue
                dmin, _ = cKDTree(a.points[:, :2]).query(b.points[:, :2], k=1)
                if dmin.min() > touch_m:
                    continue
                pts = np.vstack([a.points, b.points])
                n, d = _fit_plane(pts)
                facets[i] = Facet(a.id, n, d, pts)
                facets.pop(j)
                changed = True
                break
            if changed:
                break
    for k, f in enumerate(facets):
        f.id = k
    return facets


# ------------------------------------------------------------------ step 4: edges
def _line_length(mids: np.ndarray, hdir: np.ndarray, footprint: Polygon, gap: float = 1.5) -> float:
    """Horizontal length of the contact between two facets along their meeting line.

    Contacts are split where there are gaps; each piece is stretched to the roof edge if it
    stops just short of it (the raster loses the last half-metre or so at the eaves).
    """
    proj = mids @ hdir
    order = np.argsort(proj)
    proj, pts = proj[order], mids[order]
    pieces, start = [], 0
    for k in range(1, len(proj)):
        if proj[k] - proj[k - 1] > gap:
            pieces.append((start, k - 1))
            start = k
    pieces.append((start, len(proj) - 1))
    ring = footprint.exterior
    total = 0.0
    for a, b in pieces:
        length = proj[b] - proj[a] + CELL
        for idx, sign in ((a, -1), (b, 1)):
            end = pts[idx] + sign * hdir * CELL / 2
            reach = LineString([end, end + sign * hdir * 1.5])
            hit = reach.intersection(ring)
            if not hit.is_empty:
                length += Point(end).distance(hit)
        total += length
    return total


def _smooth_runs(ids: list[int], min_run: int) -> list[tuple[int, int]]:
    """Collapse a sequence into (id, count) runs, folding short runs into a neighbour.

    Near a corner the sample can land on the next facet over; that blip is not an edge.
    """
    runs: list[list[int]] = []
    for v in ids:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    changed = True
    while changed and len(runs) > 1:
        changed = False
        k = min(range(len(runs)), key=lambda i: runs[i][1])
        if runs[k][1] >= min_run:
            break
        nb = [i for i in (k - 1, k + 1) if 0 <= i < len(runs)]
        tgt = max(nb, key=lambda i: runs[i][1])
        runs[tgt][1] += runs[k][1]
        runs.pop(k)
        merged = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][1] += r[1]
            else:
                merged.append(r)
        runs = merged
        changed = True
    return [(a, b) for a, b in runs]


def measure_edges(facets: list[Facet], footprint: Polygon, grid: Grid) -> dict[str, float]:
    edges = {"ridge": 0.0, "hip": 0.0, "valley": 0.0, "eave": 0.0, "rake": 0.0, "other": 0.0}
    by_id = {f.id: f for f in facets}
    L = grid.labels

    # --- where two facets meet: ridge / hip / valley
    contacts: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for a, b, (dy, dx) in ((L[:, :-1], L[:, 1:], (0, 1)), (L[:-1, :], L[1:, :], (1, 0))):
        iy, ix = np.nonzero((a != b) & (a >= 0) & (b >= 0))
        for y, x in zip(iy, ix):
            i, j = int(a[y, x]), int(b[y, x])
            cxy = grid.center(y, x)
            mid = (cxy[0] + dx * CELL / 2, cxy[1] + dy * CELL / 2)
            contacts.setdefault((min(i, j), max(i, j)), []).append(mid)

    for (i, j), mids in contacts.items():
        if len(mids) < 3 or i not in by_id or j not in by_id:
            continue
        fa, fb = by_id[i], by_id[j]
        line = np.cross(fa.normal, fb.normal)
        if np.linalg.norm(line) < 1e-6:
            continue
        line /= np.linalg.norm(line)
        h = np.hypot(line[0], line[1])
        if h < 1e-6:
            continue
        hdir = line[:2] / h
        m = np.asarray(mids)
        horiz = _line_length(m, hdir, footprint)
        tan = abs(line[2]) / h
        length = horiz * np.sqrt(1 + tan ** 2)
        # Convex (ridge/hip) if each facet, extended across the line, rises above the other.
        tree = cKDTree(m)
        def near(f):
            dist, _ = tree.query(f.points[:, :2])
            return f.points[dist < 2.0]
        pa, pb = near(fa), near(fb)
        if len(pa) < 3 or len(pb) < 3:
            continue
        ga = float(np.mean(fb.z_at(pa[:, 0], pa[:, 1]) - pa[:, 2]))
        gb = float(np.mean(fa.z_at(pb[:, 0], pb[:, 1]) - pb[:, 2]))
        if ga > 0.03 and gb > 0.03:
            edges["ridge" if np.degrees(np.arctan(tan)) < 10 else "hip"] += length
        elif ga < -0.03 and gb < -0.03:
            edges["valley"] += length
        else:
            edges["other"] += length                     # pitch change, step, wall

    # --- the outside edge: eave or rake, depending on which way that facet slopes
    ring = footprint.exterior
    ccw = ring.is_ccw
    coords = list(ring.coords)
    step = 0.25
    for p, q in zip(coords[:-1], coords[1:]):
        seg = np.subtract(q, p)
        seg_len = float(np.hypot(*seg))
        if seg_len < 0.1:
            continue
        d = seg / seg_len
        outward = np.array([d[1], -d[0]]) if ccw else np.array([-d[1], d[0]])
        n_s = max(1, int(seg_len / step))
        fids = []
        for k in range(n_s):
            t = (k + 0.5) * seg_len / n_s
            fid = -1
            for inset in (0.6, 1.2, 2.0):                 # step inside the roof
                x, y = np.add(p, d * t) - outward * inset
                fid = grid.facet_at(x, y)
                if fid >= 0:
                    break
            fids.append(fid)
        per_facet: dict[int, float] = {}
        for fid, n_run in _smooth_runs(fids, min_run=int(1.5 / step)):
            if fid >= 0:
                per_facet[fid] = per_facet.get(fid, 0.0) + n_run * seg_len / n_s
        for fid, length in per_facet.items():
            f = by_id.get(fid)
            if f is None:
                continue
            down = f.normal[:2]
            if np.linalg.norm(down) < 0.05:              # flat roof: all edges are eaves
                edges["eave"] += length
                continue
            down = down / np.linalg.norm(down)
            along = abs(d @ down)          # 0 = edge runs across the slope, 1 = up the slope
            if along > 0.94:
                along = 1.0
            elif along < 0.35:
                along = 0.0
            across = float(np.sqrt(1 - along ** 2))
            # a diagonal piece (e.g. a cut corner) is split into its eave and rake parts
            if across:
                edges["eave" if outward @ down > 0 else "other"] += length * across
            if along:
                edges["rake"] += length * along / np.cos(f.slope_rad)
    return edges


# ------------------------------------------------------------------ main entry
def measure_roof(xyz, cls, target_xy) -> RoofResult:
    """Measure the building nearest target_xy (falling back to the next nearest ones)."""
    clusters, classified = building_clusters(xyz, cls)
    near = sorted(((float(np.hypot(c[:, 0] - target_xy[0], c[:, 1] - target_xy[1]).min()), k)
                   for k, c in enumerate(clusters)))
    near = [(d, k) for d, k in near if d <= 15.0]
    if not near:
        raise NoRoofFound("No building within 15 m of this location.")
    last = None
    for _, k in near[:4]:
        try:
            return measure_cluster(clusters[k], classified)
        except NoRoofFound as e:
            last = e
    raise last


def measure_cluster(roof: np.ndarray, classified: bool = True) -> RoofResult:
    """Measure one structure's roof from its LiDAR points."""
    warnings = []
    if not classified:
        warnings.append("LiDAR here has no building class; trees near the roof may affect results.")

    # Real surveys are noisier than they look; loosen the fit step by step if needed.
    facets = []
    for dist_tol, angle in ((0.15, 15.0), (0.25, 25.0), (0.4, 35.0)):
        facets = segment_planes(roof, dist_tol=dist_tol, angle_tol_deg=angle)
        if facets:
            if dist_tol > 0.15:
                warnings.append("LiDAR here is noisy; pitch may be off by about 1/12.")
            break
    if not facets:
        raise NoRoofFound("Could not find roof planes in the LiDAR points.")
    facets = merge_coplanar(facets)
    # drop tiny planes (vents, chimneys, noise); their cells go to the nearest real facet
    min_pts = max(12, int(0.02 * len(roof)))
    facets = [f for f in facets if len(f.points) >= min_pts] or facets
    for k, f in enumerate(facets):
        f.id = k
    footprint, grid = rasterize(roof, facets)
    edges = measure_edges(facets, footprint, grid)

    density = len(roof) / max(footprint.area, 1.0)
    if density < 2:
        warnings.append(f"Low LiDAR density ({density:.1f} pts/m²); small facets may be missed.")
    return RoofResult(facets, footprint, edges, len(roof), density, warnings)


def cluster_outline(pts: np.ndarray) -> Polygon:
    """Outline of one structure's LiDAR points (same edge correction as the measurements)."""
    xy = pts[:, :2]
    x0, y0 = xy.min(axis=0) - 3 * CELL
    nx = int(np.ceil((xy[:, 0].max() - x0) / CELL)) + 4
    ny = int(np.ceil((xy[:, 1].max() - y0) / CELL)) + 4
    occ = np.zeros((ny, nx), bool)
    occ[((xy[:, 1] - y0) / CELL).astype(int), ((xy[:, 0] - x0) / CELL).astype(int)] = True
    occ = ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=2))
    cy, cx = np.nonzero(occ)
    raw = unary_union([box(x0 + i * CELL, y0 + j * CELL, x0 + (i + 1) * CELL, y0 + (j + 1) * CELL)
                       for j, i in zip(cy, cx)])
    if raw.geom_type == "MultiPolygon":
        raw = max(raw.geoms, key=lambda g: g.area)
    raw = Polygon(raw.exterior)
    spacing = float(np.sqrt(raw.area / len(pts)))
    offset = OFFSET_A * spacing + OFFSET_B
    out = _regularize(raw.simplify(CELL * 0.7))
    return out.buffer(offset, join_style=2) if offset > 0 else out


def plane_for_polygon(xyz, cls, poly: Polygon):
    """Fit the roof plane inside a user-drawn polygon (for the tracing tool).

    Returns None when the scan has no roof there (e.g. the structure was built after the
    survey), so the caller falls back to the pitch the roofer entered.
    """
    bpts, _ = building_points(xyz, cls)
    if len(bpts) == 0:
        return None
    inside = np.array([poly.contains(Point(x, y)) for x, y in bpts[:, :2]])
    pts = bpts[inside]
    all_density = len(xyz) / max(MultiPoint(xyz[:, :2]).convex_hull.area, 1.0)
    if len(pts) < 8 or len(pts) < 0.4 * all_density * poly.area:
        return None                                    # roof not (fully) in the scan
    # robust: drop points far from a first fit (chimneys, overhanging branches)
    n, d = _fit_plane(pts)
    r = np.abs(pts @ n + d)
    pts = pts[r < max(0.3, np.percentile(r, 80))]
    n, d = _fit_plane(pts)
    return Facet(0, n, d, pts, poly, poly.area)
