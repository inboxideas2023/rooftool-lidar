"""End-to-end test: a fake USGS EPT dataset served over local HTTP, then the real API.

Run:  python -m pytest -q   (from the project folder)
"""
import json
import os
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import laspy
import numpy as np
import pytest
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(__file__))
from synthetic import gable_roof, hip_roof  # noqa: E402

UTM = "EPSG:32611"
CENTER_UTM = (500000.0, 3770000.0)


def _write_laz(path, xyz, cls):
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.scales = [0.01, 0.01, 0.01]
    hdr.offsets = xyz.min(axis=0)
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = cls
    las.write(path)


def build_fake_ept(root, xyz_utm, cls):
    """Write a 3-level EPT in Web Mercator (like USGS), including a -1 sub-hierarchy file."""
    to_merc = Transformer.from_crs(UTM, "EPSG:3857", always_xy=True)
    mx, my = to_merc.transform(xyz_utm[:, 0], xyz_utm[:, 1])
    pts = np.column_stack([mx, my, xyz_utm[:, 2]])
    c = pts.mean(axis=0)
    half = 512.0
    bounds = [c[0] - half, c[1] - half, c[2] - half, c[0] + half, c[1] + half, c[2] + half]
    os.makedirs(f"{root}/ept-data", exist_ok=True)
    os.makedirs(f"{root}/ept-hierarchy", exist_ok=True)

    rng = np.random.default_rng(0)
    level = rng.integers(0, 3, len(pts))           # spread points over depths 0..2
    hier, sub = {}, {}

    def key_for(p, d):
        n = 2 ** d
        ix = np.clip(((p[:, 0] - bounds[0]) / (2 * half) * n).astype(int), 0, n - 1)
        iy = np.clip(((p[:, 1] - bounds[1]) / (2 * half) * n).astype(int), 0, n - 1)
        iz = np.clip(((p[:, 2] - bounds[2]) / (2 * half) * n).astype(int), 0, n - 1)
        return [f"{d}-{a}-{b}-{z}" for a, b, z in zip(ix, iy, iz)]

    for d in range(3):
        sel = level == d
        keys = np.array(key_for(pts[sel], d))
        for k in set(keys):
            m = keys == k
            _write_laz(f"{root}/ept-data/{k}.laz", pts[sel][m], cls[sel][m])
            (sub if d == 2 else hier)[k] = int(m.sum())
    # depth-1 nodes that have depth-2 children point to a sub-hierarchy file
    for k2 in list(sub):
        d, x, y, z = map(int, k2.split("-"))
        parent = f"1-{x // 2}-{y // 2}-{z // 2}"
        hier.setdefault(parent, 0)
    for parent in [k for k in hier if k.startswith("1-")]:
        _, x, y, z = map(int, parent.split("-"))
        kids = {k: v for k, v in sub.items()
                if tuple(int(t) // 2 for t in k.split("-")[1:]) == (x, y, z)}
        if kids:
            with open(f"{root}/ept-hierarchy/{parent}.json", "w") as f:
                json.dump({parent: hier[parent], **kids}, f)
            hier[parent] = -1
    hier.setdefault("0-0-0-0", 0)
    with open(f"{root}/ept-hierarchy/0-0-0-0.json", "w") as f:
        json.dump(hier, f)
    with open(f"{root}/ept.json", "w") as f:
        json.dump({"bounds": bounds, "span": 128, "dataType": "laszip", "points": len(pts),
                   "srs": {"authority": "EPSG", "horizontal": "3857", "vertical": "5703"}}, f)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("lidar")
    xyz, cls = hip_roof()
    build_fake_ept(f"{root}/TEST_Hip_2022", xyz, cls)

    to_ll = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(*CENTER_UTM)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"

    d = 0.01
    index = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {"name": "TEST_Hip_2022", "id": 1, "count": len(xyz),
                       "url": f"{base}/TEST_Hip_2022/ept.json"},
        "geometry": {"type": "Polygon", "coordinates": [[[lon - d, lat - d], [lon + d, lat - d],
                                                         [lon + d, lat + d], [lon - d, lat + d],
                                                         [lon - d, lat - d]]]}}]}
    with open(f"{root}/resources.geojson", "w") as f:
        json.dump(index, f)

    os.environ["LIDAR_INDEX_URL"] = f"{base}/resources.geojson"
    os.environ["CACHE_DIR"] = str(root / "cache")
    os.environ["SERVICE_API_KEY"] = "test-key"
    for m in [m for m in sys.modules if m.startswith("app")]:
        del sys.modules[m]
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from fastapi.testclient import TestClient
    from app.main import app
    yield TestClient(app), lat, lon
    srv.shutdown()


def test_requires_key(client):
    c, lat, lon = client
    assert c.post("/measure", json={"lat": lat, "lon": lon}).status_code == 401


def test_measure_hip_roof(client):
    c, lat, lon = client
    r = c.post("/measure", json={"lat": lat, "lon": lon}, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200, r.text
    j = r.json()
    true_sqft = 160 * np.sqrt(1 + 0.25) * 10.7639          # 16 x 10 m hip roof at 6/12
    assert abs(j["totals"]["roof_sqft"] - true_sqft) / true_sqft < 0.05
    assert j["totals"]["predominant_pitch"] == "6/12"
    assert j["totals"]["facet_count"] == 4
    assert abs(j["edges_ft"]["ridge"] - 19.7) < 4
    assert abs(j["edges_ft"]["hip"] - 98.4) / 98.4 < 0.1
    assert abs(j["edges_ft"]["eave"] - 170.6) / 170.6 < 0.08
    assert j["lidar"]["dataset"] == "TEST_Hip_2022"
    # second call is served from cache
    assert c.post("/measure", json={"lat": lat, "lon": lon},
                  headers={"X-API-Key": "test-key"}).json()["cached"] is True


def test_measure_traced_uses_lidar_pitch(client):
    c, lat, lon = client
    to_ll = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    x0, y0 = CENTER_UTM
    # the roofer traces the south face of the hip roof (a trapezoid)
    corners = [(x0 - 8, y0 - 5), (x0 + 8, y0 - 5), (x0 + 3, y0), (x0 - 3, y0)]
    pts = [[la, lo] for lo, la in (to_ll.transform(x, y) for x, y in corners)]
    r = c.post("/measure-traced", json={"facets": [{"points": pts, "pitch_over_12": 4}]},
               headers={"X-API-Key": "test-key"})
    assert r.status_code == 200, r.text
    f = r.json()["facets"][0]
    assert f["pitch_source"] == "lidar"
    assert abs(f["pitch_over_12"] - 6) < 0.3              # LiDAR overrides the guessed 4/12
    true = (16 + 6) / 2 * 5 * np.sqrt(1.25) * 10.7639
    assert abs(f["roof_sqft"] - true) / true < 0.02


def test_no_coverage(client):
    c, _, _ = client
    r = c.post("/measure", json={"lat": 40.0, "lon": -100.0}, headers={"X-API-Key": "test-key"})
    assert r.status_code == 404


def test_gable_and_unclassified_algorithms():
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app.roof import FT_PER_M, SQFT_PER_M2, measure_roof
    xyz, cls = gable_roof(classify=False)
    r = measure_roof(xyz, cls, CENTER_UTM)
    sqft = sum(f.plan_m2 / np.cos(f.slope_rad) for f in r.facets) * SQFT_PER_M2
    assert len(r.facets) == 2
    assert abs(sqft - 1242) / 1242 < 0.05
    assert abs(r.edges_m["ridge"] * FT_PER_M - 39.4) < 4
    assert abs(r.edges_m["rake"] * FT_PER_M - 63.1) / 63.1 < 0.1
