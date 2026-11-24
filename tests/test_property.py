"""End-to-end test of /measure-property on a synthetic lot (fake LiDAR server + fake building map)."""
import json
import os
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import box
import shapely
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(__file__))
from synthetic import property_scene  # noqa: E402
from test_service import CENTER_UTM, UTM, build_fake_ept  # noqa: E402

KEY = {"X-API-Key": "test-key"}


def _outlines_lonlat(rects):
    to_ll = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    polys = [box(a + CENTER_UTM[0], b + CENTER_UTM[1], c + CENTER_UTM[0], d + CENTER_UTM[1])
             for a, b, c, d in rects]
    polys = [unary_union(polys[0:2])] + polys[2:]       # house + addition = one map outline
    return [shapely.transform(p, lambda x, y: to_ll.transform(x, y), interleaved=False) for p in polys]


@pytest.fixture(scope="module")
def prop(tmp_path_factory):
    root = tmp_path_factory.mktemp("lidar_prop")
    xyz, cls, rects = property_scene()
    build_fake_ept(f"{root}/TEST_Lot_2019", xyz, cls)
    to_ll = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(CENTER_UTM[0], CENTER_UTM[1] + 12)   # address point at the street

    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    d = 0.01
    with open(f"{root}/resources.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "properties": {"name": "TEST_Lot_2019", "id": 2, "count": len(xyz),
                           "url": f"{base}/TEST_Lot_2019/ept.json"},
            "geometry": {"type": "Polygon", "coordinates": [[[lon - d, lat - d], [lon + d, lat - d],
                         [lon + d, lat + d], [lon - d, lat + d], [lon - d, lat - d]]]}}]}, f)

    os.environ["LIDAR_INDEX_URL"] = f"{base}/resources.geojson"
    os.environ["CACHE_DIR"] = str(root / "cache")
    os.environ["SERVICE_API_KEY"] = "test-key"
    for m in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[m]
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from fastapi.testclient import TestClient
    import app.main as main

    outlines = _outlines_lonlat(rects)
    main.buildings_in_bbox = lambda bbox: (outlines, "test-map")    # stand-in for Overture/OSM
    yield TestClient(main.app), lat, lon, main
    srv.shutdown()


def _by_role(j):
    out = {}
    for s in j["structures"]:
        out.setdefault(s["role"], []).append(s)
    return out


def test_whole_property(prop):
    c, lat, lon, _ = prop
    r = c.post("/measure-property", json={"lat": lat, "lon": lon}, headers=KEY)
    assert r.status_code == 200, r.text
    j = r.json()
    roles = _by_role(j)

    main = roles["main"][0]
    assert main["status"] == "confirmed" and main["included"]
    assert abs(main["roof_sqft"] - 1926) / 1926 < 0.05                 # 16x10 m hip, 6/12

    acc = {s["status"]: s for s in roles["accessory"]}
    garage = acc["confirmed"]
    assert garage["included"] and abs(garage["plan_sqft"] - 452) / 452 < 0.12      # 7x6 m
    assert acc["lidar_only"]["included"]                               # old shed: flagged
    adu = acc["new_since_scan"]
    assert adu["source"] == "needs_trace" and adu["included"]
    assert abs(adu["plan_sqft"] - 517) / 517 < 0.08                    # 6x8 m ADU
    assert adu["roof_sqft_by_pitch"]["0/12"] == adu["plan_sqft"]

    addition = roles["addition_to_main"][0]
    assert addition["status"] == "addition_since_scan" and addition["included"]
    assert 250 < addition["plan_sqft"] < 450                           # 8x4 m = 344 sq ft

    assert not roles["neighbor"][0]["included"]
    assert not roles["neighbor_accessory"][0]["included"]

    t = j["totals_included"]
    assert t["structures_measured"] == 3 and t["structures_needing_pitch"] == 2
    assert t["complete"] is False
    assert j["sources"]["building_maps"] == "test-map"


def test_traced_new_structure_uses_roofer_pitch(prop):
    """The ADU isn't in the scan: LiDAR must NOT invent a pitch from the bare ground there."""
    c, lat, lon, _ = prop
    j = c.post("/measure-property", json={"lat": lat, "lon": lon}, headers=KEY).json()
    adu = next(s for s in j["structures"] if s["status"] == "new_since_scan")
    r = c.post("/measure-traced", json={"facets": [{"points": adu["outline_latlon"], "pitch_over_12": 5}]},
               headers=KEY)
    f = r.json()["facets"][0]
    assert f["pitch_source"] == "roofer" and f["pitch"] == "5/12"
    assert f["roof_sqft"] == adu["roof_sqft_by_pitch"]["5/12"]


def test_without_building_maps(prop):
    c, lat, lon, main = prop
    saved = main.buildings_in_bbox
    main.buildings_in_bbox = lambda bbox: ([], None)
    try:
        j = c.post("/measure-property", json={"lat": lat + 1e-6, "lon": lon}, headers=KEY).json()
    finally:
        main.buildings_in_bbox = saved
    assert all(s["status"] == "not_checked" for s in j["structures"])
    assert any("building maps" in w for w in j["warnings"])


def _pin(lat, lon, dx, dy):
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    to_ll = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    x, y = to_utm.transform(lon, lat)
    plon, plat = to_ll.transform(x + dx, y + dy)
    return {"lat": plat, "lon": plon}


def test_pin_adds_neighbor_shed(prop):
    """Address point is at (0, 12); the neighbour's shed is at (26, -14)."""
    c, lat, lon, _ = prop
    r = c.post("/structure-at-pin", json=_pin(lat, lon, 26, -26), headers=KEY)
    assert r.status_code == 200, r.text
    s = r.json()["structures"][0]
    assert s["role"] == "added_by_user" and s["included"] and s["source"] == "lidar"
    assert abs(s["roof_sqft"] - 172) / 172 < 0.2                      # 4x4 m flat roof


def test_pin_on_new_adu_needs_pitch(prop):
    c, lat, lon, _ = prop
    s = c.post("/structure-at-pin", json=_pin(lat, lon, -12.4, -34.4), headers=KEY).json()["structures"][0]
    assert s["source"] == "needs_trace" and s["status"] == "new_since_scan"


def test_pin_on_empty_ground(prop):
    c, lat, lon, _ = prop
    r = c.post("/structure-at-pin", json=_pin(lat, lon, -30, 20), headers=KEY)
    assert r.status_code == 404
