# RoofTool Pro – LiDAR measurement service

A small Python web service that measures roofs from **free USGS 3DEP LiDAR** instead of the
Google Solar API. Per-search data cost: **$0** (hosting is the only cost).

## What it does

| Endpoint | You send | You get back |
| --- | --- | --- |
| `POST /measure` | an address, or lat/lon | every roof face with pitch, direction and true (sloped) area; total sq ft, squares, squares with waste, predominant pitch; ridge, hip, valley, eave and rake lengths in feet; each face's outline as lat/lon points to draw on the map |
| `POST /measure-property` | an address, or lat/lon | **every structure on the property** (house, garage, ADU, sheds), each measured from LiDAR or flagged for the roofer, plus property totals. Cross-checks the LiDAR against current building maps to catch structures built after the scan. |
| `POST /structure-at-pin` | lat/lon where the roofer dropped a pin | that one structure (plus any addition on it), measured from LiDAR, or as `needs_trace` with its outline. 404 if nothing is there, so the app can offer tracing. |
| `POST /measure-traced` | the faces the roofer traced on the map (corner lat/lons, optional pitch) | pitch for each face read from LiDAR (falls back to the roofer's pitch), true area, totals |
| `GET /health` | – | `{"ok": true}` |

How `/measure` works:

1. Address → lat/lon with the free US Census geocoder (no key).
2. Finds the newest USGS LiDAR survey covering that spot and downloads only the small tiles around the house.
3. Separates the house from ground and trees.
4. Fits flat planes to the roof points; each plane is a roof face with its own pitch.
5. Measures each face's area on a 0.5 m grid and corrects it for pitch.
6. Finds where faces meet (ridge, hip, valley) and classifies the outside edges (eave, rake).

How `/measure-property` works:

1. Finds every structure in the LiDAR within ~150 ft of the address.
2. Downloads current building outlines for the same area from **Overture Maps** (OpenStreetMap +
   Microsoft + Esri outlines, refreshed monthly), falling back to **OpenStreetMap**. Both free.
3. Cross-checks the two:
   - outline + LiDAR roof → `confirmed`, measured from LiDAR
   - outline but no LiDAR roof → `new_since_scan` (built after the survey), returned with its
     outline and `roof_sqft_by_pitch`, so the roofer just picks the pitch
   - outline sticks out past the LiDAR roof → `addition_since_scan`, same treatment for the added part
   - LiDAR roof but no outline → `lidar_only` (may have been torn down; check the map)
4. Decides what's on the property without lot lines: the structure at the address is the main
   house; other 1,000+ sq ft buildings are neighbours' houses; each smaller structure belongs to
   the house it's closest to (within 100 ft). Everything comes back with an `included` flag so
   the roofer can tick or untick anything the rules got wrong.
5. Totals add up the included structures. `totals_included.complete` is `false` until every
   included `needs_trace` structure has a pitch.

Google Maps imagery is not analysed (Google's terms don't allow deriving data from it). The
final check is visual: the app draws every outline on its Google satellite map, so the roofer
can spot anything newer than both data sources and trace it.

Tip: send lat/lon from your app's Google address autocomplete instead of a plain address. It
lands on the house itself; the free Census geocoder sometimes lands in the street, which can
make a neighbour's house look like the main one.

## Accuracy (tested on synthetic roofs with known answers)

| Roof | Area error | Pitch | Edges |
| --- | --- | --- | --- |
| Hip, 16 × 10 m, 6/12 | +2% | exact | ridge, hips, eaves within ~3% |
| Gable, 12 × 8 m, 8/12 | +3% | exact | ridge, eaves, rakes within ~3% |
| L-shaped cross-gable | +3% | exact | ridges/valleys ~10% short |
| Whole lot: house, garage, old shed, new ADU, new addition, neighbour's house + shed | all 7 structures found and classified correctly | | |

Real-world accuracy depends on the LiDAR survey's density and date. The response includes
`lidar.year`, `lidar.points_per_m2` and `warnings` so the app can tell the roofer when to double-check.
Known limits: roofs changed after the survey date won't show; very small dormers or crickets
(under ~3 m²) can be missed; heavy tree cover over the roof lowers accuracy. Coverage is the
continental US (most of it), plus parts of Alaska, Hawaii and territories.

## Run it on your computer

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000/docs to try it in the browser
python scripts/smoke_test.py "123 Main St, Pasadena, CA 91101"
```

Run the tests: `pip install pytest && python -m pytest -q`

## Deploy (Render, about $7/month)

1. Put this folder in a GitHub repo.
2. On render.com: **New → Web Service → pick the repo**. Render detects the `Dockerfile`.
3. Choose the Starter instance (512 MB RAM is enough; the free tier works for testing but sleeps).
4. Add environment variables:
   - `SERVICE_API_KEY` = a long random password (your app sends this in the `X-API-Key` header)
   - `ALLOWED_ORIGINS` = `https://www.rooftoolpro.com` (comma-separate more if needed)
5. Deploy, then visit `https://<your-service>.onrender.com/health`.

Railway or Fly.io work the same way with the same Dockerfile.

## Connect it to the Lovable app

Keep the API key out of the browser by calling the service from a Supabase Edge Function
(Lovable apps use Supabase). Add `LIDAR_URL` and `LIDAR_API_KEY` as Supabase secrets, then:

```ts
// supabase/functions/measure-roof/index.ts
Deno.serve(async (req) => {
  const body = await req.json();               // { address } or { lat, lon }
  const r = await fetch(`${Deno.env.get("LIDAR_URL")}/measure`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-API-Key": Deno.env.get("LIDAR_API_KEY")! },
    body: JSON.stringify(body),
  });
  return new Response(await r.text(), {
    status: r.status,
    headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
  });
});
```

A prompt you can give Lovable:

> Change the edge function `measure-roof` to call `/measure-property` and pass `{ lat, lon }`
> from the address autocomplete. On the results screen:
> - Draw every item in `structures` on the Google satellite map using `outline_latlon`. Green
>   outline if `included`, gray if not. Each structure gets a checkbox so the user can include
>   or exclude it; recalculate the totals in the app when they change it.
> - For structures with `source: "lidar"`, show `roof_sqft`, `squares`, `predominant_pitch`
>   and `edges_ft`, and draw `facets[].polygon_latlon` labeled with `pitch`.
> - For structures with `source: "needs_trace"`, show `status_text` in a yellow box and a pitch
>   dropdown (0/12 to 12/12). When a pitch is picked, use `roof_sqft_by_pitch[pitch]` as its
>   roof area.
> - Show the property total = sum of included structures, and a "Not finished" badge until
>   every included `needs_trace` structure has a pitch.
> - Show `warnings` in a yellow note, and "LiDAR: {sources.lidar.year}" in small text.
> - Add a "Remove" button on each structure that unchecks it (same as the checkbox).
> - Add an "Add a building" button. The user drops a pin on the map; call the edge function
>   `structure-at-pin` with `{ lat, lon }` and add the returned `structures` to the list as
>   included. If it returns 404, show "No building found there — trace it" and open the
>   tracing tool.

## Example `/measure` response (trimmed)

```json
{
  "lidar": {"dataset": "<USGS survey name>", "year": 2016, "roof_points": 812, "points_per_m2": 5.1},
  "totals": {"roof_sqft": 1957, "squares": 19.6, "squares_with_waste": 21.5, "waste_pct": 10.0,
             "predominant_pitch": "6/12", "facet_count": 4, "footprint_sqft": 1750},
  "edges_ft": {"ridge": 20, "hip": 98, "valley": 0, "eave": 168, "rake": 7, "other": 0},
  "facets": [{"id": 0, "pitch": "6/12", "pitch_degrees": 26.6, "faces_azimuth_deg": 180,
              "plan_sqft": 525, "roof_sqft": 587, "polygon_latlon": [[34.06, -117.0], "..."]}],
  "warnings": ["LiDAR was flown in 2016; check for additions or re-roofs since then."]
}
```

## Files

- `app/main.py` – the web API
- `app/ept.py` – downloads USGS LiDAR tiles (no PDAL needed)
- `app/roof.py` – the roof math: house isolation, plane fitting, areas, edges
- `app/property.py` – whole-property logic: cross-check, what's on the lot, totals
- `app/footprints.py` – free current building outlines (Overture Maps, OpenStreetMap)
- `tests/` – synthetic roofs and an end-to-end test with a fake LiDAR server
- `scripts/smoke_test.py` – try a real address against a running service
