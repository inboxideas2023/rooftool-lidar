"""Try the service on a real address once it is deployed (or running locally).

  python scripts/smoke_test.py "123 Main St, Pasadena, CA 91101"
  SERVICE_URL=https://your-service.onrender.com SERVICE_API_KEY=... python scripts/smoke_test.py "..."
"""
import json
import os
import sys

import httpx

url = os.getenv("SERVICE_URL", "http://localhost:8000")
key = os.getenv("SERVICE_API_KEY", "")
address = " ".join(a for a in sys.argv[1:] if a != "--full") or "1600 Pennsylvania Ave NW, Washington, DC 20500"
r = httpx.post(f"{url}/measure", json={"address": address},
               headers={"X-API-Key": key}, timeout=120)
print(r.status_code)
j = r.json()
j.pop("facets", None) if r.status_code == 200 and "--full" not in sys.argv else None
print(json.dumps(j, indent=2))
