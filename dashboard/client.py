"""Dashboard -> API client. Uses the local HTTP API when RISKCAUSAL_API_URL is set (localhost only), otherwise the
same read-only FastAPI app in-process. Either way the dashboard only ever issues GET requests."""

from __future__ import annotations

import os
from urllib.parse import urlparse

LOCAL = {"localhost", "127.0.0.1", "::1"}


class ApiClient:
    def __init__(self, base_url: str | None = None):
        base_url = base_url if base_url is not None else os.environ.get("RISKCAUSAL_API_URL")
        if base_url:
            if urlparse(base_url).hostname not in LOCAL:
                raise ValueError(f"API URL must be local (COST_POLICY.md): {base_url!r}")
            import httpx

            self._c, self.mode = httpx.Client(base_url=base_url, timeout=60), f"http {base_url}"
        else:
            from fastapi.testclient import TestClient

            from riskcausal_api.main import app

            self._c, self.mode = TestClient(app), "in-process (read-only)"

    def get(self, path: str, **params) -> dict:
        r = self._c.get(path, params=params or None)
        r.raise_for_status()
        return r.json()
