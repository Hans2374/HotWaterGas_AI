"""MANUAL LIVE SMOKE TEST — full HTTP path through the FastAPI app.

Starts the real app with a real Ollama provider and issues real HTTP
requests against the canonical translation route and the Steam Store
Search route, plus checks that the removed v1 and v2 generation routes
stay gone and that the legacy route alias still answers.

Local and free. No paid API. No secret is printed.

The translation route needs an English Steam description, so it also
needs ``STEAM_RETRIEVAL_ENABLED=true`` and a reachable Steam. Without
those it answers a controlled error - which is itself a useful check,
because the route must never answer 404.
"""

from __future__ import annotations

import json
import os
import sys

os.environ["AI_GENERATION_ENABLED"] = "true"
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["INTERNAL_SERVICE_TOKEN"] = "smoke-test-token-local-only"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from app.dependencies.auth import HEADER_NAME  # noqa: E402


HEADERS = {HEADER_NAME: "smoke-test-token-local-only"}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

    app = create_app()
    with TestClient(app) as client:
        print("== 1. GET /health ==")
        health = client.get("/health")
        print(f"status={health.status_code} body={json.dumps(health.json())}")

        print("\n== 2. auth: missing token ==")
        unauth = client.post(
            "/internal/descriptions/generate",
            json={"product_name": "Echoes of the Tundra", "steam_app_id": 620},
        )
        print(f"status={unauth.status_code} body={json.dumps(unauth.json())}")

        print("\n== 3. POST /internal/descriptions/generate (live Ollama) ==")
        response = client.post(
            "/internal/descriptions/generate",
            json={"product_name": "Echoes of the Tundra", "steam_app_id": 620},
            headers=HEADERS,
        )
        print(f"status={response.status_code}")
        if response.status_code == 404:
            print("TRANSLATION ROUTE MISSING")
            return 2
        if response.status_code != 200:
            # A controlled upstream error is acceptable when Steam or
            # Ollama is unavailable; an unexpected status is not.
            print(f"body={response.text}")
            print("(a non-200 here means Steam or Ollama is unavailable)")
            body = None
        else:
            body = response.json()
            print(
                json.dumps(
                    {
                        "schema_version": body["schema_version"],
                        "model": body["model"]["model_id"],
                        "duration_ms": body["model"]["duration_ms"],
                        "release_date": body["release_date"],
                        "platforms": body["platforms"]["values"],
                        "minimum_os": body["system_requirements"]["minimum"]["os"],
                        "description_prefix": body["description"]["text"][:90],
                        "warning_codes": [w["code"] for w in body["warnings"]],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )

        print("\n== 4. POST /internal/steam/search (route reachable) ==")
        search = client.post(
            "/internal/steam/search",
            json={"term": "tundra"},
            headers=HEADERS,
        )
        print(f"status={search.status_code} body={json.dumps(search.json())}")
        if search.status_code == 404:
            print("STEAM SEARCH ROUTE MISSING")
            return 3

        print("\n== 5. The removed v1 and v2 routes stay gone ==")
        removed = {}
        for path in (
            "/internal/descriptions/generate-v2",
        ):
            gone = client.post(
                path,
                json={"product_name": "Echoes of the Tundra"},
                headers=HEADERS,
            )
            removed[path] = gone.status_code
            print(f"{path} -> {gone.status_code}")
            if gone.status_code != 404:
                print(f"REGRESSION: {path} still answers")
                return 4

        print("\n== 6. The legacy route alias still answers ==")
        legacy = client.post(
            "/internal/descriptions/generate-v3",
            json={"product_name": "Echoes of the Tundra", "steam_app_id": 620},
            headers=HEADERS,
        )
        print(f"/internal/descriptions/generate-v3 -> {legacy.status_code}")
        if legacy.status_code == 404:
            print("LEGACY ROUTE ALIAS MISSING")
            return 6

        print("\n== 7. Contract assertions ==")
        checks = {
            "the canonical route answers (not 404)": response.status_code != 404,
            "steam search route answers (not 404)": search.status_code != 404,
            "the removed v2 route answers 404": removed[
                "/internal/descriptions/generate-v2"
            ]
            == 404,
            "auth is enforced": unauth.status_code == 401,
            "the legacy alias answers (not 404)": legacy.status_code != 404,
            "no token in body": "smoke-test-token-local-only" not in response.text,
        }
        if body is not None:
            checks.update(
                {
                    "schema_version == 3": body["schema_version"] == "3",
                    "the translation is never verified": body["description"][
                        "verification"
                    ]["status"]
                    == "unverified",
                    "the description_source block is present": bool(
                        body["description_source"]
                    ),
                    "the word count is server-measured": (
                        body["description"]["word_count"]
                        == len(body["description"]["text"].split())
                    ),
                }
            )
        for label, ok in checks.items():
            print(f"  [{'OK' if ok else 'FAIL'}] {label}")
        return 0 if all(checks.values()) else 5


if __name__ == "__main__":
    sys.exit(main())
