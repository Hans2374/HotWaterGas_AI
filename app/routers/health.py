"""Health endpoint.

The response is intentionally small and stable. It is a *liveness* check
only: it confirms the Python process and the FastAPI app are running.
It does **not** prove that AI generation is operational — generation is
disabled by default and no provider connectivity is checked.

A future readiness check may add a separate route that verifies the
provider configuration and the internal auth token.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel


router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Stable health-check payload.

    Fields are explicit and fixed so dashboards / orchestrators can rely
    on them. Do not add fields that could leak configuration values.
    """

    status: str
    service: str
    version: str
    ai_generation_enabled: bool


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return a liveness payload.

    The endpoint never touches the network, never reads the Gemini key,
    and never logs the resolved configuration. The ``ai_generation_enabled``
    field is exposed so operators can verify the current switch state
    without inspecting server logs.
    """

    # Lazy import to keep the router module side-effect free.
    from app import __version__

    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        service="hotwatergas-ai",
        version=__version__,
        ai_generation_enabled=settings.ai_generation_enabled,
    )
