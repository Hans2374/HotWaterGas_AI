"""Internal-service-token verification dependency.

A single dependency, ``require_internal_token``, protects every route
under ``/internal/...``. The dependency:

1. Reads the expected token from the resolved ``Settings``. If the
   server-side token is empty while the route is being exercised, the
   request is rejected with ``MissingInternalTokenConfigError`` — the
   service fails closed.
2. Reads the presented token from the ``X-Internal-Token`` request
   header. The header is required; missing or empty values are
   rejected.
3. Compares the two values using ``hmac.compare_digest`` (constant
   time) so a timing-side-channel attacker cannot recover the token
   byte-by-byte.
4. On any failure, raises the appropriate domain exception. The route
   layer maps that to a generic HTTP error envelope; the expected token
   is never echoed back to the client.

The dependency is intentionally independent of the ``/health`` route.
The liveness endpoint is public; only internal routes require the token.
"""

from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Header, Request

from app.config import Settings
from app.errors import (
    InvalidInternalTokenError,
    MissingInternalTokenConfigError,
)


HEADER_NAME = "X-Internal-Token"


def _settings(request: Request) -> Settings:
    """Read the application ``Settings`` from the FastAPI request state.

    Falls back to a fresh ``Settings()`` instance if the lifespan did
    not populate ``app.state`` — this should never happen in production
    but keeps the dependency safe to import from tests.
    """

    settings = getattr(request.app.state, "settings", None)
    if settings is None:  # pragma: no cover - defensive
        from app.config import get_settings

        return get_settings()
    return settings


def require_internal_token(
    request: Request,
    x_internal_token: Optional[str] = Header(default=None, alias=HEADER_NAME),
) -> None:
    """Validate the ``X-Internal-Token`` header against the configured value.

    Returns ``None`` on success. Raises a domain exception on failure
    which the route layer maps to a generic HTTP error envelope.
    """

    settings = _settings(request)
    expected = settings.internal_service_token

    if not expected:
        # Fail closed. This is a server-side misconfiguration: the
        # service cannot accept any internal traffic safely.
        raise MissingInternalTokenConfigError(
            "Internal token is not configured on the server."
        )

    if not x_internal_token:
        raise InvalidInternalTokenError("Missing internal token.")

    # Constant-time comparison. The two values are coerced to bytes so
    # ``compare_digest`` cannot raise a TypeError on a stray ``None``.
    presented = x_internal_token.encode("utf-8")
    expected_bytes = expected.encode("utf-8")
    if not hmac.compare_digest(presented, expected_bytes):
        raise InvalidInternalTokenError("Invalid internal token.")
    return None
