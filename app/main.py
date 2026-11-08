"""FastAPI application entry point.

The application hosts three internal routes and a liveness endpoint:

* ``GET /health`` — the liveness check.
* ``POST /internal/descriptions/generate`` — the English description
  translation route, served by the local Ollama provider.
* ``POST /internal/steam/search`` — the Steam Store Search route.

A ``lifespan`` context manager stores the resolved ``Settings``,
providers, and services on ``app.state`` so routers can read them
without re-parsing environment variables per request.

Provider
--------
Every generation route is served by the local, free ``ollama`` provider
through the ``StructuredProvider`` contract. The legacy Gemini-backed
provider and the earlier generation pipelines have been removed: there
is no automatic fallback between providers, and no hosted or paid
backend is called.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app import __version__
from app.config import Settings, get_settings
from app.errors import MissingOllamaConfigError
from app.errors_http import register_handlers
from app.logging_config import configure_logging
from app.providers import build_structured_provider
from app.retrieval.steam import SteamAppIdRetriever
from app.retrieval.steam_mapper import STEAM_LOCALE_ENGLISH
from app.retrieval.steam_search import SteamSearchRetriever
from app.routers import (
    descriptions,
    health,
    steam_search,
)
from app.services.generation import DescriptionTranslationService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise and tear down application-wide resources."""

    settings: Settings = get_settings()
    configure_logging(app_env=settings.app_env)
    app.state.settings = settings

    # ── The structured provider (local Ollama) ─────────────────────────
    #
    # The structured provider is built once and shared by the services
    # that need it. Construction performs no network I/O: the
    # model-availability probe happens on first use and is cached by the
    # provider.
    structured_provider = None
    try:
        structured_provider = build_structured_provider(settings)
        app.state.structured_provider = structured_provider
    except MissingOllamaConfigError as exc:
        if settings.ai_generation_enabled:
            import logging

            logging.getLogger(__name__).warning(
                "structured_provider_not_configured reason=%s", exc.message
            )
        app.state.structured_provider = None

    # ── English description translation ──────────────────────────────
    #
    # The translation path reuses the *same* structured provider (the
    # local Ollama model) but has its own service, its own schema and its
    # own route, and it retrieves the **English** Steam description.
    translation_service = None
    if structured_provider is not None:
        translation_service = DescriptionTranslationService(
            settings=settings, provider=structured_provider
        )
    app.state.translation_service = translation_service

    # ── Optional Steam enrichment ───────────────────────────────────
    # The retriever is only built when the operator has opted in. The
    # construction performs no network I/O: a request is made only when
    # a translation request carries a ``steam_app_id``.
    #
    # The retriever uses the **English** locale, so the Steam
    # description it reads is the one the translation path translates.
    steam_retriever = None
    if settings.steam_retrieval_enabled:
        steam_retriever = SteamAppIdRetriever(
            connect_timeout_seconds=settings.steam_connect_timeout_seconds,
            read_timeout_seconds=settings.steam_read_timeout_seconds,
            max_response_bytes=settings.steam_max_response_bytes,
            locale=STEAM_LOCALE_ENGLISH,
        )
        if translation_service is not None:
            translation_service.attach_steam_retriever(steam_retriever)
    app.state.steam_retriever = steam_retriever

    # ── Steam Store Search (optional) ─────────────────────────────────
    # The search retriever is built independently of the appdetails
    # retriever: it serves the separate ``/internal/steam/search``
    # endpoint and shares nothing with the evidence pipeline. Like the
    # appdetails retriever it is only built when the operator has opted
    # in, and its construction performs no network I/O.
    steam_search_retriever = None
    if settings.steam_retrieval_enabled:
        steam_search_retriever = SteamSearchRetriever(
            connect_timeout_seconds=settings.steam_connect_timeout_seconds,
            read_timeout_seconds=settings.steam_read_timeout_seconds,
            max_response_bytes=settings.steam_max_response_bytes,
            max_candidates=settings.steam_search_max_candidates,
            cooldown_seconds=settings.steam_search_cooldown_seconds,
        )
    app.state.steam_search_retriever = steam_search_retriever

    # Application startup log line — never includes secrets.
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        "startup service=%s version=%s env=%s ai_generation_enabled=%s "
        "model_configured=%s llm_provider=%s",
        "hotwatergas-ai",
        __version__,
        settings.app_env,
        settings.ai_generation_enabled,
        bool(settings.ollama_model_id),
        settings.llm_provider,
    )

    yield

    # Release the reusable HTTP client held by the structured provider.
    structured_provider = getattr(app.state, "structured_provider", None)
    if structured_provider is not None:
        close = getattr(structured_provider, "close", None)
        if callable(close):
            close()

    # Release the Steam retriever's HTTP client, if one was built.
    steam_retriever = getattr(app.state, "steam_retriever", None)
    if steam_retriever is not None:
        close = getattr(steam_retriever, "close", None)
        if callable(close):
            close()

    # Release the Steam search retriever's HTTP client, if one was built.
    steam_search_retriever = getattr(app.state, "steam_search_retriever", None)
    if steam_search_retriever is not None:
        close = getattr(steam_search_retriever, "close", None)
        if callable(close):
            close()

    logger.info("shutdown service=%s version=%s", "hotwatergas-ai", __version__)


def create_app() -> FastAPI:
    """Application factory.

    Exposed as a function so tests can build isolated instances with
    custom environment variables (see ``tests/conftest.py``).
    """

    app = FastAPI(
        title="HotWaterGas AI Service",
        version=__version__,
        # API documentation endpoints are exposed unconditionally.
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(descriptions.router)
    app.include_router(steam_search.router)

    # The application-wide exception-handler bundle is registered
    # exactly once, from the version-neutral module that owns it. The
    # per-router helpers stay available for mounting a single router on
    # a standalone app, but they are not needed here: their handlers
    # are the same ones the bundle installs.
    register_handlers(app)

    return app


# Module-level instance for ``uvicorn app.main:app``.
app = create_app()
