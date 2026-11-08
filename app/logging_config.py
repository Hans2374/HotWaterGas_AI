"""Safe logging configuration.

A single function ``configure_logging`` sets up the root logger with a
stable format. The level defaults to ``INFO`` and is bumped to ``DEBUG``
when ``APP_ENV=development``.

Secrets and full request bodies must never be logged. This module only
controls format and level; route handlers are responsible for what they
emit.
"""

from __future__ import annotations

import logging
import sys


_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s — %(message)s"


def configure_logging(*, app_env: str) -> None:
    """Configure the root logger for the given environment.

    Idempotent: safe to call multiple times; existing handlers are
    replaced so a re-import in tests does not double-log.
    """

    level = logging.DEBUG if app_env == "development" else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    # Remove pre-existing handlers (e.g. uvicorn's) to keep one stream.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setLevel(level)
    stream.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(stream)
