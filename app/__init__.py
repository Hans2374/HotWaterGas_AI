"""HotWaterGas AI service package.

This package hosts the internal FastAPI service that translates an
English Steam store description into Vietnamese through a local
Ollama provider. It ships a health endpoint, a Steam Store Search
endpoint and the translation route; it has no database connectivity.
"""

__version__ = "0.1.0"
