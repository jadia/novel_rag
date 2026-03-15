"""
=============================================================================
API Package — FastAPI Web Interface
=============================================================================
This package provides the REST + WebSocket API layer that powers the web UI.

The API is a thin wrapper around the core engine (src/core/). It translates
HTTP requests and WebSocket messages into calls to the same VectorDB,
DocumentProcessor, and Embedder classes that the CLI uses.

Modules:
  - routes.py     → API endpoint definitions (REST + WebSocket)
  - chat_store.py → SQLite-backed chat session persistence
=============================================================================
"""
