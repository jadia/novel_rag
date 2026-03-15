"""
=============================================================================
Novel RAG — Core Engine Package
=============================================================================
This package contains the foundational modules that power the RAG pipeline:
  - config.py           → Settings management and environment configuration
  - document_processor.py → Chapter chunking with sliding-window overlap
  - embeddings.py       → Text-to-vector conversion (local or API)
  - vector_db.py        → ChromaDB vector storage and similarity search
  - utils.py            → Logging and performance measurement utilities

These modules are imported by both the CLI (src/cli.py) and the web API
(src/api/), keeping the core logic cleanly separated from any interface.
=============================================================================
"""

# ---------------------------------------------------------------------------
# Re-export key classes so callers can write:
#   from src.core import VectorDB, DocumentProcessor, Embedder
# instead of:
#   from src.core.vector_db import VectorDB
# ---------------------------------------------------------------------------
from src.core.vector_db import VectorDB
from src.core.document_processor import DocumentProcessor
from src.core.embeddings import Embedder
from src.core.utils import get_logger, time_it
