# =============================================================================
# Dockerfile — Novel RAG Application
# =============================================================================
# This Dockerfile builds a production-ready container for the Novel RAG app.
#
# DESIGN DECISIONS:
#   1. python:3.12-slim — Smaller than full Python image (~150MB vs ~1GB).
#      The "slim" variant has everything we need without extras like gcc.
#   2. Non-root user — Running as root inside containers is a security risk.
#      If the app is compromised, the attacker has limited permissions.
#   3. pip install with --no-cache-dir — Saves ~50MB by not caching downloaded
#      wheel files (we won't pip install again inside the container).
#
# USAGE:
#   docker build -t novel-rag .
#   docker run -p 8000:8000 -v ./data:/app/data:ro -v novel-rag-db:/app/db novel-rag
# =============================================================================

# Start from the official Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables:
#   PYTHONDONTWRITEBYTECODE=1  → Don't create .pyc files (cleaner container)
#   PYTHONUNBUFFERED=1         → Print output immediately (important for logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# --- STEP 1: Install dependencies first (Docker layer caching) ---
# By copying requirements.txt BEFORE copying the source code, Docker can
# cache the dependency installation layer. This means if you only change
# your code (not dependencies), rebuilds are nearly instant.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- STEP 2: Copy application source code ---
COPY src/ ./src/
COPY frontend/ ./frontend/

# --- STEP 3: Create non-root user for security ---
# The 'appuser' has no password, no home directory, and limited permissions.
RUN useradd --no-create-home --system appuser

# Create directories that the app needs to write to
# (data/ is mounted as a volume, but db/ needs to exist for SQLite + ChromaDB)
RUN mkdir -p /app/data /app/db && chown -R appuser:appuser /app/db

# Switch to the non-root user
USER appuser

# Expose the port that uvicorn will listen on
EXPOSE 8000

# --- STEP 4: Start the application ---
# uvicorn is the ASGI server that runs our FastAPI application.
# --host 0.0.0.0 makes it accessible from outside the container.
# --port 8000 matches our EXPOSE directive.
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
