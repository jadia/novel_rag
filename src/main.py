"""
=============================================================================
FastAPI Application Entry Point
=============================================================================
This is the main entry point for the Novel RAG web application.

WHAT THIS FILE DOES:
  1. Creates the FastAPI app instance
  2. Mounts the API router (all the /api/* and /ws/* endpoints)
  3. Serves the frontend static files (HTML, CSS, JS)
  4. Configures a health check endpoint for Docker
  5. Sets up CORS for local development

HOW STATIC FILE SERVING WORKS:
Instead of running a separate web server (Nginx, Apache) for the front-end,
FastAPI can serve static files directly. We mount the frontend/ directory
so that:
  - http://localhost:8000/          → serves index.html
  - http://localhost:8000/style.css → serves style.css
  - http://localhost:8000/app.js   → serves app.js

This keeps the deployment to a single process (uvicorn), which is simpler
for learning and development. In production, you'd put Nginx in front.

TO RUN:
  uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
=============================================================================
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.core.utils import get_logger

logger = get_logger("Main")

# ---------------------------------------------------------------------------
# Create the FastAPI application instance.
#
# FastAPI automatically generates interactive API documentation:
#   - Swagger UI:  http://localhost:8000/docs
#   - ReDoc:       http://localhost:8000/redoc
# These are incredibly useful for testing your endpoints during development.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Novel RAG",
    description=(
        "A Retrieval-Augmented Generation system for querying novel lore. "
        "Ask questions about characters, relationships, and plot events."
    ),
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing) Middleware
#
# WHY DO WE NEED THIS?
# When developing locally, your browser may open the HTML file from a
# different port (or even file://) than the API server. Browsers block
# "cross-origin" requests by default as a security measure. CORS headers
# tell the browser: "It's OK, this origin is allowed to talk to me."
#
# In production (when frontend and API are on the same origin), this
# is less critical, but we keep it for development convenience.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (restrict in production!)
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP methods (GET, POST, PUT, etc.)
    allow_headers=["*"],       # Allow all headers
)

# ---------------------------------------------------------------------------
# Mount the API router.
# All routes defined in routes.py are now available under the app.
# For example: router.get("/api/novels") → app serves GET /api/novels
# ---------------------------------------------------------------------------
app.include_router(router)

# ---------------------------------------------------------------------------
# Health check endpoint for Docker.
#
# Docker Compose can be configured to ping this endpoint to verify the
# container is healthy. If it returns a non-200 status, Docker marks
# the container as "unhealthy" and can restart it.
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Simple health check endpoint for Docker and monitoring."""
    return {"status": "healthy"}

# ---------------------------------------------------------------------------
# Serve the frontend static files.
#
# HOW THIS WORKS:
# We have two layers of static file serving:
#   1. StaticFiles mount catches requests for specific files (style.css, app.js)
#   2. The catch-all route below handles "/" and any other path, returning
#      index.html — this is how Single Page Applications (SPAs) work.
#
# IMPORTANT: The StaticFiles mount must come AFTER the API router, otherwise
# it would try to serve /api/novels as a file and return 404.
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend"
)

# Only mount if the frontend directory exists (it won't during testing)
if os.path.exists(FRONTEND_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static"
    )


@app.get("/")
async def serve_index():
    """
    Serve the main index.html file for the web UI.

    This is the entry point for the Single Page Application. All navigation
    happens in JavaScript without full page reloads.
    """
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. API is running at /docs"}
