"""
=============================================================================
Configuration & Settings Manager
=============================================================================
This module handles all configuration for the Novel RAG application.

CONFIGURATION HIERARCHY (from lowest to highest priority):
  1. Default values hardcoded in this file (fallback)
  2. Environment variables loaded from .env files
  3. Runtime settings from settings.json (set via the web UI)

WHY THIS 3-TIER APPROACH?
  - Defaults ensure the app always has sensible values
  - .env files are the standard way to inject secrets (especially in Docker)
  - settings.json lets users change config at runtime without restarting

The SettingsManager class handles reading/writing settings.json and provides
a unified interface that respects this priority chain.
=============================================================================
"""

import os
import json
from dotenv import load_dotenv
from src.core.utils import get_logger

logger = get_logger("Config")

# ---------------------------------------------------------------------------
# STEP 1: Load environment variables from the .env file.
# load_dotenv() reads key-value pairs from a .env file and sets them as
# environment variables. The 'override=True' means .env values win over
# any existing system environment variables.
# ---------------------------------------------------------------------------
# Try .env first, then .env.paid, then .env.free — first one that exists wins
for env_file in [".env"]:
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)
        break
else:
    # If no .env file found, load_dotenv will silently do nothing
    load_dotenv()

# ---------------------------------------------------------------------------
# STEP 2: Read values from environment (with defaults).
# os.getenv("KEY", "default") tries the environment first, falls back to
# the default string if the KEY isn't set.
# ---------------------------------------------------------------------------

# --- GEMINI API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- CHUNKING STRATEGY ---
# 600 characters ≈ 100-150 words (a few paragraphs in a novel).
# 150-char overlap ensures pronoun references don't get lost between chunks.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150

# --- EMBEDDINGS CONFIGURATION ---
# Controls whether we use free local embeddings or paid Gemini API embeddings.
EMBEDDING_CONFIG = {
    # True = process text locally (free, uses HuggingFace model)
    # False = use Gemini Embedding API (costs quota, more accurate)
    "USE_LOCAL": os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true",

    # The HuggingFace model for local embeddings.
    # all-MiniLM-L6-v2 is excellent for speed/accuracy balance.
    "LOCAL_MODEL_NAME": os.getenv("LOCAL_MODEL_NAME", "all-MiniLM-L6-v2"),

    # The Gemini model for API-based embeddings.
    "API_MODEL_NAME": os.getenv("API_MODEL_NAME", "models/gemini-embedding-001"),
}

# --- GENERATION MODEL ---
# The Gemini model used for generating answers (not embeddings!).
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "models/gemini-2.0-flash")

# --- DIRECTORIES ---
# We compute absolute paths from this file's location so the app works
# regardless of which directory you run it from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Where novel chapter markdown files are stored (each novel in its own subfolder)
DATA_DIR = os.path.join(BASE_DIR, "data")

# Where ChromaDB, SQLite chat history, and settings.json are stored
DB_DIR = os.path.join(BASE_DIR, "db")

# Path to the runtime settings file
SETTINGS_FILE = os.path.join(DB_DIR, "settings.json")


def get_available_novels() -> list:
    """
    Scans the data directory and returns a sorted list of novel folder names.

    Each subfolder in DATA_DIR represents one novel. The folder name
    becomes the novel's identifier throughout the system.

    Returns:
        Sorted list of folder names, e.g. ['became-the-patron-of-villains', 'another-novel']
    """
    if not os.path.exists(DATA_DIR):
        return []

    # List only directories (not files) inside DATA_DIR
    novels = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    return sorted(novels)


class SettingsManager:
    """
    Manages runtime-configurable settings with a 3-tier priority system.

    DESIGN PATTERN: This follows the "Configuration Object" pattern — instead
    of scattering config reads throughout the codebase, we centralize them
    in one object that any module can query.

    TIER PRIORITY (highest wins):
      3. settings.json  → User changed something via the web UI
      2. .env file      → Set at deploy time (Docker secrets, CI/CD)
      1. Hardcoded      → Sensible defaults in this file

    SECURITY NOTE:
    API keys are stored in settings.json but are MASKED when sent to the
    front-end (e.g., "sk-...abc123" → "sk-...***"). The full key is only
    ever used server-side.
    """

    def __init__(self):
        """Load settings from settings.json if it exists, else use env defaults."""
        self._settings = self._load()

    def _load(self) -> dict:
        """
        Load settings from the JSON file, falling back to environment defaults.

        Returns:
            A dict with all configurable settings.
        """
        # Start with environment/default values
        defaults = {
            "gemini_api_key": GEMINI_API_KEY,
            "use_local_embeddings": EMBEDDING_CONFIG["USE_LOCAL"],
            "local_model_name": EMBEDDING_CONFIG["LOCAL_MODEL_NAME"],
            "api_model_name": EMBEDDING_CONFIG["API_MODEL_NAME"],
            "generation_model": GENERATION_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }

        # If a settings.json exists, it overrides the defaults
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    saved = json.load(f)
                # Merge: saved values override defaults
                defaults.update(saved)
                logger.info("Loaded runtime settings from settings.json")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read settings.json, using defaults: {e}")

        return defaults

    def save(self) -> None:
        """
        Persist current settings to settings.json.

        This is called whenever the user changes settings via the web UI.
        The JSON file lives in db/ which is gitignored, so secrets
        never accidentally end up in version control.
        """
        # Ensure the db/ directory exists
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)

        with open(SETTINGS_FILE, 'w') as f:
            json.dump(self._settings, f, indent=2)

        logger.info("Settings saved to settings.json")

    def get(self, key: str, default=None):
        """Get a setting value by key."""
        return self._settings.get(key, default)

    def get_all(self) -> dict:
        """
        Get all settings, with API keys MASKED for safe front-end display.

        WHY MASK?
        The front-end needs to show that an API key IS configured, but
        should never display the full key (it could be captured by browser
        extensions, screenshots, or shoulder-surfing).
        """
        safe = self._settings.copy()
        # Mask the API key: show first 4 and last 4 chars only
        key = safe.get("gemini_api_key", "")
        if key and len(key) > 8:
            safe["gemini_api_key"] = key[:4] + "***" + key[-4:]
        elif key:
            safe["gemini_api_key"] = "***"
        return safe

    def update(self, new_settings: dict) -> None:
        """
        Update settings with new values from the web UI.

        IMPORTANT: If the API key field contains the masked value (with ***),
        we skip updating it — the user didn't change it, they just saw
        the masked version.

        Args:
            new_settings: Dict of setting keys to their new values.
        """
        for key, value in new_settings.items():
            # Don't overwrite the real key with a masked placeholder
            if key == "gemini_api_key" and "***" in str(value):
                continue
            self._settings[key] = value

        self.save()

    def reload(self) -> None:
        """Re-read settings from disk (useful after external changes)."""
        self._settings = self._load()


# ---------------------------------------------------------------------------
# Create a singleton instance that the rest of the app imports.
# This avoids re-reading the settings file on every import.
# ---------------------------------------------------------------------------
settings = SettingsManager()
