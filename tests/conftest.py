"""
=============================================================================
Test Configuration — Shared Fixtures
=============================================================================
pytest fixtures are reusable setup/teardown functions that tests can request.
By defining them in conftest.py, ALL test files automatically have access
to them without importing anything.

WHY FIXTURES?
Instead of each test creating its own temp directory, database, and client,
fixtures centralize this setup. Benefits:
  - DRY: Setup code is defined once
  - Isolation: Each test gets a fresh environment (no cross-contamination)
  - Teardown: Cleanup happens automatically even if a test fails
=============================================================================
"""

import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, patch

# Ensure project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory that is automatically cleaned up after the test.

    WHAT THIS DOES:
    tempfile.mkdtemp() creates a real directory on disk with a random name
    like /tmp/pytest-abcdef123/. After the test completes (yield), we
    clean it up.

    Yields:
        Path to the temporary directory.
    """
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Cleanup: remove all files and the directory
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_markdown_dir(temp_dir):
    """
    Create a temp directory with sample markdown files for testing the
    document processor.

    This fixture depends on temp_dir (fixture chaining). pytest automatically
    creates the temp_dir first, then passes its path here.

    Yields:
        Path to the directory containing sample .md files.
    """
    # Create sample chapter files
    chapters = {
        "chapter-1.md": (
            "Chapter 1: The Beginning\n\n"
            "Yutia stood at the edge of the cliff, her sword gleaming in the moonlight. "
            "The demon king's army stretched across the valley below, their torches "
            "flickering like fallen stars. She knew this would be her last battle. "
            "Behind her, Kaelen placed a reassuring hand on her shoulder. "
            "'We fight together,' he said, his voice steady despite the fear in his eyes."
        ),
        "chapter-2.md": (
            "Chapter 2: The Battle\n\n"
            "Steel clashed against steel as Yutia charged into the demon horde. "
            "Kaelen fought beside her, his magic barriers deflecting bolts of dark energy. "
            "The demon general, Vorath, watched from his obsidian throne, amused by their defiance."
        ),
    }

    for filename, content in chapters.items():
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)

    yield temp_dir


@pytest.fixture
def chat_store_db(temp_dir):
    """
    Create a ChatStore instance backed by a temporary SQLite database.

    This ensures each test gets a clean database with no leftover data
    from previous tests.

    Yields:
        A ChatStore instance connected to a temp database.
    """
    from src.api.chat_store import ChatStore
    db_path = os.path.join(temp_dir, "test_chat.sqlite3")
    store = ChatStore(db_path=db_path)
    yield store


@pytest.fixture
def mock_settings(temp_dir):
    """
    Provide mock settings that don't require real API keys or model files.

    This patches the SettingsManager to return test values, preventing
    tests from needing a real Gemini API key or HuggingFace model.
    """
    mock = {
        "gemini_api_key": "test-api-key-12345678",
        "use_local_embeddings": False,
        "local_model_name": "all-MiniLM-L6-v2",
        "api_model_name": "models/gemini-embedding-001",
        "generation_model": "models/gemini-2.0-flash",
        "chunk_size": 600,
        "chunk_overlap": 150,
    }
    return mock


@pytest.fixture
def test_client():
    """
    Create a FastAPI TestClient for testing HTTP endpoints.

    The TestClient lets us send HTTP requests to our API without
    actually starting a server. It's like having a fake browser
    that talks directly to FastAPI.

    Yields:
        A TestClient instance configured for our app.
    """
    from fastapi.testclient import TestClient
    from src.main import app
    client = TestClient(app)
    yield client
