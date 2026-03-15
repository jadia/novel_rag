"""
=============================================================================
Tests for API Routes — REST & WebSocket Endpoints
=============================================================================
These tests verify our FastAPI endpoints work correctly.

TESTING STRATEGY:
  - REST endpoints: Use FastAPI's TestClient (synchronous HTTP)
  - WebSocket endpoints: Use TestClient's WebSocket context manager
  - We mock heavy dependencies (VectorDB, Gemini) to keep tests fast
    and avoid requiring real API keys or model files

WHY MOCK?
The API tests should verify that the ROUTES work correctly (correct
HTTP methods, status codes, JSON shapes). They should NOT test whether
ChromaDB or Gemini work — those are tested separately or manually.
=============================================================================
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestNovelsEndpoints:
    """Tests for the /api/novels endpoints."""

    def test_list_novels_empty(self, test_client):
        """When no novels exist, should return an empty list."""
        with patch('src.api.routes.config.get_available_novels', return_value=[]):
            response = test_client.get("/api/novels")
        assert response.status_code == 200
        assert response.json()["novels"] == []

    def test_list_novels_with_data(self, test_client):
        """Should return novels with their embedding status."""
        mock_novels = ["novel-a", "novel-b"]
        mock_db = MagicMock()
        mock_db.has_documents.return_value = True

        with patch('src.api.routes.config.get_available_novels', return_value=mock_novels):
            with patch('src.api.routes.VectorDB', return_value=mock_db):
                response = test_client.get("/api/novels")

        assert response.status_code == 200
        data = response.json()
        assert len(data["novels"]) == 2
        assert data["novels"][0]["name"] == "novel-a"
        assert data["novels"][0]["has_embeddings"] is True

    def test_novel_status_not_found(self, test_client):
        """Requesting status for a non-existent novel should return 404."""
        with patch('src.api.routes.config.get_available_novels', return_value=[]):
            response = test_client.get("/api/novels/nonexistent/status")
        assert response.status_code == 404


class TestSessionEndpoints:
    """Tests for the /api/sessions endpoints."""

    def test_create_session(self, test_client):
        """Creating a session should return a session_id."""
        with patch('src.api.routes.chat_store') as mock_store:
            mock_store.create_session.return_value = "test-session-id-123"
            response = test_client.post(
                "/api/sessions",
                json={"novel_name": "test-novel"}
            )

        assert response.status_code == 200
        assert response.json()["session_id"] == "test-session-id-123"

    def test_list_sessions(self, test_client):
        """Should return sessions for the specified novel."""
        mock_sessions = [
            {"id": "s1", "title": "Chat 1", "novel_name": "test",
             "created_at": "2026-01-01", "updated_at": "2026-01-01"},
        ]
        with patch('src.api.routes.chat_store') as mock_store:
            mock_store.list_sessions.return_value = mock_sessions
            response = test_client.get("/api/sessions/test-novel")

        assert response.status_code == 200
        assert len(response.json()["sessions"]) == 1

    def test_get_session_messages(self, test_client):
        """Should return session metadata and messages."""
        mock_session = {"id": "s1", "title": "Test", "novel_name": "test",
                       "created_at": "2026-01-01", "updated_at": "2026-01-01"}
        mock_messages = [
            {"id": 1, "role": "user", "content": "Hello", "timestamp": "2026-01-01"},
        ]
        with patch('src.api.routes.chat_store') as mock_store:
            mock_store.get_session.return_value = mock_session
            mock_store.get_session_messages.return_value = mock_messages
            response = test_client.get("/api/sessions/s1/messages")

        assert response.status_code == 200
        assert response.json()["session"]["id"] == "s1"
        assert len(response.json()["messages"]) == 1

    def test_get_session_messages_not_found(self, test_client):
        """Getting messages for a non-existent session should return 404."""
        with patch('src.api.routes.chat_store') as mock_store:
            mock_store.get_session.return_value = None
            response = test_client.get("/api/sessions/nonexistent/messages")

        assert response.status_code == 404

    def test_delete_session(self, test_client):
        """Deleting a session should return success."""
        with patch('src.api.routes.chat_store') as mock_store:
            mock_store.delete_session.return_value = True
            response = test_client.delete("/api/sessions/s1")

        assert response.status_code == 200
        assert response.json()["deleted"] is True

    def test_delete_session_not_found(self, test_client):
        """Deleting a non-existent session should return 404."""
        with patch('src.api.routes.chat_store') as mock_store:
            mock_store.delete_session.return_value = False
            response = test_client.delete("/api/sessions/nonexistent")

        assert response.status_code == 404


class TestSettingsEndpoints:
    """Tests for the /api/settings endpoints."""

    def test_get_settings(self, test_client):
        """Should return current settings with masked API key."""
        mock_settings = {
            "gemini_api_key": "sk-1***5678",
            "generation_model": "models/gemini-2.0-flash",
            "use_local_embeddings": False,
        }
        with patch('src.api.routes.config.settings') as mock:
            mock.get_all.return_value = mock_settings
            response = test_client.get("/api/settings")

        assert response.status_code == 200
        settings = response.json()["settings"]
        assert "***" in settings["gemini_api_key"]

    def test_update_settings(self, test_client):
        """Updating settings should save and return the new values."""
        updated = {
            "gemini_api_key": "sk-1***5678",
            "generation_model": "models/new-model",
        }
        with patch('src.api.routes.config.settings') as mock:
            mock.get_all.return_value = updated
            response = test_client.put(
                "/api/settings",
                json={"generation_model": "models/new-model"}
            )

        assert response.status_code == 200
        assert response.json()["saved"] is True


class TestHealthCheck:
    """Tests for the /api/health endpoint."""

    def test_health_check(self, test_client):
        """Health check should always return 200 with status 'healthy'."""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestWebSocketChat:
    """Tests for the /ws/chat WebSocket endpoint."""

    def test_chat_websocket_error_on_missing_fields(self, test_client):
        """
        Sending a message without required fields should return an error
        message, not crash the connection.
        """
        with test_client.websocket_connect("/ws/chat") as ws:
            # Send a message missing the 'question' field
            ws.send_json({"novel_name": "test", "session_id": "s1"})
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "Missing" in response["content"]
