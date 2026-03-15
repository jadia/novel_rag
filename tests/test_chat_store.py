"""
=============================================================================
Tests for Chat Store — SQLite Session Persistence
=============================================================================
These tests verify that the ChatStore correctly manages sessions and messages,
including creation, retrieval, deletion, and auto-titling.

We use the chat_store_db fixture (from conftest.py) which creates a
temporary SQLite database for each test — no leftover data between tests.
=============================================================================
"""

import pytest


class TestSessionCRUD:
    """Test Create-Read-Update-Delete operations for sessions."""

    def test_create_session(self, chat_store_db):
        """Creating a session should return a valid UUID string."""
        session_id = chat_store_db.create_session("test-novel")
        assert session_id is not None
        assert len(session_id) == 36  # UUID format: 8-4-4-4-12

    def test_create_session_default_title(self, chat_store_db):
        """New sessions should have the default title 'New Chat'."""
        session_id = chat_store_db.create_session("test-novel")
        session = chat_store_db.get_session(session_id)
        assert session["title"] == "New Chat"

    def test_list_sessions_filters_by_novel(self, chat_store_db):
        """
        list_sessions should only return sessions for the specified novel,
        not sessions from other novels.
        """
        chat_store_db.create_session("novel-a")
        chat_store_db.create_session("novel-a")
        chat_store_db.create_session("novel-b")

        sessions_a = chat_store_db.list_sessions("novel-a")
        sessions_b = chat_store_db.list_sessions("novel-b")

        assert len(sessions_a) == 2
        assert len(sessions_b) == 1

    def test_list_sessions_ordered_by_updated_at(self, chat_store_db):
        """Sessions should be returned newest-first (by updated_at)."""
        id1 = chat_store_db.create_session("test-novel")
        id2 = chat_store_db.create_session("test-novel")

        # Add a message to session 1 (updates its updated_at)
        chat_store_db.add_message(id1, "user", "Hello!")

        sessions = chat_store_db.list_sessions("test-novel")
        # Session 1 was updated more recently, so it should be first
        assert sessions[0]["id"] == id1

    def test_get_session_returns_none_for_invalid_id(self, chat_store_db):
        """Getting a non-existent session should return None."""
        result = chat_store_db.get_session("nonexistent-id")
        assert result is None

    def test_delete_session(self, chat_store_db):
        """Deleting a session should remove it from the database."""
        session_id = chat_store_db.create_session("test-novel")

        # Add a message so we can verify cascade delete
        chat_store_db.add_message(session_id, "user", "Test message")

        deleted = chat_store_db.delete_session(session_id)
        assert deleted is True

        # Verify it's gone
        session = chat_store_db.get_session(session_id)
        assert session is None

    def test_delete_nonexistent_session(self, chat_store_db):
        """Deleting a non-existent session should return False."""
        deleted = chat_store_db.delete_session("nonexistent-id")
        assert deleted is False


class TestMessages:
    """Test message operations within sessions."""

    def test_add_and_retrieve_messages(self, chat_store_db):
        """Messages should be stored and retrieved in order."""
        session_id = chat_store_db.create_session("test-novel")
        chat_store_db.add_message(session_id, "user", "What is Yutia's weapon?")
        chat_store_db.add_message(session_id, "assistant", "Yutia wields a sword.")

        messages = chat_store_db.get_session_messages(session_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is Yutia's weapon?"
        assert messages[1]["role"] == "assistant"

    def test_messages_ordered_chronologically(self, chat_store_db):
        """Messages should always be returned in insertion order (ASC by ID)."""
        session_id = chat_store_db.create_session("test-novel")
        chat_store_db.add_message(session_id, "user", "First")
        chat_store_db.add_message(session_id, "assistant", "Second")
        chat_store_db.add_message(session_id, "user", "Third")

        messages = chat_store_db.get_session_messages(session_id)
        contents = [m["content"] for m in messages]
        assert contents == ["First", "Second", "Third"]

    def test_auto_title_on_first_user_message(self, chat_store_db):
        """
        The session title should auto-update when the first user message
        is added to a 'New Chat' session.
        """
        session_id = chat_store_db.create_session("test-novel")

        # Before: default title
        session = chat_store_db.get_session(session_id)
        assert session["title"] == "New Chat"

        # Add first user message
        chat_store_db.add_message(session_id, "user", "Tell me about Yutia's backstory")

        # After: title should be the message (or first 60 chars)
        session = chat_store_db.get_session(session_id)
        assert session["title"] == "Tell me about Yutia's backstory"

    def test_auto_title_truncates_long_messages(self, chat_store_db):
        """
        If the first user message is longer than 60 chars, the title
        should be truncated with '...' appended.
        """
        session_id = chat_store_db.create_session("test-novel")
        long_msg = "A" * 100  # 100 chars
        chat_store_db.add_message(session_id, "user", long_msg)

        session = chat_store_db.get_session(session_id)
        assert len(session["title"]) == 63  # 60 chars + "..."
        assert session["title"].endswith("...")

    def test_auto_title_not_overwritten_by_second_message(self, chat_store_db):
        """
        Only the FIRST user message should set the title. Subsequent
        messages should not change it.
        """
        session_id = chat_store_db.create_session("test-novel")
        chat_store_db.add_message(session_id, "user", "First question")
        chat_store_db.add_message(session_id, "user", "Second question")

        session = chat_store_db.get_session(session_id)
        assert session["title"] == "First question"

    def test_empty_session_has_no_messages(self, chat_store_db):
        """A newly created session should have no messages."""
        session_id = chat_store_db.create_session("test-novel")
        messages = chat_store_db.get_session_messages(session_id)
        assert messages == []
