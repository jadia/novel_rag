"""
=============================================================================
Chat Store — SQLite-Backed Session Persistence
=============================================================================
This module stores chat sessions and their messages in a local SQLite database.

WHY SQLITE?
SQLite is perfect for this use case because:
  1. Zero setup — it's built into Python (no external database server needed)
  2. Single file — the entire chat history is one .sqlite3 file in db/
  3. ACID compliant — even if the app crashes, your data is safe
  4. Perfect scale — handles thousands of sessions without breaking a sweat

DESIGN PATTERN: Repository Pattern
This module acts as a "Repository" — it hides all SQL details behind simple
Python methods. The API routes never write SQL directly; they call methods
like create_session() and add_message(). This means if you ever wanted to
switch from SQLite to PostgreSQL, you'd only change THIS file.

DATABASE SCHEMA:
  sessions:
    - id (UUID primary key)
    - novel_name (which novel this chat is about)
    - title (auto-generated from the first user message)
    - created_at (ISO-8601 timestamp)
    - updated_at (ISO-8601 timestamp, auto-updates)

  messages:
    - id (auto-incrementing primary key)
    - session_id (foreign key → sessions.id)
    - role ('user' or 'assistant')
    - content (the message text)
    - timestamp (ISO-8601 timestamp)
=============================================================================
"""

import os
import uuid
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Optional
import src.core.config as config
from src.core.utils import get_logger

logger = get_logger("ChatStore")


class ChatStore:
    """
    Provides CRUD operations for chat sessions and messages.

    THREAD SAFETY NOTE:
    SQLite supports concurrent reads but only one writer at a time.
    For a single-user application like this, that's perfectly fine.
    For multi-user scenarios, you'd want to add connection pooling
    or switch to PostgreSQL.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the chat store and create tables if they don't exist.

        Args:
            db_path: Path to the SQLite database file. Defaults to
                    db/chat_history.sqlite3 in the project root.
        """
        if db_path is None:
            # Store alongside ChromaDB in the db/ directory
            os.makedirs(config.DB_DIR, exist_ok=True)
            db_path = os.path.join(config.DB_DIR, "chat_history.sqlite3")

        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """
        Create a new database connection.

        WHY create new connections instead of reusing one?
        SQLite3 connections in Python are not thread-safe by default.
        Creating a fresh connection for each operation is the simplest
        way to avoid issues in an async web server like FastAPI/Uvicorn.

        row_factory = sqlite3.Row makes rows behave like dicts, so you
        can access columns by name (row['id']) instead of index (row[0]).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable foreign key enforcement (off by default in SQLite!)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        """
        Create the sessions and messages tables if they don't exist.

        IF NOT EXISTS is crucial — it makes this method idempotent,
        meaning you can call it 100 times and it only creates the
        tables on the first call.
        """
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    novel_name TEXT NOT NULL,
                    title TEXT DEFAULT 'New Chat',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                -- Index for fast lookups: "give me all sessions for novel X"
                CREATE INDEX IF NOT EXISTS idx_sessions_novel
                    ON sessions(novel_name);

                -- Index for fast lookups: "give me all messages for session X"
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id);
            """)
            conn.commit()
        finally:
            conn.close()

    def create_session(self, novel_name: str) -> str:
        """
        Create a new chat session for a specific novel.

        We use UUID4 for session IDs because:
          - They're globally unique (no collisions even across devices)
          - They don't reveal information (vs. auto-incrementing integers)
          - They're URL-safe when used as path parameters

        Args:
            novel_name: The novel this chat session is about.

        Returns:
            The newly created session's UUID string.
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO sessions (id, novel_name, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, novel_name, "New Chat", now, now)
            )
            conn.commit()
            logger.info(f"Created session {session_id} for novel '{novel_name}'")
        finally:
            conn.close()

        return session_id

    def list_sessions(self, novel_name: str) -> List[Dict]:
        """
        List all chat sessions for a specific novel, newest first.

        Args:
            novel_name: Filter sessions to this novel only.

        Returns:
            List of session dicts with id, novel_name, title, timestamps.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT id, novel_name, title, created_at, updated_at "
                "FROM sessions WHERE novel_name = ? "
                "ORDER BY updated_at DESC",
                (novel_name,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get a single session's metadata by its ID.

        Args:
            session_id: The UUID of the session.

        Returns:
            Session dict or None if not found.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT id, novel_name, title, created_at, updated_at "
                "FROM sessions WHERE id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_session_messages(self, session_id: str) -> List[Dict]:
        """
        Get all messages for a session, ordered chronologically.

        Args:
            session_id: The UUID of the session.

        Returns:
            List of message dicts with role, content, timestamp.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT id, role, content, timestamp "
                "FROM messages WHERE session_id = ? "
                "ORDER BY id ASC",
                (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to an existing session and auto-update the title.

        AUTO-TITLING: When the first user message is added to a "New Chat"
        session, we automatically set the title to the first 60 characters
        of that message. This provides a meaningful preview in the session
        sidebar without requiring the user to manually name their chats.

        Args:
            session_id: The UUID of the session.
            role: Either 'user' or 'assistant'.
            content: The message text.
        """
        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        try:
            # Insert the message
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, now)
            )

            # Update the session's updated_at timestamp
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id)
            )

            # Auto-title: If this is the first user message and the title
            # is still the default "New Chat", use the message as the title.
            if role == "user":
                cursor = conn.execute(
                    "SELECT title FROM sessions WHERE id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                if row and row["title"] == "New Chat":
                    # Truncate to 60 chars for a clean sidebar preview
                    title = content[:60] + ("..." if len(content) > 60 else "")
                    conn.execute(
                        "UPDATE sessions SET title = ? WHERE id = ?",
                        (title, session_id)
                    )

            conn.commit()
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its messages (CASCADE handles messages).

        Args:
            session_id: The UUID of the session to delete.

        Returns:
            True if a session was deleted, False if it didn't exist.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted session {session_id}")
            return deleted
        finally:
            conn.close()
