"""
Session Store - SQLite-Backed Session Management.

CONCEPT: What is a Session Store?
==================================
A session store keeps track of user conversations and context.
It's like a notepad where we write down important information
so we don't have to ask the user to repeat themselves.

This implementation uses SQLite for persistent storage.
Sessions survive application restarts - great for demos!

The database file is stored at: data/sessions.db

Usage:
    store = get_session_store()
    
    # Create or get a session
    session = store.get_or_create("user123")
    
    # Update session with new info
    store.update("user123", account_id="ACC-DEMO-001")
    
    # Get session data
    data = store.get("user123")
    print(data.account_id)  # "ACC-DEMO-001"
"""

import sqlite3
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import uuid

from app.config import DATA_DIR
from app.utils.logging import get_logger

logger = get_logger("session_store")

# Database path
DB_PATH = DATA_DIR / "sessions.db"


@dataclass
class ConversationTurn:
    """A single turn in the conversation (user message + AI response)."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SessionData:
    """
    Data stored for a single user session.
    
    CONCEPT: Entity Memory
    ======================
    Instead of just storing raw messages, we extract and store
    specific ENTITIES (structured data) like:
    - account_id: The customer's account number
    - customer_name: The customer's name
    - billing_period: What time period they're asking about
    
    This makes it easier for agents to use the information.
    """
    
    # Session identification
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Extracted entities (the "memory")
    account_id: Optional[str] = None
    customer_name: Optional[str] = None
    billing_period: Optional[str] = None
    current_topic: Optional[str] = None  # e.g., "billing", "plans", "dispute"
    
    # Last context from BillingAgent (for follow-ups)
    last_query: Optional[str] = None
    last_response: Optional[str] = None
    last_doc_ids: List[str] = field(default_factory=list)
    
    # Conversation history (last N turns)
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    
    # Maximum conversation turns to keep
    max_history: int = 10
    
    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn, keeping only the last N turns."""
        self.conversation_history.append(
            ConversationTurn(role=role, content=content)
        )
        # Trim to max history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        self.last_activity = datetime.now()
    
    def get_context_summary(self) -> str:
        """
        Get a summary of the session context for the AI.
        
        This is injected into prompts so agents know the context.
        """
        parts = []
        
        if self.account_id:
            parts.append(f"Account: {self.account_id}")
        if self.customer_name:
            parts.append(f"Customer: {self.customer_name}")
        if self.billing_period:
            parts.append(f"Discussing: {self.billing_period}")
        if self.current_topic:
            parts.append(f"Topic: {self.current_topic}")
        
        if parts:
            return "Session Context: " + " | ".join(parts)
        return ""
    
    def get_conversation_for_prompt(self, last_n: int = 3) -> str:
        """
        Format recent conversation history for including in prompts.
        
        This helps the AI understand the flow of conversation.
        """
        if not self.conversation_history:
            return ""
        
        recent = self.conversation_history[-last_n:]
        lines = ["Recent conversation:"]
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            # Truncate long messages
            content = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
            lines.append(f"  {prefix}: {content}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary (for debugging/logging)."""
        return {
            "session_id": self.session_id,
            "account_id": self.account_id,
            "customer_name": self.customer_name,
            "billing_period": self.billing_period,
            "current_topic": self.current_topic,
            "turns": len(self.conversation_history),
            "last_activity": self.last_activity.isoformat(),
        }


class SessionStore:
    """
    SQLite-Backed Session Store.
    
    Provides persistent session storage that survives application restarts.
    The database file is stored at data/sessions.db.
    
    Schema:
        sessions: Stores session metadata and extracted entities
        conversation_turns: Stores individual conversation messages
    
    Usage:
        store = SessionStore()
        session = store.get_or_create("user123")
        store.update("user123", account_id="ACC-DEMO-001")
    """
    
    def __init__(self, db_path: Path = None):
        self._db_path = str(db_path or DB_PATH)
        self._access_count = 0
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        return conn
    
    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id     TEXT PRIMARY KEY,
                    created_at     TEXT NOT NULL,
                    last_activity  TEXT NOT NULL,
                    account_id     TEXT,
                    customer_name  TEXT,
                    billing_period TEXT,
                    current_topic  TEXT,
                    last_query     TEXT,
                    last_response  TEXT,
                    last_doc_ids   TEXT DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    timestamp  TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session 
                    ON conversation_turns(session_id);
            """)
            conn.commit()
            logger.info(f"SQLite session store initialized at {self._db_path}")
        finally:
            conn.close()
    
    def _row_to_session(self, row: sqlite3.Row, conn: sqlite3.Connection) -> SessionData:
        """Convert a database row + its turns into a SessionData object."""
        session = SessionData(
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_activity=datetime.fromisoformat(row["last_activity"]),
            account_id=row["account_id"],
            customer_name=row["customer_name"],
            billing_period=row["billing_period"],
            current_topic=row["current_topic"],
            last_query=row["last_query"],
            last_response=row["last_response"],
            last_doc_ids=json.loads(row["last_doc_ids"] or "[]"),
        )
        
        # Load conversation history (last 10 turns)
        turns = conn.execute(
            "SELECT role, content, timestamp FROM conversation_turns "
            "WHERE session_id = ? ORDER BY id DESC LIMIT 10",
            (row["session_id"],)
        ).fetchall()
        
        session.conversation_history = [
            ConversationTurn(
                role=t["role"],
                content=t["content"],
                timestamp=datetime.fromisoformat(t["timestamp"])
            )
            for t in reversed(turns)  # Reverse so oldest is first
        ]
        
        return session
    
    def create(self, session_id: Optional[str] = None) -> SessionData:
        """
        Create a new session.
        
        Args:
            session_id: Optional custom ID. If not provided, generates UUID.
        
        Returns:
            New SessionData instance
        """
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        now = datetime.now().isoformat()
        
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO sessions (session_id, created_at, last_activity) VALUES (?, ?, ?)",
                (session_id, now, now)
            )
            conn.commit()
        finally:
            conn.close()
        
        logger.info(f"Created new session: {session_id}")
        return SessionData(session_id=session_id)
    
    def get(self, session_id: str) -> Optional[SessionData]:
        """
        Get a session by ID.
        
        Args:
            session_id: The session ID to look up
        
        Returns:
            SessionData if found, None otherwise
        """
        self._access_count += 1
        
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            return self._row_to_session(row, conn)
        finally:
            conn.close()
    
    def get_or_create(self, session_id: str) -> SessionData:
        """
        Get an existing session or create a new one.
        
        This is the most common operation - you usually want to
        reuse an existing session if one exists.
        
        Args:
            session_id: The session ID
        
        Returns:
            SessionData (existing or newly created)
        """
        session = self.get(session_id)
        if session is None:
            session = self.create(session_id)
        return session
    
    def update(self, session_id: str, **kwargs) -> Optional[SessionData]:
        """
        Update session with new entity values.
        
        Args:
            session_id: The session to update
            **kwargs: Entity values to update (e.g., account_id="ACC-...")
        
        Returns:
            Updated SessionData, or None if session not found
        
        Example:
            store.update("user123", 
                account_id="ACC-DEMO-001",
                customer_name="Dileep"
            )
        """
        # Map of allowed fields to their DB column names
        allowed_fields = {
            "account_id", "customer_name", "billing_period",
            "current_topic", "last_query", "last_response", "last_doc_ids"
        }
        
        updates = {}
        for key, value in kwargs.items():
            if key in allowed_fields:
                if key == "last_doc_ids" and isinstance(value, list):
                    updates[key] = json.dumps(value)
                else:
                    updates[key] = value
        
        if not updates:
            return self.get(session_id)
        
        updates["last_activity"] = datetime.now().isoformat()
        
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [session_id]
        
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                f"UPDATE sessions SET {set_clause} WHERE session_id = ?",
                values
            )
            conn.commit()
            
            if cursor.rowcount == 0:
                return None
        finally:
            conn.close()
        
        return self.get(session_id)
    
    def add_conversation_turn(
        self, 
        session_id: str, 
        role: str, 
        content: str
    ) -> Optional[SessionData]:
        """
        Add a conversation turn to the session.
        
        Args:
            session_id: The session ID
            role: "user" or "assistant"
            content: The message content
        
        Returns:
            Updated SessionData
        """
        now = datetime.now().isoformat()
        
        conn = self._get_conn()
        try:
            # Verify session exists
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            # Insert the turn
            conn.execute(
                "INSERT INTO conversation_turns (session_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, now)
            )
            
            # Update last_activity
            conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (now, session_id)
            )
            
            # Trim to keep only last 10 turns per session
            conn.execute(
                "DELETE FROM conversation_turns WHERE id NOT IN ("
                "  SELECT id FROM conversation_turns WHERE session_id = ? "
                "  ORDER BY id DESC LIMIT 10"
                ") AND session_id = ?",
                (session_id, session_id)
            )
            
            conn.commit()
        finally:
            conn.close()
        
        return self.get(session_id)
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a session and its conversation history.
        
        Args:
            session_id: The session to delete
        
        Returns:
            True if deleted, False if not found
        """
        conn = self._get_conn()
        try:
            conn.execute(
                "DELETE FROM conversation_turns WHERE session_id = ?",
                (session_id,)
            )
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def list_sessions(self) -> List[str]:
        """Get list of all session IDs."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT session_id FROM sessions ORDER BY last_activity DESC"
            ).fetchall()
            return [r["session_id"] for r in rows]
        finally:
            conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics (for debugging)."""
        conn = self._get_conn()
        try:
            session_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM sessions"
            ).fetchone()["cnt"]
            
            turn_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM conversation_turns"
            ).fetchone()["cnt"]
            
            sessions = []
            for row in conn.execute("SELECT * FROM sessions ORDER BY last_activity DESC").fetchall():
                sessions.append(self._row_to_session(row, conn).to_dict())
            
            return {
                "total_sessions": session_count,
                "total_turns": turn_count,
                "access_count": self._access_count,
                "db_path": self._db_path,
                "sessions": sessions
            }
        finally:
            conn.close()


# =============================================================================
# GLOBAL SESSION STORE INSTANCE
# =============================================================================
# We use a global instance so all parts of the app share the same store.
# In a web app, you'd typically use dependency injection instead.

_global_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """
    Get the global session store instance.
    
    This is a singleton pattern - ensures we have only one store.
    """
    global _global_store
    if _global_store is None:
        _global_store = SessionStore()
    return _global_store


def reset_session_store() -> None:
    """Reset the global store (useful for testing)."""
    global _global_store
    _global_store = None
