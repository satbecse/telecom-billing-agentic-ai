"""
Session Store - In-Memory Session Management.

CONCEPT: What is a Session Store?
==================================
A session store keeps track of user conversations and context.
It's like a notepad where we write down important information
so we don't have to ask the user to repeat themselves.

This implementation uses a simple Python dictionary.
In production, you might use Redis, a database, or LangGraph's
built-in checkpointing.

Usage:
    store = SessionStore()
    
    # Create or get a session
    session = store.get_or_create("user123")
    
    # Update session with new info
    store.update("user123", account_id="ACC-DEMO-001")
    
    # Get session data
    data = store.get("user123")
    print(data.account_id)  # "ACC-DEMO-001"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid


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
    In-Memory Session Store.
    
    IMPORTANT: This is an in-memory store!
    - Data is lost when the application restarts
    - Good for demos and development
    - For production, use Redis or a database
    
    Usage:
        store = SessionStore()
        session = store.get_or_create("user123")
        store.update("user123", account_id="ACC-DEMO-001")
    """
    
    def __init__(self):
        # The actual storage - a dict mapping session_id -> SessionData
        self._sessions: Dict[str, SessionData] = {}
        
        # For debugging
        self._access_count = 0
    
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
        
        session = SessionData(session_id=session_id)
        self._sessions[session_id] = session
        
        return session
    
    def get(self, session_id: str) -> Optional[SessionData]:
        """
        Get a session by ID.
        
        Args:
            session_id: The session ID to look up
        
        Returns:
            SessionData if found, None otherwise
        """
        self._access_count += 1
        return self._sessions.get(session_id)
    
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
        session = self.get(session_id)
        if session is None:
            return None
        
        # Update only the fields that are provided
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        session.last_activity = datetime.now()
        return session
    
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
        session = self.get(session_id)
        if session is None:
            return None
        
        session.add_turn(role, content)
        return session
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The session to delete
        
        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[str]:
        """Get list of all session IDs."""
        return list(self._sessions.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics (for debugging)."""
        return {
            "total_sessions": len(self._sessions),
            "access_count": self._access_count,
            "sessions": [s.to_dict() for s in self._sessions.values()]
        }


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
