"""
Conversation Memory Models for Persistent Context Storage

These models enable:
- Multi-session conversation continuity
- Entity tracking with position support ("the second one")
- Smart context pruning with importance scoring
- Server restart survival
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class Conversation(Base):
    """
    Master conversation record per user.

    Each user has one active conversation at a time.
    Old conversations are marked inactive but preserved for history.
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    title = Column(String(500), nullable=True)  # Auto-generated from first message
    summary = Column(Text, nullable=True)  # AI-generated summary of older messages
    meta_data = Column(JSON, nullable=True)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "ConversationMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.created_at"
    )
    tracked_entities = relationship(
        "EntityTracking",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id={self.user_id}, messages={self.message_count})>"


class ConversationMessage(Base):
    """
    Individual conversation messages with importance scoring.

    Importance scoring determines what gets kept during context pruning:
    - 1.0: Critical (entity definitions, data payloads)
    - 0.7: Important (tool calls, confirmations)
    - 0.5: Normal (regular messages)
    - 0.3: Low (greetings, acknowledgments)
    """
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    importance_score = Column(Float, default=0.5, nullable=False)
    has_data_payload = Column(Boolean, default=False, nullable=False)
    has_tool_call = Column(Boolean, default=False, nullable=False)
    intent = Column(String(100), nullable=True)  # Detected intent
    entities_mentioned = Column(JSON, nullable=True)  # ["contact", "email"]
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    meta_data = Column(JSON, nullable=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    entities = relationship("EntityTracking", back_populates="message")

    def __repr__(self):
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<ConversationMessage(id={self.id}, role='{self.role}', content='{content_preview}')>"


class EntityTracking(Base):
    """
    Track ALL mentioned entities with positions for pronoun resolution.

    Enables:
    - "the second one" -> finds entity at list_position=1
    - "the one from John" -> searches entity_data for sender
    - "that contact" -> finds most recent contact with is_current_focus=True
    - Multi-day context: "the contact we discussed yesterday"
    """
    __tablename__ = "entity_tracking"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    entity_type = Column(String(50), nullable=False)  # 'emails', 'contacts', 'projects', 'tasks', 'organizations'
    entity_id = Column(String(255), nullable=True)  # Internal ID (e.g., contact.id from our DB)
    external_id = Column(String(500), nullable=True)  # External ID (e.g., email ID from Microsoft Graph)
    entity_name = Column(String(500), nullable=True)  # Name/subject for quick lookup
    entity_data = Column(JSON, nullable=True)  # Full entity data (from, email, phone, etc.)
    list_position = Column(Integer, nullable=True)  # Position in list (0-indexed)
    is_current_focus = Column(Boolean, default=False, nullable=False)
    mention_count = Column(Integer, default=1, nullable=False)
    first_mentioned_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_mentioned_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    message_id = Column(Integer, ForeignKey("conversation_messages.id", ondelete="SET NULL"), nullable=True)
    meta_data = Column(JSON, nullable=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="tracked_entities")
    message = relationship("ConversationMessage", back_populates="entities")

    def __repr__(self):
        return f"<EntityTracking(id={self.id}, type='{self.entity_type}', name='{self.entity_name}', pos={self.list_position})>"


# Constants for importance scoring
class ImportanceScores:
    """Standard importance scores for message prioritization during context pruning"""
    CRITICAL = 1.0      # Entity definitions, data payloads, confirmations
    HIGH = 0.8          # Tool calls, important updates
    NORMAL = 0.5        # Regular conversation
    LOW = 0.3           # Greetings, acknowledgments, small talk
    MINIMAL = 0.1       # Can be safely pruned


# Valid entity types
ENTITY_TYPES = {
    "emails",
    "contacts",
    "projects",
    "tasks",
    "organizations"
}
