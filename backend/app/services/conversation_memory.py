"""
Conversation Memory Service for CRM AI Assistant

Provides BOTH:
- In-memory fast access (for within-request operations)
- Database persistence (for cross-session continuity)

Key Features:
- Entity tracking with position support ("the second one")
- Accurate token counting with tiktoken
- Smart pruning with importance scoring
- Multi-day conversation continuity
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque
import structlog
import re
import json

# Database imports
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, desc
from sqlalchemy.orm import selectinload

# Models
from ..models.conversation import (
    Conversation, ConversationMessage, EntityTracking,
    ImportanceScores, ENTITY_TYPES
)

logger = structlog.get_logger()

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    # Claude uses cl100k_base encoding (similar to GPT-4)
    _encoding = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    _encoding = None
    logger.warning("tiktoken not available, using approximate token counting")


class TokenCounter:
    """Accurate token counting for Claude/OpenAI models"""

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens in text using tiktoken or approximation"""
        if not text:
            return 0
        if TIKTOKEN_AVAILABLE and _encoding:
            try:
                return len(_encoding.encode(text))
            except Exception:
                pass
        # Fallback: rough approximation (1 token ≈ 4 characters)
        return len(text) // 4

    @staticmethod
    def count_messages(messages: List[Dict[str, str]]) -> int:
        """Count total tokens in message list including overhead"""
        total = 0
        for msg in messages:
            # Add overhead for message formatting (~4 tokens per message)
            total += TokenCounter.count_tokens(msg.get("content", "")) + 4
        return total

    @staticmethod
    def estimate_budget(model: str = "claude") -> int:
        """Get context budget for different models"""
        # Conservative estimates to leave room for response
        budgets = {
            "claude": 180000,       # Claude 3.5 has 200K context
            "claude-haiku": 180000,
            "gpt-4": 100000,        # GPT-4 Turbo has 128K
            "gpt-3.5": 12000,       # GPT-3.5 has 16K
        }
        return budgets.get(model, 8000)


class SmartContextPruner:
    """
    Intelligent context pruning with multiple strategies.

    Strategies:
    1. Recency - Most recent messages are most relevant
    2. Importance - Data payloads and tool results are critical
    3. Relevance - Messages related to current topic
    4. Summarization - Old messages can be summarized
    """

    # Minimum tokens to always preserve
    MIN_RECENT_TOKENS = 2000

    # Priority tiers for messages
    PRIORITY_CRITICAL = 4  # Must keep (recent data payloads, current context)
    PRIORITY_HIGH = 3      # Should keep (tool results, entity definitions)
    PRIORITY_NORMAL = 2    # Can keep if space (regular conversation)
    PRIORITY_LOW = 1       # Can drop (greetings, confirmations)

    @classmethod
    def prune_messages(
        cls,
        messages: List[Dict[str, Any]],
        max_tokens: int = 8000,
        preserve_recent: int = 10
    ) -> Tuple[List[Dict[str, Any]], int, bool]:
        """
        Prune message list to fit within token budget.

        Args:
            messages: List of messages (can include 'importance' key)
            max_tokens: Maximum token budget
            preserve_recent: Number of recent messages to always keep

        Returns:
            Tuple of (pruned_messages, total_tokens, was_pruned)
        """
        if not messages:
            return [], 0, False

        # Count tokens for all messages
        messages_with_tokens = []
        for msg in messages:
            content = msg.get("content", "")
            tokens = TokenCounter.count_tokens(content)
            priority = cls._calculate_priority(msg)
            messages_with_tokens.append({
                **msg,
                "_tokens": tokens,
                "_priority": priority
            })

        total_tokens = sum(m["_tokens"] for m in messages_with_tokens)

        # If within budget, return as-is
        if total_tokens <= max_tokens:
            # Remove internal fields before returning
            return [
                {k: v for k, v in m.items() if not k.startswith("_")}
                for m in messages_with_tokens
            ], total_tokens, False

        # Need to prune - apply smart strategy
        pruned = cls._smart_prune(messages_with_tokens, max_tokens, preserve_recent)

        final_tokens = sum(m["_tokens"] for m in pruned)

        # Clean up internal fields
        result = [
            {k: v for k, v in m.items() if not k.startswith("_")}
            for m in pruned
        ]

        logger.info(
            "Pruned context",
            original_messages=len(messages),
            pruned_messages=len(result),
            original_tokens=total_tokens,
            final_tokens=final_tokens
        )

        return result, final_tokens, True

    @classmethod
    def _calculate_priority(cls, msg: Dict[str, Any]) -> int:
        """Calculate priority score for a message"""
        content = msg.get("content", "").lower()
        importance = msg.get("importance", msg.get("importance_score", 0.5))

        # Critical: data payloads
        if msg.get("has_data_payload") or importance >= 0.9:
            return cls.PRIORITY_CRITICAL

        # Critical indicators in content
        critical_patterns = [
            "email data start", "query result", "exact count",
            "confirmed:", "successfully created", "successfully updated"
        ]
        if any(p in content for p in critical_patterns):
            return cls.PRIORITY_CRITICAL

        # High: tool calls or high importance
        if msg.get("has_tool_call") or importance >= 0.7:
            return cls.PRIORITY_HIGH

        # Low: greetings and simple responses
        low_patterns = [
            "hello", "hi", "thanks", "thank you", "ok", "okay",
            "sure", "got it", "understood", "great", "perfect"
        ]
        content_words = content.split()[:3]  # Check first 3 words
        if any(p in " ".join(content_words) for p in low_patterns) and len(content) < 50:
            return cls.PRIORITY_LOW

        return cls.PRIORITY_NORMAL

    @classmethod
    def _smart_prune(
        cls,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        preserve_recent: int
    ) -> List[Dict[str, Any]]:
        """
        Apply smart pruning strategy.

        Strategy:
        1. Always keep last N messages
        2. Keep all CRITICAL priority messages
        3. Fill remaining budget with HIGH priority, then NORMAL
        4. Drop LOW priority first
        """
        if len(messages) <= preserve_recent:
            return messages

        # Split into recent and older
        recent = messages[-preserve_recent:]
        older = messages[:-preserve_recent]

        recent_tokens = sum(m["_tokens"] for m in recent)
        remaining_budget = max_tokens - recent_tokens

        if remaining_budget <= 0:
            # Even recent messages exceed budget - truncate from oldest
            result = []
            current_tokens = 0
            for msg in reversed(recent):
                if current_tokens + msg["_tokens"] <= max_tokens:
                    result.insert(0, msg)
                    current_tokens += msg["_tokens"]
            return result

        # Select from older messages by priority
        selected = []

        # First pass: CRITICAL messages
        for msg in older:
            if msg["_priority"] >= cls.PRIORITY_CRITICAL:
                if remaining_budget >= msg["_tokens"]:
                    selected.append(msg)
                    remaining_budget -= msg["_tokens"]

        # Second pass: HIGH priority (if space)
        for msg in older:
            if msg["_priority"] == cls.PRIORITY_HIGH and msg not in selected:
                if remaining_budget >= msg["_tokens"]:
                    selected.append(msg)
                    remaining_budget -= msg["_tokens"]

        # Third pass: NORMAL priority (if space)
        for msg in older:
            if msg["_priority"] == cls.PRIORITY_NORMAL and msg not in selected:
                if remaining_budget >= msg["_tokens"]:
                    selected.append(msg)
                    remaining_budget -= msg["_tokens"]

        # Sort selected by original order
        selected_indices = {id(m): i for i, m in enumerate(older)}
        selected.sort(key=lambda m: selected_indices.get(id(m), 0))

        return selected + recent

    @classmethod
    def create_summary_prompt(cls, messages: List[Dict[str, Any]]) -> str:
        """Create a prompt for AI to summarize older messages"""
        if len(messages) < 5:
            return ""

        # Extract key information
        entities = set()
        actions = set()

        for msg in messages:
            content = msg.get("content", "")

            # Extract entity mentions
            if "contact" in content.lower():
                entities.add("contacts")
            if "email" in content.lower():
                entities.add("emails")
            if "project" in content.lower():
                entities.add("projects")
            if "task" in content.lower():
                entities.add("tasks")

            # Extract actions
            for action in ["created", "updated", "deleted", "searched", "listed"]:
                if action in content.lower():
                    actions.add(action)

        summary_parts = []
        if entities:
            summary_parts.append(f"Discussed: {', '.join(sorted(entities))}")
        if actions:
            summary_parts.append(f"Actions: {', '.join(sorted(actions))}")

        return ". ".join(summary_parts) if summary_parts else "Previous CRM conversation."


class PersistentConversationMemory:
    """
    Database-backed conversation memory with in-memory cache.

    Usage:
        memory = await PersistentConversationMemory.get_or_create(user_id, db)
        await memory.add_message("user", "Hello")
        await memory.track_entity("contacts", {...}, position=0)
        context = await memory.get_context_window(max_tokens=8000)
    """

    def __init__(self, conversation: Conversation, db: AsyncSession):
        self.conversation = conversation
        self.conversation_id = conversation.id
        self.user_id = conversation.user_id
        self.db = db

        # In-memory cache for fast access
        self._messages_cache: List[ConversationMessage] = []
        self._entities_cache: Dict[str, List[EntityTracking]] = {
            et: [] for et in ENTITY_TYPES
        }
        self._current_focus: Dict[str, EntityTracking] = {}
        self._cache_loaded = False

        # Recent data for backward compatibility
        self.recent_data = {
            "current_focus": None,
            "current_email_id": None,
            "current_email_subject": None,
            "current_email_from": None,
            "current_email_timestamp": None,
            "current_contact_id": None,
            "current_contact_name": None,
            "current_project_id": None,
            "current_project_name": None,
            "current_task_id": None,
            "current_task_name": None,
            "current_organization_id": None,
            "current_organization_name": None,
            "last_ai_offer": None,
            "offer_timestamp": None,
        }

    @classmethod
    async def get_or_create(cls, user_id: int, db: AsyncSession) -> "PersistentConversationMemory":
        """Get active conversation for user, or create new one"""
        # Try to find active conversation
        result = await db.execute(
            select(Conversation)
            .where(and_(
                Conversation.user_id == user_id,
                Conversation.is_active == True
            ))
            .order_by(desc(Conversation.last_activity_at))
            .limit(1)
        )
        conversation = result.scalar_one_or_none()

        if not conversation:
            # Create new conversation
            conversation = Conversation(
                user_id=user_id,
                is_active=True,
                message_count=0,
                total_tokens=0
            )
            db.add(conversation)
            await db.flush()
            logger.info(f"Created new conversation for user {user_id}", conversation_id=conversation.id)
        else:
            logger.debug(f"Loaded existing conversation for user {user_id}", conversation_id=conversation.id)

        memory = cls(conversation, db)
        await memory._load_cache()
        return memory

    async def _load_cache(self):
        """Load recent messages and entities into memory cache"""
        if self._cache_loaded:
            return

        # Load recent messages (last 50)
        result = await self.db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == self.conversation_id)
            .order_by(desc(ConversationMessage.created_at))
            .limit(50)
        )
        self._messages_cache = list(reversed(result.scalars().all()))

        # Load recent entities (last 100 per type)
        result = await self.db.execute(
            select(EntityTracking)
            .where(EntityTracking.conversation_id == self.conversation_id)
            .order_by(desc(EntityTracking.last_mentioned_at))
            .limit(500)
        )
        entities = result.scalars().all()

        for entity in entities:
            if entity.entity_type in self._entities_cache:
                self._entities_cache[entity.entity_type].append(entity)
            if entity.is_current_focus:
                self._current_focus[entity.entity_type] = entity

        # Update recent_data from current focus entities
        self._sync_recent_data_from_focus()

        self._cache_loaded = True
        logger.debug(f"Loaded cache: {len(self._messages_cache)} messages, {sum(len(v) for v in self._entities_cache.values())} entities")

    def _sync_recent_data_from_focus(self):
        """Sync recent_data dict from current focus entities (backward compatibility)"""
        for entity_type, entity in self._current_focus.items():
            if entity_type == "emails":
                self.recent_data["current_email_id"] = entity.external_id or entity.entity_id
                self.recent_data["current_email_subject"] = entity.entity_name
                data = entity.entity_data or {}
                self.recent_data["current_email_from"] = data.get("from")
                self.recent_data["current_email_timestamp"] = entity.last_mentioned_at.isoformat() if entity.last_mentioned_at else None
            elif entity_type == "contacts":
                self.recent_data["current_contact_id"] = entity.entity_id
                self.recent_data["current_contact_name"] = entity.entity_name
            elif entity_type == "projects":
                self.recent_data["current_project_id"] = entity.entity_id
                self.recent_data["current_project_name"] = entity.entity_name
            elif entity_type == "tasks":
                self.recent_data["current_task_id"] = entity.entity_id
                self.recent_data["current_task_name"] = entity.entity_name
            elif entity_type == "organizations":
                self.recent_data["current_organization_id"] = entity.entity_id
                self.recent_data["current_organization_name"] = entity.entity_name

    async def add_message(
        self,
        role: str,
        content: str,
        intent: Optional[str] = None,
        has_data_payload: bool = False,
        has_tool_call: bool = False,
        entities_mentioned: Optional[List[str]] = None
    ) -> ConversationMessage:
        """Add a message to the conversation"""
        # Calculate importance score
        importance = self._calculate_importance(content, has_data_payload, has_tool_call)

        # Count tokens
        token_count = TokenCounter.count_tokens(content)

        message = ConversationMessage(
            conversation_id=self.conversation_id,
            role=role,
            content=content,
            token_count=token_count,
            importance_score=importance,
            has_data_payload=has_data_payload,
            has_tool_call=has_tool_call,
            intent=intent,
            entities_mentioned=entities_mentioned
        )
        self.db.add(message)

        # Update conversation stats
        self.conversation.message_count += 1
        self.conversation.total_tokens += token_count
        self.conversation.last_activity_at = datetime.now(timezone.utc)

        # Auto-generate title from first user message
        if not self.conversation.title and role == "user":
            self.conversation.title = content[:100] + ("..." if len(content) > 100 else "")

        await self.db.flush()

        # Update cache
        self._messages_cache.append(message)

        logger.debug(f"Added message: role={role}, tokens={token_count}, importance={importance}")
        return message

    def _calculate_importance(self, content: str, has_data_payload: bool, has_tool_call: bool) -> float:
        """Calculate importance score for a message"""
        if has_data_payload:
            return ImportanceScores.CRITICAL
        if has_tool_call:
            return ImportanceScores.HIGH

        content_lower = content.lower()

        # High importance indicators
        if any(ind in content_lower for ind in [
            "created successfully", "updated successfully", "deleted successfully",
            "confirmed", "email data start", "query result", "exact count"
        ]):
            return ImportanceScores.HIGH

        # Low importance indicators
        if any(ind in content_lower for ind in [
            "hello", "hi", "thanks", "thank you", "ok", "okay", "sure", "got it"
        ]):
            return ImportanceScores.LOW

        return ImportanceScores.NORMAL

    async def track_entity(
        self,
        entity_type: str,
        entity_data: Dict[str, Any],
        position: Optional[int] = None,
        is_primary: bool = True,
        message_id: Optional[int] = None
    ) -> EntityTracking:
        """
        Track an entity mention with optional position.

        Args:
            entity_type: 'emails', 'contacts', 'projects', 'tasks', 'organizations'
            entity_data: Full entity data dict
            position: Position in list (0-indexed) for "the second one" references
            is_primary: If True, set as current focus
            message_id: Optional message this entity was mentioned in
        """
        if entity_type not in ENTITY_TYPES:
            logger.warning(f"Unknown entity type: {entity_type}")
            entity_type = entity_type.rstrip('s') + 's'  # Normalize to plural

        entity_id = str(entity_data.get("id", "")) if entity_data.get("id") else None
        external_id = entity_data.get("external_id") or entity_data.get("email_id")
        entity_name = entity_data.get("name") or entity_data.get("subject") or entity_data.get("title")

        # Check if entity already tracked
        existing = None
        for e in self._entities_cache.get(entity_type, []):
            if (entity_id and e.entity_id == entity_id) or (external_id and e.external_id == external_id):
                existing = e
                break

        now = datetime.now(timezone.utc)

        if existing:
            # Update existing entity
            existing.mention_count += 1
            existing.last_mentioned_at = now
            existing.entity_data = entity_data
            if position is not None:
                existing.list_position = position
            if is_primary:
                # Clear old focus for this type
                await self._clear_focus(entity_type)
                existing.is_current_focus = True
                self._current_focus[entity_type] = existing
            entity = existing
        else:
            # Create new entity tracking
            if is_primary:
                await self._clear_focus(entity_type)

            entity = EntityTracking(
                conversation_id=self.conversation_id,
                entity_type=entity_type,
                entity_id=entity_id,
                external_id=external_id,
                entity_name=entity_name,
                entity_data=entity_data,
                list_position=position,
                is_current_focus=is_primary,
                message_id=message_id
            )
            self.db.add(entity)
            await self.db.flush()

            # Update cache
            if entity_type in self._entities_cache:
                self._entities_cache[entity_type].insert(0, entity)

            if is_primary:
                self._current_focus[entity_type] = entity

        # Update backward-compatible recent_data
        self._update_recent_data(entity_type, entity_data)

        logger.debug(f"Tracked entity: type={entity_type}, name={entity_name}, position={position}, is_primary={is_primary}")
        return entity

    async def track_entity_list(
        self,
        entity_type: str,
        entities: List[Dict[str, Any]],
        message_id: Optional[int] = None
    ) -> List[EntityTracking]:
        """
        Track ALL entities from a list with positions.

        This enables "the second one", "the third contact", etc.
        """
        tracked = []
        for idx, entity_data in enumerate(entities):
            entity = await self.track_entity(
                entity_type=entity_type,
                entity_data=entity_data,
                position=idx,
                is_primary=(idx == 0),  # First item is primary focus
                message_id=message_id
            )
            tracked.append(entity)

        logger.info(f"Tracked {len(tracked)} {entity_type} with positions")
        return tracked

    async def _clear_focus(self, entity_type: str):
        """Clear current focus for an entity type"""
        await self.db.execute(
            update(EntityTracking)
            .where(and_(
                EntityTracking.conversation_id == self.conversation_id,
                EntityTracking.entity_type == entity_type,
                EntityTracking.is_current_focus == True
            ))
            .values(is_current_focus=False)
        )
        if entity_type in self._current_focus:
            self._current_focus[entity_type].is_current_focus = False

    def _update_recent_data(self, entity_type: str, entity_data: Dict[str, Any]):
        """Update backward-compatible recent_data dict"""
        if entity_type == "emails":
            self.recent_data["current_email_id"] = entity_data.get("id") or entity_data.get("email_id")
            self.recent_data["current_email_subject"] = entity_data.get("subject")
            self.recent_data["current_email_from"] = entity_data.get("from")
            self.recent_data["current_email_timestamp"] = datetime.now(timezone.utc).isoformat()
        elif entity_type == "contacts":
            self.recent_data["current_contact_id"] = entity_data.get("id")
            self.recent_data["current_contact_name"] = entity_data.get("name")
        elif entity_type == "projects":
            self.recent_data["current_project_id"] = entity_data.get("id")
            self.recent_data["current_project_name"] = entity_data.get("name")
        elif entity_type == "tasks":
            self.recent_data["current_task_id"] = entity_data.get("id")
            self.recent_data["current_task_name"] = entity_data.get("name")
        elif entity_type == "organizations":
            self.recent_data["current_organization_id"] = entity_data.get("id")
            self.recent_data["current_organization_name"] = entity_data.get("name")

        # Update current_focus for backward compatibility
        self.recent_data["current_focus"] = {
            "type": entity_type,
            "data": entity_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def resolve_reference(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Resolve entity references including positions and attributes.

        Handles:
        - "it", "that", "this one" -> current focus
        - "the first one", "second", "#2" -> position-based
        - "the one from John" -> attribute-based matching
        - "his email", "her contact" -> gendered pronoun + type
        """
        text_lower = text.lower().strip()

        # Position patterns
        position_patterns = [
            (r"\b(first|1st|#1)\b", 0),
            (r"\b(second|2nd|#2)\b", 1),
            (r"\b(third|3rd|#3)\b", 2),
            (r"\b(fourth|4th|#4)\b", 3),
            (r"\b(fifth|5th|#5)\b", 4),
            (r"\b(last|final|latest)\b", -1),
        ]

        for pattern, position in position_patterns:
            if re.search(pattern, text_lower):
                entity = await self._get_entity_by_position(position)
                if entity:
                    logger.debug(f"Resolved position reference '{pattern}' to entity: {entity.entity_name}")
                    return {
                        "type": entity.entity_type,
                        "data": entity.entity_data or {"id": entity.entity_id, "name": entity.entity_name}
                    }

        # Attribute-based: "the one from <name>"
        from_match = re.search(r"(?:from|by|sent by)\s+([A-Za-z\s\.]+?)(?:\s|$|')", text_lower)
        if from_match:
            sender_name = from_match.group(1).strip()
            entity = await self._get_entity_by_attribute("from", sender_name)
            if entity:
                logger.debug(f"Resolved 'from {sender_name}' to entity: {entity.entity_name}")
                return {
                    "type": entity.entity_type,
                    "data": entity.entity_data or {"id": entity.entity_id, "name": entity.entity_name}
                }

        # Simple pronouns -> current focus
        if any(pronoun in text_lower for pronoun in ["it", "that one", "this one", "that", "this"]):
            # Find most recent focus
            for entity_type in ["emails", "contacts", "projects", "tasks", "organizations"]:
                if entity_type in self._current_focus:
                    entity = self._current_focus[entity_type]
                    logger.debug(f"Resolved pronoun to current focus: {entity.entity_name}")
                    return {
                        "type": entity.entity_type,
                        "data": entity.entity_data or {"id": entity.entity_id, "name": entity.entity_name}
                    }

        return None

    async def _get_entity_by_position(self, position: int) -> Optional[EntityTracking]:
        """Get entity by list position"""
        # Combine all entity types and sort by last_mentioned
        all_entities = []
        for entities in self._entities_cache.values():
            all_entities.extend(entities)

        if not all_entities:
            return None

        # Filter to only those with positions
        positioned = [e for e in all_entities if e.list_position is not None]
        if not positioned:
            return None

        # Sort by last_mentioned to get most recent list
        positioned.sort(key=lambda e: e.last_mentioned_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

        # Get the entity type of the most recent positioned entity
        most_recent_type = positioned[0].entity_type

        # Filter to same type
        same_type = [e for e in positioned if e.entity_type == most_recent_type]

        if position == -1:
            # "last" -> highest position
            return max(same_type, key=lambda e: e.list_position or 0)

        # Find entity at position
        for entity in same_type:
            if entity.list_position == position:
                return entity

        return None

    async def _get_entity_by_attribute(self, attr: str, value: str) -> Optional[EntityTracking]:
        """Get entity by attribute value (e.g., 'from' for emails)"""
        value_lower = value.lower()

        for entity_type, entities in self._entities_cache.items():
            for entity in entities:
                data = entity.entity_data or {}
                if attr in data:
                    attr_value = str(data[attr]).lower()
                    if value_lower in attr_value or attr_value in value_lower:
                        return entity

        return None

    async def get_context_window(
        self,
        max_tokens: int = 8000,
        include_summary: bool = True
    ) -> List[Dict[str, str]]:
        """
        Build optimized context window for AI.

        Priority order:
        1. Most recent 10 messages (always included)
        2. High importance messages (data payloads, tool calls)
        3. Entity-defining messages
        4. Older summary
        """
        if not self._messages_cache:
            return []

        # Always keep last 10 messages
        recent_count = min(10, len(self._messages_cache))
        recent = self._messages_cache[-recent_count:]
        recent_tokens = sum(m.token_count or 0 for m in recent)
        remaining_budget = max_tokens - recent_tokens

        # Get older messages sorted by importance
        older = self._messages_cache[:-recent_count] if len(self._messages_cache) > recent_count else []
        older_sorted = sorted(older, key=lambda m: m.importance_score or 0.5, reverse=True)

        # Select important older messages that fit
        selected_older = []
        for msg in older_sorted:
            tokens = msg.token_count or TokenCounter.count_tokens(msg.content)
            if remaining_budget >= tokens:
                selected_older.append(msg)
                remaining_budget -= tokens

        # Sort selected by timestamp
        selected_older.sort(key=lambda m: m.created_at)

        # Build messages list
        messages = []

        # Add summary if exists and space permits
        if include_summary and self.conversation.summary and remaining_budget > 100:
            summary_tokens = TokenCounter.count_tokens(self.conversation.summary)
            if remaining_budget >= summary_tokens:
                messages.append({
                    "role": "system",
                    "content": f"[CONVERSATION SUMMARY]: {self.conversation.summary}"
                })

        # Add older selected messages
        for msg in selected_older:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Add recent messages
        for msg in recent:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        logger.debug(f"Built context: {len(messages)} messages, ~{max_tokens - remaining_budget} tokens")
        return messages

    def get_current_focus_entities(self) -> Dict[str, Dict[str, Any]]:
        """Get all current focus entities"""
        result = {}
        for entity_type, entity in self._current_focus.items():
            result[entity_type] = {
                "id": entity.entity_id,
                "external_id": entity.external_id,
                "name": entity.entity_name,
                "data": entity.entity_data
            }
        return result

    async def generate_summary(self, messages_to_summarize: int = 50) -> str:
        """
        Generate summary of older messages (to be called by AI service).

        Returns a prompt that should be sent to AI for summarization.
        """
        if len(self._messages_cache) <= messages_to_summarize:
            return ""

        # Get messages to summarize
        to_summarize = self._messages_cache[:-20]  # Keep last 20 unsummarized

        # Build simple summary based on entities and actions
        entities_mentioned = set()
        actions_taken = []

        for msg in to_summarize:
            if msg.entities_mentioned:
                entities_mentioned.update(msg.entities_mentioned)
            if msg.intent:
                actions_taken.append(msg.intent)

        summary_parts = []
        if entities_mentioned:
            summary_parts.append(f"Entities discussed: {', '.join(entities_mentioned)}")
        if actions_taken:
            # Deduplicate and limit
            unique_actions = list(dict.fromkeys(actions_taken))[:10]
            summary_parts.append(f"Actions: {', '.join(unique_actions)}")

        summary = ". ".join(summary_parts) if summary_parts else "Previous conversation about CRM operations."

        # Update conversation summary
        self.conversation.summary = summary
        await self.db.flush()

        return summary

    async def commit(self):
        """Commit all changes to database"""
        await self.db.commit()

    # Backward compatibility methods
    def track_entity_mention(self, entity_type: str, entity_data: Dict[str, Any]):
        """Backward compatible sync wrapper - schedules async tracking"""
        # Update in-memory data immediately
        self._update_recent_data(entity_type, entity_data)
        logger.debug(f"[Sync] Tracked {entity_type} mention: {entity_data.get('name') or entity_data.get('subject')}")

    def get_current_email_id(self) -> Optional[str]:
        """Backward compatible method"""
        return self.recent_data.get("current_email_id")

    def get_current_contact(self) -> Optional[Dict[str, Any]]:
        """Backward compatible method"""
        return {
            "id": self.recent_data.get("current_contact_id"),
            "name": self.recent_data.get("current_contact_name")
        }

    def get_current_project(self) -> Optional[Dict[str, Any]]:
        """Backward compatible method"""
        return {
            "id": self.recent_data.get("current_project_id"),
            "name": self.recent_data.get("current_project_name")
        }

    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """Backward compatible method"""
        return {
            "id": self.recent_data.get("current_task_id"),
            "name": self.recent_data.get("current_task_name")
        }

    def get_current_organization(self) -> Optional[Dict[str, Any]]:
        """Backward compatible method"""
        return {
            "id": self.recent_data.get("current_organization_id"),
            "name": self.recent_data.get("current_organization_name")
        }

    def get_tracked_entities(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        Get all tracked entities of a specific type.

        Args:
            entity_type: Type of entity (contacts, emails, projects, tasks, organizations)

        Returns:
            List of entity dictionaries with id, name, and any stored data
        """
        entities = self._entities_cache.get(entity_type, [])
        result = []

        for entity in entities:
            entity_dict = {
                "id": entity.entity_id,
                "name": entity.entity_name,
                "position": entity.position,
                "is_current_focus": entity.is_current_focus
            }
            # Include any additional stored data
            if entity.entity_data:
                entity_dict.update(entity.entity_data)
            if entity.external_id:
                entity_dict["external_id"] = entity.external_id

            result.append(entity_dict)

        return result

    def get_recent_entities_for_parser(self, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get recent entities from conversation context for LLM parser injection.

        This enables context-aware intent detection - e.g., when user says
        "update Gabriel with his email", the parser knows Gabriel is a contact.

        Args:
            limit: Maximum entities per type to return

        Returns:
            Dict mapping entity_type to list of recent entities with name/id
        """
        result = {}

        for entity_type in ENTITY_TYPES:
            entities = self._entities_cache.get(entity_type, [])
            if not entities:
                continue

            # Sort by last_mentioned_at descending
            sorted_entities = sorted(
                entities,
                key=lambda e: e.last_mentioned_at or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True
            )[:limit]

            # Build simplified list for parser
            entity_list = []
            for entity in sorted_entities:
                entity_info = {
                    "name": entity.entity_name,
                    "id": entity.entity_id
                }
                # Add key details for better matching
                if entity.entity_data:
                    if entity_type == "contacts":
                        if entity.entity_data.get("email"):
                            entity_info["email"] = entity.entity_data["email"]
                    elif entity_type == "organizations":
                        if entity.entity_data.get("industry"):
                            entity_info["industry"] = entity.entity_data["industry"]

                entity_list.append(entity_info)

            if entity_list:
                result[entity_type] = entity_list

        return result

    def set_last_offer(self, offer_message: str):
        """Store the AI's last offer/question"""
        self.recent_data["last_ai_offer"] = offer_message
        self.recent_data["offer_timestamp"] = datetime.utcnow()

    def get_last_offer(self, max_age_seconds: int = 120) -> Optional[str]:
        """Get the last AI offer if recent enough"""
        offer = self.recent_data.get("last_ai_offer")
        timestamp = self.recent_data.get("offer_timestamp")

        if not offer or not timestamp:
            return None

        age = datetime.utcnow() - timestamp
        if age > timedelta(seconds=max_age_seconds):
            self.recent_data["last_ai_offer"] = None
            self.recent_data["offer_timestamp"] = None
            return None

        return offer

    def store_data_payload(self, data_type: str, data_list: List[Dict[str, Any]]):
        """Backward compatible - store data payload and track entities"""
        for idx, item in enumerate(data_list):
            self.track_entity_mention(data_type, item)
        logger.info(f"[Sync] Stored {len(data_list)} {data_type} in memory")

    def resolve_pronoun(self, text: str) -> Optional[Dict[str, Any]]:
        """Backward compatible sync pronoun resolution"""
        text_lower = text.lower().strip()

        # Position patterns
        if "first" in text_lower or "1st" in text_lower:
            return self.recent_data.get("current_focus")
        if "second" in text_lower or "2nd" in text_lower:
            # Need async for position-based - return focus as fallback
            return self.recent_data.get("current_focus")

        # Simple pronouns
        if any(p in text_lower for p in ["it", "that one", "this one", "that", "this"]):
            return self.recent_data.get("current_focus")

        return None

    def mark_message_as_important(self, message: Dict[str, str], reason: str = None):
        """
        Mark a message as important for context preservation.

        This updates the importance score of recent messages that match
        the given content, ensuring they are preserved during context pruning.

        Args:
            message: Message dict with 'role' and 'content'
            reason: Optional reason for importance (for logging)
        """
        content = message.get("content", "")
        role = message.get("role", "user")

        # Find matching message in cache and update importance
        for msg in reversed(self._messages_cache):
            if msg.role == role and msg.content == content:
                msg.importance_score = ImportanceScores.CRITICAL
                msg.has_data_payload = True
                logger.debug(f"Marked message as important: {reason or 'manual'}")
                break

        # Also mark as important if this is a recent data-bearing exchange
        if reason and "data" in reason.lower():
            self.recent_data["last_important_exchange"] = {
                "message": content[:200],
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_relevant_context(self, user_message: str) -> Optional[str]:
        """Backward compatible context injection"""
        # This is handled by get_context_window now
        return None


# ============================================================================
# LEGACY SUPPORT - Keep old ConversationMemory class for backward compatibility
# ============================================================================

class ConversationMemory:
    """
    LEGACY: In-memory conversation memory.

    This class is kept for backward compatibility.
    New code should use PersistentConversationMemory.
    """

    def __init__(self, max_recent_messages: int = 20):
        self.max_recent_messages = max_recent_messages

        # Entity tracking (legacy)
        self.mentioned_entities = {
            "emails": deque(maxlen=50),
            "contacts": deque(maxlen=50),
            "projects": deque(maxlen=50),
            "tasks": deque(maxlen=50),
            "organizations": deque(maxlen=50)
        }

        # Recent data (legacy)
        self.recent_data = {
            "last_emails": deque(maxlen=20),
            "last_contacts": deque(maxlen=20),
            "last_projects": deque(maxlen=20),
            "last_tasks": deque(maxlen=20),
            "last_organizations": deque(maxlen=20),
            "current_focus": None,
            "current_email_id": None,
            "current_email_subject": None,
            "current_email_from": None,
            "current_email_timestamp": None,
            "current_contact_id": None,
            "current_contact_name": None,
            "current_project_id": None,
            "current_project_name": None,
            "current_task_id": None,
            "current_task_name": None,
            "current_organization_id": None,
            "current_organization_name": None,
            "last_ai_offer": None,
            "offer_timestamp": None,
        }

        self.conversation_summaries = deque(maxlen=10)
        self.important_messages = deque(maxlen=30)

    def track_entity_mention(self, entity_type: str, entity_data: Dict[str, Any]):
        """Track entity mention (legacy)"""
        if entity_type not in self.mentioned_entities:
            logger.warning(f"Unknown entity type: {entity_type}")
            return

        self.mentioned_entities[entity_type].append({
            "data": entity_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        self.recent_data["current_focus"] = {
            "type": entity_type,
            "data": entity_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Update type-specific tracking
        if entity_type == "emails" and entity_data.get("id"):
            self.recent_data["current_email_id"] = entity_data["id"]
            self.recent_data["current_email_subject"] = entity_data.get("subject")
            self.recent_data["current_email_from"] = entity_data.get("from")
            self.recent_data["current_email_timestamp"] = datetime.now(timezone.utc).isoformat()
        elif entity_type == "contacts":
            if entity_data.get("id"):
                self.recent_data["current_contact_id"] = entity_data["id"]
            if entity_data.get("name"):
                self.recent_data["current_contact_name"] = entity_data["name"]
        elif entity_type == "projects":
            if entity_data.get("id"):
                self.recent_data["current_project_id"] = entity_data["id"]
            if entity_data.get("name"):
                self.recent_data["current_project_name"] = entity_data["name"]
        elif entity_type == "tasks":
            if entity_data.get("id"):
                self.recent_data["current_task_id"] = entity_data["id"]
            if entity_data.get("name"):
                self.recent_data["current_task_name"] = entity_data["name"]
        elif entity_type == "organizations":
            if entity_data.get("id"):
                self.recent_data["current_organization_id"] = entity_data["id"]
            if entity_data.get("name"):
                self.recent_data["current_organization_name"] = entity_data["name"]

    def get_current_email_id(self) -> Optional[str]:
        return self.recent_data.get("current_email_id")

    def get_current_contact(self) -> Optional[Dict[str, Any]]:
        return {
            "id": self.recent_data.get("current_contact_id"),
            "name": self.recent_data.get("current_contact_name")
        }

    def get_current_project(self) -> Optional[Dict[str, Any]]:
        return {
            "id": self.recent_data.get("current_project_id"),
            "name": self.recent_data.get("current_project_name")
        }

    def get_current_task(self) -> Optional[Dict[str, Any]]:
        return {
            "id": self.recent_data.get("current_task_id"),
            "name": self.recent_data.get("current_task_name")
        }

    def get_current_organization(self) -> Optional[Dict[str, Any]]:
        return {
            "id": self.recent_data.get("current_organization_id"),
            "name": self.recent_data.get("current_organization_name")
        }

    def set_last_offer(self, offer_message: str):
        self.recent_data["last_ai_offer"] = offer_message
        self.recent_data["offer_timestamp"] = datetime.utcnow()

    def get_last_offer(self, max_age_seconds: int = 120) -> Optional[str]:
        offer = self.recent_data.get("last_ai_offer")
        timestamp = self.recent_data.get("offer_timestamp")

        if not offer or not timestamp:
            return None

        age = datetime.utcnow() - timestamp
        if age > timedelta(seconds=max_age_seconds):
            self.recent_data["last_ai_offer"] = None
            self.recent_data["offer_timestamp"] = None
            return None

        return offer

    def store_data_payload(self, data_type: str, data_list: List[Dict[str, Any]]):
        storage_key = f"last_{data_type}"
        if storage_key not in self.recent_data:
            return

        self.recent_data[storage_key].clear()
        for item in data_list:
            self.recent_data[storage_key].append(item)
            self.track_entity_mention(data_type, item)

    def resolve_pronoun(self, text: str) -> Optional[Dict[str, Any]]:
        text_lower = text.lower().strip()

        if any(p in text_lower for p in ["it", "that one", "this one", "that", "this"]):
            if self.recent_data["current_focus"]:
                return self.recent_data["current_focus"]

        if "first" in text_lower:
            for key, data_list in self.recent_data.items():
                if key.startswith("last_") and len(data_list) > 0:
                    return {
                        "type": key.replace("last_", "").rstrip('s'),
                        "data": list(data_list)[0]
                    }

        if "second" in text_lower:
            for key, data_list in self.recent_data.items():
                if key.startswith("last_") and len(data_list) >= 2:
                    return {
                        "type": key.replace("last_", "").rstrip('s'),
                        "data": list(data_list)[1]
                    }

        return None

    def get_relevant_context(self, user_message: str) -> Optional[str]:
        return None

    def get_tiered_context(
        self,
        conversation_history: List[Dict[str, str]],
        max_tokens: int = 8000
    ) -> Tuple[List[Dict[str, str]], bool]:
        if not conversation_history:
            return [], False

        if len(conversation_history) <= self.max_recent_messages:
            return conversation_history, False

        # Simple tiering
        return conversation_history[-self.max_recent_messages:], True

    def estimate_tokens(self, text: str) -> int:
        return TokenCounter.count_tokens(text)

    def clear_memory(self):
        for entity_list in self.mentioned_entities.values():
            entity_list.clear()
        for key, value in self.recent_data.items():
            if isinstance(value, deque):
                value.clear()
        self.recent_data["current_focus"] = None
        self.conversation_summaries.clear()
        self.important_messages.clear()


# ============================================================================
# GLOBAL MEMORY MANAGEMENT
# ============================================================================

# Legacy in-memory store
_conversation_memory_store: Dict[int, ConversationMemory] = {}

# Persistent memory cache (loaded from DB)
_persistent_memory_cache: Dict[int, PersistentConversationMemory] = {}


def get_conversation_memory(user_id: int) -> ConversationMemory:
    """
    LEGACY: Get in-memory conversation memory for user.

    For new code, use: await get_persistent_memory(user_id, db)
    """
    if user_id not in _conversation_memory_store:
        _conversation_memory_store[user_id] = ConversationMemory()
        logger.info(f"Created new (legacy) conversation memory for user {user_id}")
    return _conversation_memory_store[user_id]


async def get_persistent_memory(user_id: int, db: AsyncSession) -> PersistentConversationMemory:
    """
    Get persistent conversation memory for user.

    This is the recommended method for new code.
    Memory is loaded from DB and cached for the session.
    """
    if user_id in _persistent_memory_cache:
        # Update db session reference
        _persistent_memory_cache[user_id].db = db
        return _persistent_memory_cache[user_id]

    memory = await PersistentConversationMemory.get_or_create(user_id, db)
    _persistent_memory_cache[user_id] = memory
    return memory


def clear_user_memory(user_id: int) -> None:
    """Clear all memory for user"""
    if user_id in _conversation_memory_store:
        del _conversation_memory_store[user_id]
    if user_id in _persistent_memory_cache:
        del _persistent_memory_cache[user_id]
    logger.info(f"Cleared conversation memory for user {user_id}")


def clear_persistent_cache():
    """Clear the persistent memory cache (on server restart, etc.)"""
    _persistent_memory_cache.clear()
    logger.info("Cleared persistent memory cache")


# Legacy global instance
conversation_memory = ConversationMemory()
