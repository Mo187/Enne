"""
Conversation Memory Service for CRM AI Assistant

Manages conversation context, entity tracking, and data preservation across long conversations.
Implements tiered context management to handle conversations that exceed token limits.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from collections import deque
import structlog
import re
import json

logger = structlog.get_logger()


class ConversationMemory:
    """
    Manages conversation history with intelligent context preservation.

    Features:
    - Entity tracking across conversation (emails, contacts, projects, tasks)
    - Data payload preservation for follow-up questions
    - Tiered context management (recent/older/ancient messages)
    - Token-aware context window sizing
    - Pronoun resolution ("it", "that one", "the first one")
    """

    def __init__(self, max_recent_messages: int = 20):
        """
        Initialize conversation memory.

        Args:
            max_recent_messages: Number of recent messages to keep in full detail
        """
        self.max_recent_messages = max_recent_messages

        # Entity tracking
        self.mentioned_entities = {
            "emails": deque(maxlen=50),       # Recent email subjects/IDs
            "contacts": deque(maxlen=50),     # Recent contact names
            "projects": deque(maxlen=50),     # Recent project names
            "tasks": deque(maxlen=50),        # Recent task names
            "organizations": deque(maxlen=50) # Recent organization names
        }

        # Data payload storage for follow-up questions
        self.recent_data = {
            "last_emails": deque(maxlen=20),      # Last N emails shown
            "last_contacts": deque(maxlen=20),    # Last N contacts shown
            "last_projects": deque(maxlen=20),    # Last N projects shown
            "last_tasks": deque(maxlen=20),       # Last N tasks shown
            "last_organizations": deque(maxlen=20), # Last N organizations shown
            "current_focus": None,                # Entity user is discussing
            "current_email_id": None,             # ID of email currently being discussed
            "current_email_subject": None,        # Subject of email currently being discussed
            "current_email_from": None,           # Sender of email currently being discussed
            "current_email_timestamp": None,      # Timestamp of email tracking
            "current_contact_id": None,           # ID of contact currently being discussed
            "current_project_id": None,           # ID of project currently being discussed
            "current_task_id": None,              # ID of task currently being discussed
            "current_organization_id": None,      # ID of organization currently being discussed
            "current_contact_name": None,         # Name for fallback
            "current_project_name": None,         # Name for fallback
            "current_task_name": None,            # Name for fallback
            "current_organization_name": None,    # Name for fallback
            "last_ai_offer": None,                # Track AI's last offer/question
            "offer_timestamp": None,              # When the offer was made
        }

        # Conversation summaries for ancient messages
        self.conversation_summaries = deque(maxlen=10)

        # Message importance tracking
        self.important_messages = deque(maxlen=30)  # Messages with data payloads

    def track_entity_mention(self, entity_type: str, entity_data: Dict[str, Any]):
        """
        Track when an entity is mentioned in conversation.

        Args:
            entity_type: Type of entity (email, contact, project, task, organization)
            entity_data: Entity information (id, name, subject, etc.)
        """
        if entity_type not in self.mentioned_entities:
            logger.warning(f"Unknown entity type: {entity_type}")
            return

        self.mentioned_entities[entity_type].append({
            "data": entity_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Update current focus
        self.recent_data["current_focus"] = {
            "type": entity_type,
            "data": entity_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Track current IDs for each entity type
        if entity_type == "emails" and entity_data.get("id"):
            old_email_id = self.recent_data.get("current_email_id")
            self.recent_data["current_email_id"] = entity_data["id"]
            self.recent_data["current_email_subject"] = entity_data.get("subject")
            self.recent_data["current_email_from"] = entity_data.get("from")
            self.recent_data["current_email_timestamp"] = datetime.now(timezone.utc).isoformat()

            logger.info(
                f"Email tracking updated: old_id={old_email_id}, new_id={entity_data['id']}, "
                f"subject={entity_data.get('subject')[:50] if entity_data.get('subject') else 'None'}, "
                f"from={entity_data.get('from')}"
            )

        elif entity_type == "contacts":
            if entity_data.get("id"):
                self.recent_data["current_contact_id"] = entity_data["id"]
                logger.info(f"Set current contact ID to: {entity_data['id']}")
            if entity_data.get("name"):
                self.recent_data["current_contact_name"] = entity_data["name"]
                logger.info(f"Set current contact name to: {entity_data['name']}")

        elif entity_type == "projects":
            if entity_data.get("id"):
                self.recent_data["current_project_id"] = entity_data["id"]
                logger.info(f"Set current project ID to: {entity_data['id']}")
            if entity_data.get("name"):
                self.recent_data["current_project_name"] = entity_data["name"]
                logger.info(f"Set current project name to: {entity_data['name']}")

        elif entity_type == "tasks":
            if entity_data.get("id"):
                self.recent_data["current_task_id"] = entity_data["id"]
                logger.info(f"Set current task ID to: {entity_data['id']}")
            if entity_data.get("name"):
                self.recent_data["current_task_name"] = entity_data["name"]
                logger.info(f"Set current task name to: {entity_data['name']}")

        elif entity_type == "organizations":
            if entity_data.get("id"):
                self.recent_data["current_organization_id"] = entity_data["id"]
                logger.info(f"Set current organization ID to: {entity_data['id']}")
            if entity_data.get("name"):
                self.recent_data["current_organization_name"] = entity_data["name"]
                logger.info(f"Set current organization name to: {entity_data['name']}")

        logger.debug(
            f"Tracked {entity_type} mention",
            entity_type=entity_type,
            entity_id=entity_data.get("id"),
            entity_name=entity_data.get("name") or entity_data.get("subject")
        )

    def get_current_email_id(self) -> Optional[str]:
        """Get the ID of the email currently being discussed."""
        return self.recent_data.get("current_email_id")

    def get_current_contact(self) -> Optional[Dict[str, Any]]:
        """Get the contact currently being discussed."""
        return {
            "id": self.recent_data.get("current_contact_id"),
            "name": self.recent_data.get("current_contact_name")
        }

    def get_current_project(self) -> Optional[Dict[str, Any]]:
        """Get the project currently being discussed."""
        return {
            "id": self.recent_data.get("current_project_id"),
            "name": self.recent_data.get("current_project_name")
        }

    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """Get the task currently being discussed."""
        return {
            "id": self.recent_data.get("current_task_id"),
            "name": self.recent_data.get("current_task_name")
        }

    def get_current_organization(self) -> Optional[Dict[str, Any]]:
        """Get the organization currently being discussed."""
        return {
            "id": self.recent_data.get("current_organization_id"),
            "name": self.recent_data.get("current_organization_name")
        }

    def set_last_offer(self, offer_message: str):
        """Store the AI's last offer/question for follow-up detection."""
        from datetime import datetime
        self.recent_data["last_ai_offer"] = offer_message
        self.recent_data["offer_timestamp"] = datetime.utcnow()
        logger.info(f"Stored AI offer: {offer_message[:50]}...")

    def get_last_offer(self, max_age_seconds: int = 120) -> Optional[str]:
        """Get the last AI offer if it's recent enough (default: 2 minutes)."""
        from datetime import datetime, timedelta

        offer = self.recent_data.get("last_ai_offer")
        timestamp = self.recent_data.get("offer_timestamp")

        if not offer or not timestamp:
            return None

        # Check if offer is still fresh (< max_age_seconds old)
        age = datetime.utcnow() - timestamp
        if age > timedelta(seconds=max_age_seconds):
            # Offer expired, clear it
            self.recent_data["last_ai_offer"] = None
            self.recent_data["offer_timestamp"] = None
            return None

        return offer

    def store_data_payload(self, data_type: str, data_list: List[Dict[str, Any]]):
        """
        Store data payload for follow-up reference.

        Args:
            data_type: Type of data (emails, contacts, projects, tasks)
            data_list: List of data items
        """
        storage_key = f"last_{data_type}"
        if storage_key not in self.recent_data:
            logger.warning(f"Unknown data type: {data_type}")
            return

        # Store the data
        self.recent_data[storage_key].clear()
        for item in data_list:
            self.recent_data[storage_key].append(item)

            # Also track individual entity mentions
            # Entity types should remain PLURAL (emails, contacts, etc.)
            self.track_entity_mention(data_type, item)

        logger.info(
            f"Stored {len(data_list)} {data_type} in memory",
            data_type=data_type,
            count=len(data_list)
        )

    def resolve_pronoun(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Resolve pronouns to specific entities from conversation history.

        Args:
            text: User message containing pronoun

        Returns:
            Entity data if resolved, None otherwise
        """
        text_lower = text.lower().strip()

        # "it" / "that one" / "this one" -> most recent entity mentioned
        if any(pronoun in text_lower for pronoun in ["it", "that one", "this one", "that", "this"]):
            if self.recent_data["current_focus"]:
                logger.debug("Resolved pronoun to current focus entity")
                return self.recent_data["current_focus"]

        # "the first one" / "first" -> first item in last shown list
        if "first" in text_lower:
            for data_type, data_list in self.recent_data.items():
                if data_type.startswith("last_") and len(data_list) > 0:
                    logger.debug(f"Resolved 'first' to first item in {data_type}")
                    return {
                        "type": data_type.replace("last_", "").rstrip('s'),
                        "data": list(data_list)[0]
                    }

        # "the second one" / "second" -> second item in last shown list
        if "second" in text_lower:
            for data_type, data_list in self.recent_data.items():
                if data_type.startswith("last_") and len(data_list) >= 2:
                    logger.debug(f"Resolved 'second' to second item in {data_type}")
                    return {
                        "type": data_type.replace("last_", "").rstrip('s'),
                        "data": list(data_list)[1]
                    }

        # "the last one" -> most recent entity of any type
        if "last one" in text_lower or "latest" in text_lower:
            if self.recent_data["current_focus"]:
                logger.debug("Resolved 'last one' to current focus entity")
                return self.recent_data["current_focus"]

        return None

    def get_relevant_context(self, user_message: str) -> Optional[str]:
        """
        Get relevant context to inject based on user message.

        Args:
            user_message: Current user message

        Returns:
            Context string to inject, or None
        """
        user_message_lower = user_message.lower()

        # Check if user is asking about emails
        if any(keyword in user_message_lower for keyword in ["email", "message", "inbox"]):
            if len(self.recent_data["last_emails"]) > 0:
                email_context = self._format_email_context(list(self.recent_data["last_emails"]))
                logger.debug("Injecting email context for follow-up question")
                return email_context

        # Check if user is asking about contacts
        if any(keyword in user_message_lower for keyword in ["contact", "person", "people"]):
            if len(self.recent_data["last_contacts"]) > 0:
                contact_context = self._format_contact_context(list(self.recent_data["last_contacts"]))
                logger.debug("Injecting contact context for follow-up question")
                return contact_context

        # Check if user is asking about projects
        if "project" in user_message_lower:
            if len(self.recent_data["last_projects"]) > 0:
                project_context = self._format_project_context(list(self.recent_data["last_projects"]))
                logger.debug("Injecting project context for follow-up question")
                return project_context

        # Check if user is asking about tasks
        if "task" in user_message_lower:
            if len(self.recent_data["last_tasks"]) > 0:
                task_context = self._format_task_context(list(self.recent_data["last_tasks"]))
                logger.debug("Injecting task context for follow-up question")
                return task_context

        # Check for pronoun references
        resolved_entity = self.resolve_pronoun(user_message)
        if resolved_entity:
            return self._format_entity_context(resolved_entity)

        return None

    def _format_email_context(self, emails: List[Dict[str, Any]]) -> str:
        """Format email list for context injection"""
        context = f"\n[CONTEXT: Recently retrieved {len(emails)} email(s)]\n"
        for i, email in enumerate(emails, 1):
            context += f"{i}. {email.get('subject', 'No subject')} - from {email.get('from', 'Unknown')}\n"
        return context

    def _format_contact_context(self, contacts: List[Dict[str, Any]]) -> str:
        """Format contact list for context injection"""
        context = f"\n[CONTEXT: Recently retrieved {len(contacts)} contact(s)]\n"
        for i, contact in enumerate(contacts, 1):
            context += f"{i}. {contact.get('name', 'Unknown')} ({contact.get('email', 'no email')})\n"
        return context

    def _format_project_context(self, projects: List[Dict[str, Any]]) -> str:
        """Format project list for context injection"""
        context = f"\n[CONTEXT: Recently retrieved {len(projects)} project(s)]\n"
        for i, project in enumerate(projects, 1):
            context += f"{i}. {project.get('name', 'Unknown')} (Status: {project.get('status', 'unknown')})\n"
        return context

    def _format_task_context(self, tasks: List[Dict[str, Any]]) -> str:
        """Format task list for context injection"""
        context = f"\n[CONTEXT: Recently retrieved {len(tasks)} task(s)]\n"
        for i, task in enumerate(tasks, 1):
            context += f"{i}. {task.get('name', 'Unknown')} (Priority: {task.get('priority', 'medium')})\n"
        return context

    def _format_entity_context(self, entity: Dict[str, Any]) -> str:
        """Format single entity for context injection"""
        entity_type = entity.get("type", "unknown")
        entity_data = entity.get("data", {})

        context = f"\n[CONTEXT: User is referring to {entity_type}]\n"
        context += f"{entity_type.capitalize()}: {entity_data.get('name') or entity_data.get('subject', 'Unknown')}\n"

        return context

    def mark_message_as_important(self, message: Dict[str, str], reason: str):
        """
        Mark a message as important (contains data payload, entity references, etc.)

        Args:
            message: Message dict with role and content
            reason: Why this message is important
        """
        self.important_messages.append({
            "message": message,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        logger.debug("Marked message as important", reason=reason)

    def get_tiered_context(
        self,
        conversation_history: List[Dict[str, str]],
        max_tokens: int = 8000
    ) -> Tuple[List[Dict[str, str]], bool]:
        """
        Get tiered conversation context that fits within token limit.

        Implements three tiers:
        - Recent messages (last 20): Full detail
        - Older messages (21-50): Summarized or filtered
        - Ancient messages (51+): Key facts only or omitted

        Args:
            conversation_history: Full conversation history
            max_tokens: Maximum tokens to use (rough estimate)

        Returns:
            Tuple of (filtered_history, was_compressed)
        """
        if not conversation_history:
            return [], False

        total_messages = len(conversation_history)

        # If conversation is short, return all messages
        if total_messages <= self.max_recent_messages:
            return conversation_history, False

        # Tier 1: Recent messages (full detail)
        recent_messages = conversation_history[-self.max_recent_messages:]

        # Tier 2: Older messages (21-50) - keep only important ones
        if total_messages > self.max_recent_messages:
            older_start = max(0, total_messages - 50)
            older_end = total_messages - self.max_recent_messages
            older_messages = conversation_history[older_start:older_end]

            # Filter to keep only important messages
            filtered_older = self._filter_important_messages(older_messages)
        else:
            filtered_older = []

        # Tier 3: Ancient messages (51+) - create summary
        if total_messages > 50:
            ancient_messages = conversation_history[:total_messages - 50]
            summary = self._create_conversation_summary(ancient_messages)

            # Add summary as synthetic message
            summary_message = {
                "role": "system",
                "content": f"[CONVERSATION SUMMARY - {len(ancient_messages)} older messages]: {summary}"
            }

            final_history = [summary_message] + filtered_older + recent_messages
        else:
            final_history = filtered_older + recent_messages

        logger.info(
            "Built tiered conversation context",
            original_count=total_messages,
            final_count=len(final_history),
            recent_count=len(recent_messages),
            older_count=len(filtered_older),
            was_compressed=True
        )

        return final_history, True

    def _filter_important_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filter messages to keep only important ones (data payloads, entity references).

        Args:
            messages: List of messages to filter

        Returns:
            Filtered list of important messages
        """
        important = []

        for msg in messages:
            content = msg.get("content", "").lower()

            # Keep messages with data indicators
            if any(indicator in content for indicator in [
                "email data start", "query result", "exact count",
                "contact", "project", "task", "organization",
                "created successfully", "updated successfully"
            ]):
                important.append(msg)

            # Keep messages with entity mentions
            elif any(entity_type in content for entity_type in [
                "email", "contact", "project", "task", "organization"
            ]):
                # Only keep if it has specific details (not just the word)
                if re.search(r"(from|to|subject|name|status|priority):", content):
                    important.append(msg)

        return important

    def _create_conversation_summary(self, messages: List[Dict[str, str]]) -> str:
        """
        Create a summary of ancient messages.

        Args:
            messages: List of messages to summarize

        Returns:
            Summary string
        """
        summary_parts = []

        # Extract key entities mentioned
        contacts = set()
        projects = set()
        organizations = set()

        for msg in messages:
            content = msg.get("content", "")

            # Extract contact names (simple pattern matching)
            if "contact" in content.lower():
                contact_matches = re.findall(r"contact[: ]+'?([A-Z][a-z]+ [A-Z][a-z]+)", content)
                contacts.update(contact_matches[:3])  # Limit to 3

            # Extract project names
            if "project" in content.lower():
                project_matches = re.findall(r"project[: ]+'?([A-Z][A-Za-z\s]+)", content)
                projects.update(p.strip() for p in project_matches[:3])

            # Extract organization names
            if "organization" in content.lower():
                org_matches = re.findall(r"organization[: ]+'?([A-Z][A-Za-z\s]+)", content)
                organizations.update(o.strip() for o in org_matches[:3])

        # Build summary
        if contacts:
            summary_parts.append(f"Discussed contacts: {', '.join(list(contacts)[:5])}")

        if projects:
            summary_parts.append(f"Discussed projects: {', '.join(list(projects)[:5])}")

        if organizations:
            summary_parts.append(f"Discussed organizations: {', '.join(list(organizations)[:5])}")

        if not summary_parts:
            summary_parts.append(f"Previous conversation covered CRM operations")

        return ". ".join(summary_parts)

    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimate of tokens in text (approximation: 1 token ≈ 4 characters).

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def clear_memory(self):
        """Clear all conversation memory (for new conversation)"""
        for entity_list in self.mentioned_entities.values():
            entity_list.clear()

        for data_list in self.recent_data.values():
            if isinstance(data_list, deque):
                data_list.clear()

        self.recent_data["current_focus"] = None
        self.conversation_summaries.clear()
        self.important_messages.clear()

        logger.info("Conversation memory cleared")


# Per-user conversation memory storage
_conversation_memory_store: Dict[int, ConversationMemory] = {}

def get_conversation_memory(user_id: int) -> ConversationMemory:
    """
    Get or create conversation memory for a specific user.

    Args:
        user_id: The user's ID

    Returns:
        ConversationMemory instance for this user
    """
    if user_id not in _conversation_memory_store:
        _conversation_memory_store[user_id] = ConversationMemory()
        logger.info(f"Created new conversation memory for user {user_id}")
    return _conversation_memory_store[user_id]

def clear_user_memory(user_id: int) -> None:
    """Clear conversation memory for a specific user."""
    if user_id in _conversation_memory_store:
        del _conversation_memory_store[user_id]
        logger.info(f"Cleared conversation memory for user {user_id}")

# Backward compatibility - default instance
conversation_memory = ConversationMemory()
