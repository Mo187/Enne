"""
Clarification Manager - Handles pending clarification requests across HTTP requests.

When the system finds multiple matches (e.g., 2 contacts named "Gabriel"), it asks the user
to clarify which one they want. This manager stores that pending clarification and matches
the user's response to resume the original operation.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import structlog

logger = structlog.get_logger()


class ClarificationManager:
    """Manages pending clarification requests across HTTP requests"""

    def __init__(self, timeout_minutes: int = 5):
        """
        Initialize the clarification manager.

        Args:
            timeout_minutes: How long to keep clarifications before expiring
        """
        self.pending_clarifications: Dict[str, Dict[str, Any]] = {}
        self.timeout = timedelta(minutes=timeout_minutes)
        # Also track by user for quick lookup
        self.user_clarifications: Dict[int, str] = {}  # user_id -> clarification_id

    def create_clarification(
        self,
        user_id: int,
        clarification_type: str,
        matches: list,
        original_action: Dict[str, Any]
    ) -> str:
        """
        Store a pending clarification request.

        Args:
            user_id: ID of user who needs to clarify
            clarification_type: Type (e.g., "multiple_contacts")
            matches: List of matching entities
            original_action: The original API action to resume

        Returns:
            clarification_id: Unique ID for this clarification
        """
        clarification_id = str(uuid.uuid4())

        self.pending_clarifications[clarification_id] = {
            "user_id": user_id,
            "clarification_type": clarification_type,
            "matches": matches,
            "original_action": original_action,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + self.timeout
        }

        # Track latest for this user
        self.user_clarifications[user_id] = clarification_id

        logger.info(
            "Created clarification request",
            clarification_id=clarification_id,
            user_id=user_id,
            clarification_type=clarification_type,
            num_matches=len(matches)
        )

        return clarification_id

    def resolve_clarification(
        self,
        clarification_id: str,
        selected_identifier: str,
        user_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a clarification by matching user's selection to one of the options.

        Args:
            clarification_id: ID of pending clarification
            selected_identifier: What user said (e.g., "Gabriel Jones", "the first one")
            user_id: User making the selection

        Returns:
            Dict with original_action, selected_id, and clarification_type, or None if failed
        """
        if clarification_id not in self.pending_clarifications:
            logger.warning("Clarification not found", clarification_id=clarification_id)
            return None

        clarif = self.pending_clarifications[clarification_id]

        # Verify user
        if clarif["user_id"] != user_id:
            logger.error(
                "User mismatch for clarification",
                clarification_id=clarification_id,
                expected_user=clarif["user_id"],
                actual_user=user_id
            )
            return None

        # Check expiration
        if datetime.utcnow() > clarif["expires_at"]:
            del self.pending_clarifications[clarification_id]
            if user_id in self.user_clarifications:
                del self.user_clarifications[user_id]
            logger.warning("Clarification expired", clarification_id=clarification_id)
            return None

        # Match selection to one of the options
        selected_id = self._match_selection(selected_identifier, clarif["matches"])

        if not selected_id:
            logger.warning(
                "Could not match selection to any option",
                selection=selected_identifier,
                num_matches=len(clarif["matches"])
            )
            return None

        # Clean up
        del self.pending_clarifications[clarification_id]
        if user_id in self.user_clarifications:
            del self.user_clarifications[user_id]

        logger.info(
            "Clarification resolved successfully",
            clarification_id=clarification_id,
            selected_id=selected_id
        )

        return {
            "original_action": clarif["original_action"],
            "selected_id": selected_id,
            "clarification_type": clarif["clarification_type"]
        }

    def _match_selection(self, selection: str, matches: list) -> Optional[int]:
        """
        Match user's selection to one of the presented options.

        Handles:
        - Exact name match: "Gabriel Jones" → match by name
        - Positional: "the first one", "second", "1", "2"
        - Partial match: "Jones" → match if unique
        - Email match: "gab@email.com"

        Args:
            selection: What the user said
            matches: List of entity dicts with 'id', 'name', 'email' etc.

        Returns:
            Entity ID if matched, None otherwise
        """
        selection_lower = selection.lower().strip()

        # Check for positional selection
        positional_patterns = {
            "first": 0, "1": 0, "1st": 0, "one": 0,
            "second": 1, "2": 1, "2nd": 1, "two": 1,
            "third": 2, "3": 2, "3rd": 2, "three": 2,
            "fourth": 3, "4": 3, "4th": 4, "four": 3,
            "fifth": 4, "5": 4, "5th": 4, "five": 4
        }

        for pattern, index in positional_patterns.items():
            if pattern in selection_lower:
                if index < len(matches):
                    logger.info(f"Matched positional selection '{pattern}' to index {index}")
                    return matches[index]["id"]
                else:
                    logger.warning(
                        f"Position '{pattern}' out of range",
                        num_matches=len(matches)
                    )
                    return None

        # Check for exact name/email match
        for match in matches:
            match_name = match.get("name", "").lower()
            match_email = match.get("email", "").lower() if match.get("email") else ""

            if selection_lower == match_name or selection_lower == match_email:
                logger.info(f"Matched exact selection to: {match.get('name')}")
                return match["id"]

        # Check for partial match (if unique)
        partial_matches = []
        for match in matches:
            match_name = match.get("name", "").lower()
            match_email = match.get("email", "").lower() if match.get("email") else ""

            if selection_lower in match_name or selection_lower in match_email:
                partial_matches.append(match)

        if len(partial_matches) == 1:
            logger.info(f"Matched partial selection to: {partial_matches[0].get('name')}")
            return partial_matches[0]["id"]
        elif len(partial_matches) > 1:
            logger.warning(f"Ambiguous partial match - {len(partial_matches)} matches found")
            return None

        logger.warning("No match found for selection", selection=selection)
        return None

    def get_latest_clarification_for_user(self, user_id: int) -> Optional[str]:
        """
        Get the most recent clarification ID for a user.

        Args:
            user_id: User ID

        Returns:
            Clarification ID or None
        """
        return self.user_clarifications.get(user_id)

    def cleanup_expired(self):
        """Remove expired clarifications"""
        now = datetime.utcnow()
        expired = [
            cid for cid, clarif in self.pending_clarifications.items()
            if now > clarif["expires_at"]
        ]

        for cid in expired:
            user_id = self.pending_clarifications[cid]["user_id"]
            del self.pending_clarifications[cid]
            if user_id in self.user_clarifications and self.user_clarifications[user_id] == cid:
                del self.user_clarifications[user_id]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired clarifications")


# Global instance
clarification_manager = ClarificationManager()
