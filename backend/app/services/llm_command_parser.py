from typing import Dict, Any, List, Optional
import json
import structlog
from enum import Enum
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta
import re

from .ai_service import ai_service

logger = structlog.get_logger()


class LLMCommandParser:
    """
    Advanced command parser using LLM (Claude/GPT) for intent classification and entity extraction.
    This replaces the regex-based parser with more intelligent natural language understanding.
    """

    def __init__(self):
        self.available_intents = [
            "create_contact",
            "update_contact",
            "delete_contact",
            "search_contacts",
            "list_contacts",
            "create_organization",
            "update_organization",
            "delete_organization",
            "search_organizations",
            "list_organizations",
            "create_project",
            "update_project",
            "delete_project",
            "search_projects",
            "list_projects",
            "create_task",
            "update_task",
            "delete_task",
            "search_tasks",
            "list_tasks",
            "export_data",
            "get_stats",
            "help",
            "unknown",
            "clarification_response",
            # Microsoft 365 MCP Integration
            "send_email",
            "search_emails",
            "read_email",
            "get_latest_email",
            "get_new_emails",
            "create_draft",
            "list_folders",
            "get_calendar_events",
            "create_meeting",
            "schedule_meeting",
            "join_meeting",
            "list_sharepoint_sites",
            "search_sharepoint_documents",
            "upload_sharepoint_document",
            "share_document",
            "list_onedrive_files",
            "upload_onedrive_file",
            "list_teams",
            "list_team_files",
            "extract_document_content",
            "connect_microsoft365",
            "authenticate_microsoft365"
        ]

        # Cache key for prompt caching
        self.cached_system_prompt = None

    async def parse_command(
        self,
        text: str,
        user_context: Dict[str, Any] = None,
        conversation_history: List[Dict[str, str]] = None,
        recent_entities: Dict[str, List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Parse natural language command using LLM

        Args:
            text: Natural language command from user
            user_context: Optional context about the user
            conversation_history: Optional conversation history for follow-up detection
            recent_entities: Optional dict of recently discussed entities for context-aware parsing
                             Format: {"contacts": [{"name": "Gabriel", "id": 1}, ...], "projects": [...]}

        Returns:
            Dict with intent, entities, and confidence
        """

        if not text or not text.strip():
            return {
                "intent": "unknown",
                "entities": {},
                "confidence": 0.0,
                "raw_text": text
            }

        # Check if this is a follow-up question before parsing
        if conversation_history and self._is_follow_up_question(text, conversation_history):
            return {
                "intent": "answer_from_context",
                "entities": {"question": text},
                "confidence": 0.95,
                "raw_text": text
            }

        # Build prompt for LLM to extract intent and entities
        prompt = self._build_extraction_prompt(recent_entities)

        try:
            # Use Claude/GPT to extract structured data with caching
            response = await ai_service.generate_response(
                messages=[
                    {"role": "system", "content": prompt, "cache_control": {"type": "ephemeral"}},
                    {"role": "user", "content": f"Extract intent and entities from: '{text}'"}
                ],
                temperature=0.1,  # Low temperature for consistent parsing
                max_tokens=500
            )

            # Parse the LLM response
            parsed_result = self._parse_llm_response(response["response"], text)

            return parsed_result

        except Exception as e:
            logger.error("LLM command parsing failed", error=str(e), text=text)
            return {
                "intent": "unknown",
                "entities": {"error": str(e)},
                "confidence": 0.0,
                "raw_text": text
            }

    def _build_extraction_prompt(self, recent_entities: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> str:
        """Build prompt for LLM to extract intent and entities, with optional context injection"""

        base_prompt = """You are an expert at understanding CRM commands. Extract INTENT and ENTITIES from user input.

ACTION SYNONYMS:
- CREATE: add, create, new, make, register
- UPDATE: update, change, modify, edit, set, add [field] to
- DELETE: delete, remove, drop, erase
- SEARCH: find, search, look for, show me [specific]
- LIST: list, show all, display all, how many

AVAILABLE INTENTS:
Contacts: create_contact, update_contact, delete_contact, search_contacts, list_contacts
Organizations: create_organization, update_organization, delete_organization, search_organizations, list_organizations
Projects: create_project, update_project, delete_project, search_projects, list_projects
Tasks: create_task, update_task, delete_task, search_tasks, list_tasks
Email: send_email, search_emails, read_email, get_latest_email, get_new_emails, create_draft, list_folders
Calendar: get_calendar_events, create_meeting, schedule_meeting
Files: list_sharepoint_sites, search_sharepoint_documents, list_onedrive_files, list_teams, list_team_files, extract_document_content
Auth: connect_microsoft365, authenticate_microsoft365
Other: export_data, get_stats, help, unknown, clarification_response

LIST vs SEARCH:
- list_* = ALL items, no filter ("how many contacts", "show all projects")
- search_* = FILTERED items ("contacts named John", "completed projects")

ENTITY FIELDS:
- name: Full name (new contacts) | contact_name: Contact being updated
- email, phone, job_position, organization, industry, website
- project_name, task_name, status, priority, assignee, due_date, description
- email_to, email_subject, email_body, email_folder
- meeting_subject, meeting_start, meeting_end, meeting_attendees
- search_query, document_path, site_name, team_name

OUTPUT FORMAT (JSON only):
{{"intent": "intent_name", "entities": {{"field": "value"}}, "confidence": 0.95}}

EXAMPLES:

Input: "Create contact John Smith with email john@test.com"
Output: {{"intent": "create_contact", "entities": {{"name": "John Smith", "email": "john@test.com"}}, "confidence": 0.95}}

Input: "Update Gabriel's phone to 555-1234" (or "add phone 555-1234 to Gabriel")
Output: {{"intent": "update_contact", "entities": {{"contact_name": "Gabriel", "phone": "555-1234"}}, "confidence": 0.9}}

Input: "Delete Luke"
Output: {{"intent": "delete_contact", "entities": {{"name": "Luke"}}, "confidence": 0.95}}

Input: "Create project Website Redesign for Acme Corp"
Output: {{"intent": "create_project", "entities": {{"project_name": "Website Redesign", "organization_id": "Acme Corp"}}, "confidence": 0.9}}

Input: "Add task Review Docs to Website project"
Output: {{"intent": "create_task", "entities": {{"task_name": "Review Docs", "project_name": "Website"}}, "confidence": 0.9}}

Input: "Do I have emails from john@example.com?"
Output: {{"intent": "search_emails", "entities": {{"search_query": "john@example.com"}}, "confidence": 0.95}}

Input: "Show me my latest email"
Output: {{"intent": "get_latest_email", "entities": {{}}, "confidence": 0.9}}

Input: "yes" / "please do" / "go ahead" / "show it" (after AI offered to show email)
Output: {{"intent": "get_latest_email", "entities": {{}}, "confidence": 0.95}}

Input: "Gabriel Jones" (after "Which Gabriel did you mean?")
Output: {{"intent": "clarification_response", "entities": {{"selected_name": "Gabriel Jones"}}, "confidence": 0.95}}

Input: "yes, add email john@test.com and phone 555-1234" (after creating contact, AI offered to add more info)
Output: {{"intent": "update_contact", "entities": {{"contact_name": "him", "email": "john@test.com", "phone": "555-1234"}}, "confidence": 0.9}}

Input: "sure, his email is test@example.com" (follow-up to add info offer)
Output: {{"intent": "update_contact", "entities": {{"contact_name": "him", "email": "test@example.com"}}, "confidence": 0.9}}

Input: "add their phone 555-9876" (after discussing a contact)
Output: {{"intent": "update_contact", "entities": {{"contact_name": "them", "phone": "555-9876"}}, "confidence": 0.9}}

CRITICAL RULES:
1. Email confirmation (yes/sure/ok after email offer) → get_latest_email, NEVER extract_document_content
2. extract_document_content is ONLY for SharePoint/OneDrive files, NOT emails
3. Pronouns (him/her/it) in updates → use as contact_name, let system resolve
4. Partial names OK ("Gabriel" matches "Gabriel Kajero")
5. "How many X called Y" → search_X with search_query: "Y" """

        # Inject conversation context if available
        if recent_entities:
            context_lines = ["\n\nCONVERSATION CONTEXT (recently discussed entities):"]
            for entity_type, entities in recent_entities.items():
                if entities:
                    names = [e.get("name", "Unknown") for e in entities[:5]]
                    context_lines.append(f"- {entity_type}: {', '.join(names)}")

            context_lines.append("""
CONTEXT-AWARE RULES:
- If user mentions a name from the context above without specifying type, INFER the entity type
- Example: If "Gabriel" is in contacts context and user says "update Gabriel with email x@y.com" → update_contact
- Example: If "Acme Corp" is in organizations context and user says "delete Acme Corp" → delete_organization
- Names in context should be matched even with partial matches (e.g., "Gabriel" matches "Gabriel Kajero")
""")
            base_prompt += "\n".join(context_lines)

        return base_prompt

    def _parse_llm_response(self, response: str, original_text: str) -> Dict[str, Any]:
        """Parse and validate LLM response"""

        try:
            # Try to extract JSON from response
            response_clean = response.strip()

            # Find JSON in the response (in case LLM adds extra text)
            start_idx = response_clean.find('{')
            end_idx = response_clean.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response_clean[start_idx:end_idx]
                parsed = json.loads(json_str)

                # Validate required fields
                if "intent" in parsed and "entities" in parsed:
                    # Ensure intent is valid
                    intent = parsed["intent"]
                    if intent not in self.available_intents:
                        intent = "unknown"

                    return {
                        "intent": intent,
                        "entities": parsed.get("entities", {}),
                        "confidence": float(parsed.get("confidence", 0.7)),
                        "raw_text": original_text
                    }

            # If parsing failed, return unknown intent
            logger.warning("Failed to parse LLM response", response=response)
            return {
                "intent": "unknown",
                "entities": {"raw_response": response},
                "confidence": 0.0,
                "raw_text": original_text
            }

        except json.JSONDecodeError as e:
            logger.error("JSON parsing error", error=str(e), response=response)
            return {
                "intent": "unknown",
                "entities": {"parse_error": str(e)},
                "confidence": 0.0,
                "raw_text": original_text
            }

    def generate_api_action(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert parsed LLM result to API action

        Args:
            parsed_result: Output from parse_command

        Returns:
            Dict with API endpoint, method, and data
        """

        intent = parsed_result["intent"]
        entities = parsed_result["entities"]

        if intent == "create_contact":
            return {
                "method": "POST",
                "endpoint": "/api/v1/contacts",
                "data": {
                    "name": entities.get("name"),
                    "email": entities.get("email"),
                    "phone": entities.get("phone"),
                    "job_position": entities.get("job_position"),
                    "organization": entities.get("organization"),
                    "notes": entities.get("notes")
                },
                "description": f"Create new contact: {entities.get('name', 'Unknown')}"
            }

        elif intent == "create_organization":
            return {
                "method": "POST",
                "endpoint": "/api/v1/organizations",
                "data": {
                    "name": entities.get("name"),
                    "industry": entities.get("industry"),
                    "website": entities.get("website"),
                    "email": entities.get("email"),
                    "description": entities.get("description")
                },
                "description": f"Create new organization: {entities.get('name', 'Unknown')}"
            }

        elif intent == "update_contact":
            # Priority for identifier: contact_name > name > email
            # contact_name is the contact being updated (e.g., "John Smith")
            # name/email/phone are the NEW values being set
            identifier = entities.get("contact_name") or entities.get("name") or entities.get("email")

            # Build update data (exclude the identifier from updates)
            update_data = {k: v for k, v in entities.items() if k != "contact_name" and v is not None}

            return {
                "method": "PUT",
                "endpoint": "/api/v1/contacts/update",
                "data": update_data,
                "identifier": identifier,
                "description": f"Update contact: {identifier or 'Unknown'}"
            }

        elif intent == "update_organization":
            return {
                "method": "PUT",
                "endpoint": "/api/v1/organizations/update",
                "data": entities,
                "identifier": entities.get("name"),
                "description": f"Update organization: {entities.get('name', 'Unknown')}"
            }

        elif intent == "delete_contact":
            return {
                "method": "DELETE",
                "endpoint": "/api/v1/contacts/delete",
                "identifier": entities.get("name") or entities.get("email"),
                "description": f"Delete contact: {entities.get('name', 'Unknown')}"
            }

        elif intent == "delete_organization":
            return {
                "method": "DELETE",
                "endpoint": "/api/v1/organizations/delete",
                "identifier": entities.get("name"),
                "description": f"Delete organization: {entities.get('name', 'Unknown')}"
            }

        elif intent == "delete_project":
            return {
                "method": "DELETE",
                "endpoint": "/api/v1/projects/delete",
                "identifier": entities.get("project_name") or entities.get("name"),
                "description": f"Delete project: {entities.get('project_name') or entities.get('name', 'Unknown')}"
            }

        elif intent == "delete_task":
            return {
                "method": "DELETE",
                "endpoint": "/api/v1/tasks/delete",
                "identifier": entities.get("task_name") or entities.get("name"),
                "description": f"Delete task: {entities.get('task_name') or entities.get('name', 'Unknown')}"
            }

        elif intent == "search_contacts":
            params = {}
            if entities.get("search_query"):
                params["search"] = entities["search_query"]
            if entities.get("organization"):
                params["organization"] = entities["organization"]

            return {
                "method": "GET",
                "endpoint": "/api/v1/contacts",
                "params": params,
                "description": f"Search contacts: {entities.get('search_query', 'all')}"
            }

        elif intent == "list_contacts":
            return {
                "method": "GET",
                "endpoint": "/api/v1/contacts",
                "params": {},
                "description": "List all contacts"
            }

        elif intent == "create_project":
            # Convert relative dates
            due_date = self._parse_date(entities.get("due_date"))
            start_date = self._parse_date(entities.get("start_date"))

            return {
                "method": "POST",
                "endpoint": "/api/v1/projects",
                "data": {
                    "name": entities.get("project_name") or entities.get("name"),
                    "description": entities.get("description"),
                    "status": entities.get("status", "planned"),
                    "priority": entities.get("priority", "medium"),
                    "due_date": due_date,
                    "start_date": start_date,
                    "organization_id": entities.get("organization_id"),
                    "notes": entities.get("notes")
                },
                "description": f"Create new project: {entities.get('project_name') or entities.get('name', 'Unknown')}"
            }

        elif intent == "update_project":
            # Convert relative dates
            due_date = self._parse_date(entities.get("due_date"))
            start_date = self._parse_date(entities.get("start_date"))

            return {
                "method": "PUT",
                "endpoint": "/api/v1/projects/update",
                "data": {
                    "name": entities.get("project_name") or entities.get("name"),
                    "description": entities.get("description"),
                    "status": entities.get("status"),
                    "priority": entities.get("priority"),
                    "due_date": due_date,
                    "start_date": start_date,
                    "organization_id": entities.get("organization_id"),
                    "notes": entities.get("notes")
                },
                "identifier": entities.get("project_name") or entities.get("name"),
                "description": f"Update project: {entities.get('project_name') or entities.get('name', 'Unknown')}"
            }

        elif intent == "create_task":
            # Convert relative dates
            due_date = self._parse_date(entities.get("due_date"))

            return {
                "method": "POST",
                "endpoint": "/api/v1/tasks",
                "data": {
                    "name": entities.get("task_name") or entities.get("name"),
                    "description": entities.get("description"),
                    "status": entities.get("status", "pending"),
                    "priority": entities.get("priority", "medium"),
                    "due_date": due_date,
                    "assignee": entities.get("assignee"),
                    "project_name": entities.get("project_name"),
                    "notes": entities.get("notes")
                },
                "description": f"Create new task: {entities.get('task_name') or entities.get('name', 'Unknown')}"
            }

        elif intent == "update_task":
            # Convert relative dates
            due_date = self._parse_date(entities.get("due_date"))

            return {
                "method": "PUT",
                "endpoint": "/api/v1/tasks/update",
                "data": {
                    "name": entities.get("task_name") or entities.get("name"),
                    "description": entities.get("description"),
                    "status": entities.get("status"),
                    "priority": entities.get("priority"),
                    "due_date": due_date,
                    "assignee": entities.get("assignee"),
                    "project_name": entities.get("project_name"),
                    "notes": entities.get("notes")
                },
                "identifier": entities.get("task_name") or entities.get("name"),
                "description": f"Update task: {entities.get('task_name') or entities.get('name', 'Unknown')}"
            }

        elif intent == "search_organizations":
            params = {}
            if entities.get("search_query"):
                params["search"] = entities["search_query"]
            if entities.get("industry"):
                params["industry"] = entities["industry"]

            return {
                "method": "GET",
                "endpoint": "/api/v1/organizations",
                "params": params,
                "description": f"Search organizations: {entities.get('search_query', 'all')}"
            }

        elif intent == "list_organizations":
            return {
                "method": "GET",
                "endpoint": "/api/v1/organizations",
                "params": {},
                "description": "List all organizations"
            }

        elif intent == "search_projects":
            params = {}
            if entities.get("search_query"):
                params["search"] = entities["search_query"]
            if entities.get("status"):
                params["status"] = entities["status"]
            if entities.get("priority"):
                params["priority"] = entities["priority"]

            return {
                "method": "GET",
                "endpoint": "/api/v1/projects",
                "params": params,
                "description": f"Search projects: {entities.get('search_query', 'all')}"
            }

        elif intent == "list_projects":
            return {
                "method": "GET",
                "endpoint": "/api/v1/projects",
                "params": {},
                "description": "List all projects"
            }

        elif intent == "search_tasks":
            params = {}
            if entities.get("search_query"):
                params["search"] = entities["search_query"]
            if entities.get("status"):
                params["status"] = entities["status"]
            if entities.get("priority"):
                params["priority"] = entities["priority"]
            if entities.get("assignee"):
                params["assignee"] = entities["assignee"]

            return {
                "method": "GET",
                "endpoint": "/api/v1/tasks",
                "params": params,
                "description": f"Search tasks: {entities.get('search_query', 'all')}"
            }

        elif intent == "list_tasks":
            return {
                "method": "GET",
                "endpoint": "/api/v1/tasks",
                "params": {},
                "description": "List all tasks"
            }

        elif intent == "export_data":
            export_type = "contacts"  # default
            if "organization" in entities.get("search_query", "").lower():
                export_type = "organizations"

            endpoint = f"/api/v1/{export_type}/export/csv"
            return {
                "method": "GET",
                "endpoint": endpoint,
                "params": {},
                "description": f"Export {export_type} to CSV"
            }

        elif intent == "clarification_response":
            # This is a follow-up response to a clarification request
            # We'll handle this specially in the chat handler
            return {
                "method": "CLARIFICATION",
                "endpoint": "CLARIFICATION",
                "data": entities,
                "description": f"Clarification response: {entities}"
            }

        # Microsoft 365 MCP Tool Integrations
        elif intent == "send_email":
            # Convert single email to list if needed (MCP server expects arrays)
            to_list = entities.get("email_to")
            if to_list and isinstance(to_list, str):
                to_list = [to_list]

            cc_list = entities.get("email_cc")
            if cc_list and isinstance(cc_list, str):
                cc_list = [cc_list]

            return {
                "method": "MCP_TOOL",
                "tool_name": "outlook_send_email",
                "parameters": {
                    "to_recipients": to_list,  # MCP server expects array
                    "subject": entities.get("email_subject"),
                    "body": entities.get("email_body"),
                    "cc_recipients": cc_list,  # MCP server expects array
                    "importance": entities.get("priority", "normal")
                },
                "description": f"Send email to {entities.get('email_to', 'recipient')}",
                "raw_text": parsed_result.get("raw_text", "")  # For AI content generation detection
            }

        elif intent == "search_emails":
            # Detect if user wants recent emails
            message_lower = ""
            params = {
                "query": entities.get("search_query"),
                "folder": entities.get("email_folder", "inbox"),
                "limit": entities.get("limit", 20),
                "unread_only": entities.get("unread_only", False)
            }

            # Add date filter for recent requests (last 7 days)
            if not entities.get("from_date"):
                search_query_lower = (entities.get("search_query") or "").lower()
                is_recent_request = any(word in search_query_lower for word in [
                    "recent", "latest", "new", "today", "this week", "last"
                ])

                if is_recent_request:
                    from datetime import datetime, timedelta
                    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                    params["from_date"] = seven_days_ago
                    logger.info(f"Added date filter for recent request: from_date={seven_days_ago}")
            else:
                params["from_date"] = entities.get("from_date")

            return {
                "method": "MCP_TOOL",
                "tool_name": "outlook_search_emails",
                "parameters": params,
                "description": f"Search emails: {entities.get('search_query', 'all')}"
            }

        elif intent == "read_email":
            return {
                "method": "MCP_TOOL",
                "tool_name": "outlook_get_email",
                "parameters": {
                    "email_id": entities.get("email_id"),
                    "include_attachments": True
                },
                "description": f"Read email: {entities.get('email_id', 'unknown')}"
            }
        elif intent == "get_latest_email":
            # Check if there's a tracked email from conversation memory
            # NOTE: We cannot access per-user memory here because generate_api_action
            # doesn't receive user_id. The tracking will be handled in chat.py instead.
            # For now, don't try to retrieve tracked email ID here.
            tracked_email_id = None  # Will be set by chat.py if needed

            return {
                "method": "MCP_TOOL",
                "tool_name": "outlook_get_email",
                "parameters": {
                    "email_id": tracked_email_id,  # Will be injected by chat.py
                    "include_attachments": True
                },
                "description": "Get latest email"
            }
        elif intent == "get_new_emails":
            return {
                "method": "MCP_TOOL",
                "tool_name": "outlook_search_emails",
                "parameters": {
                    "query": None,
                    "folder": "inbox",
                    "limit": 20,
                    "unread_only": True  # New parameter for filtering unread emails
                },
                "description": "Get unread/new emails"
            }

        elif intent == "create_draft":
            return {
                "method": "MCP_TOOL",
                "tool_name": "outlook_create_draft",
                "parameters": {
                    "to": entities.get("email_to"),
                    "subject": entities.get("email_subject"),
                    "body": entities.get("email_body"),
                    "cc": entities.get("email_cc"),
                    "bcc": entities.get("email_bcc")
                },
                "description": f"Create email draft to {entities.get('email_to', 'recipient')}"
            }

        elif intent == "list_folders":
            return {
                "method": "MCP_TOOL",
                "tool_name": "outlook_list_folders",
                "parameters": {},
                "description": "List email folders"
            }

        elif intent == "get_calendar_events":
            return {
                "method": "MCP_TOOL",
                "tool_name": "outlook_get_calendar_events",
                "parameters": {
                    "start_date": self._parse_date(entities.get("meeting_start")) or self._parse_date(entities.get("search_query")),
                    "limit": 50
                },
                "description": f"Get calendar events: {entities.get('search_query', 'all')}"
            }

        elif intent in ["create_meeting", "schedule_meeting"]:
            meeting_start = self._parse_date(entities.get("meeting_start"))
            meeting_end = self._parse_date(entities.get("meeting_end"))

            return {
                "method": "MCP_TOOL",
                "tool_name": "teams_create_meeting",
                "parameters": {
                    "subject": entities.get("meeting_subject") or "Meeting",
                    "start_time": meeting_start,
                    "end_time": meeting_end,
                    "attendees": entities.get("meeting_attendees"),
                    "location": entities.get("meeting_location")
                },
                "description": f"Create meeting: {entities.get('meeting_subject', 'Meeting')}"
            }

        elif intent == "join_meeting":
            return {
                "method": "MCP_TOOL",
                "tool_name": "teams_join_meeting",
                "parameters": {
                    "meeting_id": entities.get("meeting_id")
                },
                "description": f"Join meeting: {entities.get('meeting_id', 'unknown')}"
            }

        elif intent == "list_sharepoint_sites":
            return {
                "method": "MCP_TOOL",
                "tool_name": "sharepoint_list_sites",
                "parameters": {},
                "description": "List SharePoint sites"
            }

        elif intent == "search_sharepoint_documents":
            return {
                "method": "MCP_TOOL",
                "tool_name": "sharepoint_search_documents",
                "parameters": {
                    "query": entities.get("search_query"),
                    "site_name": entities.get("site_name"),
                    "library_name": entities.get("library_name"),
                    "limit": 50
                },
                "description": f"Search SharePoint documents: {entities.get('search_query', 'all')}"
            }

        elif intent == "upload_sharepoint_document":
            return {
                "method": "MCP_TOOL",
                "tool_name": "sharepoint_upload_document",
                "parameters": {
                    "file_path": entities.get("document_path"),
                    "site_name": entities.get("site_name"),
                    "library_name": entities.get("library_name"),
                    "folder_name": entities.get("folder_name")
                },
                "description": f"Upload document to SharePoint: {entities.get('document_path', 'file')}"
            }

        elif intent == "share_document":
            return {
                "method": "MCP_TOOL",
                "tool_name": "sharepoint_share_document",
                "parameters": {
                    "document_path": entities.get("document_path"),
                    "site_name": entities.get("site_name"),
                    "permission": entities.get("permission", "view")
                },
                "description": f"Share document: {entities.get('document_path', 'file')}"
            }

        elif intent == "list_onedrive_files":
            return {
                "method": "MCP_TOOL",
                "tool_name": "onedrive_list_files",
                "parameters": {
                    "folder_path": entities.get("folder_name", "/"),
                    "limit": 50
                },
                "description": f"List OneDrive files in {entities.get('folder_name', 'root')}"
            }

        elif intent == "upload_onedrive_file":
            return {
                "method": "MCP_TOOL",
                "tool_name": "onedrive_upload_document",
                "parameters": {
                    "file_path": entities.get("document_path"),
                    "folder_path": entities.get("folder_name", "/")
                },
                "description": f"Upload file to OneDrive: {entities.get('document_path', 'file')}"
            }

        elif intent == "list_teams":
            return {
                "method": "MCP_TOOL",
                "tool_name": "teams_list_teams",
                "parameters": {},
                "description": "List Microsoft Teams"
            }

        elif intent == "list_team_files":
            return {
                "method": "MCP_TOOL",
                "tool_name": "teams_list_files",
                "parameters": {
                    "team_name": entities.get("team_name"),
                    "channel_name": entities.get("channel_name")
                },
                "description": f"List files in team: {entities.get('team_name', 'unknown')}"
            }

        elif intent == "extract_document_content":
            return {
                "method": "MCP_TOOL",
                "tool_name": "extract_document_content_stateless",
                "parameters": {
                    "document_path": entities.get("document_path"),
                    "file_type": entities.get("file_type", "auto")
                },
                "description": f"Extract content from: {entities.get('document_path', 'document')}"
            }

        elif intent in ["connect_microsoft365", "authenticate_microsoft365"]:
            return {
                "method": "MCP_TOOL",
                "tool_name": "authenticate",
                "parameters": {},
                "description": "Connect Microsoft 365 account"
            }

        else:
            return {
                "method": "UNKNOWN",
                "endpoint": "UNKNOWN",
                "data": entities,
                "description": f"Intent '{intent}' not yet implemented"
            }

    def _parse_date(self, date_input: str) -> Optional[str]:
        """Parse relative and absolute dates into ISO format"""
        if not date_input or not isinstance(date_input, str):
            return None

        date_input = date_input.lower().strip()

        try:
            # Handle relative dates
            if date_input in ['today']:
                return datetime.now().date().isoformat()
            elif date_input in ['tomorrow']:
                return (datetime.now() + timedelta(days=1)).date().isoformat()
            elif date_input in ['yesterday']:
                return (datetime.now() - timedelta(days=1)).date().isoformat()
            elif 'next week' in date_input:
                return (datetime.now() + timedelta(weeks=1)).date().isoformat()
            elif 'next month' in date_input:
                return (datetime.now() + relativedelta(months=1)).date().isoformat()
            elif 'next year' in date_input:
                return (datetime.now() + relativedelta(years=1)).date().isoformat()

            # Handle "in X days/weeks/months" patterns
            time_patterns = [
                r'in (\d+) days?',
                r'in (\d+) weeks?',
                r'in (\d+) months?'
            ]

            for i, pattern in enumerate(time_patterns):
                match = re.search(pattern, date_input)
                if match:
                    num = int(match.group(1))
                    if i == 0:  # days
                        return (datetime.now() + timedelta(days=num)).date().isoformat()
                    elif i == 1:  # weeks
                        return (datetime.now() + timedelta(weeks=num)).date().isoformat()
                    elif i == 2:  # months
                        return (datetime.now() + relativedelta(months=num)).date().isoformat()

            # Handle specific day names (next Monday, etc.)
            weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            for i, day in enumerate(weekdays):
                if day in date_input:
                    current_weekday = datetime.now().weekday()
                    days_ahead = i - current_weekday
                    if days_ahead <= 0:  # Target day already happened this week
                        days_ahead += 7
                    return (datetime.now() + timedelta(days=days_ahead)).date().isoformat()

            # Try to parse absolute dates
            parsed_date = parse_date(date_input, fuzzy=True)
            return parsed_date.date().isoformat()

        except Exception as e:
            logger.warning("Failed to parse date", date_input=date_input, error=str(e))
            return None

    def _is_follow_up_question(self, text: str, conversation_history: List[Dict[str, str]]) -> bool:
        """
        Detect if this is a follow-up question using semantic understanding.

        This method uses a combination of:
        1. Quick checks for obvious action patterns (retrieval, CRUD)
        2. Semantic analysis for follow-up questions about previous data

        Args:
            text: Current user message
            conversation_history: Previous conversation messages

        Returns:
            True if this is a follow-up question, False otherwise
        """
        text_lower = text.lower().strip()

        # =====================================================
        # QUICK PATH: Detect obvious ACTIONS (not follow-ups)
        # =====================================================

        # 1. Retrieval actions - user confirming they want to retrieve something
        retrieval_patterns = [
            "retrieve it", "retrieve", "get it", "get the full",
            "show me the full", "show me the body", "show the body",
            "yes please", "yes get", "yes retrieve", "yes show",
            "go ahead", "please retrieve", "please get"
        ]

        # Short affirmative responses after AI offer = action, not follow-up
        if text_lower in ["yes", "sure", "ok", "okay", "yeah", "yep", "please", "do it", "go ahead"]:
            if conversation_history and len(conversation_history) > 0:
                recent = conversation_history[-3:]
                for msg in recent:
                    content = msg.get("content", "").lower()
                    if any(ind in content for ind in [
                        "retrieve", "full email", "full body", "would you like",
                        "do you want me to", "shall i", "want me to"
                    ]):
                        return False  # This is confirming an action

        # Explicit retrieval action
        if any(p in text_lower for p in retrieval_patterns):
            return False  # Parse as action

        # 2. CRUD action patterns - these should be parsed, not treated as follow-ups
        crud_action_words = [
            "create", "add", "new", "make", "register",
            "update", "change", "modify", "edit", "set",
            "delete", "remove", "drop", "erase",
            "list", "show", "display", "get", "fetch"  # List/retrieval actions
        ]

        # If message starts with or contains strong action verbs, it's likely an action
        for action in crud_action_words:
            if text_lower.startswith(action) or f" {action} " in f" {text_lower} ":
                # Check if it's referencing something from context (e.g., "delete him")
                if not any(ref in text_lower for ref in ["him", "her", "it", "that", "this", "the first", "the second"]):
                    return False  # Likely a new action, not a follow-up

        # =====================================================
        # SEMANTIC PATH: Detect follow-up questions
        # =====================================================

        # Check for pronoun/reference patterns that indicate follow-up
        follow_up_patterns = [
            # Pronouns referring to entities
            r"\b(it|that|this|those|these|them)\b",
            # Ordinal references
            r"\b(first|second|third|last|latest|previous)\s*(one|email|contact|project|task|organization)?\b",
            # Possessive references
            r"\b(his|her|their|its)\s+(email|phone|name|address|status)\b",
            # "The" + entity type (the email, the contact)
            r"\bthe\s+(email|contact|project|task|organization|one)\b",
            # Content questions about something discussed
            r"\b(tell me more|what does it|what's in|summarize|explain|details about)\b",
            # About/from references
            r"\b(about that|about the|from that|from the|about it)\b",
        ]

        has_reference_pattern = any(re.search(p, text_lower) for p in follow_up_patterns)

        if not has_reference_pattern:
            return False  # No follow-up indicators found

        # Verify there's relevant context in conversation history
        if not conversation_history or len(conversation_history) == 0:
            return False

        # Look for entity data in recent messages
        recent_messages = conversation_history[-5:]
        conversation_text = " ".join(msg.get("content", "").lower() for msg in recent_messages)

        # Check if conversation has discussed entities
        entity_evidence = [
            "email", "contact", "project", "task", "organization",
            "subject:", "from:", "name:", "status:", "priority:",
            "found", "retrieved", "showing", "created", "updated"
        ]

        has_entity_context = any(evidence in conversation_text for evidence in entity_evidence)

        if has_entity_context:
            logger.debug(
                "Follow-up detected: reference pattern with entity context",
                text=text[:50],
                has_pattern=has_reference_pattern
            )
            return True

        return False


# Global LLM command parser instance
llm_command_parser = LLMCommandParser()