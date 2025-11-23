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
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Parse natural language command using LLM

        Args:
            text: Natural language command from user
            user_context: Optional context about the user
            conversation_history: Optional conversation history for follow-up detection

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
        prompt = self._build_extraction_prompt()

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

    def _build_extraction_prompt(self) -> str:
        """Build cached prompt for LLM to extract intent and entities"""

        # Use cached prompt if available
        if self.cached_system_prompt:
            return self.cached_system_prompt

        prompt = """You are an expert at understanding CRM (Customer Relationship Management) commands.

Your task is to extract the INTENT and ENTITIES from user commands, using natural language understanding.

CRITICAL: Recognize natural language variations for actions:
- CREATE: "add", "create", "new", "register", "make", "set up"
- UPDATE: "update", "change", "modify", "edit", "alter", "fix", "correct", "set"
- DELETE: "delete", "remove", "drop", "erase", "get rid of", "eliminate"
- SEARCH: "find", "search", "look for", "show me", "get", "retrieve", "display"
- LIST: "list", "show all", "display all", "what are my", "how many"

IMPORTANT: Commands without explicit action words should infer intent from context:
- "Luke" or "Delete Luke" or "Remove Luke" → delete_contact (if Luke exists in context)
- "Mike's email is new@email.com" → update_contact
- "Add John Smith" → create_contact
- "Contact for Sarah" → create_contact

Available intents:
- create_contact: User wants to add a new contact
- update_contact: User wants to modify existing contact
- delete_contact: User wants to remove a contact
- search_contacts: User wants to find specific contacts
- list_contacts: User wants to see all/filtered contacts
- create_organization: User wants to add a new company/organization
- update_organization: User wants to modify existing organization
- delete_organization: User wants to remove an organization
- search_organizations: User wants to find specific organizations
- list_organizations: User wants to see all/filtered organizations
- create_project: User wants to create a new project
- update_project: User wants to modify existing project
- delete_project: User wants to remove a project
- search_projects: User wants to find projects
- list_projects: User wants to see all/filtered projects
- create_task: User wants to create a new task
- update_task: User wants to modify existing task
- delete_task: User wants to remove a task
- search_tasks: User wants to find tasks
- list_tasks: User wants to see all/filtered tasks
- export_data: User wants to export data to file
- get_stats: User wants statistics/analytics
- help: User needs help/instructions
- unknown: Intent cannot be determined
- send_email: User wants to send an email
- search_emails: User wants to find, view, list, show, retrieve, or access emails (includes generic requests like "show my emails", "get my recent emails")
- read_email: User wants to read a specific email
- get_latest_email: User specifically wants the most recent, latest, or newest email
- get_new_emails: User wants unread, new, or unseen emails
- create_draft: User wants to create an email draft
- list_folders: User wants to see email folders
- get_calendar_events: User wants to see calendar events
- create_meeting: User wants to create/schedule a meeting
- schedule_meeting: User wants to schedule a meeting (alias for create_meeting)
- join_meeting: User wants to join a meeting
- list_sharepoint_sites: User wants to see SharePoint sites
- search_sharepoint_documents: User wants to find SharePoint documents
- upload_sharepoint_document: User wants to upload a file to SharePoint
- share_document: User wants to share a document
- list_onedrive_files: User wants to see OneDrive files
- upload_onedrive_file: User wants to upload a file to OneDrive
- list_teams: User wants to see Microsoft Teams
- list_team_files: User wants to see files in Teams channels
- extract_document_content: User wants to extract text from a document
- connect_microsoft365: User wants to connect Microsoft 365 account
- authenticate_microsoft365: User wants to authenticate with Microsoft 365

CRITICAL: Distinguish between list and search intents:
- list_* : User wants ALL items or a count of ALL items (no specific criteria)
- search_* : User wants to find SPECIFIC items with criteria/filters

Examples:
- "How many contacts do I have" → list_contacts (wants total count of all contacts)
- "How many contacts called Gabriel" → search_contacts (wants filtered count with criteria)
- "Show all organizations" → list_organizations (wants all organizations)
- "Show tech organizations" → search_organizations (wants filtered organizations)
- "List my projects" → list_projects (wants all user's projects)
- "Find completed projects" → search_projects (wants projects with status filter)

For ENTITIES, extract these fields when relevant:
- name: Person's full name
- email: Email address
- phone: Phone number
- job_position: Job title/position
- organization: Company/organization name
- industry: Business industry/sector
- website: Website URL
- search_query: What to search for
- export_format: csv, excel, etc.
- project_name: Name of project
- task_name: Name of task
- due_date: Date information (absolute dates or relative like 'tomorrow', 'next week')
- start_date: Start date information
- priority: high, medium, low, urgent
- status: For projects: planned, in_progress, completed, on_hold, cancelled
- status: For tasks: pending, in_progress, completed, blocked, cancelled
- assignee: Person assigned to task
- description: Detailed description
- notes: Additional notes
- organization_id: Organization associated with project
- project_id: Project associated with task
- email_to: Email recipient address
- email_cc: Email CC recipients
- email_bcc: Email BCC recipients
- email_subject: Email subject line
- email_body: Email message content
- email_folder: Email folder name (inbox, sent, etc.)
- meeting_subject: Meeting title/subject
- meeting_start: Meeting start time
- meeting_end: Meeting end time
- meeting_attendees: Meeting participants
- meeting_location: Meeting location or Teams link
- document_path: File or document path
- folder_name: Folder or directory name
- file_name: Name of file
- search_query: Search terms for finding items
- site_name: SharePoint site name
- library_name: Document library name
- team_name: Microsoft Teams team name
- channel_name: Teams channel name

IMPORTANT: Respond with ONLY a valid JSON object in this exact format:
{{
  "intent": "intent_name",
  "entities": {{
    "field_name": "extracted_value"
  }},
  "confidence": 0.95
}}

Examples:

Input: "Create a contact named Luke with email luke@email.com"
Output: {{
  "intent": "create_contact",
  "entities": {{
    "name": "Luke",
    "email": "luke@email.com"
  }},
  "confidence": 0.95
}}

Input: "Create a contact called Wesley Waka, his phone number is 1234567890"
Output: {{
  "intent": "create_contact",
  "entities": {{
    "name": "Wesley Waka",
    "phone": "1234567890"
  }},
  "confidence": 0.95
}}

Input: "Add Sarah Johnson as a contact, her email is sarah@email.com"
Output: {{
  "intent": "create_contact",
  "entities": {{
    "name": "Sarah Johnson",
    "email": "sarah@email.com"
  }},
  "confidence": 0.95
}}

Input: "Update Gabriel's phone number to 123456789"
Output: {{
  "intent": "update_contact",
  "entities": {{
    "name": "Gabriel",
    "phone": "123456789"
  }},
  "confidence": 0.9
}}

Input: "update his phone to 555-1234"
Output: {{
  "intent": "update_contact",
  "entities": {{
    "name": "him",
    "phone": "555-1234"
  }},
  "confidence": 0.9
}}

Input: "add his email which is test@example.com"
Output: {{
  "intent": "update_contact",
  "entities": {{
    "name": "him",
    "email": "test@example.com"
  }},
  "confidence": 0.9
}}

Input: "update his name to Wesley Waka"
Output: {{
  "intent": "update_contact",
  "entities": {{
    "name": "Wesley Waka"
  }},
  "confidence": 0.9
}}

Input: "change her name to Sarah Miller"
Output: {{
  "intent": "update_contact",
  "entities": {{
    "name": "Sarah Miller"
  }},
  "confidence": 0.9
}}

Input: "Delete Luke"
Output: {{
  "intent": "delete_contact",
  "entities": {{
    "name": "Luke"
  }},
  "confidence": 0.95
}}

Input: "Search for contacts named John"
Output: {{
  "intent": "search_contacts",
  "entities": {{
    "search_query": "John"
  }},
  "confidence": 0.9
}}

Input: "Create organization Acme Corp"
Output: {{
  "intent": "create_organization",
  "entities": {{
    "name": "Acme Corp"
  }},
  "confidence": 0.9
}}

Input: "update its website to example.com"
Output: {{
  "intent": "update_organization",
  "entities": {{
    "name": "it",
    "website": "example.com"
  }},
  "confidence": 0.9
}}

Input: "Delete Acme Corp organization"
Output: {{
  "intent": "delete_organization",
  "entities": {{
    "name": "Acme Corp"
  }},
  "confidence": 0.9
}}

Input: "Create project Website Redesign for Acme Corp"
Output: {{
  "intent": "create_project",
  "entities": {{
    "project_name": "Website Redesign",
    "organization_id": "Acme Corp"
  }},
  "confidence": 0.9
}}

Input: "Update Website Redesign status to completed"
Output: {{
  "intent": "update_project",
  "entities": {{
    "project_name": "Website Redesign",
    "status": "completed"
  }},
  "confidence": 0.9
}}

Input: "Delete Website Redesign project"
Output: {{
  "intent": "delete_project",
  "entities": {{
    "project_name": "Website Redesign"
  }},
  "confidence": 0.9
}}

Input: "Create task Review Documentation"
Output: {{
  "intent": "create_task",
  "entities": {{
    "task_name": "Review Documentation"
  }},
  "confidence": 0.9
}}

Input: "Update task status to completed"
Output: {{
  "intent": "update_task",
  "entities": {{
    "status": "completed"
  }},
  "confidence": 0.85
}}

Input: "Do I have any emails from momoagoumar@gmail.com?"
Output: {{
  "intent": "search_emails",
  "entities": {{
    "search_query": "momoagoumar@gmail.com"
  }},
  "confidence": 0.95
}}

Input: "Show me my latest email"
Output: {{
  "intent": "get_latest_email",
  "entities": {{}},
  "confidence": 0.9
}}

Input: "Any new emails?"
Output: {{
  "intent": "get_new_emails",
  "entities": {{}},
  "confidence": 0.9
}}

Input: "Send email to john@example.com"
Output: {{
  "intent": "send_email",
  "entities": {{
    "email_to": "john@example.com"
  }},
  "confidence": 0.9
}}

Input: "Gabriel Jones"
Output: {{
  "intent": "clarification_response",
  "entities": {{
    "selected_name": "Gabriel Jones"
  }},
  "confidence": 0.95
}}

Input: "yes" (after being asked if user wants full email retrieval)
Output: {{
  "intent": "get_latest_email",
  "entities": {{}},
  "confidence": 0.85
}}

Input: "yes" (after being asked if user wants email details)
Output: {{
  "intent": "get_latest_email",
  "entities": {{}},
  "confidence": 0.95
}}

Input: "yes please" (after being offered to show email)
Output: {{
  "intent": "get_latest_email",
  "entities": {{}},
  "confidence": 0.95
}}

Input: "sure" (after email retrieval offer)
Output: {{
  "intent": "get_latest_email",
  "entities": {{}},
  "confidence": 0.9
}}

Input: "ok" (after being asked about showing email)
Output: {{
  "intent": "get_latest_email",
  "entities": {{}},
  "confidence": 0.9
}}

Input: "yeah" (after email summary offer)
Output: {{
  "intent": "get_latest_email",
  "entities": {{}},
  "confidence": 0.9
}}

Input: "show it" (after finding emails)
Output: {{
  "intent": "get_latest_email",
  "entities": {{}},
  "confidence": 0.95
}}

IMPORTANT NOTES:
- For updates/searches, accept PARTIAL NAMES (e.g., "Gabriel" could match "Gabriel Kajero")
- Extract the most specific identifier available (name, email, project name, etc.)
- Be confident about common update/search patterns even with partial information
- When updating, include the identifier field and the fields being updated
- For counting/filtering questions, extract the entity name being searched for
- Questions like "How many X called Y" → search_X with search_query: "Y"
- Questions like "Count X with status Y" → search_X with status: "Y"
- For clarification responses after multiple matches, detect simple name responses
- "Gabriel Jones" after asking "Which Gabriel?" → clarification_response with selected_name"""

        self.cached_system_prompt = prompt
        return prompt

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
            return {
                "method": "PUT",
                "endpoint": "/api/v1/contacts/update",
                "data": entities,
                "identifier": entities.get("name") or entities.get("email"),
                "description": f"Update contact: {entities.get('name', 'Unknown')}"
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
            return {
                "method": "MCP_TOOL",
                "tool_name": "outlook_send_email",
                "parameters": {
                    "to": entities.get("email_to"),
                    "subject": entities.get("email_subject"),
                    "body": entities.get("email_body"),
                    "cc": entities.get("email_cc"),
                    "bcc": entities.get("email_bcc"),
                    "importance": entities.get("priority", "normal")
                },
                "description": f"Send email to {entities.get('email_to', 'recipient')}"
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
        Detect if this is a follow-up question to previous conversation.

        Enhanced version that tracks:
        - Email references
        - Contact references
        - Project/task references
        - Pronoun usage ("it", "that one", "the first one")
        - Generic follow-up patterns

        Args:
            text: Current user message
            conversation_history: Previous conversation messages

        Returns:
            True if this is a follow-up question, False otherwise
        """
        # Check if this is an email retrieval ACTION (not a follow-up question)
        # User says "yes", "retrieve it", "get it", etc. after AI offered to retrieve
        retrieval_action_indicators = [
            "retrieve it", "retrieve", "get it", "get the full",
            "show me the full", "show me the body", "show the body",
            "yes please", "yes get", "yes retrieve", "yes show",
            "ok retrieve", "ok get", "okay get", "sure retrieve",
            "go ahead", "please retrieve", "please get"
        ]

        text_lower = text.lower().strip()

        # Short affirmative responses after email discussion = likely retrieval action
        if text_lower in ["yes", "sure", "ok", "okay", "yeah", "yep", "please", "do it"]:
            # Check if conversation mentioned emails recently
            if conversation_history and len(conversation_history) > 0:
                recent_messages = conversation_history[-3:]
                for msg in recent_messages:
                    content = msg.get("content", "").lower()
                    if any(indicator in content for indicator in [
                        "retrieve", "full email", "full body", "full content",
                        "body preview", "truncated", "would you like me to retrieve"
                    ]):
                        # This is likely "yes" to retrieval offer, NOT a follow-up question
                        return False  # Don't treat as follow-up, let it be parsed as action

        # Check if this is an explicit retrieval action
        is_retrieval_action = any(indicator in text_lower for indicator in retrieval_action_indicators)
        if is_retrieval_action:
            # Check if conversation mentioned emails recently
            if conversation_history and len(conversation_history) > 0:
                recent_messages = conversation_history[-3:]
                for msg in recent_messages:
                    content = msg.get("content", "").lower()
                    if any(indicator in content for indicator in ["email", "subject:", "from:", "inbox", "message"]):
                        # This is an email retrieval action, not a follow-up question
                        return False  # Don't treat as follow-up, let it be parsed as action

        # Universal follow-up indicators (applies to all entity types)
        follow_up_indicators = [
            # Email-specific (but NOT retrieval actions)
            "that email", "the email", "this email", "those emails",
            # Contact-specific
            "that contact", "the contact", "this person", "him", "her",
            # Project/task-specific
            "that project", "the project", "that task", "the task",
            # Organization-specific
            "that organization", "that company", "the organization",
            # Generic references
            "that one", "this one", "it", "that", "this", "those",
            "the one", "them", "the first one", "first one", "the second",
            "the last one", "last one", "latest one",
            # Content requests (passive, not active retrieval)
            "tell me more", "what does it say",
            "what's in it", "what about", "summarize it",
            "what's the content", "what is it",
            "from that", "from the", "about that", "about the", "about it",
            # Follow-up actions
            "more info", "additional details", "what else", "anything else",
            "can you tell me", "do you know", "what do you know about"
        ]

        # Check if message contains follow-up indicators
        has_indicator = any(indicator in text_lower for indicator in follow_up_indicators)

        if not has_indicator:
            return False

        # Check if conversation history mentions relevant entities in recent messages
        if conversation_history and len(conversation_history) > 0:
            # Look at last 5 messages for entity-related content
            recent_messages = conversation_history[-5:]

            for msg in recent_messages:
                content = msg.get("content", "").lower()

                # Check if previous messages were about emails
                email_indicators = [
                    "email", "subject:", "from:", "inbox", "message",
                    "received_date", "body content", "outlook", "sender"
                ]

                # Check if previous messages were about contacts
                contact_indicators = [
                    "contact", "person", "name:", "phone:", "email:",
                    "job_position", "organization:"
                ]

                # Check if previous messages were about projects
                project_indicators = [
                    "project", "status:", "priority:", "due date:",
                    "planned", "in progress", "completed"
                ]

                # Check if previous messages were about tasks
                task_indicators = [
                    "task", "assignee:", "priority:", "pending",
                    "blocked", "cancelled"
                ]

                # Check if previous messages were about organizations
                org_indicators = [
                    "organization", "company", "industry:", "website:"
                ]

                # Data payload indicators (suggests data was recently shown)
                data_indicators = [
                    "query result", "exact count", "found", "showing",
                    "retrieved", "created successfully", "updated successfully"
                ]

                all_indicators = (
                    email_indicators + contact_indicators + project_indicators +
                    task_indicators + org_indicators + data_indicators
                )

                if any(indicator in content for indicator in all_indicators):
                    logger.debug(
                        "Follow-up detected: found entity reference in recent messages",
                        text=text,
                        matched_in_history=True
                    )
                    return True

        return False


# Global LLM command parser instance
llm_command_parser = LLMCommandParser()