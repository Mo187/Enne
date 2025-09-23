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
            "clarification_response"
        ]

        # Cache key for prompt caching
        self.cached_system_prompt = None

    async def parse_command(self, text: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Parse natural language command using LLM

        Args:
            text: Natural language command from user
            user_context: Optional context about the user

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

        # Force rebuild to include new examples
        if False and self.cached_system_prompt:
            return self.cached_system_prompt

        prompt = """You are an expert at understanding CRM (Customer Relationship Management) commands.

Your task is to extract the INTENT and ENTITIES from user commands.

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

IMPORTANT: Respond with ONLY a valid JSON object in this exact format:
{{
  "intent": "intent_name",
  "entities": {{
    "field_name": "extracted_value"
  }},
  "confidence": 0.95
}}

Examples:

Input: "Can you create a contact for Mike Ogego. He's Enne Co-founder, his number is 079990000."
Output: {{
  "intent": "create_contact",
  "entities": {{
    "name": "Mike Ogego",
    "organization": "Enne",
    "job_position": "Co-founder",
    "phone": "079990000"
  }},
  "confidence": 0.95
}}

Input: "Add a company called Acme Inc in the technology industry"
Output: {{
  "intent": "create_organization",
  "entities": {{
    "name": "Acme Inc",
    "industry": "technology"
  }},
  "confidence": 0.9
}}

Input: "Create a Project called Website Redesign, Due date should be tomorrow"
Output: {{
  "intent": "create_project",
  "entities": {{
    "project_name": "Website Redesign",
    "due_date": "tomorrow"
  }},
  "confidence": 0.95
}}

Input: "Add task Review wireframes to Website Redesign project with high priority"
Output: {{
  "intent": "create_task",
  "entities": {{
    "task_name": "Review wireframes",
    "project_name": "Website Redesign",
    "priority": "high"
  }},
  "confidence": 0.9
}}

Input: "Show me all projects"
Output: {{
  "intent": "list_projects",
  "entities": {{}},
  "confidence": 0.9
}}

Input: "How many contacts do I have"
Output: {{
  "intent": "list_contacts",
  "entities": {{}},
  "confidence": 0.9
}}

Input: "Show me all my contacts"
Output: {{
  "intent": "list_contacts",
  "entities": {{}},
  "confidence": 0.9
}}

Input: "How many organizations do I have"
Output: {{
  "intent": "list_organizations",
  "entities": {{}},
  "confidence": 0.9
}}

Input: "List all my companies"
Output: {{
  "intent": "list_organizations",
  "entities": {{}},
  "confidence": 0.9
}}

Input: "Show me all contacts from tech companies"
Output: {{
  "intent": "search_contacts",
  "entities": {{
    "search_query": "tech companies"
  }},
  "confidence": 0.85
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

Input: "Change Mike's email to new@email.com"
Output: {{
  "intent": "update_contact",
  "entities": {{
    "name": "Mike",
    "email": "new@email.com"
  }},
  "confidence": 0.9
}}

Input: "Update the Acme project status to completed"
Output: {{
  "intent": "update_project",
  "entities": {{
    "project_name": "Acme",
    "status": "completed"
  }},
  "confidence": 0.85
}}

Input: "How many contacts called Gabriel"
Output: {{
  "intent": "search_contacts",
  "entities": {{
    "search_query": "Gabriel"
  }},
  "confidence": 0.9
}}

Input: "Show me contacts named Mike"
Output: {{
  "intent": "search_contacts",
  "entities": {{
    "search_query": "Mike"
  }},
  "confidence": 0.9
}}

Input: "Count how many projects are completed"
Output: {{
  "intent": "search_projects",
  "entities": {{
    "status": "completed"
  }},
  "confidence": 0.85
}}

Input: "Find all tasks assigned to John"
Output: {{
  "intent": "search_tasks",
  "entities": {{
    "search_query": "John"
  }},
  "confidence": 0.9
}}

Input: "List organizations in tech industry"
Output: {{
  "intent": "search_organizations",
  "entities": {{
    "search_query": "tech"
  }},
  "confidence": 0.85
}}

Input: "Gabriel Jones"
Output: {{
  "intent": "clarification_response",
  "entities": {{
    "selected_name": "Gabriel Jones"
  }},
  "confidence": 0.95
}}

Input: "The first one"
Output: {{
  "intent": "clarification_response",
  "entities": {{
    "selected_option": "first"
  }},
  "confidence": 0.9
}}

Input: "Update Gabriel Jones email to new@email.com"
Output: {{
  "intent": "update_contact",
  "entities": {{
    "name": "Gabriel Jones",
    "email": "new@email.com"
  }},
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


# Global LLM command parser instance
llm_command_parser = LLMCommandParser()