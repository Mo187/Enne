from typing import Dict, Any, List, Optional, Tuple
import re
import json
import structlog
from enum import Enum

logger = structlog.get_logger()


class CommandType(Enum):
    """Types of CRM commands"""
    CREATE_CONTACT = "create_contact"
    UPDATE_CONTACT = "update_contact"
    DELETE_CONTACT = "delete_contact"
    SEARCH_CONTACTS = "search_contacts"
    LIST_CONTACTS = "list_contacts"

    CREATE_ORGANIZATION = "create_organization"
    UPDATE_ORGANIZATION = "update_organization"
    DELETE_ORGANIZATION = "delete_organization"
    SEARCH_ORGANIZATIONS = "search_organizations"
    LIST_ORGANIZATIONS = "list_organizations"

    CREATE_PROJECT = "create_project"
    UPDATE_PROJECT = "update_project"
    SEARCH_PROJECTS = "search_projects"

    CREATE_TASK = "create_task"
    UPDATE_TASK = "update_task"
    SEARCH_TASKS = "search_tasks"

    EXPORT_DATA = "export_data"
    GET_STATS = "get_stats"
    HELP = "help"
    UNKNOWN = "unknown"


class CommandParser:
    """Parser for natural language CRM commands"""

    def __init__(self):
        self.command_patterns = self._build_command_patterns()

    def _build_command_patterns(self) -> Dict[CommandType, List[str]]:
        """Build regex patterns for command recognition"""
        return {
            # Contact commands
            CommandType.CREATE_CONTACT: [
                r"(?:add|create|new)\s+(?:a\s+)?(?:contact|person)?\s*(?:for\s+)?(.+)",
                r"(?:save|store)\s+(?:a\s+)?(?:contact|person)\s+(.+)",
                r"(?:enter|input)\s+(?:new\s+)?(?:a\s+)?(?:contact|person)\s+(.+)",
                r"(?:can\s+you\s+)?(?:create|add|make)\s+(?:a\s+)?(?:contact|person)\s+(?:for\s+)?(.+)"
            ],
            CommandType.UPDATE_CONTACT: [
                r"(?:update|edit|change|modify)\s+(?:contact|person)?\s+(.+)",
                r"(?:set|change)\s+(.+?)(?:'s|s)\s+(.+)",
                r"(?:update|modify)\s+(.+?)(?:\s+to\s+|\s+with\s+)(.+)"
            ],
            CommandType.DELETE_CONTACT: [
                r"(?:delete|remove|drop)\s+(?:contact|person)?\s+(.+)",
                r"(?:get\s+rid\s+of|eliminate)\s+(?:contact|person)?\s+(.+)"
            ],
            CommandType.SEARCH_CONTACTS: [
                r"(?:find|search|look\s+for|show\s+me)\s+(?:contacts?|people?)\s+(.+)",
                r"(?:who\s+is|where\s+is)\s+(.+)",
                r"(?:contacts?|people?)\s+(?:from|at|in|with)\s+(.+)"
            ],
            CommandType.LIST_CONTACTS: [
                r"(?:show|list|display)\s+(?:all\s+)?(?:contacts?|people?)(?:\s+(.+))?",
                r"(?:get|give\s+me)\s+(?:all\s+)?(?:contacts?|people?)(?:\s+(.+))?"
            ],

            # Organization commands
            CommandType.CREATE_ORGANIZATION: [
                r"(?:add|create|new)\s+(?:an?\s+)?(?:organization|company|business|org)\s+(.+)",
                r"(?:save|store)\s+(?:an?\s+)?(?:organization|company|business)\s+(.+)",
                r"(?:can\s+you\s+)?(?:create|add|make)\s+(?:an?\s+)?(?:organization|company|business|org)\s+(?:called\s+)?(.+)",
                r"(?:add|create)\s+(.+?)\s+(?:as\s+)?(?:an?\s+)?(?:organization|company|business|org)(?:\s+in\s+(.+))?",
                r"(?:new|create)\s+(?:organization|company|business):\s*(.+)"
            ],
            CommandType.UPDATE_ORGANIZATION: [
                r"(?:update|edit|change|modify)\s+(?:organization|company|business|org)\s+(.+)",
                r"(?:change|set)\s+(.+?)(?:'s|s)\s+(.+)\s+(?:to|as)\s+(.+)"
            ],
            CommandType.SEARCH_ORGANIZATIONS: [
                r"(?:find|search|show\s+me)\s+(?:organizations?|companies|businesses)\s+(.+)",
                r"(?:organizations?|companies|businesses)\s+(?:in|from|with)\s+(.+)"
            ],
            CommandType.LIST_ORGANIZATIONS: [
                r"(?:show|list|display)\s+(?:all\s+)?(?:organizations?|companies|businesses)(?:\s+(.+))?",
                r"(?:get|give\s+me)\s+(?:all\s+)?(?:organizations?|companies|businesses)(?:\s+(.+))?"
            ],

            # Project commands
            CommandType.CREATE_PROJECT: [
                r"(?:add|create|new)\s+(?:a\s+)?(?:project)\s+(?:called\s+)?(.+)",
                r"(?:start|begin)\s+(?:new\s+)?(?:project)\s+(.+)",
                r"(?:can\s+you\s+)?(?:create|add|make)\s+(?:a\s+)?(?:project)\s+(?:called|named)?\s*(.+)",
                r"(?:new|create)\s+project:\s*(.+)",
                r"(?:project|make\s+project)\s+(.+)",
                r"(?:add|create)\s+(.+?)\s+(?:as\s+)?(?:a\s+)?(?:project)(?:\s+for\s+(.+))?",
                r"(?:i\s+want\s+to\s+create|let's\s+create)\s+(?:a\s+)?(?:project)\s+(.+)"
            ],
            CommandType.UPDATE_PROJECT: [
                r"(?:update|edit|change|modify)\s+(?:project)\s+(.+)",
                r"(?:change|set)\s+(.+?)(?:'s|s)\s+(.+)\s+(?:to|as)\s+(.+)",
                r"(?:mark|set)\s+(?:project)\s+(.+?)\s+(?:as|to)\s+(.+)"
            ],
            CommandType.SEARCH_PROJECTS: [
                r"(?:find|search|show\s+me)\s+(?:projects?)\s+(.+)",
                r"(?:projects?)\s+(?:for|from|with|in)\s+(.+)",
                r"(?:list|show)\s+(?:projects?)\s+(.+)"
            ],

            # Task commands
            CommandType.CREATE_TASK: [
                r"(?:add|create|new)\s+(?:a\s+)?(?:task)\s+(?:called\s+)?(.+)",
                r"(?:can\s+you\s+)?(?:create|add|make)\s+(?:a\s+)?(?:task)\s+(?:called|named)?\s*(.+)",
                r"(?:new|create)\s+task:\s*(.+)",
                r"(?:task|make\s+task)\s+(.+)",
                r"(?:add|create)\s+(.+?)\s+(?:as\s+)?(?:a\s+)?(?:task)(?:\s+(?:for|to|in)\s+(.+))?",
                r"(?:add\s+task|create\s+task)\s+(.+?)\s+(?:to|for)\s+(?:project\s+)?(.+)",
                r"(?:i\s+need\s+to|let's)\s+(?:add|create)\s+(?:a\s+)?(?:task)\s+(.+)"
            ],
            CommandType.UPDATE_TASK: [
                r"(?:update|edit|change|modify)\s+(?:task)\s+(.+)",
                r"(?:mark|set)\s+(?:task)\s+(.+?)\s+(?:as|to)\s+(.+)",
                r"(?:complete|finish)\s+(?:task)\s+(.+)"
            ],
            CommandType.SEARCH_TASKS: [
                r"(?:find|search|show\s+me)\s+(?:tasks?)\s+(.+)",
                r"(?:tasks?)\s+(?:for|from|with|in)\s+(.+)",
                r"(?:list|show)\s+(?:tasks?)\s+(.+)"
            ],

            # Export and stats
            CommandType.EXPORT_DATA: [
                r"(?:export|download|save)\s+(.+?)\s+(?:to\s+)?(?:csv|excel|file)",
                r"(?:generate|create)\s+(?:csv|excel|file)\s+(?:of|for|with)\s+(.+)"
            ],
            CommandType.GET_STATS: [
                r"(?:show|get|give\s+me)\s+(?:stats|statistics|numbers|summary|overview)",
                r"(?:how\s+many|count)\s+(.+)",
                r"(?:stats|statistics)\s+(?:for|of|about)\s+(.+)"
            ],

            # Help
            CommandType.HELP: [
                r"(?:help|what\s+can\s+you\s+do|commands|options)",
                r"(?:how\s+do\s+i|how\s+to)\s+(.+)",
                r"(?:what\s+is|explain)\s+(.+)"
            ]
        }

    def parse_command(self, text: str) -> Dict[str, Any]:
        """
        Parse natural language text into a structured command

        Args:
            text: Natural language command

        Returns:
            Dict with command type, extracted data, and confidence
        """
        text = text.strip().lower()

        if not text:
            return {
                "command_type": CommandType.UNKNOWN,
                "data": {},
                "confidence": 0.0,
                "raw_text": text
            }

        # Try to match against command patterns
        best_match = None
        best_confidence = 0.0

        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    confidence = self._calculate_confidence(pattern, text, match)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = {
                            "command_type": command_type,
                            "match": match,
                            "pattern": pattern
                        }

        if best_match and best_confidence > 0.3:  # Minimum confidence threshold
            data = self._extract_data(best_match["command_type"], best_match["match"], text)
            return {
                "command_type": best_match["command_type"],
                "data": data,
                "confidence": best_confidence,
                "raw_text": text
            }
        else:
            return {
                "command_type": CommandType.UNKNOWN,
                "data": {"raw_text": text},
                "confidence": 0.0,
                "raw_text": text
            }

    def _calculate_confidence(self, pattern: str, text: str, match: re.Match) -> float:
        """Calculate confidence score for a pattern match"""
        # Base confidence from pattern specificity
        base_confidence = 0.5

        # Boost for exact keyword matches
        keywords = ["contact", "organization", "company", "project", "task", "add", "create", "update", "find", "search"]
        keyword_boost = sum(0.1 for keyword in keywords if keyword in text) / len(keywords)

        # Extra boost for specific entity keywords
        entity_keywords = {
            "project": 0.2,
            "task": 0.2,
            "organization": 0.15,
            "company": 0.15,
            "contact": 0.1
        }
        entity_boost = sum(boost for keyword, boost in entity_keywords.items() if keyword in text)

        # Boost for captured groups (indicates structured data)
        group_boost = len(match.groups()) * 0.1

        # Penalty for very short matches
        length_penalty = max(0, (10 - len(text)) * 0.05)

        confidence = base_confidence + keyword_boost + entity_boost + group_boost - length_penalty
        return min(1.0, max(0.0, confidence))

    def _extract_data(self, command_type: CommandType, match: re.Match, original_text: str) -> Dict[str, Any]:
        """Extract structured data from the matched command"""

        if command_type in [CommandType.CREATE_CONTACT, CommandType.UPDATE_CONTACT]:
            return self._extract_contact_data(match.group(1) if match.groups() else original_text)

        elif command_type in [CommandType.CREATE_ORGANIZATION, CommandType.UPDATE_ORGANIZATION]:
            return self._extract_organization_data(match.group(1) if match.groups() else original_text)

        elif command_type in [CommandType.CREATE_PROJECT, CommandType.UPDATE_PROJECT]:
            return self._extract_project_data(match.group(1) if match.groups() else original_text)

        elif command_type in [CommandType.CREATE_TASK, CommandType.UPDATE_TASK]:
            return self._extract_task_data(match.group(1) if match.groups() else original_text)

        elif command_type in [CommandType.SEARCH_CONTACTS, CommandType.SEARCH_ORGANIZATIONS, CommandType.SEARCH_PROJECTS, CommandType.SEARCH_TASKS]:
            return self._extract_search_data(match.group(1) if match.groups() else original_text)

        elif command_type == CommandType.EXPORT_DATA:
            return self._extract_export_data(match.group(1) if match.groups() else original_text)

        else:
            return {"query": match.group(1) if match.groups() else original_text}

    def _extract_contact_data(self, text: str) -> Dict[str, Any]:
        """Extract contact information from text"""
        data = {}

        # Extract name - handle multiple patterns including "for [name]"
        name_patterns = [
            r"(?:for|contact)\s+([A-Za-z\s]+?)(?:\.|;|,|\s+(?:from|at|with|email|phone|number|he's|she's|who|works))",
            r"^([A-Za-z\s]+?)(?:\s+from|\s+at|\s+with|\s+email|\s+phone|\s+number|\.|;|,|$)",
            r"(?:add|create|new)\s+([A-Za-z\s]+?)(?:\s+(?:from|at|with|email|phone|number)|$)"
        ]

        for pattern in name_patterns:
            name_match = re.search(pattern, text, re.IGNORECASE)
            if name_match:
                potential_name = name_match.group(1).strip()
                # Filter out common words that aren't names
                if potential_name and not re.match(r'(?:contact|person|user|someone|anybody)', potential_name, re.IGNORECASE):
                    data["name"] = potential_name
                    break

        # Extract email - enhanced patterns
        email_patterns = [
            r"(?:email|e-mail|mail)[\s:]+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"  # Direct email pattern
        ]

        for pattern in email_patterns:
            email_match = re.search(pattern, text, re.IGNORECASE)
            if email_match:
                data["email"] = email_match.group(1)
                break

        # Extract phone - enhanced patterns for natural language
        phone_patterns = [
            r"(?:phone|tel|telephone)[\s:]+([0-9\s\-\+\(\)]+)",
            r"(?:number|phone\s+number)[\s:is]+([0-9\s\-\+\(\)]+)",
            r"(?:his|her|their)\s+(?:number|phone)[\s:is]+([0-9\s\-\+\(\)]+)",
            r"(?:call|reach)\s+(?:him|her|them)[\s:at]+([0-9\s\-\+\(\)]+)"
        ]

        for pattern in phone_patterns:
            phone_match = re.search(pattern, text, re.IGNORECASE)
            if phone_match:
                data["phone"] = phone_match.group(1).strip()
                break

        # Extract organization and job position - handle "He's/She's Company Position" pattern
        org_job_patterns = [
            r"(?:he's|she's|they're)\s+([A-Za-z0-9\s&\.]+?)\s+([A-Za-z\s\-]+?)(?:\.|,|$|\s+(?:his|her|their|phone|number|email))",
            r"(?:works?\s+at|employed\s+by|from)\s+([A-Za-z0-9\s&\.]+?)(?:\s+as\s+([A-Za-z\s\-]+?))?(?:\.|,|$|\s+(?:phone|number|email))",
            r"(?:at|with)\s+([A-Za-z0-9\s&\.]+?)(?:\s+as\s+([A-Za-z\s\-]+?))?(?:\.|,|$|\s+(?:phone|number|email))"
        ]

        for pattern in org_job_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if match.group(1):
                    data["organization"] = match.group(1).strip()
                if len(match.groups()) > 1 and match.group(2):
                    data["job_position"] = match.group(2).strip()
                break

        # If no organization found from patterns above, try simpler org patterns
        if "organization" not in data:
            org_patterns = [
                r"(?:from|at|with|works?\s+at|employed\s+by)\s+([A-Za-z0-9\s&\.]+?)(?:\s+(?:email|phone|number)|$|\.|,)",
                r"(?:company|organization|firm|business)[\s:]+([A-Za-z0-9\s&\.]+?)(?:\s+(?:email|phone|number)|$|\.|,)"
            ]

            for pattern in org_patterns:
                org_match = re.search(pattern, text, re.IGNORECASE)
                if org_match:
                    data["organization"] = org_match.group(1).strip()
                    break

        # If no job position found, try additional job patterns
        if "job_position" not in data:
            job_patterns = [
                r"(?:as|is|works?\s+as|job\s+title|position|role)[\s:]+([A-Za-z\s\-]+?)(?:\s+(?:at|from|with)|$|\.|,)",
                r"(?:title|position|role)[\s:]+([A-Za-z\s\-]+?)(?:\s+(?:at|from|with)|$|\.|,)"
            ]

            for pattern in job_patterns:
                job_match = re.search(pattern, text, re.IGNORECASE)
                if job_match:
                    data["job_position"] = job_match.group(1).strip()
                    break

        return data

    def _extract_organization_data(self, text: str) -> Dict[str, Any]:
        """Extract organization information from text"""
        data = {}

        # Extract name - handle multiple patterns
        name_patterns = [
            r"(?:called|named)\s+([A-Za-z0-9\s&\.]+?)(?:\s+(?:in|industry|website|email)|$|\.|,)",
            r"(?:company|organization|business)\s+([A-Za-z0-9\s&\.]+?)(?:\s+(?:in|industry|website|email)|$|\.|,)",
            r"^([A-Za-z0-9\s&\.]+?)(?:\s+(?:in|industry|website|email|as|is)|$|\.|,)",
            r"([A-Za-z0-9\s&\.]+?)\s+(?:as\s+)?(?:an?\s+)?(?:organization|company|business)",
            r"add\s+([A-Za-z0-9\s&\.]+?)(?:\s+(?:in|industry|website|email)|$|\.|,)"
        ]

        for pattern in name_patterns:
            name_match = re.search(pattern, text, re.IGNORECASE)
            if name_match:
                potential_name = name_match.group(1).strip()
                # Filter out common words that aren't organization names
                if potential_name and not re.match(r'(?:organization|company|business|the|a|an)$', potential_name, re.IGNORECASE):
                    data["name"] = potential_name
                    break

        # Extract industry - enhanced patterns
        industry_patterns = [
            r"(?:in|industry)[\s:]+([A-Za-z\s]+?)(?:\s+(?:website|email|sector)|$|\.|,)",
            r"(?:in\s+the\s+)([A-Za-z\s]+?)\s+(?:industry|sector|business|field)(?:\s+(?:website|email)|$|\.|,)",
            r"([A-Za-z\s]+?)\s+(?:industry|sector|company|business)(?:\s+(?:website|email)|$|\.|,)"
        ]

        for pattern in industry_patterns:
            industry_match = re.search(pattern, text, re.IGNORECASE)
            if industry_match:
                data["industry"] = industry_match.group(1).strip()
                break

        # Extract website
        website_patterns = [
            r"(?:website|url|site)[\s:]+([a-zA-Z0-9\.\-\/\:]+)",
            r"(?:www\.|https?://)([a-zA-Z0-9\.\-\/\:]+)"
        ]

        for pattern in website_patterns:
            website_match = re.search(pattern, text, re.IGNORECASE)
            if website_match:
                website = website_match.group(1).strip()
                if not website.startswith(('http://', 'https://')):
                    website = f"https://{website}"
                data["website"] = website
                break

        # Extract email
        email_match = re.search(r"(?:email|e-mail|mail)[\s:]+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text, re.IGNORECASE)
        if email_match:
            data["email"] = email_match.group(1)

        return data

    def _extract_project_data(self, text: str) -> Dict[str, Any]:
        """Extract project information from text"""
        data = {}

        # Extract project name - handle multiple patterns
        name_patterns = [
            r"(?:called|named)\s+([A-Za-z0-9\s&\.]+?)(?:\s+(?:for|with|status|priority|due)|$|\.|,)",
            r"^([A-Za-z0-9\s&\.]+?)(?:\s+(?:for|with|status|priority|due|organization)|$|\.|,)",
            r"(?:project)\s+([A-Za-z0-9\s&\.]+?)(?:\s+(?:for|with|status|priority|due)|$|\.|,)",
            r"([A-Za-z0-9\s&\.]+?)\s+(?:as\s+)?(?:a\s+)?(?:project)"
        ]

        for pattern in name_patterns:
            name_match = re.search(pattern, text, re.IGNORECASE)
            if name_match:
                potential_name = name_match.group(1).strip()
                # Filter out common words that aren't project names
                if potential_name and not re.match(r'(?:project|the|a|an)$', potential_name, re.IGNORECASE):
                    data["name"] = potential_name
                    break

        # Extract organization
        org_patterns = [
            r"(?:for|with)\s+(?:organization|company)\s+([A-Za-z0-9\s&\.]+?)(?:\s+(?:status|priority|due)|$|\.|,)",
            r"(?:for|at|with)\s+([A-Za-z0-9\s&\.]+?)(?:\s+(?:status|priority|due|organization)|$|\.|,)"
        ]

        for pattern in org_patterns:
            org_match = re.search(pattern, text, re.IGNORECASE)
            if org_match:
                data["organization"] = org_match.group(1).strip()
                break

        # Extract status
        status_patterns = [
            r"(?:status|set\s+as)\s+([a-z_]+)(?:\s|$|\.|,)",
            r"(?:mark|set)\s+(?:as|to)\s+([a-z_]+)(?:\s|$|\.|,)"
        ]

        for pattern in status_patterns:
            status_match = re.search(pattern, text, re.IGNORECASE)
            if status_match:
                status = status_match.group(1).lower().replace(' ', '_')
                if status in ['planned', 'in_progress', 'completed', 'on_hold', 'cancelled']:
                    data["status"] = status
                break

        # Extract priority
        priority_patterns = [
            r"(?:priority)\s+(low|medium|high|urgent)(?:\s|$|\.|,)",
            r"(?:with|set)\s+(low|medium|high|urgent)\s+priority(?:\s|$|\.|,)"
        ]

        for pattern in priority_patterns:
            priority_match = re.search(pattern, text, re.IGNORECASE)
            if priority_match:
                data["priority"] = priority_match.group(1).lower()
                break

        # Extract due date
        due_date_patterns = [
            r"(?:due|deadline|by)\s+([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})",
            r"(?:due|deadline|by)\s+(today|tomorrow|next\s+week|next\s+month)"
        ]

        for pattern in due_date_patterns:
            due_match = re.search(pattern, text, re.IGNORECASE)
            if due_match:
                data["due_date"] = due_match.group(1)
                break

        return data

    def _extract_task_data(self, text: str) -> Dict[str, Any]:
        """Extract task information from text"""
        data = {}

        # Extract task name - handle multiple patterns
        name_patterns = [
            r"(?:called|named)\s+([A-Za-z0-9\s&\.]+?)(?:\s+(?:for|to|in|with|status|priority|due)|$|\.|,)",
            r"^([A-Za-z0-9\s&\.]+?)(?:\s+(?:for|to|in|with|status|priority|due|project)|$|\.|,)",
            r"(?:task)\s+([A-Za-z0-9\s&\.]+?)(?:\s+(?:for|to|in|with|status|priority|due)|$|\.|,)",
            r"([A-Za-z0-9\s&\.]+?)\s+(?:as\s+)?(?:a\s+)?(?:task)"
        ]

        for pattern in name_patterns:
            name_match = re.search(pattern, text, re.IGNORECASE)
            if name_match:
                potential_name = name_match.group(1).strip()
                # Filter out common words that aren't task names
                if potential_name and not re.match(r'(?:task|the|a|an)$', potential_name, re.IGNORECASE):
                    data["name"] = potential_name
                    break

        # Extract project
        project_patterns = [
            r"(?:for|to|in)\s+(?:project\s+)?([A-Za-z0-9\s&\.]+?)(?:\s+(?:status|priority|due|with)|$|\.|,)",
            r"(?:project)\s+([A-Za-z0-9\s&\.]+?)(?:\s+(?:status|priority|due)|$|\.|,)"
        ]

        for pattern in project_patterns:
            project_match = re.search(pattern, text, re.IGNORECASE)
            if project_match:
                data["project"] = project_match.group(1).strip()
                break

        # Extract assignee
        assignee_patterns = [
            r"(?:assign|assigned\s+to|for)\s+([A-Za-z\s]+?)(?:\s+(?:status|priority|due|with)|$|\.|,)",
            r"(?:assignee)\s+([A-Za-z\s]+?)(?:\s+(?:status|priority|due)|$|\.|,)"
        ]

        for pattern in assignee_patterns:
            assignee_match = re.search(pattern, text, re.IGNORECASE)
            if assignee_match:
                data["assignee"] = assignee_match.group(1).strip()
                break

        # Extract status
        status_patterns = [
            r"(?:status|set\s+as)\s+([a-z_]+)(?:\s|$|\.|,)",
            r"(?:mark|set)\s+(?:as|to)\s+([a-z_]+)(?:\s|$|\.|,)"
        ]

        for pattern in status_patterns:
            status_match = re.search(pattern, text, re.IGNORECASE)
            if status_match:
                status = status_match.group(1).lower().replace(' ', '_')
                if status in ['pending', 'in_progress', 'completed', 'blocked', 'cancelled']:
                    data["status"] = status
                break

        # Extract priority
        priority_patterns = [
            r"(?:priority)\s+(low|medium|high|urgent)(?:\s|$|\.|,)",
            r"(?:with|set)\s+(low|medium|high|urgent)\s+priority(?:\s|$|\.|,)"
        ]

        for pattern in priority_patterns:
            priority_match = re.search(pattern, text, re.IGNORECASE)
            if priority_match:
                data["priority"] = priority_match.group(1).lower()
                break

        # Extract due date
        due_date_patterns = [
            r"(?:due|deadline|by)\s+([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})",
            r"(?:due|deadline|by)\s+(today|tomorrow|next\s+week|next\s+month)"
        ]

        for pattern in due_date_patterns:
            due_match = re.search(pattern, text, re.IGNORECASE)
            if due_match:
                data["due_date"] = due_match.group(1)
                break

        return data

    def _extract_search_data(self, text: str) -> Dict[str, Any]:
        """Extract search criteria from text"""
        data = {"search": text.strip()}

        # Extract specific field searches
        if "from" in text or "at" in text:
            org_match = re.search(r"(?:from|at)\s+([A-Za-z0-9\s&\.]+)", text)
            if org_match:
                data["organization"] = org_match.group(1).strip()

        if "industry" in text:
            industry_match = re.search(r"industry[\s:]+([A-Za-z\s]+)", text)
            if industry_match:
                data["industry"] = industry_match.group(1).strip()

        return data

    def _extract_export_data(self, text: str) -> Dict[str, Any]:
        """Extract export specifications from text"""
        data = {"format": "csv"}  # Default format

        if "contacts" in text:
            data["type"] = "contacts"
        elif "organizations" in text or "companies" in text:
            data["type"] = "organizations"
        elif "projects" in text:
            data["type"] = "projects"
        elif "tasks" in text:
            data["type"] = "tasks"
        else:
            data["type"] = "contacts"  # Default

        return data

    def generate_api_action(self, parsed_command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate API action from parsed command

        Args:
            parsed_command: Output from parse_command

        Returns:
            Dict with API endpoint, method, and data
        """
        command_type = parsed_command["command_type"]
        data = parsed_command["data"]

        if command_type == CommandType.CREATE_CONTACT:
            return {
                "method": "POST",
                "endpoint": "/api/v1/contacts",
                "data": data,
                "description": f"Create new contact: {data.get('name', 'Unknown')}"
            }

        elif command_type == CommandType.SEARCH_CONTACTS:
            params = {}
            if "search" in data:
                params["search"] = data["search"]
            if "organization" in data:
                params["organization"] = data["organization"]

            return {
                "method": "GET",
                "endpoint": "/api/v1/contacts",
                "params": params,
                "description": f"Search contacts: {data.get('search', 'all')}"
            }

        elif command_type == CommandType.CREATE_ORGANIZATION:
            return {
                "method": "POST",
                "endpoint": "/api/v1/organizations",
                "data": data,
                "description": f"Create new organization: {data.get('name', 'Unknown')}"
            }

        elif command_type == CommandType.SEARCH_ORGANIZATIONS:
            params = {}
            if "search" in data:
                params["search"] = data["search"]
            if "industry" in data:
                params["industry"] = data["industry"]

            return {
                "method": "GET",
                "endpoint": "/api/v1/organizations",
                "params": params,
                "description": f"Search organizations: {data.get('search', 'all')}"
            }

        elif command_type == CommandType.CREATE_PROJECT:
            return {
                "method": "POST",
                "endpoint": "/api/v1/projects",
                "data": data,
                "description": f"Create new project: {data.get('name', 'Unknown')}"
            }

        elif command_type == CommandType.UPDATE_PROJECT:
            return {
                "method": "PUT",
                "endpoint": f"/api/v1/projects/{data.get('project_id', 'ID')}",
                "data": data,
                "description": f"Update project: {data.get('name', 'Unknown')}"
            }

        elif command_type == CommandType.SEARCH_PROJECTS:
            params = {}
            if "search" in data:
                params["search"] = data["search"]
            if "organization" in data:
                params["organization"] = data["organization"]
            if "status" in data:
                params["status"] = data["status"]

            return {
                "method": "GET",
                "endpoint": "/api/v1/projects",
                "params": params,
                "description": f"Search projects: {data.get('search', 'all')}"
            }

        elif command_type == CommandType.CREATE_TASK:
            return {
                "method": "POST",
                "endpoint": "/api/v1/tasks",
                "data": data,
                "description": f"Create new task: {data.get('name', 'Unknown')}"
            }

        elif command_type == CommandType.UPDATE_TASK:
            return {
                "method": "PUT",
                "endpoint": f"/api/v1/tasks/{data.get('task_id', 'ID')}",
                "data": data,
                "description": f"Update task: {data.get('name', 'Unknown')}"
            }

        elif command_type == CommandType.SEARCH_TASKS:
            params = {}
            if "search" in data:
                params["search"] = data["search"]
            if "project" in data:
                params["project"] = data["project"]
            if "status" in data:
                params["status"] = data["status"]
            if "assignee" in data:
                params["assignee"] = data["assignee"]

            return {
                "method": "GET",
                "endpoint": "/api/v1/tasks",
                "params": params,
                "description": f"Search tasks: {data.get('search', 'all')}"
            }

        elif command_type == CommandType.EXPORT_DATA:
            export_type = data.get("type", "contacts")
            endpoint_map = {
                "contacts": "/api/v1/contacts/export/csv",
                "organizations": "/api/v1/organizations/export/csv",
                "projects": "/api/v1/projects/export/csv",
                "tasks": "/api/v1/tasks/export/csv"
            }
            endpoint = endpoint_map.get(export_type, "/api/v1/contacts/export/csv")

            return {
                "method": "GET",
                "endpoint": endpoint,
                "params": {},
                "description": f"Export {export_type} to CSV"
            }

        elif command_type == CommandType.GET_STATS:
            return {
                "method": "GET",
                "endpoint": "/api/v1/organizations/stats",
                "params": {},
                "description": "Get CRM statistics"
            }

        else:
            return {
                "method": "UNKNOWN",
                "endpoint": "UNKNOWN",
                "data": data,
                "description": "Command not recognized or not yet implemented"
            }


# Global command parser instance
command_parser = CommandParser()