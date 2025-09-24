from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import structlog
from datetime import datetime, timezone
import asyncio

from ...core.database import get_db
from ...models.user import User
from ...models.contact import Contact
from ...models.organization import Organization
from ...models.project import Project
from ...models.task import Task
from ..dependencies import get_current_active_user
from ...services.ai_service import ai_service
from ...services.llm_command_parser import llm_command_parser
from ...services.tool_interface import tool_registry
from ...integrations.mcp_client import get_mcp_client

logger = structlog.get_logger()

router = APIRouter()


# Pydantic models for chat
class ChatMessage(BaseModel):
    message: str
    provider: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    message: str
    response: str
    command_detected: bool = False
    command_type: Optional[str] = None
    api_action: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None
    provider: str
    timestamp: str
    duration_seconds: float


class WebSocketManager:
    """Manage WebSocket connections for real-time chat"""

    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info("WebSocket connected", user_id=user_id)

    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info("WebSocket disconnected", user_id=user_id)

    async def send_personal_message(self, message: dict, user_id: int):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error("Failed to send WebSocket message", user_id=user_id, error=str(e))
                self.disconnect(user_id)


manager = WebSocketManager()


def detect_query_intent(original_message: str, parsed_intent: str) -> str:
    """
    Detect the type of query for better AI response formatting.

    Returns:
        "count" - User wants statistics/counts
        "list" - User wants to see data items
        "search" - User wants to find specific items
        "action" - User wants to perform an action (create/update/delete)
    """
    message_lower = original_message.lower()

    # Count/Statistics queries
    count_indicators = [
        "how many", "count", "total", "number of", "statistics", "stats",
        "do i have", "are there", "show me the count", "tell me how many",
        "what's my total", "how much", "give me a count", "any idea how many"
    ]

    if any(indicator in message_lower for indicator in count_indicators):
        return "count"

    # Search queries
    search_indicators = [
        "find", "search for", "look for", "show me", "where is", "who is"
    ]

    if any(indicator in message_lower for indicator in search_indicators):
        return "search"

    # List queries
    list_indicators = [
        "list", "show all", "display", "view", "see all"
    ]

    if any(indicator in message_lower for indicator in list_indicators):
        return "list"

    # Default to action for create/update/delete intents
    action_intents = [
        "create_", "update_", "delete_", "add", "remove", "modify"
    ]

    if any(intent in parsed_intent for intent in action_intents):
        return "action"

    # Default to search for other read operations
    return "search"


async def execute_api_action(
    action: Dict[str, Any],
    user: User,
    db: AsyncSession
) -> Dict[str, Any]:
    """Execute API action from parsed command"""

    try:
        method = action.get("method")
        endpoint = action.get("endpoint")
        data = action.get("data", {})
        params = action.get("params", {})

        if method == "POST" and "/contacts" in endpoint:
            # Create contact
            if not data.get("name"):
                return {"success": False, "error": "Contact name is required"}

            # Check for existing contact
            if data.get("email"):
                existing_query = select(Contact).where(
                    and_(
                        Contact.user_id == user.id,
                        Contact.email == data["email"]
                    )
                )
                existing_result = await db.execute(existing_query)
                if existing_result.scalar_one_or_none():
                    return {"success": False, "error": "Contact with this email already exists"}

            # Create new contact
            contact = Contact(
                user_id=user.id,
                name=data["name"],
                email=data.get("email"),
                phone=data.get("phone"),
                job_position=data.get("job_position"),
                organization=data.get("organization"),
                notes=data.get("notes")
            )

            db.add(contact)
            await db.commit()
            await db.refresh(contact)

            return {
                "success": True,
                "result": {
                    "id": contact.id,
                    "name": contact.name,
                    "email": contact.email,
                    "organization": contact.organization
                },
                "message": f"Contact '{contact.name}' created successfully"
            }

        elif method == "POST" and "/organizations" in endpoint:
            # Create organization
            if not data.get("name"):
                return {"success": False, "error": "Organization name is required"}

            # Check for existing organization
            existing_query = select(Organization).where(
                and_(
                    Organization.user_id == user.id,
                    Organization.name.ilike(data["name"])
                )
            )
            existing_result = await db.execute(existing_query)
            if existing_result.scalar_one_or_none():
                return {"success": False, "error": "Organization with this name already exists"}

            # Create new organization
            organization = Organization(
                user_id=user.id,
                name=data["name"],
                industry=data.get("industry"),
                website=data.get("website"),
                email=data.get("email"),
                description=data.get("description")
            )

            db.add(organization)
            await db.commit()
            await db.refresh(organization)

            return {
                "success": True,
                "result": {
                    "id": organization.id,
                    "name": organization.name,
                    "industry": organization.industry
                },
                "message": f"Organization '{organization.name}' created successfully"
            }

        elif method == "POST" and "/projects" in endpoint:
            # Create project
            if not data.get("name"):
                return {"success": False, "error": "Project name is required"}

            # Check for existing project
            existing_query = select(Project).where(
                and_(
                    Project.user_id == user.id,
                    Project.name.ilike(data["name"])
                )
            )
            existing_result = await db.execute(existing_query)
            if existing_result.scalar_one_or_none():
                return {"success": False, "error": "Project with this name already exists"}

            # Handle organization lookup if specified
            organization_id = None
            if data.get("organization"):
                org_query = select(Organization).where(
                    and_(
                        Organization.user_id == user.id,
                        Organization.name.ilike(data["organization"])
                    )
                )
                org_result = await db.execute(org_query)
                organization = org_result.scalar_one_or_none()
                if organization:
                    organization_id = organization.id

            # Handle dates
            due_date = None
            start_date = None
            if data.get("due_date"):
                try:
                    due_date = datetime.fromisoformat(data["due_date"]).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            if data.get("start_date"):
                try:
                    start_date = datetime.fromisoformat(data["start_date"]).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            # Create new project
            project = Project(
                user_id=user.id,
                name=data["name"],
                description=data.get("description"),
                status=data.get("status", "planned"),
                priority=data.get("priority", "medium"),
                organization_id=organization_id,
                due_date=due_date,
                start_date=start_date,
                notes=data.get("notes")
            )

            db.add(project)
            await db.commit()
            await db.refresh(project)

            return {
                "success": True,
                "result": {
                    "id": project.id,
                    "name": project.name,
                    "status": project.status,
                    "priority": project.priority
                },
                "message": f"Project '{project.name}' created successfully"
            }

        elif method == "POST" and "/tasks" in endpoint:
            # Create task
            if not data.get("name"):
                return {"success": False, "error": "Task name is required"}

            # Handle project lookup if specified
            project_id = None
            project_name = data.get("project_name") or data.get("project")
            if project_name:
                project_query = select(Project).where(
                    and_(
                        Project.user_id == user.id,
                        Project.name.ilike(project_name)
                    )
                )
                project_result = await db.execute(project_query)
                project = project_result.scalar_one_or_none()
                if project:
                    project_id = project.id
                else:
                    return {"success": False, "error": f"Project '{project_name}' not found"}
            else:
                # If no project specified, try to find a default project or require one
                return {"success": False, "error": "Project is required for task creation"}

            # Handle due date for task
            due_date = None
            if data.get("due_date"):
                try:
                    due_date = datetime.fromisoformat(data["due_date"]).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            # Create new task
            task = Task(
                project_id=project_id,
                name=data["name"],
                description=data.get("description"),
                status=data.get("status", "pending"),
                priority=data.get("priority", "medium"),
                assignee=data.get("assignee"),
                due_date=due_date
            )

            db.add(task)
            await db.commit()
            await db.refresh(task)

            return {
                "success": True,
                "result": {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status,
                    "priority": task.priority,
                    "project_id": task.project_id
                },
                "message": f"Task '{task.name}' created successfully"
            }

        elif method == "GET" and "/contacts" in endpoint:
            # Search contacts
            query = select(Contact).where(Contact.user_id == user.id)

            if params.get("search"):
                search_term = f"%{params['search']}%"
                query = query.where(
                    or_(
                        Contact.name.ilike(search_term),
                        Contact.email.ilike(search_term),
                        Contact.organization.ilike(search_term)
                    )
                )

            if params.get("organization"):
                query = query.where(Contact.organization.ilike(f"%{params['organization']}%"))

            # Add total count for accurate reporting
            count_query = select(func.count()).select_from(Contact).where(Contact.user_id == user.id)
            if params.get("search"):
                search_term = f"%{params['search']}%"
                count_query = count_query.where(
                    or_(
                        Contact.name.ilike(search_term),
                        Contact.email.ilike(search_term),
                        Contact.organization.ilike(search_term)
                    )
                )
            if params.get("organization"):
                count_query = count_query.where(Contact.organization.ilike(f"%{params['organization']}%"))

            total_count = await db.scalar(count_query)

            # Limit results for chat display (but still report total)
            query = query.limit(50 if params.get("search") else 10)

            result = await db.execute(query)
            contacts = result.scalars().all()

            return {
                "success": True,
                "result": [
                    {
                        "id": contact.id,
                        "name": contact.name,
                        "email": contact.email,
                        "organization": contact.organization,
                        "phone": contact.phone
                    }
                    for contact in contacts
                ],
                "total_count": total_count,
                "message": f"Found {total_count} contact(s) total" + (f", showing first {len(contacts)}" if total_count > len(contacts) else "")
            }

        elif method == "GET" and "/organizations" in endpoint:
            # Search organizations
            query = select(Organization).where(Organization.user_id == user.id)

            if params.get("search"):
                search_term = f"%{params['search']}%"
                query = query.where(Organization.name.ilike(search_term))

            if params.get("industry"):
                query = query.where(Organization.industry.ilike(f"%{params['industry']}%"))

            # Add total count for accurate reporting
            count_query = select(func.count()).select_from(Organization).where(Organization.user_id == user.id)
            if params.get("search"):
                search_term = f"%{params['search']}%"
                count_query = count_query.where(Organization.name.ilike(search_term))
            if params.get("industry"):
                count_query = count_query.where(Organization.industry.ilike(f"%{params['industry']}%"))

            total_count = await db.scalar(count_query)

            # Limit results for chat display
            query = query.limit(50 if params.get("search") else 10)

            result = await db.execute(query)
            organizations = result.scalars().all()

            return {
                "success": True,
                "result": [
                    {
                        "id": org.id,
                        "name": org.name,
                        "industry": org.industry,
                        "website": org.website
                    }
                    for org in organizations
                ],
                "total_count": total_count,
                "message": f"Found {total_count} organization(s) total" + (f", showing first {len(organizations)}" if total_count > len(organizations) else "")
            }

        elif method == "GET" and "/projects" in endpoint:
            # Search projects
            query = select(Project).where(Project.user_id == user.id)

            if params.get("search"):
                search_term = f"%{params['search']}%"
                query = query.where(
                    or_(
                        Project.name.ilike(search_term),
                        Project.description.ilike(search_term)
                    )
                )

            if params.get("status"):
                query = query.where(Project.status == params["status"])

            if params.get("organization"):
                # Join with organization table to search by organization name
                query = query.join(Organization).where(Organization.name.ilike(f"%{params['organization']}%"))

            # Add total count for accurate reporting
            count_query = select(func.count()).select_from(Project).where(Project.user_id == user.id)
            if params.get("search"):
                search_term = f"%{params['search']}%"
                count_query = count_query.where(
                    or_(
                        Project.name.ilike(search_term),
                        Project.description.ilike(search_term)
                    )
                )
            if params.get("status"):
                count_query = count_query.where(Project.status == params["status"])
            if params.get("organization"):
                count_query = count_query.join(Organization).where(Organization.name.ilike(f"%{params['organization']}%"))

            total_count = await db.scalar(count_query)

            # Limit results for chat display
            query = query.limit(50 if params.get("search") else 10)

            result = await db.execute(query)
            projects = result.scalars().all()

            return {
                "success": True,
                "result": [
                    {
                        "id": project.id,
                        "name": project.name,
                        "status": project.status,
                        "priority": project.priority,
                        "organization_id": project.organization_id
                    }
                    for project in projects
                ],
                "total_count": total_count,
                "message": f"Found {total_count} project(s) total" + (f", showing first {len(projects)}" if total_count > len(projects) else "")
            }

        elif method == "GET" and "/tasks" in endpoint:
            # Search tasks
            query = select(Task).join(Project).where(Project.user_id == user.id)

            if params.get("search"):
                search_term = f"%{params['search']}%"
                query = query.where(
                    or_(
                        Task.name.ilike(search_term),
                        Task.description.ilike(search_term),
                        Task.assignee.ilike(search_term)
                    )
                )

            if params.get("status"):
                query = query.where(Task.status == params["status"])

            if params.get("project"):
                query = query.where(Project.name.ilike(f"%{params['project']}%"))

            if params.get("assignee"):
                query = query.where(Task.assignee.ilike(f"%{params['assignee']}%"))

            # Add total count for accurate reporting
            count_query = select(func.count(Task.id)).select_from(Task).join(Project).where(Project.user_id == user.id)
            if params.get("search"):
                search_term = f"%{params['search']}%"
                count_query = count_query.where(
                    or_(
                        Task.name.ilike(search_term),
                        Task.description.ilike(search_term),
                        Task.assignee.ilike(search_term)
                    )
                )
            if params.get("status"):
                count_query = count_query.where(Task.status == params["status"])
            if params.get("project"):
                count_query = count_query.where(Project.name.ilike(f"%{params['project']}%"))
            if params.get("assignee"):
                count_query = count_query.where(Task.assignee.ilike(f"%{params['assignee']}%"))

            total_count = await db.scalar(count_query)

            # Limit results for chat display
            query = query.limit(50 if params.get("search") else 10)

            result = await db.execute(query)
            tasks = result.scalars().all()

            return {
                "success": True,
                "result": [
                    {
                        "id": task.id,
                        "name": task.name,
                        "status": task.status,
                        "priority": task.priority,
                        "assignee": task.assignee,
                        "project_id": task.project_id
                    }
                    for task in tasks
                ],
                "total_count": total_count,
                "message": f"Found {total_count} task(s) total" + (f", showing first {len(tasks)}" if total_count > len(tasks) else "")
            }

        elif method == "PUT" and "/contacts/update" in endpoint:
            # Update contact with smart search
            identifier = action.get("identifier")
            if not identifier:
                return {"success": False, "error": "Contact identifier required"}

            # Search for all possible contacts using partial matching
            search_term = f"%{identifier}%"
            query = select(Contact).where(
                and_(
                    Contact.user_id == user.id,
                    or_(
                        Contact.name.ilike(search_term),
                        Contact.email.ilike(search_term)
                    )
                )
            ).limit(10)  # Limit to prevent too many matches

            result = await db.execute(query)
            contacts = result.scalars().all()

            if not contacts:
                return {"success": False, "error": f"No contacts found matching '{identifier}'"}

            if len(contacts) > 1:
                # Multiple matches found - return them for AI clarification
                return {
                    "success": False,
                    "requires_clarification": True,
                    "clarification_type": "multiple_contacts",
                    "matches": [
                        {
                            "id": contact.id,
                            "name": contact.name,
                            "email": contact.email,
                            "phone": contact.phone,
                            "organization": contact.organization
                        }
                        for contact in contacts
                    ],
                    "message": f"Found {len(contacts)} contacts matching '{identifier}'. Please specify which one you want to update.",
                    "ai_context": f"I found {len(contacts)} contacts matching '{identifier}':\n" +
                                "\n".join([f"- {contact.name} ({contact.email if contact.email else 'no email'})" for contact in contacts]) +
                                "\n\nWhich contact did you want to update?"
                }

            # Single match found - proceed with update
            contact = contacts[0]

            # Update fields that are provided and valid
            for field, value in data.items():
                if field not in ["identifier"] and value is not None and hasattr(contact, field):
                    setattr(contact, field, value)

            await db.commit()
            await db.refresh(contact)

            return {
                "success": True,
                "result": {
                    "id": contact.id,
                    "name": contact.name,
                    "email": contact.email,
                    "phone": contact.phone
                },
                "message": f"Contact '{contact.name}' updated successfully"
            }

        elif method == "PUT" and "/organizations/update" in endpoint:
            # Update organization with smart search
            identifier = action.get("identifier")
            if not identifier:
                return {"success": False, "error": "Organization identifier required"}

            # Search for all possible organizations using partial matching
            search_term = f"%{identifier}%"
            query = select(Organization).where(
                and_(
                    Organization.user_id == user.id,
                    Organization.name.ilike(search_term)
                )
            ).limit(10)

            result = await db.execute(query)
            organizations = result.scalars().all()

            if not organizations:
                return {"success": False, "error": f"No organizations found matching '{identifier}'"}

            if len(organizations) > 1:
                # Multiple matches found - return them for AI clarification
                return {
                    "success": False,
                    "requires_clarification": True,
                    "clarification_type": "multiple_organizations",
                    "matches": [
                        {
                            "id": org.id,
                            "name": org.name,
                            "industry": org.industry,
                            "website": org.website
                        }
                        for org in organizations
                    ],
                    "message": f"Found {len(organizations)} organizations matching '{identifier}'. Please specify which one you want to update.",
                    "ai_context": f"I found {len(organizations)} organizations matching '{identifier}':\n" +
                                "\n".join([f"- {org.name} ({org.industry if org.industry else 'no industry specified'})" for org in organizations]) +
                                "\n\nWhich organization did you want to update?"
                }

            # Single match found - proceed with update
            organization = organizations[0]

            # Update fields that are provided and valid
            for field, value in data.items():
                if field not in ["identifier"] and value is not None and hasattr(organization, field):
                    setattr(organization, field, value)

            await db.commit()
            await db.refresh(organization)

            return {
                "success": True,
                "result": {
                    "id": organization.id,
                    "name": organization.name,
                    "industry": organization.industry
                },
                "message": f"Organization '{organization.name}' updated successfully"
            }

        elif method == "PUT" and "/projects/update" in endpoint:
            # Update project with smart search
            identifier = action.get("identifier")
            if not identifier:
                return {"success": False, "error": "Project identifier required"}

            # Search for all possible projects using partial matching
            search_term = f"%{identifier}%"
            query = select(Project).where(
                and_(
                    Project.user_id == user.id,
                    Project.name.ilike(search_term)
                )
            ).limit(10)

            result = await db.execute(query)
            projects = result.scalars().all()

            if not projects:
                return {"success": False, "error": f"No projects found matching '{identifier}'"}

            if len(projects) > 1:
                # Multiple matches found - return them for AI clarification
                return {
                    "success": False,
                    "requires_clarification": True,
                    "clarification_type": "multiple_projects",
                    "matches": [
                        {
                            "id": proj.id,
                            "name": proj.name,
                            "status": proj.status,
                            "priority": proj.priority
                        }
                        for proj in projects
                    ],
                    "message": f"Found {len(projects)} projects matching '{identifier}'. Please specify which one you want to update.",
                    "ai_context": f"I found {len(projects)} projects matching '{identifier}':\n" +
                                "\n".join([f"- {proj.name} (Status: {proj.status})" for proj in projects]) +
                                "\n\nWhich project did you want to update?"
                }

            # Single match found - proceed with update
            project = projects[0]

            # Handle dates
            if data.get("due_date"):
                try:
                    project.due_date = datetime.fromisoformat(data["due_date"]).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            if data.get("start_date"):
                try:
                    project.start_date = datetime.fromisoformat(data["start_date"]).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            # Update other fields that are provided and valid
            for field, value in data.items():
                if field not in ["identifier", "due_date", "start_date"] and value is not None and hasattr(project, field):
                    setattr(project, field, value)

            await db.commit()
            await db.refresh(project)

            return {
                "success": True,
                "result": {
                    "id": project.id,
                    "name": project.name,
                    "status": project.status,
                    "priority": project.priority
                },
                "message": f"Project '{project.name}' updated successfully"
            }

        elif method == "PUT" and "/tasks/update" in endpoint:
            # Update task with smart search
            identifier = action.get("identifier")
            if not identifier:
                return {"success": False, "error": "Task identifier required"}

            # Search for all possible tasks using partial matching
            search_term = f"%{identifier}%"
            query = select(Task).join(Project).where(
                and_(
                    Project.user_id == user.id,
                    Task.name.ilike(search_term)
                )
            ).limit(10)

            result = await db.execute(query)
            tasks = result.scalars().all()

            if not tasks:
                return {"success": False, "error": f"No tasks found matching '{identifier}'"}

            if len(tasks) > 1:
                # Multiple matches found - return them for AI clarification
                # Load projects for context
                project_ids = [task.project_id for task in tasks]
                projects_query = select(Project).where(Project.id.in_(project_ids))
                projects_result = await db.execute(projects_query)
                projects = {proj.id: proj.name for proj in projects_result.scalars().all()}

                return {
                    "success": False,
                    "requires_clarification": True,
                    "clarification_type": "multiple_tasks",
                    "matches": [
                        {
                            "id": task.id,
                            "name": task.name,
                            "status": task.status,
                            "priority": task.priority,
                            "project": projects.get(task.project_id, "Unknown"),
                            "assignee": task.assignee
                        }
                        for task in tasks
                    ],
                    "message": f"Found {len(tasks)} tasks matching '{identifier}'. Please specify which one you want to update.",
                    "ai_context": f"I found {len(tasks)} tasks matching '{identifier}':\n" +
                                "\n".join([f"- {task.name} (Project: {projects.get(task.project_id, 'Unknown')})" for task in tasks]) +
                                "\n\nWhich task did you want to update?"
                }

            # Single match found - proceed with update
            task = tasks[0]

            # Handle due date
            if data.get("due_date"):
                try:
                    task.due_date = datetime.fromisoformat(data["due_date"]).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            # Update other fields that are provided and valid
            for field, value in data.items():
                if field not in ["identifier", "due_date", "project_name"] and value is not None and hasattr(task, field):
                    setattr(task, field, value)

            await db.commit()
            await db.refresh(task)

            return {
                "success": True,
                "result": {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status,
                    "priority": task.priority
                },
                "message": f"Task '{task.name}' updated successfully"
            }

        elif method == "CLARIFICATION":
            # Handle clarification responses for multiple matches
            clarification_type = action.get("clarification_type")
            selected_id = action.get("selected_id")
            original_action = action.get("original_action", {})

            if not clarification_type or not selected_id or not original_action:
                return {
                    "success": False,
                    "error": "Invalid clarification request - missing type, ID, or original action"
                }

            # Execute the original action with the specific selected ID
            if clarification_type == "multiple_contacts":
                # Update the original action to target specific contact
                original_action["method"] = "PUT"
                original_action["endpoint"] = "/contacts/update"
                original_action["identifier"] = str(selected_id)

                # Execute the update on the specific contact
                contact_query = select(Contact).where(
                    and_(Contact.user_id == user.id, Contact.id == selected_id)
                )
                contact_result = await db.execute(contact_query)
                contact = contact_result.scalar_one_or_none()

                if not contact:
                    return {"success": False, "error": f"Contact with ID {selected_id} not found"}

                # Update the contact with provided data
                update_data = original_action.get("data", {})
                for field, value in update_data.items():
                    if field not in ["identifier"] and value is not None and hasattr(contact, field):
                        setattr(contact, field, value)

                await db.commit()
                await db.refresh(contact)

                return {
                    "success": True,
                    "result": {
                        "id": contact.id,
                        "name": contact.name,
                        "email": contact.email,
                        "phone": contact.phone
                    },
                    "message": f"Contact '{contact.name}' updated successfully"
                }

            elif clarification_type == "multiple_organizations":
                # Similar logic for organizations
                org_query = select(Organization).where(
                    and_(Organization.user_id == user.id, Organization.id == selected_id)
                )
                org_result = await db.execute(org_query)
                organization = org_result.scalar_one_or_none()

                if not organization:
                    return {"success": False, "error": f"Organization with ID {selected_id} not found"}

                update_data = original_action.get("data", {})
                for field, value in update_data.items():
                    if field not in ["identifier"] and value is not None and hasattr(organization, field):
                        setattr(organization, field, value)

                await db.commit()
                await db.refresh(organization)

                return {
                    "success": True,
                    "result": {
                        "id": organization.id,
                        "name": organization.name,
                        "industry": organization.industry
                    },
                    "message": f"Organization '{organization.name}' updated successfully"
                }

            elif clarification_type == "multiple_projects":
                # Similar logic for projects
                project_query = select(Project).where(
                    and_(Project.user_id == user.id, Project.id == selected_id)
                )
                project_result = await db.execute(project_query)
                project = project_result.scalar_one_or_none()

                if not project:
                    return {"success": False, "error": f"Project with ID {selected_id} not found"}

                update_data = original_action.get("data", {})

                # Handle dates
                if update_data.get("due_date"):
                    try:
                        project.due_date = datetime.fromisoformat(update_data["due_date"]).replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        pass

                if update_data.get("start_date"):
                    try:
                        project.start_date = datetime.fromisoformat(update_data["start_date"]).replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        pass

                # Update other fields
                for field, value in update_data.items():
                    if field not in ["identifier", "due_date", "start_date"] and value is not None and hasattr(project, field):
                        setattr(project, field, value)

                await db.commit()
                await db.refresh(project)

                return {
                    "success": True,
                    "result": {
                        "id": project.id,
                        "name": project.name,
                        "status": project.status,
                        "priority": project.priority
                    },
                    "message": f"Project '{project.name}' updated successfully"
                }

            elif clarification_type == "multiple_tasks":
                # Similar logic for tasks
                task_query = select(Task).join(Project).where(
                    and_(Project.user_id == user.id, Task.id == selected_id)
                )
                task_result = await db.execute(task_query)
                task = task_result.scalar_one_or_none()

                if not task:
                    return {"success": False, "error": f"Task with ID {selected_id} not found"}

                update_data = original_action.get("data", {})

                # Handle due date
                if update_data.get("due_date"):
                    try:
                        task.due_date = datetime.fromisoformat(update_data["due_date"]).replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        pass

                # Update other fields
                for field, value in update_data.items():
                    if field not in ["identifier", "due_date", "project_name"] and value is not None and hasattr(task, field):
                        setattr(task, field, value)

                await db.commit()
                await db.refresh(task)

                return {
                    "success": True,
                    "result": {
                        "id": task.id,
                        "name": task.name,
                        "status": task.status,
                        "priority": task.priority
                    },
                    "message": f"Task '{task.name}' updated successfully"
                }

            else:
                return {
                    "success": False,
                    "error": f"Unknown clarification type: {clarification_type}"
                }

        elif method == "MCP_TOOL":
            # Handle MCP tool execution
            tool_name = action.get("tool_name")
            parameters = action.get("parameters", {})

            if not tool_name:
                return {
                    "success": False,
                    "error": "MCP tool name is required"
                }

            # Check if it's a Microsoft 365 MCP tool
            if tool_name.startswith(("outlook_", "sharepoint_", "onedrive_", "teams_", "authenticate", "extract_document")):
                try:
                    # Execute via tool registry (using our MCP adapter)
                    tool_result = await tool_registry.execute_tool(
                        tool_name=tool_name,
                        parameters=parameters,
                        user_context={
                            "user": user,
                            "user_id": user.id,
                            "db": db
                        }
                    )

                    # Convert ToolResult to API action result format
                    result = {
                        "success": tool_result.success,
                        "message": tool_result.message or f"Executed {tool_name}",
                    }

                    if tool_result.data is not None:
                        result["result"] = tool_result.data

                    if tool_result.total_count is not None:
                        result["total_count"] = tool_result.total_count

                    if tool_result.error:
                        result["error"] = tool_result.error

                    if tool_result.requires_clarification:
                        result["requires_clarification"] = True
                        result["clarification_type"] = tool_result.clarification_type
                        result["clarification_data"] = tool_result.clarification_data

                    logger.info(
                        "MCP tool executed via registry",
                        tool_name=tool_name,
                        success=tool_result.success,
                        user_id=user.id
                    )

                    return result

                except Exception as e:
                    logger.error(
                        "MCP tool execution failed",
                        tool_name=tool_name,
                        error=str(e),
                        user_id=user.id
                    )
                    return {
                        "success": False,
                        "error": f"Failed to execute Microsoft 365 tool: {str(e)}"
                    }
            else:
                return {
                    "success": False,
                    "error": f"Unknown MCP tool: {tool_name}"
                }

        else:
            return {
                "success": False,
                "error": f"API action not yet implemented: {method} {endpoint}"
            }

    except Exception as e:
        logger.error("API action execution failed", error=str(e), action=action)
        return {
            "success": False,
            "error": f"Failed to execute action: {str(e)}"
        }


@router.post("/message", response_model=ChatResponse)
async def send_chat_message(
    chat_data: ChatMessage,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Send a message to the AI assistant and get response"""

    try:
        # Parse the command using LLM
        parsed_command = await llm_command_parser.parse_command(chat_data.message)
        command_detected = parsed_command["confidence"] > 0.7
        api_action = None
        execution_result = None

        # If it's a recognized command, try to execute it
        if command_detected and parsed_command["intent"] != "unknown":
            api_action = llm_command_parser.generate_api_action(parsed_command)

            # Execute the action if it's supported
            if api_action["method"] != "UNKNOWN":
                execution_result = await execute_api_action(api_action, current_user, db)

        # Build user context for AI
        user_context = {
            "user": current_user,  # Include full user object for integration checking
            "name": current_user.name,
            "email": current_user.email,
            "company": current_user.company,
            "preferred_ai_model": current_user.preferred_ai_model
        }

        # Prepare message for AI with execution context
        user_message = chat_data.message
        if execution_result:
            if execution_result["success"]:
                # Detect query intent for better response formatting
                query_intent = detect_query_intent(chat_data.message, parsed_command.get("intent", ""))

                if execution_result.get("result"):
                    result_data = execution_result["result"]
                    total_count = execution_result.get("total_count", 0)

                    if query_intent == "count":
                        # For count queries, emphasize the exact number
                        user_message = f"User asked: '{chat_data.message}'\n\nQUERY RESULT: EXACT COUNT = {total_count}\n\nTell the user they have exactly {total_count} items. Be friendly and natural, like Claude would respond."

                    elif isinstance(result_data, list) and result_data:
                        # For list/search queries, show both count and data
                        data_summary = "Here is the data I found:\n"
                        for i, item in enumerate(result_data, 1):
                            if isinstance(item, dict):
                                # Format based on type of data
                                if 'name' in item and 'email' in item:  # Contact
                                    data_summary += f"{i}. {item.get('name', 'Unknown')} ({item.get('email', 'no email')})\n"
                                elif 'name' in item and 'status' in item:  # Project
                                    data_summary += f"{i}. {item.get('name', 'Unknown')} (Status: {item.get('status', 'unknown')})\n"
                                elif 'name' in item and 'industry' in item:  # Organization
                                    data_summary += f"{i}. {item.get('name', 'Unknown')} (Industry: {item.get('industry', 'unknown')})\n"
                                else:
                                    data_summary += f"{i}. {item}\n"

                        if query_intent == "search":
                            user_message += f"\n\nSearch completed successfully. Found {total_count} total matches. {data_summary.strip()}\n\nPresent these search results in a helpful, natural way."
                        else:  # list intent
                            user_message += f"\n\nList completed successfully. Showing {len(result_data)} items (total: {total_count}). {data_summary.strip()}\n\nShow this data to the user in a clear, friendly way."

                    else:
                        # Empty results
                        if query_intent == "count":
                            user_message = f"User asked: '{chat_data.message}'\n\nQUERY RESULT: EXACT COUNT = 0\n\nTell the user they don't have any items yet. Be encouraging and suggest what they can do next."
                        else:
                            user_message += f"\n\nQuery completed but no results found. {execution_result['message']}. Be encouraging and suggest helpful next steps."
                else:
                    # No result data (like successful creation)
                    user_message += f"\n\nAction completed successfully: {execution_result['message']}. Celebrate the success and suggest helpful next steps in a friendly way."

            elif execution_result.get("requires_clarification"):
                # Handle clarification scenarios (like multiple matches)
                ai_context = execution_result.get("ai_context", "")
                user_message = f"User said: '{chat_data.message}'\n\n{ai_context}"
            else:
                user_message += f"\n\nI tried to execute this command but encountered an error: {execution_result['error']}. Please help me understand what went wrong and suggest how to fix it."

        # Get AI response
        ai_response = await ai_service.generate_crm_response(
            user_message=user_message,
            user_context=user_context,
            conversation_history=chat_data.conversation_history,
            provider=chat_data.provider
        )

        response = ChatResponse(
            message=chat_data.message,
            response=ai_response["response"],
            command_detected=command_detected,
            command_type=parsed_command["intent"] if command_detected else None,
            api_action=api_action,
            execution_result=execution_result,
            provider=ai_response["provider"],
            timestamp=ai_response["timestamp"],
            duration_seconds=ai_response["duration_seconds"]
        )

        # Send real-time update via WebSocket if connected
        await manager.send_personal_message({
            "type": "chat_response",
            "data": response.model_dump()
        }, current_user.id)

        return response

    except Exception as e:
        logger.error("Chat message processing failed", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message"
        )


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str,
    db: AsyncSession = Depends(get_db)
):
    """WebSocket endpoint for real-time chat"""

    # Authenticate user via token
    try:
        from ...core.security import verify_token
        from sqlalchemy import select

        user_id = verify_token(token)
        if not user_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Get user from database
        result = await db.execute(select(User).where(User.id == int(user_id)))
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    except Exception as e:
        logger.error("WebSocket authentication failed", error=str(e))
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Connect user
    await manager.connect(websocket, user.id)

    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            if message_data.get("type") == "chat_message":
                # Process chat message
                chat_message = ChatMessage(**message_data.get("data", {}))

                # This would trigger the same logic as the REST endpoint
                # For now, just echo back
                await manager.send_personal_message({
                    "type": "chat_response",
                    "data": {
                        "message": chat_message.message,
                        "response": "WebSocket message received (processing via REST endpoint recommended)",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }, user.id)

            elif message_data.get("type") == "ping":
                # Respond to ping
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }, user.id)

    except WebSocketDisconnect:
        manager.disconnect(user.id)
    except Exception as e:
        logger.error("WebSocket error", error=str(e), user_id=user.id)
        manager.disconnect(user.id)


@router.get("/providers")
async def get_available_providers(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of available AI providers"""
    return {
        "providers": ai_service.get_available_providers(),
        "default": ai_service.default_provider,
        "user_preference": current_user.preferred_ai_model
    }


@router.post("/provider")
async def set_user_provider(
    provider: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Set user's preferred AI provider"""

    available_providers = ai_service.get_available_providers()
    if provider not in available_providers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider}' not available. Available: {available_providers}"
        )

    current_user.preferred_ai_model = provider
    await db.commit()

    return {
        "message": f"Default AI provider set to {provider}",
        "provider": provider
    }


@router.get("/test-command")
async def test_command_parsing(
    command: str,
    current_user: User = Depends(get_current_active_user)
):
    """Test command parsing (for development/debugging)"""

    parsed = command_parser.parse_command(command)
    action = command_parser.generate_api_action(parsed)

    return {
        "command": command,
        "parsed": {
            "command_type": parsed["command_type"].value,
            "data": parsed["data"],
            "confidence": parsed["confidence"]
        },
        "action": action
    }