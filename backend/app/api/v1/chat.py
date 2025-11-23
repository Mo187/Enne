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
from ...services.conversation_memory import get_conversation_memory, clear_user_memory
from ...services.clarification_manager import clarification_manager

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


async def safe_db_commit_with_verification(
    db: AsyncSession,
    entity,
    entity_type: str,
    operation: str,
    user_id: int,
    verify_query = None
) -> Dict[str, Any]:
    """
    Safely commit a database operation with verification.

    Args:
        db: Database session
        entity: The entity being created/updated
        entity_type: Type of entity (e.g., "contact", "organization")
        operation: Operation type (e.g., "creation", "update")
        user_id: User ID for logging
        verify_query: Optional custom verification query

    Returns:
        Dict with success status and any error messages
    """
    try:
        await db.commit()
        await db.refresh(entity)

        logger.info(
            f"{entity_type.capitalize()} {operation} committed to database",
            entity_id=entity.id,
            entity_type=entity_type,
            operation=operation,
            user_id=user_id
        )

        # Verification step (if query provided)
        if verify_query is not None:
            verify_result = await db.execute(verify_query)
            verified_entity = verify_result.scalar_one_or_none()

            if not verified_entity:
                logger.error(
                    f"CRITICAL: {entity_type.capitalize()} verification failed after {operation}",
                    entity_id=entity.id,
                    entity_type=entity_type,
                    user_id=user_id
                )
                return {
                    "success": False,
                    "error": f"{entity_type.capitalize()} {operation} verification failed - database inconsistency detected"
                }

            logger.info(
                f"✓ {entity_type.capitalize()} {operation} verified successfully",
                entity_id=verified_entity.id,
                entity_type=entity_type,
                user_id=user_id
            )

        return {"success": True}

    except Exception as commit_error:
        logger.error(
            f"{entity_type.capitalize()} {operation} commit failed - rolling back",
            error=str(commit_error),
            entity_type=entity_type,
            user_id=user_id
        )
        await db.rollback()
        return {
            "success": False,
            "error": f"Failed to {operation} {entity_type} in database: {str(commit_error)}"
        }


async def execute_api_action(
    action: Dict[str, Any],
    user: User,
    db: AsyncSession
) -> Dict[str, Any]:
    """Execute API action from parsed command with transaction handling and verification"""

    method = action.get("method")
    endpoint = action.get("endpoint")
    data = action.get("data", {})
    params = action.get("params", {})

    # Log action start
    logger.info(
        "Starting API action execution",
        method=method,
        endpoint=endpoint,
        user_id=user.id,
        has_data=bool(data),
        has_params=bool(params)
    )

    try:
        if method == "POST" and "/contacts" in endpoint:
            # Create contact with explicit transaction handling
            if not data.get("name"):
                logger.warning("Contact creation failed: name required", user_id=user.id)
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
                    logger.warning(
                        "Contact creation failed: duplicate email",
                        email=data["email"],
                        user_id=user.id
                    )
                    return {"success": False, "error": "Contact with this email already exists"}

            # Create new contact with explicit transaction boundary
            contact = None
            try:
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
                await db.flush()  # Flush to get ID before commit

                # Commit transaction
                await db.commit()
                await db.refresh(contact)

                logger.info(
                    "Contact creation committed to database",
                    contact_id=contact.id,
                    contact_name=contact.name,
                    user_id=user.id
                )

            except Exception as commit_error:
                logger.error(
                    "Contact creation commit failed - rolling back",
                    error=str(commit_error),
                    user_id=user.id,
                    contact_name=data.get("name")
                )
                await db.rollback()
                return {
                    "success": False,
                    "error": f"Failed to save contact to database: {str(commit_error)}"
                }

            # Post-operation verification: Confirm contact exists in database
            try:
                verify_query = select(Contact).where(
                    and_(
                        Contact.user_id == user.id,
                        Contact.id == contact.id
                    )
                )
                verify_result = await db.execute(verify_query)
                verified_contact = verify_result.scalar_one_or_none()

                if not verified_contact:
                    logger.error(
                        "CRITICAL: Contact verification failed - not found after commit",
                        contact_id=contact.id,
                        user_id=user.id
                    )
                    return {
                        "success": False,
                        "error": "Contact creation verification failed - database inconsistency detected"
                    }

                logger.info(
                    "✓ Contact creation verified successfully",
                    contact_id=verified_contact.id,
                    contact_name=verified_contact.name,
                    user_id=user.id
                )

            except Exception as verify_error:
                logger.error(
                    "Contact verification query failed",
                    error=str(verify_error),
                    contact_id=contact.id if contact else None,
                    user_id=user.id
                )
                # Contact may exist but verification failed - still return success
                # since commit succeeded
                pass

            # Track created contact in conversation memory
            user_memory = get_conversation_memory(user.id)
            user_memory.track_entity_mention("contacts", {
                "id": contact.id,
                "name": contact.name,
                "email": contact.email
            })

            return {
                "success": True,
                "result": {
                    "id": contact.id,
                    "name": contact.name,
                    "email": contact.email,
                    "organization": contact.organization
                },
                "message": f"✓ CONFIRMED: Contact '{contact.name}' created successfully with ID {contact.id}"
            }

        elif method == "POST" and "/organizations" in endpoint:
            # Create organization with explicit transaction handling
            if not data.get("name"):
                logger.warning("Organization creation failed: name required", user_id=user.id)
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
                logger.warning(
                    "Organization creation failed: duplicate name",
                    name=data["name"],
                    user_id=user.id
                )
                return {"success": False, "error": "Organization with this name already exists"}

            # Create new organization with explicit transaction boundary
            organization = None
            try:
                organization = Organization(
                    user_id=user.id,
                    name=data["name"],
                    industry=data.get("industry"),
                    website=data.get("website"),
                    email=data.get("email"),
                    description=data.get("description")
                )

                db.add(organization)
                await db.flush()  # Flush to get ID before commit

                # Commit transaction
                await db.commit()
                await db.refresh(organization)

                logger.info(
                    "Organization creation committed to database",
                    org_id=organization.id,
                    org_name=organization.name,
                    user_id=user.id
                )

            except Exception as commit_error:
                logger.error(
                    "Organization creation commit failed - rolling back",
                    error=str(commit_error),
                    user_id=user.id,
                    org_name=data.get("name")
                )
                await db.rollback()
                return {
                    "success": False,
                    "error": f"Failed to save organization to database: {str(commit_error)}"
                }

            # Post-operation verification: Confirm organization exists in database
            try:
                verify_query = select(Organization).where(
                    and_(
                        Organization.user_id == user.id,
                        Organization.id == organization.id
                    )
                )
                verify_result = await db.execute(verify_query)
                verified_org = verify_result.scalar_one_or_none()

                if not verified_org:
                    logger.error(
                        "CRITICAL: Organization verification failed - not found after commit",
                        org_id=organization.id,
                        user_id=user.id
                    )
                    return {
                        "success": False,
                        "error": "Organization creation verification failed - database inconsistency detected"
                    }

                logger.info(
                    "✓ Organization creation verified successfully",
                    org_id=verified_org.id,
                    org_name=verified_org.name,
                    user_id=user.id
                )

            except Exception as verify_error:
                logger.error(
                    "Organization verification query failed",
                    error=str(verify_error),
                    org_id=organization.id if organization else None,
                    user_id=user.id
                )
                # Organization may exist but verification failed - still return success
                # since commit succeeded
                pass

            # Track created organization in conversation memory
            user_memory = get_conversation_memory(user.id)
            user_memory.track_entity_mention("organizations", {
                "id": organization.id,
                "name": organization.name
            })

            return {
                "success": True,
                "result": {
                    "id": organization.id,
                    "name": organization.name,
                    "industry": organization.industry
                },
                "message": f"✓ CONFIRMED: Organization '{organization.name}' created successfully with ID {organization.id}"
            }

        elif method == "POST" and "/projects" in endpoint:
            # Create project with explicit transaction handling
            if not data.get("name"):
                logger.warning("Project creation failed: name required", user_id=user.id)
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
                logger.warning(
                    "Project creation failed: duplicate name",
                    name=data["name"],
                    user_id=user.id
                )
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

            # Create new project with explicit transaction boundary
            project = None
            try:
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
                await db.flush()  # Flush to get ID before commit

                # Commit transaction
                await db.commit()
                await db.refresh(project)

                logger.info(
                    "Project creation committed to database",
                    project_id=project.id,
                    project_name=project.name,
                    user_id=user.id
                )

            except Exception as commit_error:
                logger.error(
                    "Project creation commit failed - rolling back",
                    error=str(commit_error),
                    user_id=user.id,
                    project_name=data.get("name")
                )
                await db.rollback()
                return {
                    "success": False,
                    "error": f"Failed to save project to database: {str(commit_error)}"
                }

            # Post-operation verification: Confirm project exists in database
            try:
                verify_query = select(Project).where(
                    and_(
                        Project.user_id == user.id,
                        Project.id == project.id
                    )
                )
                verify_result = await db.execute(verify_query)
                verified_project = verify_result.scalar_one_or_none()

                if not verified_project:
                    logger.error(
                        "CRITICAL: Project verification failed - not found after commit",
                        project_id=project.id,
                        user_id=user.id
                    )
                    return {
                        "success": False,
                        "error": "Project creation verification failed - database inconsistency detected"
                    }

                logger.info(
                    "✓ Project creation verified successfully",
                    project_id=verified_project.id,
                    project_name=verified_project.name,
                    user_id=user.id
                )

            except Exception as verify_error:
                logger.error(
                    "Project verification query failed",
                    error=str(verify_error),
                    project_id=project.id if project else None,
                    user_id=user.id
                )
                # Project may exist but verification failed - still return success
                # since commit succeeded
                pass

            # Track created project in conversation memory
            user_memory = get_conversation_memory(user.id)
            user_memory.track_entity_mention("projects", {
                "id": project.id,
                "name": project.name
            })

            return {
                "success": True,
                "result": {
                    "id": project.id,
                    "name": project.name,
                    "status": project.status,
                    "priority": project.priority
                },
                "message": f"✓ CONFIRMED: Project '{project.name}' created successfully with ID {project.id}"
            }

        elif method == "POST" and "/tasks" in endpoint:
            # Create task with explicit transaction handling
            if not data.get("name"):
                logger.warning("Task creation failed: name required", user_id=user.id)
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
                    logger.warning(
                        "Task creation failed: project not found",
                        project_name=project_name,
                        user_id=user.id
                    )
                    return {"success": False, "error": f"Project '{project_name}' not found"}
            else:
                # If no project specified, try to find a default project or require one
                logger.warning("Task creation failed: project required", user_id=user.id)
                return {"success": False, "error": "Project is required for task creation"}

            # Handle due date for task
            due_date = None
            if data.get("due_date"):
                try:
                    due_date = datetime.fromisoformat(data["due_date"]).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            # Create new task with explicit transaction boundary
            task = None
            try:
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
                await db.flush()  # Flush to get ID before commit

                # Commit transaction
                await db.commit()
                await db.refresh(task)

                logger.info(
                    "Task creation committed to database",
                    task_id=task.id,
                    task_name=task.name,
                    project_id=task.project_id,
                    user_id=user.id
                )

            except Exception as commit_error:
                logger.error(
                    "Task creation commit failed - rolling back",
                    error=str(commit_error),
                    user_id=user.id,
                    task_name=data.get("name")
                )
                await db.rollback()
                return {
                    "success": False,
                    "error": f"Failed to save task to database: {str(commit_error)}"
                }

            # Post-operation verification: Confirm task exists in database
            try:
                verify_query = select(Task).join(Project).where(
                    and_(
                        Project.user_id == user.id,
                        Task.id == task.id
                    )
                )
                verify_result = await db.execute(verify_query)
                verified_task = verify_result.scalar_one_or_none()

                if not verified_task:
                    logger.error(
                        "CRITICAL: Task verification failed - not found after commit",
                        task_id=task.id,
                        user_id=user.id
                    )
                    return {
                        "success": False,
                        "error": "Task creation verification failed - database inconsistency detected"
                    }

                logger.info(
                    "✓ Task creation verified successfully",
                    task_id=verified_task.id,
                    task_name=verified_task.name,
                    user_id=user.id
                )

            except Exception as verify_error:
                logger.error(
                    "Task verification query failed",
                    error=str(verify_error),
                    task_id=task.id if task else None,
                    user_id=user.id
                )
                # Task may exist but verification failed - still return success
                # since commit succeeded
                pass

            # Track created task in conversation memory
            user_memory = get_conversation_memory(user.id)
            user_memory.track_entity_mention("tasks", {
                "id": task.id,
                "name": task.name
            })

            return {
                "success": True,
                "result": {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status,
                    "priority": task.priority,
                    "project_id": task.project_id
                },
                "message": f"✓ CONFIRMED: Task '{task.name}' created successfully with ID {task.id}"
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
            user_memory = get_conversation_memory(user.id)

            identifier = action.get("identifier")

            # Check if identifier is a pronoun - use tracked contact
            if not identifier or (isinstance(identifier, str) and identifier.lower() in ["it", "that", "that one", "him", "her", "them"]):
                tracked_contact = user_memory.get_current_contact()
                if tracked_contact and tracked_contact.get("name"):
                    identifier = tracked_contact["name"]
                    logger.info(f"Resolved pronoun to tracked contact for UPDATE: {identifier}")
                elif not identifier:
                    return {"success": False, "error": "Contact identifier required"}

            # Try exact match first
            exact_query = select(Contact).where(
                and_(
                    Contact.user_id == user.id,
                    or_(
                        Contact.name.ilike(identifier),
                        Contact.email.ilike(identifier)
                    )
                )
            ).limit(10)

            exact_result = await db.execute(exact_query)
            contacts = exact_result.scalars().all()

            # If no exact match, try partial matching
            if not contacts:
                search_term = f"%{identifier}%"
                partial_query = select(Contact).where(
                    and_(
                        Contact.user_id == user.id,
                        or_(
                            Contact.name.ilike(search_term),
                            Contact.email.ilike(search_term)
                        )
                    )
                ).limit(10)

                partial_result = await db.execute(partial_query)
                contacts = partial_result.scalars().all()

            if not contacts:
                return {"success": False, "error": f"No contacts found matching '{identifier}'"}

            if len(contacts) > 1:
                # Multiple matches found - create clarification with manager
                matches_list = [
                    {
                        "id": contact.id,
                        "name": contact.name,
                        "email": contact.email,
                        "phone": contact.phone,
                        "organization": contact.organization
                    }
                    for contact in contacts
                ]

                # Store clarification for multi-turn resolution
                clarification_id = clarification_manager.create_clarification(
                    user_id=user.id,
                    clarification_type="multiple_contacts",
                    matches=matches_list,
                    original_action={
                        "method": "PUT",
                        "endpoint": "/contacts/update",
                        "data": data,
                        "identifier": identifier
                    }
                )

                return {
                    "success": False,
                    "requires_clarification": True,
                    "clarification_id": clarification_id,
                    "clarification_type": "multiple_contacts",
                    "matches": matches_list,
                    "original_action": {
                        "method": "PUT",
                        "endpoint": "/contacts/update",
                        "data": data,
                        "identifier": identifier
                    },
                    "message": f"Found {len(contacts)} contacts matching '{identifier}'. Please specify which one you want to update.",
                    "ai_context": f"I found {len(contacts)} contacts matching '{identifier}':\n" +
                                "\n".join([f"- {contact.name} ({contact.email if contact.email else 'no email'})" for contact in contacts]) +
                                "\n\nWhich contact did you want to update?"
                }

            # Single match found - proceed with update
            contact = contacts[0]

            # Store original values for logging
            original_data = {field: getattr(contact, field, None) for field in data.keys() if field != "identifier"}

            # Filter out pronouns from name updates (they're used for identification, not as new values)
            pronouns = ["him", "her", "them", "it", "that", "this", "he", "she", "they"]
            if "name" in data and isinstance(data["name"], str) and data["name"].lower() in pronouns:
                # Remove pronoun name from data - it's the identifier, not the new value
                data = {k: v for k, v in data.items() if k != "name"}
                logger.info(f"Filtered out pronoun name '{data.get('name')}' from UPDATE data")

            # Update fields with explicit transaction handling
            try:
                # Update fields that are provided and valid
                for field, value in data.items():
                    if field not in ["identifier"] and value is not None and hasattr(contact, field):
                        setattr(contact, field, value)

                # Commit transaction
                await db.commit()
                await db.refresh(contact)

                # Track updated contact
                user_memory.track_entity_mention("contacts", {
                    "id": contact.id,
                    "name": contact.name,
                    "email": contact.email
                })

                logger.info(
                    "Contact update committed to database",
                    contact_id=contact.id,
                    contact_name=contact.name,
                    updated_fields=list(data.keys()),
                    user_id=user.id
                )

            except Exception as commit_error:
                logger.error(
                    "Contact update commit failed - rolling back",
                    error=str(commit_error),
                    contact_id=contact.id,
                    user_id=user.id
                )
                await db.rollback()
                return {
                    "success": False,
                    "error": f"Failed to update contact in database: {str(commit_error)}"
                }

            # Post-operation verification: Confirm update was applied
            try:
                verify_query = select(Contact).where(
                    and_(
                        Contact.user_id == user.id,
                        Contact.id == contact.id
                    )
                )
                verify_result = await db.execute(verify_query)
                verified_contact = verify_result.scalar_one_or_none()

                if not verified_contact:
                    logger.error(
                        "CRITICAL: Contact verification failed after update",
                        contact_id=contact.id,
                        user_id=user.id
                    )
                    return {
                        "success": False,
                        "error": "Contact update verification failed - contact not found"
                    }

                logger.info(
                    "✓ Contact update verified successfully",
                    contact_id=verified_contact.id,
                    contact_name=verified_contact.name,
                    user_id=user.id
                )

            except Exception as verify_error:
                logger.error(
                    "Contact update verification query failed",
                    error=str(verify_error),
                    contact_id=contact.id,
                    user_id=user.id
                )
                # Update likely succeeded but verification failed
                pass

            return {
                "success": True,
                "result": {
                    "id": contact.id,
                    "name": contact.name,
                    "email": contact.email,
                    "phone": contact.phone
                },
                "message": f"✓ CONFIRMED: Contact '{contact.name}' updated successfully"
            }

        elif method == "PUT" and "/organizations/update" in endpoint:
            # Update organization with smart search
            user_memory = get_conversation_memory(user.id)

            identifier = action.get("identifier")
            if not identifier:
                return {"success": False, "error": "Organization identifier required"}

            # Try exact match first
            exact_query = select(Organization).where(
                and_(
                    Organization.user_id == user.id,
                    Organization.name.ilike(identifier)
                )
            ).limit(10)

            exact_result = await db.execute(exact_query)
            organizations = exact_result.scalars().all()

            # If no exact match, try partial matching
            if not organizations:
                search_term = f"%{identifier}%"
                partial_query = select(Organization).where(
                    and_(
                        Organization.user_id == user.id,
                        Organization.name.ilike(search_term)
                    )
                ).limit(10)

                partial_result = await db.execute(partial_query)
                organizations = partial_result.scalars().all()

            if not organizations:
                return {"success": False, "error": f"No organizations found matching '{identifier}'"}

            if len(organizations) > 1:
                # Multiple matches found - create clarification with manager
                matches_list = [
                    {
                        "id": org.id,
                        "name": org.name,
                        "industry": org.industry,
                        "website": org.website
                    }
                    for org in organizations
                ]

                # Store clarification for multi-turn resolution
                clarification_id = clarification_manager.create_clarification(
                    user_id=user.id,
                    clarification_type="multiple_organizations",
                    matches=matches_list,
                    original_action={
                        "method": "PUT",
                        "endpoint": "/organizations/update",
                        "data": data,
                        "identifier": identifier
                    }
                )

                return {
                    "success": False,
                    "requires_clarification": True,
                    "clarification_id": clarification_id,
                    "clarification_type": "multiple_organizations",
                    "matches": matches_list,
                    "original_action": {
                        "method": "PUT",
                        "endpoint": "/organizations/update",
                        "data": data,
                        "identifier": identifier
                    },
                    "message": f"Found {len(organizations)} organizations matching '{identifier}'. Please specify which one you want to update.",
                    "ai_context": f"I found {len(organizations)} organizations matching '{identifier}':\n" +
                                "\n".join([f"- {org.name} ({org.industry if org.industry else 'no industry specified'})" for org in organizations]) +
                                "\n\nWhich organization did you want to update?"
                }

            # Single match found - proceed with update
            organization = organizations[0]

            # Filter out pronouns from name updates (they're used for identification, not as new values)
            pronouns = ["it", "that", "this", "them"]
            if "name" in data and isinstance(data["name"], str) and data["name"].lower() in pronouns:
                # Remove pronoun name from data - it's the identifier, not the new value
                data = {k: v for k, v in data.items() if k != "name"}
                logger.info(f"Filtered out pronoun name from organization UPDATE data")

            # Update fields that are provided and valid
            for field, value in data.items():
                if field not in ["identifier"] and value is not None and hasattr(organization, field):
                    setattr(organization, field, value)

            await db.commit()
            await db.refresh(organization)

            # Track updated organization
            user_memory.track_entity_mention("organizations", {
                "id": organization.id,
                "name": organization.name
            })

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
            user_memory = get_conversation_memory(user.id)

            identifier = action.get("identifier")
            if not identifier:
                return {"success": False, "error": "Project identifier required"}

            # Try exact match first
            exact_query = select(Project).where(
                and_(
                    Project.user_id == user.id,
                    Project.name.ilike(identifier)
                )
            ).limit(10)

            exact_result = await db.execute(exact_query)
            projects = exact_result.scalars().all()

            # If no exact match, try partial matching
            if not projects:
                search_term = f"%{identifier}%"
                partial_query = select(Project).where(
                    and_(
                        Project.user_id == user.id,
                        Project.name.ilike(search_term)
                    )
                ).limit(10)

                partial_result = await db.execute(partial_query)
                projects = partial_result.scalars().all()

            if not projects:
                return {"success": False, "error": f"No projects found matching '{identifier}'"}

            if len(projects) > 1:
                # Multiple matches found - create clarification with manager
                matches_list = [
                    {
                        "id": proj.id,
                        "name": proj.name,
                        "status": proj.status,
                        "priority": proj.priority
                    }
                    for proj in projects
                ]

                # Store clarification for multi-turn resolution
                clarification_id = clarification_manager.create_clarification(
                    user_id=user.id,
                    clarification_type="multiple_projects",
                    matches=matches_list,
                    original_action={
                        "method": "PUT",
                        "endpoint": "/projects/update",
                        "data": data,
                        "identifier": identifier
                    }
                )

                return {
                    "success": False,
                    "requires_clarification": True,
                    "clarification_id": clarification_id,
                    "clarification_type": "multiple_projects",
                    "matches": matches_list,
                    "original_action": {
                        "method": "PUT",
                        "endpoint": "/projects/update",
                        "data": data,
                        "identifier": identifier
                    },
                    "message": f"Found {len(projects)} projects matching '{identifier}'. Please specify which one you want to update.",
                    "ai_context": f"I found {len(projects)} projects matching '{identifier}':\n" +
                                "\n".join([f"- {proj.name} (Status: {proj.status})" for proj in projects]) +
                                "\n\nWhich project did you want to update?"
                }

            # Single match found - proceed with update
            project = projects[0]

            # Filter out pronouns from name updates (they're used for identification, not as new values)
            pronouns = ["it", "that", "this", "them"]
            if "name" in data and isinstance(data["name"], str) and data["name"].lower() in pronouns:
                # Remove pronoun name from data - it's the identifier, not the new value
                data = {k: v for k, v in data.items() if k != "name"}
                logger.info(f"Filtered out pronoun name from project UPDATE data")

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

            # Track updated project
            user_memory.track_entity_mention("projects", {
                "id": project.id,
                "name": project.name
            })

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
            user_memory = get_conversation_memory(user.id)

            identifier = action.get("identifier")
            if not identifier:
                return {"success": False, "error": "Task identifier required"}

            # Try exact match first
            exact_query = select(Task).join(Project).where(
                and_(
                    Project.user_id == user.id,
                    Task.name.ilike(identifier)
                )
            ).limit(10)

            exact_result = await db.execute(exact_query)
            tasks = exact_result.scalars().all()

            # If no exact match, try partial matching
            if not tasks:
                search_term = f"%{identifier}%"
                partial_query = select(Task).join(Project).where(
                    and_(
                        Project.user_id == user.id,
                        Task.name.ilike(search_term)
                    )
                ).limit(10)

                partial_result = await db.execute(partial_query)
                tasks = partial_result.scalars().all()

            if not tasks:
                return {"success": False, "error": f"No tasks found matching '{identifier}'"}

            if len(tasks) > 1:
                # Multiple matches found - create clarification with manager
                # Load projects for context
                project_ids = [task.project_id for task in tasks]
                projects_query = select(Project).where(Project.id.in_(project_ids))
                projects_result = await db.execute(projects_query)
                projects = {proj.id: proj.name for proj in projects_result.scalars().all()}

                matches_list = [
                    {
                        "id": task.id,
                        "name": task.name,
                        "status": task.status,
                        "priority": task.priority,
                        "project": projects.get(task.project_id, "Unknown"),
                        "assignee": task.assignee
                    }
                    for task in tasks
                ]

                # Store clarification for multi-turn resolution
                clarification_id = clarification_manager.create_clarification(
                    user_id=user.id,
                    clarification_type="multiple_tasks",
                    matches=matches_list,
                    original_action={
                        "method": "PUT",
                        "endpoint": "/tasks/update",
                        "data": data,
                        "identifier": identifier
                    }
                )

                return {
                    "success": False,
                    "requires_clarification": True,
                    "clarification_id": clarification_id,
                    "clarification_type": "multiple_tasks",
                    "matches": matches_list,
                    "original_action": {
                        "method": "PUT",
                        "endpoint": "/tasks/update",
                        "data": data,
                        "identifier": identifier
                    },
                    "message": f"Found {len(tasks)} tasks matching '{identifier}'. Please specify which one you want to update.",
                    "ai_context": f"I found {len(tasks)} tasks matching '{identifier}':\n" +
                                "\n".join([f"- {task.name} (Project: {projects.get(task.project_id, 'Unknown')})" for task in tasks]) +
                                "\n\nWhich task did you want to update?"
                }

            # Single match found - proceed with update
            task = tasks[0]

            # Filter out pronouns from name updates (they're used for identification, not as new values)
            pronouns = ["it", "that", "this", "them"]
            if "name" in data and isinstance(data["name"], str) and data["name"].lower() in pronouns:
                # Remove pronoun name from data - it's the identifier, not the new value
                data = {k: v for k, v in data.items() if k != "name"}
                logger.info(f"Filtered out pronoun name from task UPDATE data")

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

            # Track updated task
            user_memory.track_entity_mention("tasks", {
                "id": task.id,
                "name": task.name
            })

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

        elif method == "DELETE" and "/contacts/delete" in endpoint:
            # Delete contact with smart search
            user_memory = get_conversation_memory(user.id)

            identifier = action.get("identifier")

            # Check if identifier is a pronoun - use tracked contact
            if not identifier or identifier.lower() in ["it", "that", "that one", "him", "her", "them"]:
                tracked_contact = user_memory.get_current_contact()
                if tracked_contact and tracked_contact.get("name"):
                    identifier = tracked_contact["name"]
                    logger.info(f"Resolved pronoun to tracked contact: {identifier}")
                elif not identifier:
                    return {"success": False, "error": "Contact identifier required"}

            # Search for contact using partial matching
            search_term = f"%{identifier}%"
            query = select(Contact).where(
                and_(
                    Contact.user_id == user.id,
                    or_(
                        Contact.name.ilike(search_term),
                        Contact.email.ilike(search_term)
                    )
                )
            ).limit(10)

            result = await db.execute(query)
            contacts = result.scalars().all()

            if not contacts:
                return {"success": False, "error": f"No contacts found matching '{identifier}'"}

            if len(contacts) > 1:
                # Multiple matches - create clarification with manager
                matches_list = [
                    {
                        "id": contact.id,
                        "name": contact.name,
                        "email": contact.email,
                        "phone": contact.phone,
                        "organization": contact.organization
                    }
                    for contact in contacts
                ]

                # Store clarification for multi-turn resolution
                clarification_id = clarification_manager.create_clarification(
                    user_id=user.id,
                    clarification_type="multiple_contacts",
                    matches=matches_list,
                    original_action={
                        "method": "DELETE",
                        "endpoint": "/contacts/delete",
                        "identifier": identifier
                    }
                )

                return {
                    "success": False,
                    "requires_clarification": True,
                    "clarification_id": clarification_id,
                    "clarification_type": "multiple_contacts",
                    "matches": matches_list,
                    "original_action": {
                        "method": "DELETE",
                        "endpoint": "/contacts/delete",
                        "identifier": identifier
                    },
                    "message": f"Found {len(contacts)} contacts matching '{identifier}'. Which one do you want to delete?",
                    "ai_context": f"I found {len(contacts)} contacts matching '{identifier}':\n" +
                                "\n".join([f"- {contact.name} ({contact.email if contact.email else 'no email'})" for contact in contacts]) +
                                "\n\nWhich contact did you want to delete?"
                }

            # Single match - delete it
            contact = contacts[0]
            contact_name = contact.name
            contact_id = contact.id

            try:
                await db.delete(contact)
                await db.flush()
                await db.commit()
                logger.info("Contact deletion committed", contact_id=contact_id, contact_name=contact_name)

                # Verify deletion
                verify_query = select(Contact).where(Contact.id == contact_id)
                verify_result = await db.execute(verify_query)
                if verify_result.scalar_one_or_none():
                    logger.error("CRITICAL: Verification failed - contact still exists after deletion")
                    return {"success": False, "error": "Deletion verification failed"}

                logger.info("Contact deletion verified successfully", contact_id=contact_id)

                return {
                    "success": True,
                    "result": {
                        "id": contact_id,
                        "name": contact_name
                    },
                    "message": f"✓ CONFIRMED: Contact '{contact_name}' deleted successfully"
                }

            except Exception as commit_error:
                logger.error("Commit failed during deletion - rolling back", error=str(commit_error))
                await db.rollback()
                return {"success": False, "error": f"Failed to delete: {str(commit_error)}"}

        elif method == "DELETE" and "/organizations/delete" in endpoint:
            # Delete organization with smart search
            user_memory = get_conversation_memory(user.id)

            identifier = action.get("identifier")

            # Check if identifier is a pronoun - use tracked organization
            if not identifier or identifier.lower() in ["it", "that", "that one"]:
                tracked_org = user_memory.get_current_organization()
                if tracked_org and tracked_org.get("name"):
                    identifier = tracked_org["name"]
                    logger.info(f"Resolved pronoun to tracked organization: {identifier}")
                elif not identifier:
                    return {"success": False, "error": "Organization identifier required"}

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
                matches_list = [
                    {"id": org.id, "name": org.name, "industry": org.industry}
                    for org in organizations
                ]

                # Store clarification for multi-turn resolution
                clarification_id = clarification_manager.create_clarification(
                    user_id=user.id,
                    clarification_type="multiple_organizations",
                    matches=matches_list,
                    original_action={
                        "method": "DELETE",
                        "endpoint": "/organizations/delete",
                        "identifier": identifier
                    }
                )

                return {
                    "success": False,
                    "requires_clarification": True,
                    "clarification_id": clarification_id,
                    "clarification_type": "multiple_organizations",
                    "matches": matches_list,
                    "original_action": {
                        "method": "DELETE",
                        "endpoint": "/organizations/delete",
                        "identifier": identifier
                    },
                    "message": f"Found {len(organizations)} organizations matching '{identifier}'. Which one do you want to delete?"
                }

            organization = organizations[0]
            org_name = organization.name
            org_id = organization.id

            try:
                await db.delete(organization)
                await db.flush()
                await db.commit()
                logger.info("Organization deletion committed", org_id=org_id, org_name=org_name)

                # Verify deletion
                verify_query = select(Organization).where(Organization.id == org_id)
                if (await db.execute(verify_query)).scalar_one_or_none():
                    logger.error("CRITICAL: Verification failed - organization still exists")
                    return {"success": False, "error": "Deletion verification failed"}

                return {
                    "success": True,
                    "result": {"id": org_id, "name": org_name},
                    "message": f"✓ CONFIRMED: Organization '{org_name}' deleted successfully"
                }

            except Exception as commit_error:
                logger.error("Commit failed during deletion", error=str(commit_error))
                await db.rollback()
                return {"success": False, "error": f"Failed to delete: {str(commit_error)}"}

        elif method == "DELETE" and "/projects/delete" in endpoint:
            # Delete project
            user_memory = get_conversation_memory(user.id)

            identifier = action.get("identifier")

            # Check if identifier is a pronoun - use tracked project
            if not identifier or identifier.lower() in ["it", "that", "that one"]:
                tracked_project = user_memory.get_current_project()
                if tracked_project and tracked_project.get("name"):
                    identifier = tracked_project["name"]
                    logger.info(f"Resolved pronoun to tracked project: {identifier}")
                elif not identifier:
                    return {"success": False, "error": "Project identifier required"}

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
                matches_list = [
                    {"id": proj.id, "name": proj.name, "status": proj.status}
                    for proj in projects
                ]

                # Store clarification for multi-turn resolution
                clarification_id = clarification_manager.create_clarification(
                    user_id=user.id,
                    clarification_type="multiple_projects",
                    matches=matches_list,
                    original_action={
                        "method": "DELETE",
                        "endpoint": "/projects/delete",
                        "identifier": identifier
                    }
                )

                return {
                    "success": False,
                    "requires_clarification": True,
                    "clarification_id": clarification_id,
                    "clarification_type": "multiple_projects",
                    "matches": matches_list,
                    "original_action": {
                        "method": "DELETE",
                        "endpoint": "/projects/delete",
                        "identifier": identifier
                    },
                    "message": f"Found {len(projects)} projects matching '{identifier}'. Which one?"
                }

            project = projects[0]
            proj_name = project.name
            proj_id = project.id

            try:
                await db.delete(project)
                await db.flush()
                await db.commit()
                logger.info("Project deletion committed", proj_id=proj_id, proj_name=proj_name)

                return {
                    "success": True,
                    "result": {"id": proj_id, "name": proj_name},
                    "message": f"✓ CONFIRMED: Project '{proj_name}' deleted successfully"
                }

            except Exception as commit_error:
                logger.error("Commit failed during deletion", error=str(commit_error))
                await db.rollback()
                return {"success": False, "error": f"Failed to delete: {str(commit_error)}"}

        elif method == "DELETE" and "/tasks/delete" in endpoint:
            # Delete task
            identifier = action.get("identifier")
            if not identifier:
                return {"success": False, "error": "Task identifier required"}

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
                # Load projects for context
                project_ids = [task.project_id for task in tasks]
                projects_query = select(Project).where(Project.id.in_(project_ids))
                projects_result = await db.execute(projects_query)
                projects = {proj.id: proj.name for proj in projects_result.scalars().all()}

                matches_list = [
                    {
                        "id": task.id,
                        "name": task.name,
                        "project": projects.get(task.project_id, "Unknown")
                    }
                    for task in tasks
                ]

                # Store clarification for multi-turn resolution
                clarification_id = clarification_manager.create_clarification(
                    user_id=user.id,
                    clarification_type="multiple_tasks",
                    matches=matches_list,
                    original_action={
                        "method": "DELETE",
                        "endpoint": "/tasks/delete",
                        "identifier": identifier
                    }
                )

                return {
                    "success": False,
                    "requires_clarification": True,
                    "clarification_id": clarification_id,
                    "clarification_type": "multiple_tasks",
                    "matches": matches_list,
                    "original_action": {
                        "method": "DELETE",
                        "endpoint": "/tasks/delete",
                        "identifier": identifier
                    },
                    "message": f"Found {len(tasks)} tasks matching '{identifier}'. Which one?"
                }

            task = tasks[0]
            task_name = task.name
            task_id = task.id

            try:
                await db.delete(task)
                await db.flush()
                await db.commit()
                logger.info("Task deletion committed", task_id=task_id, task_name=task_name)

                return {
                    "success": True,
                    "result": {"id": task_id, "name": task_name},
                    "message": f"✓ CONFIRMED: Task '{task_name}' deleted successfully"
                }

            except Exception as commit_error:
                logger.error("Commit failed during deletion", error=str(commit_error))
                await db.rollback()
                return {"success": False, "error": f"Failed to delete: {str(commit_error)}"}

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
                # Fetch the selected contact
                contact_query = select(Contact).where(
                    and_(Contact.user_id == user.id, Contact.id == selected_id)
                )
                contact_result = await db.execute(contact_query)
                contact = contact_result.scalar_one_or_none()

                if not contact:
                    return {"success": False, "error": f"Contact with ID {selected_id} not found"}

                # Check if this is UPDATE or DELETE
                if original_action.get("method") == "DELETE":
                    # DELETE operation
                    contact_name = contact.name
                    try:
                        await db.delete(contact)
                        await db.commit()
                        return {
                            "success": True,
                            "result": {"id": selected_id, "name": contact_name},
                            "message": f"✓ CONFIRMED: Contact '{contact_name}' deleted successfully"
                        }
                    except Exception as e:
                        await db.rollback()
                        return {"success": False, "error": f"Failed to delete: {str(e)}"}
                else:
                    # UPDATE operation
                    update_data = original_action.get("data", {})
                    for field, value in update_data.items():
                        if field not in ["identifier"] and value is not None and hasattr(contact, field):
                            setattr(contact, field, value)

                    await db.commit()
                    await db.refresh(contact)

                    # Track entity after clarification resolution
                    user_memory = get_conversation_memory(user.id)
                    user_memory.track_entity_mention("contacts", {
                        "id": contact.id,
                        "name": contact.name,
                        "email": contact.email
                    })

                    return {
                        "success": True,
                        "result": {
                            "id": contact.id,
                            "name": contact.name,
                            "email": contact.email,
                            "phone": contact.phone
                        },
                        "message": f"✓ CONFIRMED: Contact '{contact.name}' updated successfully"
                    }

            elif clarification_type == "multiple_organizations":
                # Fetch the selected organization
                org_query = select(Organization).where(
                    and_(Organization.user_id == user.id, Organization.id == selected_id)
                )
                org_result = await db.execute(org_query)
                organization = org_result.scalar_one_or_none()

                if not organization:
                    return {"success": False, "error": f"Organization with ID {selected_id} not found"}

                # Check if this is UPDATE or DELETE
                if original_action.get("method") == "DELETE":
                    # DELETE operation
                    org_name = organization.name
                    try:
                        await db.delete(organization)
                        await db.commit()
                        return {
                            "success": True,
                            "result": {"id": selected_id, "name": org_name},
                            "message": f"✓ CONFIRMED: Organization '{org_name}' deleted successfully"
                        }
                    except Exception as e:
                        await db.rollback()
                        return {"success": False, "error": f"Failed to delete: {str(e)}"}
                else:
                    # UPDATE operation
                    update_data = original_action.get("data", {})
                    for field, value in update_data.items():
                        if field not in ["identifier"] and value is not None and hasattr(organization, field):
                            setattr(organization, field, value)

                    await db.commit()
                    await db.refresh(organization)

                    # Track entity after clarification resolution
                    user_memory = get_conversation_memory(user.id)
                    user_memory.track_entity_mention("organizations", {
                        "id": organization.id,
                        "name": organization.name
                    })

                    return {
                        "success": True,
                        "result": {
                            "id": organization.id,
                            "name": organization.name,
                            "industry": organization.industry
                        },
                        "message": f"✓ CONFIRMED: Organization '{organization.name}' updated successfully"
                    }

            elif clarification_type == "multiple_projects":
                # Fetch the selected project
                project_query = select(Project).where(
                    and_(Project.user_id == user.id, Project.id == selected_id)
                )
                project_result = await db.execute(project_query)
                project = project_result.scalar_one_or_none()

                if not project:
                    return {"success": False, "error": f"Project with ID {selected_id} not found"}

                # Check if this is UPDATE or DELETE
                if original_action.get("method") == "DELETE":
                    # DELETE operation
                    proj_name = project.name
                    try:
                        await db.delete(project)
                        await db.commit()
                        return {
                            "success": True,
                            "result": {"id": selected_id, "name": proj_name},
                            "message": f"✓ CONFIRMED: Project '{proj_name}' deleted successfully"
                        }
                    except Exception as e:
                        await db.rollback()
                        return {"success": False, "error": f"Failed to delete: {str(e)}"}

                # UPDATE operation
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

                # Track entity after clarification resolution
                user_memory = get_conversation_memory(user.id)
                user_memory.track_entity_mention("projects", {
                    "id": project.id,
                    "name": project.name
                })

                return {
                    "success": True,
                    "result": {
                        "id": project.id,
                        "name": project.name,
                        "status": project.status,
                        "priority": project.priority
                    },
                    "message": f"✓ CONFIRMED: Project '{project.name}' updated successfully"
                }

            elif clarification_type == "multiple_tasks":
                # Fetch the selected task
                task_query = select(Task).join(Project).where(
                    and_(Project.user_id == user.id, Task.id == selected_id)
                )
                task_result = await db.execute(task_query)
                task = task_result.scalar_one_or_none()

                if not task:
                    return {"success": False, "error": f"Task with ID {selected_id} not found"}

                # Check if this is UPDATE or DELETE
                if original_action.get("method") == "DELETE":
                    # DELETE operation
                    task_name = task.name
                    try:
                        await db.delete(task)
                        await db.commit()
                        return {
                            "success": True,
                            "result": {"id": selected_id, "name": task_name},
                            "message": f"✓ CONFIRMED: Task '{task_name}' deleted successfully"
                        }
                    except Exception as e:
                        await db.rollback()
                        return {"success": False, "error": f"Failed to delete: {str(e)}"}

                # UPDATE operation
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

                # Track entity after clarification resolution
                user_memory = get_conversation_memory(user.id)
                user_memory.track_entity_mention("tasks", {
                    "id": task.id,
                    "name": task.name
                })

                return {
                    "success": True,
                    "result": {
                        "id": task.id,
                        "name": task.name,
                        "status": task.status,
                        "priority": task.priority
                    },
                    "message": f"✓ CONFIRMED: Task '{task.name}' updated successfully"
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
        logger.error(
            "API action execution failed - unhandled exception",
            error=str(e),
            error_type=type(e).__name__,
            method=method,
            endpoint=endpoint,
            user_id=user.id,
            action=action
        )
        return {
            "success": False,
            "error": f"Failed to execute action: {str(e)}"
        }

    # If we reach here, the action wasn't handled
    logger.warning(
        "API action not recognized",
        method=method,
        endpoint=endpoint,
        user_id=user.id
    )
    return {
        "success": False,
        "error": f"Unsupported API action: {method} {endpoint}"
    }


@router.post("/message", response_model=ChatResponse)
async def send_chat_message(
    chat_data: ChatMessage,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Send a message to the AI assistant and get response with intelligent context management"""

    user_memory = get_conversation_memory(current_user.id)

    try:
        # ============================================
        # CLARIFICATION RESOLUTION HANDLER
        # ============================================
        # Check if user is responding to a pending clarification
        clarification_id = clarification_manager.get_latest_clarification_for_user(current_user.id)

        if clarification_id:
            logger.info(
                "Checking for clarification resolution",
                user_id=current_user.id,
                clarification_id=clarification_id,
                user_message=chat_data.message[:50]
            )

            # Try to resolve the clarification with user's message
            resolved = clarification_manager.resolve_clarification(
                clarification_id=clarification_id,
                selected_identifier=chat_data.message,
                user_id=current_user.id
            )

            if resolved:
                # User answered clarification - execute original action with selected entity
                logger.info(
                    "Clarification resolved successfully",
                    user_id=current_user.id,
                    selected_id=resolved["selected_id"],
                    clarification_type=resolved["clarification_type"]
                )

                # Execute the original action with the selected ID
                original_action = resolved["original_action"]
                original_action["clarification_resolved"] = True
                original_action["selected_id"] = resolved["selected_id"]

                # Execute the action
                execution_result = await execute_api_action(original_action, current_user, db)

                # Build user message with execution results
                if execution_result and execution_result.get("success"):
                    user_message = f"User resolved clarification by selecting entity ID {resolved['selected_id']}.\n\n"
                    user_message += f"Action completed successfully: {execution_result.get('message', 'Operation completed')}\n\n"
                    user_message += "Acknowledge the successful operation in a friendly, conversational way."

                    if execution_result.get("result"):
                        user_message += f"\n\nResult data:\n{json.dumps(execution_result['result'], indent=2)}"
                else:
                    error_msg = execution_result.get("error", "Unknown error") if execution_result else "No result"
                    user_message = f"User resolved clarification but the operation failed: {error_msg}"

                # Generate AI response
                ai_response = await ai_service.generate_crm_response(
                    user_message=user_message,
                    conversation_history=chat_data.conversation_history,
                    user_context={
                        "user_id": current_user.id,
                        "name": current_user.name,
                        "email": current_user.email
                    },
                    provider=chat_data.provider
                )

                # Return response in ChatResponse format
                response = ChatResponse(
                    message=chat_data.message,
                    response=ai_response["response"],
                    command_detected=True,
                    command_type=original_action.get("method"),
                    api_action=original_action,
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
            else:
                # Couldn't resolve - maybe user said something else
                logger.info(
                    "Could not resolve clarification with user message",
                    user_id=current_user.id,
                    message=chat_data.message[:50]
                )
                # Continue with normal command parsing

        # ============================================
        # NORMAL COMMAND PARSING (existing code)
        # ============================================
        # Inject last AI offer into conversation history for better follow-up detection
        last_offer = user_memory.get_last_offer()
        if last_offer and chat_data.conversation_history:
            # Add as the most recent assistant message
            chat_data.conversation_history.append({
                "role": "assistant",
                "content": last_offer
            })

        # Parse the command using LLM with conversation history for follow-up detection
        parsed_command = await llm_command_parser.parse_command(
            chat_data.message,
            user_context={"user_id": current_user.id},
            conversation_history=chat_data.conversation_history
        )
        command_detected = parsed_command["confidence"] > 0.7
        api_action = None
        execution_result = None

        # If it's a recognized command, try to execute it
        if command_detected and parsed_command["intent"] != "unknown":
            # Skip execution for follow-up questions - AI will answer from conversation history
            if parsed_command["intent"] == "answer_from_context":
                # Don't execute any tools - the data is already in conversation history
                pass
            else:
                api_action = llm_command_parser.generate_api_action(parsed_command)

                # If intent is get_latest_email, inject tracked email ID from per-user memory
                if parsed_command.get("intent") == "get_latest_email" and api_action:
                    tracked_email_id = user_memory.get_current_email_id()

                    if tracked_email_id:
                        if "parameters" not in api_action:
                            api_action["parameters"] = {}
                        api_action["parameters"]["email_id"] = tracked_email_id

                        # Get email context for logging
                        email_subject = user_memory.recent_data.get("current_email_subject", "Unknown")
                        email_from = user_memory.recent_data.get("current_email_from", "Unknown")
                        email_timestamp = user_memory.recent_data.get("current_email_timestamp", "Unknown")

                        logger.info(
                            f"Injected tracked email into action: email_id={tracked_email_id}, "
                            f"subject={email_subject[:50] if email_subject and email_subject != 'Unknown' else email_subject}, "
                            f"from={email_from}, tracked_at={email_timestamp}, user_message='{chat_data.message[:50]}...'"
                        )
                    else:
                        logger.warning(
                            f"get_latest_email intent but no tracked email ID available. "
                            f"user_message='{chat_data.message}', "
                            f"conversation_length={len(chat_data.conversation_history) if chat_data.conversation_history else 0}"
                        )

                # Execute the action if it's supported
                if api_action["method"] != "UNKNOWN":
                    # Log MCP tool calls for email searches
                    if parsed_command.get("intent") == "search_emails" and api_action.get("method") == "MCP_TOOL":
                        logger.info(
                            f"🔍 EMAIL SEARCH: Calling MCP tool with parameters: {api_action.get('parameters', {})}"
                        )

                    execution_result = await execute_api_action(api_action, current_user, db)
                    logger.info(
                        "Tool execution completed",
                        success=execution_result.get("success") if execution_result else None,
                        has_result=execution_result is not None,
                        user_id=current_user.id
                    )

                    # Log email search results
                    if parsed_command.get("intent") == "search_emails" and execution_result and execution_result.get("success"):
                        result_data = execution_result.get("result", [])
                        if isinstance(result_data, list) and result_data:
                            logger.info(
                                f"🔍 EMAIL SEARCH RESULTS: Found {len(result_data)} emails. "
                                f"First email: id={result_data[0].get('id')}, "
                                f"subject='{result_data[0].get('subject', 'No subject')[:50]}...', "
                                f"from='{result_data[0].get('from', 'Unknown')[:50]}...'"
                            )
                        else:
                            logger.info("🔍 EMAIL SEARCH RESULTS: No emails found")

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

        # Handle follow-up questions that should be answered from context
        if parsed_command.get("intent") == "answer_from_context":
            user_message = f"""User asked: "{chat_data.message}"

IMPORTANT: This is a follow-up question about data from the previous conversation.
Look at the conversation history above to find the relevant information (email data, contact info, etc.).
The data was already retrieved and shown in previous messages - DO NOT say you need to retrieve it again.
Answer the question using the information from the conversation history.
Be natural and reference the data confidently since you already have access to it.
"""

        # Defensive check: ensure execution_result is not None
        if execution_result is None:
            logger.error(
                "CRITICAL: execute_api_action returned None",
                api_action=api_action,
                user_id=current_user.id
            )
            execution_result = {
                "success": False,
                "error": "Internal error: operation returned no result"
            }

        if execution_result:
            if execution_result["success"]:
                # Detect query intent for better response formatting
                query_intent = detect_query_intent(chat_data.message, parsed_command.get("intent", ""))

                # Track data payloads in conversation memory for follow-up questions
                if execution_result.get("result"):
                    result_data = execution_result["result"]

                    # Store data payloads by type
                    if isinstance(result_data, list) and result_data:
                        # Determine data type from intent
                        intent = parsed_command.get("intent", "")
                        if "contact" in intent:
                            user_memory.store_data_payload("contacts", result_data)
                        elif "organization" in intent:
                            user_memory.store_data_payload("organizations", result_data)
                        elif "project" in intent:
                            user_memory.store_data_payload("projects", result_data)
                        elif "task" in intent:
                            user_memory.store_data_payload("tasks", result_data)
                        elif "email" in intent:
                            user_memory.store_data_payload("emails", result_data)

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
                                if 'subject' in item and 'from' in item:  # Email
                                    from_name = item.get('from_name', item.get('from', 'Unknown'))
                                    received_date = item.get('received_date', '')
                                    # Format date to be more readable
                                    if received_date:
                                        try:
                                            from datetime import datetime
                                            dt = datetime.fromisoformat(received_date.replace('Z', '+00:00'))
                                            date_str = dt.strftime('%Y-%m-%d %H:%M')
                                        except:
                                            date_str = received_date[:10]  # Just the date part
                                    else:
                                        date_str = 'unknown date'

                                    data_summary += f"{i}. {item.get('subject', 'No subject')} - from {from_name} ({date_str})\n"
                                    # Add body preview if available
                                    body_preview = item.get('body_preview', '')
                                    if body_preview:
                                        preview_text = body_preview.strip()[:150]  # First 150 chars
                                        if len(body_preview.strip()) > 150:
                                            preview_text += "..."
                                        data_summary += f"    Body Preview (TRUNCATED - ~200 chars only): {preview_text}\n"
                                    else:
                                        data_summary += f"    [No body preview available]\n"
                                elif 'name' in item and 'email' in item:  # Contact
                                    data_summary += f"{i}. {item.get('name', 'Unknown')} ({item.get('email', 'no email')})\n"
                                elif 'name' in item and 'status' in item and 'priority' in item:  # Project (has priority)
                                    data_summary += f"{i}. {item.get('name', 'Unknown')} (Status: {item.get('status', 'unknown')})\n"
                                elif 'name' in item and 'status' in item:  # Task (has status but not priority at this level)
                                    data_summary += f"{i}. {item.get('name', 'Unknown')} (Status: {item.get('status', 'unknown')})\n"
                                elif 'name' in item and 'industry' in item:  # Organization
                                    data_summary += f"{i}. {item.get('name', 'Unknown')} (Industry: {item.get('industry', 'unknown')})\n"
                                else:
                                    data_summary += f"{i}. {item}\n"

                        if query_intent == "search":
                            user_message += f"\n\nSearch completed successfully. Found {total_count} total matches. {data_summary.strip()}\n\nPresent these search results in a helpful, natural way."
                        else:  # list intent
                            user_message += f"\n\nList completed successfully. Showing {len(result_data)} items (total: {total_count}). {data_summary.strip()}\n\nShow this data to the user in a clear, friendly way."

                        # Track entities for follow-up queries
                        if isinstance(result_data, list) and result_data:
                            # Track emails - SMART TRACKING based on search query
                            if 'subject' in result_data[0]:
                                email_to_track = result_data[0]  # Default to first

                                # If this was a search with a specific query, find the BEST MATCH
                                if parsed_command.get("intent") == "search_emails":
                                    entities = parsed_command.get("entities", {})
                                    search_query = (entities.get("search_query") or "").lower()

                                    if search_query:  # User searched for something specific
                                        # Try to find email that best matches the search query
                                        for email in result_data:
                                            email_from = (email.get('from', '') or '').lower()
                                            email_subject = (email.get('subject', '') or '').lower()

                                            # Check if search query matches sender or subject
                                            if search_query in email_from or search_query in email_subject:
                                                email_to_track = email
                                                logger.info(f"Found best matching email for query '{search_query}': {email.get('id')}")
                                                break

                                user_memory.track_entity_mention("emails", {
                                    "id": email_to_track.get('id'),
                                    "subject": email_to_track.get('subject'),
                                    "from": email_to_track.get('from')
                                })
                                logger.info(f"Tracked email: {email_to_track.get('id')} (subject: {email_to_track.get('subject')[:50] if email_to_track.get('subject') else 'None'})")

                            # Track contacts
                            elif 'email' in result_data[0] and 'name' in result_data[0]:
                                user_memory.track_entity_mention("contacts", {
                                    "id": result_data[0].get('id'),
                                    "name": result_data[0].get('name'),
                                    "email": result_data[0].get('email')
                                })
                                logger.info(f"Tracked first contact in list: {result_data[0].get('name')}")

                            # Track projects
                            elif 'status' in result_data[0] and 'priority' in result_data[0]:
                                user_memory.track_entity_mention("projects", {
                                    "id": result_data[0].get('id'),
                                    "name": result_data[0].get('name')
                                })
                                logger.info(f"Tracked first project in list: {result_data[0].get('name')}")

                            # Track organizations
                            elif 'industry' in result_data[0]:
                                user_memory.track_entity_mention("organizations", {
                                    "id": result_data[0].get('id'),
                                    "name": result_data[0].get('name')
                                })
                                logger.info(f"Tracked first organization in list: {result_data[0].get('name')}")

                            # Track tasks
                            elif 'status' in result_data[0] and 'project_id' in result_data[0]:
                                user_memory.track_entity_mention("tasks", {
                                    "id": result_data[0].get('id'),
                                    "name": result_data[0].get('name')
                                })
                                logger.info(f"Tracked first task in list: {result_data[0].get('name')}")

                            user_message += f"""\n\nCRITICAL CONTEXT FOR EMAIL LISTS:
- You are seeing {len(result_data)} email(s) with PREVIEW TEXT ONLY (~200 characters)
- You DO NOT have full email body content for these search results
- If user asks for full email content, explain you only have previews
- Offer to retrieve specific email(s) for full content: "Would you like me to retrieve the full content of email #1?"
- NEVER claim to know full email content when you only have previews
- NEVER make up or extrapolate content beyond the preview shown"""

                    elif isinstance(result_data, dict) and ('subject' in result_data or 'body_content' in result_data):
                        # Single email retrieved (e.g., "most recent email")
                        # Track this email for follow-up questions
                        user_memory.track_entity_mention("emails", {
                            "id": result_data.get('id'),
                            "subject": result_data.get('subject'),
                            "from": result_data.get('from')
                        })
                        logger.info(f"Tracked single email: {result_data.get('id')}")

                        from_info = result_data.get('from_name', result_data.get('from', 'Unknown'))
                        from_email = result_data.get('from', '')
                        if isinstance(result_data.get('from'), dict):
                            from_info = result_data['from'].get('name', 'Unknown')
                            from_email = result_data['from'].get('address', '')

                        # Format date
                        received_date = result_data.get('received_date', '')
                        if received_date:
                            try:
                                from datetime import datetime
                                dt = datetime.fromisoformat(received_date.replace('Z', '+00:00'))
                                date_str = dt.strftime('%B %d, %Y at %I:%M %p')
                            except:
                                date_str = received_date[:10]
                        else:
                            date_str = 'Unknown date'

                        # Get body content
                        body_content = result_data.get('body_content', '')
                        if not body_content:
                            body_content = result_data.get('body', '')

                        # Format the email for AI with extreme anti-hallucination boundaries
                        email_display = f"""
==== EMAIL DATA START ====
{{
  "email_id": "{result_data.get('id', 'N/A')}",
  "subject": "{result_data.get('subject', 'No subject')}",
  "from_name": "{from_info}",
  "from_email": "{from_email}",
  "date": "{date_str}",
  "status": "{'Read' if result_data.get('is_read') else 'Unread'}",
  "importance": "{result_data.get('importance', 'normal')}",
  "has_attachments": {'Yes' if result_data.get('has_attachments') else 'No'}
}}

==== BODY CONTENT START ====
{body_content if body_content else '[EMPTY - THIS EMAIL HAS NO BODY TEXT]'}
==== BODY CONTENT END ====

==== EMAIL DATA END ====

CRITICAL ANTI-HALLUCINATION INSTRUCTIONS:
1. The text between "==== BODY CONTENT START ====" and "==== BODY CONTENT END ====" is the COMPLETE email body
2. If it says [EMPTY], there is NO body text - do not make up any content
3. DO NOT paraphrase in a way that adds information not present
4. DO NOT extrapolate or imagine additional details
5. DO NOT reference topics, names, or details not explicitly in the body above
6. If body discusses specific topics, only mention those exact topics
7. If you add ANY information not in the body above, you are FAILING
8. Better to say "The email body is empty/brief" than to make up content

User asked: "{chat_data.message}"
Answer based ONLY on data between the ==== markers above. Nothing else exists.
"""
                        user_message = email_display

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
                # CRITICAL: Tool execution failed - strict error reporting
                error_message = execution_result.get('error', 'Unknown error')
                user_message = f"""User asked: "{chat_data.message}"

CRITICAL ERROR - TOOL EXECUTION FAILED:
Error: {error_message}

STRICT INSTRUCTIONS FOR ERROR HANDLING:
1. Inform the user that the operation failed
2. Report the error message clearly and honestly
3. DO NOT make up fake data or pretend the operation succeeded
4. DO NOT fabricate email content, contacts, or any information
5. DO NOT provide workarounds unless you can actually execute them
6. Be apologetic but honest about the failure
7. If it's an authentication error, suggest checking Microsoft 365 connection
8. If it's an API error, suggest the operation may not be supported

Example response: "I apologize, but I encountered an error while trying to [action]: {error_message}. This prevents me from retrieving the data you requested. [Suggest checking connection if auth error]"

ABSOLUTELY FORBIDDEN:
- Inventing fake data
- Pretending the operation succeeded
- Making up email subjects, senders, or content
- Estimating or guessing results
"""

        # Mark important messages in conversation memory
        if execution_result and execution_result.get("success"):
            if execution_result.get("result"):
                # Mark this exchange as important (contains data)
                user_memory.mark_message_as_important(
                    {"role": "user", "content": chat_data.message},
                    reason="Query with data payload"
                )

        # Log what's being sent to AI for email searches
        if parsed_command.get("intent") == "search_emails":
            logger.info(
                f"📧 SENDING TO AI for email search: "
                f"user_message_length={len(user_message)}, "
                f"contains_search_results={'Search completed successfully' in user_message}, "
                f"tracked_email_before_ai={user_memory.get_current_email_id()}"
            )
            # Log snippet of what AI will see
            if "Search completed successfully" in user_message:
                start_idx = user_message.find("Search completed successfully")
                snippet = user_message[start_idx:start_idx+300]
                logger.info(f"📧 AI will see: {snippet}...")

        # Get AI response with enhanced context management
        ai_response = await ai_service.generate_crm_response(
            user_message=user_message,
            user_context=user_context,
            conversation_history=chat_data.conversation_history,
            provider=chat_data.provider
        )

        # Log AI response for email searches
        if parsed_command.get("intent") == "search_emails":
            logger.info(
                f"📧 AI RESPONDED: response_length={len(ai_response['response'])}, "
                f"response_preview='{ai_response['response'][:150]}...'"
            )

        # Store AI offer for follow-up detection
        if "would you like" in ai_response["response"].lower() or "shall i" in ai_response["response"].lower():
            user_memory.set_last_offer(ai_response["response"])

        # CRITICAL FIX: If AI is offering to retrieve an email, update tracking
        # to match what AI is describing (not just what was in result_data[0])
        if execution_result and execution_result.get("success") and execution_result.get("result"):
            result_data = execution_result["result"]

            # If this was an email search that found results
            if isinstance(result_data, list) and result_data and len(result_data) > 0:
                # Check if this looks like email data
                if any(key in result_data[0] for key in ['subject', 'from', 'id']):
                    if parsed_command.get("intent") == "search_emails":
                        # Parse AI response to see which email it's describing
                        ai_text = ai_response["response"].lower()
                        logger.info(f"🔄 AI DESCRIPTION SYNC: Checking if AI response matches any search results...")

                        # Try to match email details from AI response to actual emails
                        matched = False
                        for idx, email in enumerate(result_data):
                            email_subject = (email.get('subject', '') or '').lower()
                            email_from_name = ""
                            email_from_addr = ""

                            # Extract from field (handle dict or string format)
                            from_field = email.get('from', '')
                            if isinstance(from_field, dict):
                                email_from_name = (from_field.get('name', '') or '').lower()
                                email_from_addr = (from_field.get('emailAddress', {}).get('address', '') or '').lower()
                            elif isinstance(from_field, str):
                                email_from_addr = from_field.lower()

                            # Check if AI mentioned this email's subject or sender
                            subject_match = email_subject and len(email_subject) > 3 and email_subject in ai_text
                            name_match = email_from_name and len(email_from_name) > 3 and email_from_name in ai_text
                            addr_match = email_from_addr and '@' in email_from_addr and email_from_addr in ai_text

                            if subject_match or name_match or addr_match:
                                # Update tracking to THIS email (the one AI described)
                                old_tracked = user_memory.get_current_email_id()
                                user_memory.track_entity_mention("emails", {
                                    "id": email.get('id'),
                                    "subject": email.get('subject'),
                                    "from": email_from_addr or email_from_name or from_field
                                })
                                logger.info(
                                    f"🔄 AI DESCRIPTION SYNC MATCHED: Updated tracking from {old_tracked} to {email.get('id')} "
                                    f"(match_type: {'subject' if subject_match else 'name' if name_match else 'addr'}, "
                                    f"email_idx={idx}, subject='{email.get('subject', '')[:30]}...')"
                                )
                                matched = True
                                break

                        if not matched:
                            logger.info(
                                f"🔄 AI DESCRIPTION SYNC: No match found in {len(result_data)} results. "
                                f"Tracked email remains: {user_memory.get_current_email_id()}"
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