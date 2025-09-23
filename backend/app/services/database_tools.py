"""
Database Tools for CRM - Implementing the Tool Interface

These tools handle all CRM database operations through the standardized tool interface.
This modular approach prepares the system for MCP integration while maintaining
current functionality.
"""

from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

from .tool_interface import BaseTool, ToolType, ToolResult, ToolSchema
from ..models.contact import Contact
from ..models.organization import Organization
from ..models.project import Project
from ..models.task import Task
from ..models.user import User


class ContactSearchTool(BaseTool):
    """Tool for searching and counting contacts"""

    def __init__(self, db_session: AsyncSession):
        super().__init__("contact_search", ToolType.DATABASE)
        self.db = db_session

    async def execute(self, parameters: Dict[str, Any], user_context: Dict[str, Any]) -> ToolResult:
        """Search for contacts with optional filters"""
        try:
            user_id = user_context.get("user_id")
            if not user_id:
                return ToolResult(success=False, error="User context required")

            # Build base query
            query = select(Contact).where(Contact.user_id == user_id)
            count_query = select(func.count()).select_from(Contact).where(Contact.user_id == user_id)

            # Apply filters
            search_term = parameters.get("search")
            organization_filter = parameters.get("organization")

            if search_term:
                search_pattern = f"%{search_term}%"
                search_conditions = or_(
                    Contact.name.ilike(search_pattern),
                    Contact.email.ilike(search_pattern),
                    Contact.organization.ilike(search_pattern)
                )
                query = query.where(search_conditions)
                count_query = count_query.where(search_conditions)

            if organization_filter:
                org_pattern = f"%{organization_filter}%"
                query = query.where(Contact.organization.ilike(org_pattern))
                count_query = count_query.where(Contact.organization.ilike(org_pattern))

            # Get total count
            total_count = await self.db.scalar(count_query)

            # Get limited results for display
            limit = parameters.get("limit", 50)
            query = query.limit(limit)

            result = await self.db.execute(query)
            contacts = result.scalars().all()

            # Format results
            contact_data = [
                {
                    "id": contact.id,
                    "name": contact.name,
                    "email": contact.email,
                    "phone": contact.phone,
                    "organization": contact.organization,
                    "job_position": contact.job_position
                }
                for contact in contacts
            ]

            message = f"Found {total_count} contact(s) total"
            if total_count > len(contact_data):
                message += f", showing first {len(contact_data)}"

            return ToolResult(
                success=True,
                data=contact_data,
                total_count=total_count,
                message=message
            )

        except Exception as e:
            self.logger.error("Contact search failed", error=str(e))
            return ToolResult(success=False, error=f"Contact search failed: {str(e)}")

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Search and count contacts in the CRM",
            tool_type=self.tool_type,
            parameters={
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Search term for name, email, or organization"
                    },
                    "organization": {
                        "type": "string",
                        "description": "Filter by organization name"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 50
                    }
                }
            },
            examples=[
                {"search": "john", "description": "Find contacts named John"},
                {"organization": "acme", "description": "Find contacts from Acme"},
                {"search": "gabriel", "description": "Count contacts named Gabriel"}
            ]
        )


class OrganizationSearchTool(BaseTool):
    """Tool for searching and counting organizations"""

    def __init__(self, db_session: AsyncSession):
        super().__init__("organization_search", ToolType.DATABASE)
        self.db = db_session

    async def execute(self, parameters: Dict[str, Any], user_context: Dict[str, Any]) -> ToolResult:
        """Search for organizations with optional filters"""
        try:
            user_id = user_context.get("user_id")
            if not user_id:
                return ToolResult(success=False, error="User context required")

            # Build base query
            query = select(Organization).where(Organization.user_id == user_id)
            count_query = select(func.count()).select_from(Organization).where(Organization.user_id == user_id)

            # Apply filters
            search_term = parameters.get("search")
            industry_filter = parameters.get("industry")

            if search_term:
                search_pattern = f"%{search_term}%"
                query = query.where(Organization.name.ilike(search_pattern))
                count_query = count_query.where(Organization.name.ilike(search_pattern))

            if industry_filter:
                industry_pattern = f"%{industry_filter}%"
                query = query.where(Organization.industry.ilike(industry_pattern))
                count_query = count_query.where(Organization.industry.ilike(industry_pattern))

            # Get total count
            total_count = await self.db.scalar(count_query)

            # Get limited results
            limit = parameters.get("limit", 50)
            query = query.limit(limit)

            result = await self.db.execute(query)
            organizations = result.scalars().all()

            # Format results
            org_data = [
                {
                    "id": org.id,
                    "name": org.name,
                    "industry": org.industry,
                    "website": org.website,
                    "description": org.description
                }
                for org in organizations
            ]

            message = f"Found {total_count} organization(s) total"
            if total_count > len(org_data):
                message += f", showing first {len(org_data)}"

            return ToolResult(
                success=True,
                data=org_data,
                total_count=total_count,
                message=message
            )

        except Exception as e:
            self.logger.error("Organization search failed", error=str(e))
            return ToolResult(success=False, error=f"Organization search failed: {str(e)}")

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Search and count organizations in the CRM",
            tool_type=self.tool_type,
            parameters={
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Search term for organization name"
                    },
                    "industry": {
                        "type": "string",
                        "description": "Filter by industry"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 50
                    }
                }
            },
            examples=[
                {"search": "acme", "description": "Find organizations named Acme"},
                {"industry": "technology", "description": "Find tech companies"}
            ]
        )


class ProjectSearchTool(BaseTool):
    """Tool for searching and counting projects"""

    def __init__(self, db_session: AsyncSession):
        super().__init__("project_search", ToolType.DATABASE)
        self.db = db_session

    async def execute(self, parameters: Dict[str, Any], user_context: Dict[str, Any]) -> ToolResult:
        """Search for projects with optional filters"""
        try:
            user_id = user_context.get("user_id")
            if not user_id:
                return ToolResult(success=False, error="User context required")

            # Build base query
            query = select(Project).where(Project.user_id == user_id)
            count_query = select(func.count()).select_from(Project).where(Project.user_id == user_id)

            # Apply filters
            search_term = parameters.get("search")
            status_filter = parameters.get("status")

            if search_term:
                search_pattern = f"%{search_term}%"
                search_conditions = or_(
                    Project.name.ilike(search_pattern),
                    Project.description.ilike(search_pattern)
                )
                query = query.where(search_conditions)
                count_query = count_query.where(search_conditions)

            if status_filter:
                query = query.where(Project.status == status_filter)
                count_query = count_query.where(Project.status == status_filter)

            # Get total count
            total_count = await self.db.scalar(count_query)

            # Get limited results
            limit = parameters.get("limit", 50)
            query = query.limit(limit)

            result = await self.db.execute(query)
            projects = result.scalars().all()

            # Format results
            project_data = [
                {
                    "id": project.id,
                    "name": project.name,
                    "status": project.status,
                    "priority": project.priority,
                    "description": project.description,
                    "organization_id": project.organization_id
                }
                for project in projects
            ]

            message = f"Found {total_count} project(s) total"
            if total_count > len(project_data):
                message += f", showing first {len(project_data)}"

            return ToolResult(
                success=True,
                data=project_data,
                total_count=total_count,
                message=message
            )

        except Exception as e:
            self.logger.error("Project search failed", error=str(e))
            return ToolResult(success=False, error=f"Project search failed: {str(e)}")

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Search and count projects in the CRM",
            tool_type=self.tool_type,
            parameters={
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Search term for project name or description"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["planned", "in_progress", "completed", "on_hold", "cancelled"],
                        "description": "Filter by project status"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 50
                    }
                }
            },
            examples=[
                {"search": "website", "description": "Find projects related to website"},
                {"status": "completed", "description": "Find completed projects"}
            ]
        )


class TaskSearchTool(BaseTool):
    """Tool for searching and counting tasks"""

    def __init__(self, db_session: AsyncSession):
        super().__init__("task_search", ToolType.DATABASE)
        self.db = db_session

    async def execute(self, parameters: Dict[str, Any], user_context: Dict[str, Any]) -> ToolResult:
        """Search for tasks with optional filters"""
        try:
            user_id = user_context.get("user_id")
            if not user_id:
                return ToolResult(success=False, error="User context required")

            # Build base query (tasks belong to user's projects)
            query = select(Task).join(Project).where(Project.user_id == user_id)
            count_query = select(func.count(Task.id)).select_from(Task).join(Project).where(Project.user_id == user_id)

            # Apply filters
            search_term = parameters.get("search")
            status_filter = parameters.get("status")
            assignee_filter = parameters.get("assignee")

            if search_term:
                search_pattern = f"%{search_term}%"
                search_conditions = or_(
                    Task.name.ilike(search_pattern),
                    Task.description.ilike(search_pattern),
                    Task.assignee.ilike(search_pattern)
                )
                query = query.where(search_conditions)
                count_query = count_query.where(search_conditions)

            if status_filter:
                query = query.where(Task.status == status_filter)
                count_query = count_query.where(Task.status == status_filter)

            if assignee_filter:
                assignee_pattern = f"%{assignee_filter}%"
                query = query.where(Task.assignee.ilike(assignee_pattern))
                count_query = count_query.where(Task.assignee.ilike(assignee_pattern))

            # Get total count
            total_count = await self.db.scalar(count_query)

            # Get limited results
            limit = parameters.get("limit", 50)
            query = query.limit(limit)

            result = await self.db.execute(query)
            tasks = result.scalars().all()

            # Format results
            task_data = [
                {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status,
                    "priority": task.priority,
                    "assignee": task.assignee,
                    "project_id": task.project_id,
                    "description": task.description
                }
                for task in tasks
            ]

            message = f"Found {total_count} task(s) total"
            if total_count > len(task_data):
                message += f", showing first {len(task_data)}"

            return ToolResult(
                success=True,
                data=task_data,
                total_count=total_count,
                message=message
            )

        except Exception as e:
            self.logger.error("Task search failed", error=str(e))
            return ToolResult(success=False, error=f"Task search failed: {str(e)}")

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Search and count tasks in the CRM",
            tool_type=self.tool_type,
            parameters={
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Search term for task name, description, or assignee"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "blocked", "cancelled"],
                        "description": "Filter by task status"
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Filter by assignee name"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 50
                    }
                }
            },
            examples=[
                {"search": "review", "description": "Find tasks related to review"},
                {"status": "pending", "description": "Find pending tasks"},
                {"assignee": "john", "description": "Find tasks assigned to John"}
            ]
        )


# Tool initialization function for chat.py integration
def initialize_database_tools(db_session: AsyncSession) -> Dict[str, BaseTool]:
    """
    Initialize all database tools with the current database session

    Returns:
        Dictionary of tool instances by name
    """
    tools = {
        "contact_search": ContactSearchTool(db_session),
        "organization_search": OrganizationSearchTool(db_session),
        "project_search": ProjectSearchTool(db_session),
        "task_search": TaskSearchTool(db_session)
    }

    return tools