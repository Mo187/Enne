from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timezone

from ...core.database import get_db
from ...models.user import User
from ...models.project import Project
from ...models.task import Task
from ...models.organization import Organization
from ..dependencies import get_current_active_user

templates = Jinja2Templates(directory="app/templates")

router = APIRouter()


# Pydantic schemas for request/response
class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    status: str = "planned"
    priority: str = "medium"
    organization_id: Optional[int] = None
    start_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    estimated_hours: Optional[int] = None
    budget: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


class ProjectCreate(ProjectBase):
    assignee_ids: Optional[List[int]] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    organization_id: Optional[int] = None
    start_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    estimated_hours: Optional[int] = None
    actual_hours: Optional[int] = None
    budget: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    progress_percentage: Optional[int] = None
    assignee_ids: Optional[List[int]] = None


class ProjectResponse(ProjectBase):
    id: int
    user_id: int
    progress_percentage: int
    actual_hours: int
    completed_date: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    task_count: int = 0
    completed_task_count: int = 0
    organization_name: Optional[str] = None
    assignee_names: List[str] = []

    class Config:
        from_attributes = True


class ProjectList(BaseModel):
    projects: List[ProjectResponse]
    total: int
    page: int
    per_page: int
    total_pages: int


@router.get("/")
async def list_projects(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by name or description"),
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    organization_id: Optional[int] = Query(None, description="Filter by organization"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get paginated list of projects for current user"""

    # Build base query
    query = select(Project).where(Project.user_id == current_user.id)

    # Apply search filter
    if search:
        search_filter = or_(
            Project.name.ilike(f"%{search}%"),
            Project.description.ilike(f"%{search}%")
        )
        query = query.where(search_filter)

    # Apply filters
    if status:
        query = query.where(Project.status == status)

    if priority:
        query = query.where(Project.priority == priority)

    if organization_id:
        query = query.where(Project.organization_id == organization_id)

    # Apply tags filter
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        for tag in tag_list:
            query = query.where(Project.tags.op('@>')([tag]))

    # Apply sorting
    sort_column = getattr(Project, sort_by, Project.created_at)
    if sort_order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Apply pagination
    offset = (page - 1) * per_page
    query = query.offset(offset).limit(per_page)

    # Execute query with related data
    query_with_relations = query.options(
        selectinload(Project.tasks),
        selectinload(Project.organization),
        selectinload(Project.assignees)
    )
    result = await db.execute(query_with_relations)
    projects = result.scalars().all()

    # Calculate total pages
    total_pages = (total + per_page - 1) // per_page

    # Check if this is an HTMX request
    if request.headers.get("HX-Request"):
        # Return HTML template for HTMX
        class PaginationInfo:
            def __init__(self, page, per_page, total, total_pages):
                self.page = page
                self.per_page = per_page
                self.total = total
                self.pages = total_pages
                self.has_prev = page > 1
                self.has_next = page < total_pages
                self.prev_num = page - 1 if self.has_prev else None
                self.next_num = page + 1 if self.has_next else None

            def iter_pages(self):
                start = max(1, self.page - 2)
                end = min(self.pages + 1, self.page + 3)
                return list(range(start, end))

        pagination = PaginationInfo(page, per_page, total, total_pages)

        return templates.TemplateResponse(
            "partials/projects_list.html",
            {
                "request": request,
                "projects": projects,
                "pagination": pagination,
                "now": lambda: datetime.now(timezone.utc)
            }
        )
    else:
        # Return JSON for API clients
        # Prepare response with computed fields
        project_responses = []
        for project in projects:
            project_dict = ProjectResponse.model_validate(project).model_dump()
            project_dict['task_count'] = len(project.tasks) if project.tasks else 0
            project_dict['completed_task_count'] = len([t for t in project.tasks if t.status == "completed"]) if project.tasks else 0
            project_dict['organization_name'] = project.organization.name if project.organization else None
            project_dict['assignee_names'] = [assignee.name for assignee in project.assignees] if project.assignees else []
            project_responses.append(ProjectResponse(**project_dict))

        total_pages = (total + per_page - 1) // per_page

        return ProjectList(
            projects=project_responses,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages
        )


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new project"""

    # Check if project with same name already exists for this user
    existing_query = select(Project).where(
        and_(
            Project.user_id == current_user.id,
            Project.name.ilike(project_data.name)
        )
    )
    existing_result = await db.execute(existing_query)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project with this name already exists"
        )

    # Validate organization if provided
    if project_data.organization_id:
        org_query = select(Organization).where(
            and_(
                Organization.id == project_data.organization_id,
                Organization.user_id == current_user.id
            )
        )
        org_result = await db.execute(org_query)
        if not org_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization not found"
            )

    # Create new project
    project_dict = project_data.model_dump(exclude={'assignee_ids'})
    project = Project(
        user_id=current_user.id,
        **project_dict
    )

    db.add(project)
    await db.commit()
    await db.refresh(project)

    # Add assignees if provided
    if project_data.assignee_ids:
        assignee_query = select(User).where(User.id.in_(project_data.assignee_ids))
        assignee_result = await db.execute(assignee_query)
        assignees = assignee_result.scalars().all()
        project.assignees.extend(assignees)
        await db.commit()

    # Reload with relationships
    await db.refresh(project, ["assignees", "organization"])

    return ProjectResponse.model_validate(project)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific project by ID"""

    query = select(Project).where(
        and_(
            Project.id == project_id,
            Project.user_id == current_user.id
        )
    ).options(
        selectinload(Project.tasks),
        selectinload(Project.organization),
        selectinload(Project.assignees)
    )

    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    # Prepare response with computed fields
    project_dict = ProjectResponse.model_validate(project).model_dump()
    project_dict['task_count'] = len(project.tasks) if project.tasks else 0
    project_dict['completed_task_count'] = len([t for t in project.tasks if t.status == "completed"]) if project.tasks else 0
    project_dict['organization_name'] = project.organization.name if project.organization else None
    project_dict['assignee_names'] = [assignee.name for assignee in project.assignees] if project.assignees else []

    return ProjectResponse(**project_dict)


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: int,
    project_data: ProjectUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a project"""

    # Get existing project
    query = select(Project).where(
        and_(
            Project.id == project_id,
            Project.user_id == current_user.id
        )
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    # Check for name conflicts if name is being updated
    if project_data.name and project_data.name != project.name:
        existing_query = select(Project).where(
            and_(
                Project.user_id == current_user.id,
                Project.name.ilike(project_data.name),
                Project.id != project_id
            )
        )
        existing_result = await db.execute(existing_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Project with this name already exists"
            )

    # Update fields
    update_data = project_data.model_dump(exclude_unset=True, exclude={'assignee_ids'})
    for field, value in update_data.items():
        setattr(project, field, value)

    # Update completion date if status changed to completed
    if project_data.status == "completed" and project.status != "completed":
        project.completed_date = datetime.utcnow()
    elif project_data.status != "completed":
        project.completed_date = None

    # Update assignees if provided
    if project_data.assignee_ids is not None:
        project.assignees.clear()
        if project_data.assignee_ids:
            assignee_query = select(User).where(User.id.in_(project_data.assignee_ids))
            assignee_result = await db.execute(assignee_query)
            assignees = assignee_result.scalars().all()
            project.assignees.extend(assignees)

    await db.commit()
    await db.refresh(project, ["assignees", "organization", "tasks"])

    return ProjectResponse.model_validate(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a project"""

    query = select(Project).where(
        and_(
            Project.id == project_id,
            Project.user_id == current_user.id
        )
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    await db.delete(project)
    await db.commit()


@router.get("/export/csv")
async def export_projects_csv(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Export projects as CSV"""
    from fastapi.responses import StreamingResponse
    import csv
    import io

    # Get all projects for user
    query = select(Project).where(Project.user_id == current_user.id).options(
        selectinload(Project.organization),
        selectinload(Project.assignees),
        selectinload(Project.tasks)
    )
    result = await db.execute(query)
    projects = result.scalars().all()

    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        'Name', 'Status', 'Priority', 'Organization', 'Assignees',
        'Progress %', 'Task Count', 'Completed Tasks', 'Start Date', 'Due Date',
        'Estimated Hours', 'Actual Hours', 'Budget', 'Created At', 'Description'
    ])

    # Write data
    for project in projects:
        writer.writerow([
            project.name,
            project.status or '',
            project.priority or '',
            project.organization.name if project.organization else '',
            ', '.join([assignee.name for assignee in project.assignees]) if project.assignees else '',
            project.progress_percentage or 0,
            len(project.tasks) if project.tasks else 0,
            len([t for t in project.tasks if t.status == "completed"]) if project.tasks else 0,
            project.start_date.strftime('%Y-%m-%d') if project.start_date else '',
            project.due_date.strftime('%Y-%m-%d') if project.due_date else '',
            project.estimated_hours or '',
            project.actual_hours or 0,
            project.budget or '',
            project.created_at.strftime('%Y-%m-%d %H:%M:%S') if project.created_at else '',
            project.description or ''
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=projects.csv"}
    )


@router.get("/{project_id}/tasks")
async def get_project_tasks(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all tasks for a project"""

    # First verify project exists and belongs to user
    project_query = select(Project).where(
        and_(
            Project.id == project_id,
            Project.user_id == current_user.id
        )
    )
    project_result = await db.execute(project_query)
    project = project_result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    # Get tasks for this project
    tasks_query = select(Task).where(Task.project_id == project_id).order_by(Task.sort_order, Task.created_at)
    tasks_result = await db.execute(tasks_query)
    tasks = tasks_result.scalars().all()

    return {
        "project": ProjectResponse.model_validate(project),
        "tasks": [
            {
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "status": task.status,
                "priority": task.priority,
                "assignee": task.assignee,
                "due_date": task.due_date,
                "estimated_hours": task.estimated_hours,
                "actual_hours": task.actual_hours,
                "created_at": task.created_at,
                "updated_at": task.updated_at
            }
            for task in tasks
        ]
    }