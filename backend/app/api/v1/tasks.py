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
from ...models.task import Task
from ...models.project import Project
from ..dependencies import get_current_active_user

templates = Jinja2Templates(directory="app/templates")

router = APIRouter()


# Pydantic schemas for request/response
class TaskBase(BaseModel):
    name: str
    description: Optional[str] = None
    status: str = "pending"
    priority: str = "medium"
    assignee: Optional[str] = None
    estimated_hours: Optional[int] = None
    start_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    parent_task_id: Optional[int] = None
    sort_order: int = 0


class TaskCreate(TaskBase):
    project_id: int


class TaskUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    assignee: Optional[str] = None
    estimated_hours: Optional[int] = None
    actual_hours: Optional[int] = None
    start_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    parent_task_id: Optional[int] = None
    sort_order: Optional[int] = None


class TaskResponse(TaskBase):
    id: int
    project_id: int
    actual_hours: int
    completed_date: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    project_name: Optional[str] = None
    subtask_count: int = 0

    class Config:
        from_attributes = True


class TaskList(BaseModel):
    tasks: List[TaskResponse]
    total: int
    page: int
    per_page: int
    total_pages: int


@router.get("/")
async def list_tasks(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by name, description, or assignee"),
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    project_id: Optional[int] = Query(None, description="Filter by project"),
    assignee: Optional[str] = Query(None, description="Filter by assignee"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get paginated list of tasks for current user"""

    # Build base query - tasks from projects owned by current user
    query = select(Task).join(Project).where(Project.user_id == current_user.id)

    # Apply search filter
    if search:
        search_filter = or_(
            Task.name.ilike(f"%{search}%"),
            Task.description.ilike(f"%{search}%"),
            Task.assignee.ilike(f"%{search}%")
        )
        query = query.where(search_filter)

    # Apply filters
    if status:
        query = query.where(Task.status == status)

    if priority:
        query = query.where(Task.priority == priority)

    if project_id:
        query = query.where(Task.project_id == project_id)

    if assignee:
        query = query.where(Task.assignee.ilike(f"%{assignee}%"))

    # Apply tags filter
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        for tag in tag_list:
            query = query.where(Task.tags.op('@>')([tag]))

    # Apply sorting
    sort_column = getattr(Task, sort_by, Task.created_at)
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
        selectinload(Task.project),
        selectinload(Task.subtasks)
    )
    result = await db.execute(query_with_relations)
    tasks = result.scalars().all()

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
            "partials/tasks_list.html",
            {
                "request": request,
                "tasks": tasks,
                "pagination": pagination,
                "now": lambda: datetime.now(timezone.utc)
            }
        )
    else:
        # Return JSON for API clients
        # Prepare response with computed fields
        task_responses = []
        for task in tasks:
            task_dict = TaskResponse.model_validate(task).model_dump()
            task_dict['project_name'] = task.project.name if task.project else None
            task_dict['subtask_count'] = len(task.subtasks) if hasattr(task, 'subtasks') and task.subtasks else 0
            task_responses.append(TaskResponse(**task_dict))

        total_pages = (total + per_page - 1) // per_page

        return TaskList(
            tasks=task_responses,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages
        )


@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    task_data: TaskCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new task"""

    # Verify project exists and belongs to user
    project_query = select(Project).where(
        and_(
            Project.id == task_data.project_id,
            Project.user_id == current_user.id
        )
    )
    project_result = await db.execute(project_query)
    project = project_result.scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project not found"
        )

    # Verify parent task if provided
    if task_data.parent_task_id:
        parent_query = select(Task).where(
            and_(
                Task.id == task_data.parent_task_id,
                Task.project_id == task_data.project_id
            )
        )
        parent_result = await db.execute(parent_query)
        if not parent_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parent task not found in the same project"
            )

    # Create new task
    task = Task(**task_data.model_dump())

    db.add(task)
    await db.commit()
    await db.refresh(task)

    # Reload with project relationship
    await db.refresh(task, ["project"])

    return TaskResponse.model_validate(task)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific task by ID"""

    query = select(Task).join(Project).where(
        and_(
            Task.id == task_id,
            Project.user_id == current_user.id
        )
    ).options(
        selectinload(Task.project),
        selectinload(Task.subtasks)
    )

    result = await db.execute(query)
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    # Prepare response with computed fields
    task_dict = TaskResponse.model_validate(task).model_dump()
    task_dict['project_name'] = task.project.name if task.project else None
    task_dict['subtask_count'] = len(task.subtasks) if hasattr(task, 'subtasks') and task.subtasks else 0

    return TaskResponse(**task_dict)


@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: int,
    task_data: TaskUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a task"""

    # Get existing task
    query = select(Task).join(Project).where(
        and_(
            Task.id == task_id,
            Project.user_id == current_user.id
        )
    )
    result = await db.execute(query)
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    # Verify parent task if being updated
    if task_data.parent_task_id and task_data.parent_task_id != task.parent_task_id:
        parent_query = select(Task).where(
            and_(
                Task.id == task_data.parent_task_id,
                Task.project_id == task.project_id
            )
        )
        parent_result = await db.execute(parent_query)
        if not parent_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parent task not found in the same project"
            )

    # Update fields
    update_data = task_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(task, field, value)

    # Update completion date if status changed to completed
    if task_data.status == "completed" and task.status != "completed":
        task.completed_date = datetime.utcnow()
    elif task_data.status != "completed":
        task.completed_date = None

    await db.commit()
    await db.refresh(task, ["project"])

    return TaskResponse.model_validate(task)


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a task"""

    query = select(Task).join(Project).where(
        and_(
            Task.id == task_id,
            Project.user_id == current_user.id
        )
    )
    result = await db.execute(query)
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    await db.delete(task)
    await db.commit()


@router.post("/{task_id}/mark-completed")
async def mark_task_completed(
    task_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Mark task as completed"""

    query = select(Task).join(Project).where(
        and_(
            Task.id == task_id,
            Project.user_id == current_user.id
        )
    )
    result = await db.execute(query)
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    task.status = "completed"
    task.completed_date = datetime.utcnow()
    await db.commit()

    return {"message": "Task marked as completed", "completed_date": task.completed_date}


@router.get("/export/csv")
async def export_tasks_csv(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Export tasks as CSV"""
    from fastapi.responses import StreamingResponse
    import csv
    import io

    # Get all tasks for user's projects
    query = select(Task).join(Project).where(Project.user_id == current_user.id).options(
        selectinload(Task.project)
    )
    result = await db.execute(query)
    tasks = result.scalars().all()

    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        'Name', 'Status', 'Priority', 'Project', 'Assignee',
        'Start Date', 'Due Date', 'Estimated Hours', 'Actual Hours',
        'Completed Date', 'Created At', 'Description'
    ])

    # Write data
    for task in tasks:
        writer.writerow([
            task.name,
            task.status or '',
            task.priority or '',
            task.project.name if task.project else '',
            task.assignee or '',
            task.start_date.strftime('%Y-%m-%d') if task.start_date else '',
            task.due_date.strftime('%Y-%m-%d') if task.due_date else '',
            task.estimated_hours or '',
            task.actual_hours or 0,
            task.completed_date.strftime('%Y-%m-%d') if task.completed_date else '',
            task.created_at.strftime('%Y-%m-%d %H:%M:%S') if task.created_at else '',
            task.description or ''
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tasks.csv"}
    )


@router.get("/{task_id}/subtasks")
async def get_task_subtasks(
    task_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all subtasks for a task"""

    # First verify task exists and belongs to user's project
    task_query = select(Task).join(Project).where(
        and_(
            Task.id == task_id,
            Project.user_id == current_user.id
        )
    )
    task_result = await db.execute(task_query)
    task = task_result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    # Get subtasks for this task
    subtasks_query = select(Task).where(Task.parent_task_id == task_id).order_by(Task.sort_order, Task.created_at)
    subtasks_result = await db.execute(subtasks_query)
    subtasks = subtasks_result.scalars().all()

    return {
        "task": TaskResponse.model_validate(task),
        "subtasks": [
            {
                "id": subtask.id,
                "name": subtask.name,
                "description": subtask.description,
                "status": subtask.status,
                "priority": subtask.priority,
                "assignee": subtask.assignee,
                "due_date": subtask.due_date,
                "estimated_hours": subtask.estimated_hours,
                "actual_hours": subtask.actual_hours,
                "created_at": subtask.created_at,
                "updated_at": subtask.updated_at
            }
            for subtask in subtasks
        ]
    }