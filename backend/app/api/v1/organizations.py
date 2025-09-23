from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload
from typing import List, Optional
from pydantic import BaseModel, EmailStr
from datetime import datetime

from ...core.database import get_db
from ...models.user import User
from ...models.organization import Organization
from ...models.project import Project
from ..dependencies import get_current_active_user

templates = Jinja2Templates(directory="app/templates")

router = APIRouter()


# Pydantic schemas for request/response
class OrganizationBase(BaseModel):
    name: str
    description: Optional[str] = None
    industry: Optional[str] = None
    website: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None
    company_size: Optional[str] = None
    annual_revenue: Optional[str] = None
    founded_year: Optional[int] = None
    linkedin_url: Optional[str] = None
    twitter_url: Optional[str] = None
    relationship_status: Optional[str] = "prospect"
    priority: Optional[str] = "medium"
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


class OrganizationCreate(OrganizationBase):
    pass


class OrganizationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None
    website: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None
    company_size: Optional[str] = None
    annual_revenue: Optional[str] = None
    founded_year: Optional[int] = None
    linkedin_url: Optional[str] = None
    twitter_url: Optional[str] = None
    relationship_status: Optional[str] = None
    priority: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


class OrganizationResponse(OrganizationBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_interaction: Optional[datetime] = None
    full_address: str
    project_count: int = 0

    class Config:
        from_attributes = True

    @property
    def full_address(self) -> str:
        """Return formatted full address"""
        address_parts = [
            self.address_line1,
            self.address_line2,
            self.city,
            self.state,
            self.postal_code,
            self.country
        ]
        return ", ".join([part for part in address_parts if part])


class OrganizationList(BaseModel):
    organizations: List[OrganizationResponse]
    total: int
    page: int
    per_page: int
    total_pages: int


class OrganizationStats(BaseModel):
    total_organizations: int
    by_industry: dict
    by_relationship_status: dict
    by_priority: dict
    by_company_size: dict


@router.get("/")
async def list_organizations(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by name, industry, or description"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    relationship_status: Optional[str] = Query(None, description="Filter by relationship status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    company_size: Optional[str] = Query(None, description="Filter by company size"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get paginated list of organizations for current user"""

    # Build base query
    query = select(Organization).where(Organization.user_id == current_user.id)

    # Apply search filter
    if search:
        search_filter = or_(
            Organization.name.ilike(f"%{search}%"),
            Organization.industry.ilike(f"%{search}%"),
            Organization.description.ilike(f"%{search}%")
        )
        query = query.where(search_filter)

    # Apply filters
    if industry:
        query = query.where(Organization.industry.ilike(f"%{industry}%"))

    if relationship_status:
        query = query.where(Organization.relationship_status == relationship_status)

    if priority:
        query = query.where(Organization.priority == priority)

    if company_size:
        query = query.where(Organization.company_size == company_size)

    # Apply tags filter
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        for tag in tag_list:
            query = query.where(Organization.tags.op('@>')([tag]))

    # Apply sorting
    sort_column = getattr(Organization, sort_by, Organization.created_at)
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

    # Execute query with project count
    query_with_projects = query.options(selectinload(Organization.projects))
    result = await db.execute(query_with_projects)
    organizations = result.scalars().all()

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
                # Simple pagination logic
                start = max(1, self.page - 2)
                end = min(self.pages + 1, self.page + 3)
                return list(range(start, end))

        pagination = PaginationInfo(page, per_page, total, total_pages)

        return templates.TemplateResponse(
            "partials/organizations_list.html",
            {
                "request": request,
                "organizations": organizations,
                "pagination": pagination
            }
        )
    else:
        # Return JSON for API clients
        # Prepare response with project counts
        org_responses = []
        for org in organizations:
            org_dict = OrganizationResponse.model_validate(org).model_dump()
            org_dict['project_count'] = len(org.projects) if org.projects else 0
            org_responses.append(OrganizationResponse(**org_dict))

        return OrganizationList(
            organizations=org_responses,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages
        )


@router.post("/", response_model=OrganizationResponse, status_code=status.HTTP_201_CREATED)
async def create_organization(
    org_data: OrganizationCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new organization"""

    # Check if organization with same name already exists for this user
    existing_query = select(Organization).where(
        and_(
            Organization.user_id == current_user.id,
            Organization.name.ilike(org_data.name)
        )
    )
    existing_result = await db.execute(existing_query)
    if existing_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization with this name already exists"
        )

    # Create new organization
    organization = Organization(
        user_id=current_user.id,
        **org_data.model_dump()
    )

    db.add(organization)
    await db.commit()
    await db.refresh(organization)

    return OrganizationResponse.model_validate(organization)


@router.get("/stats", response_model=OrganizationStats)
async def get_organization_stats(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get organization statistics for current user"""

    # Get all organizations for user
    query = select(Organization).where(Organization.user_id == current_user.id)
    result = await db.execute(query)
    organizations = result.scalars().all()

    # Calculate stats
    total_organizations = len(organizations)

    by_industry = {}
    by_relationship_status = {}
    by_priority = {}
    by_company_size = {}

    for org in organizations:
        # Industry stats
        industry = org.industry or "Unknown"
        by_industry[industry] = by_industry.get(industry, 0) + 1

        # Relationship status stats
        status = org.relationship_status or "Unknown"
        by_relationship_status[status] = by_relationship_status.get(status, 0) + 1

        # Priority stats
        priority = org.priority or "Unknown"
        by_priority[priority] = by_priority.get(priority, 0) + 1

        # Company size stats
        size = org.company_size or "Unknown"
        by_company_size[size] = by_company_size.get(size, 0) + 1

    return OrganizationStats(
        total_organizations=total_organizations,
        by_industry=by_industry,
        by_relationship_status=by_relationship_status,
        by_priority=by_priority,
        by_company_size=by_company_size
    )


@router.get("/{organization_id}", response_model=OrganizationResponse)
async def get_organization(
    organization_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific organization by ID"""

    query = select(Organization).where(
        and_(
            Organization.id == organization_id,
            Organization.user_id == current_user.id
        )
    ).options(selectinload(Organization.projects))

    result = await db.execute(query)
    organization = result.scalar_one_or_none()

    if not organization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    org_dict = OrganizationResponse.model_validate(organization).model_dump()
    org_dict['project_count'] = len(organization.projects) if organization.projects else 0

    return OrganizationResponse(**org_dict)


@router.put("/{organization_id}", response_model=OrganizationResponse)
async def update_organization(
    organization_id: int,
    org_data: OrganizationUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update an organization"""

    # Get existing organization
    query = select(Organization).where(
        and_(
            Organization.id == organization_id,
            Organization.user_id == current_user.id
        )
    )
    result = await db.execute(query)
    organization = result.scalar_one_or_none()

    if not organization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    # Check for name conflicts if name is being updated
    if org_data.name and org_data.name != organization.name:
        existing_query = select(Organization).where(
            and_(
                Organization.user_id == current_user.id,
                Organization.name.ilike(org_data.name),
                Organization.id != organization_id
            )
        )
        existing_result = await db.execute(existing_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization with this name already exists"
            )

    # Update fields
    update_data = org_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(organization, field, value)

    await db.commit()
    await db.refresh(organization)

    return OrganizationResponse.model_validate(organization)


@router.delete("/{organization_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_organization(
    organization_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete an organization"""

    query = select(Organization).where(
        and_(
            Organization.id == organization_id,
            Organization.user_id == current_user.id
        )
    )
    result = await db.execute(query)
    organization = result.scalar_one_or_none()

    if not organization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    await db.delete(organization)
    await db.commit()


@router.post("/{organization_id}/mark-interaction")
async def mark_interaction(
    organization_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Mark organization interaction (updates last_interaction timestamp)"""

    query = select(Organization).where(
        and_(
            Organization.id == organization_id,
            Organization.user_id == current_user.id
        )
    )
    result = await db.execute(query)
    organization = result.scalar_one_or_none()

    if not organization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    organization.last_interaction = datetime.utcnow()
    await db.commit()

    return {"message": "Organization interaction recorded", "last_interaction": organization.last_interaction}


@router.get("/export/csv")
async def export_organizations_csv(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Export organizations as CSV"""
    from fastapi.responses import StreamingResponse
    import csv
    import io

    # Get all organizations for user
    query = select(Organization).where(Organization.user_id == current_user.id)
    result = await db.execute(query)
    organizations = result.scalars().all()

    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        'Name', 'Industry', 'Website', 'Email', 'Phone',
        'City', 'State', 'Country', 'Company Size', 'Annual Revenue',
        'Founded Year', 'Relationship Status', 'Priority',
        'Created At', 'Last Interaction', 'Description'
    ])

    # Write data
    for org in organizations:
        writer.writerow([
            org.name,
            org.industry or '',
            org.website or '',
            org.email or '',
            org.phone or '',
            org.city or '',
            org.state or '',
            org.country or '',
            org.company_size or '',
            org.annual_revenue or '',
            org.founded_year or '',
            org.relationship_status or '',
            org.priority or '',
            org.created_at.strftime('%Y-%m-%d %H:%M:%S') if org.created_at else '',
            org.last_interaction.strftime('%Y-%m-%d %H:%M:%S') if org.last_interaction else '',
            org.description or ''
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=organizations.csv"}
    )


@router.get("/{organization_id}/projects")
async def get_organization_projects(
    organization_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all projects for an organization"""

    # First verify organization exists and belongs to user
    org_query = select(Organization).where(
        and_(
            Organization.id == organization_id,
            Organization.user_id == current_user.id
        )
    )
    org_result = await db.execute(org_query)
    organization = org_result.scalar_one_or_none()

    if not organization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )

    # Get projects for this organization
    projects_query = select(Project).where(
        and_(
            Project.organization_id == organization_id,
            Project.user_id == current_user.id
        )
    ).options(selectinload(Project.tasks))

    projects_result = await db.execute(projects_query)
    projects = projects_result.scalars().all()

    return {
        "organization": OrganizationResponse.model_validate(organization),
        "projects": [
            {
                "id": project.id,
                "name": project.name,
                "status": project.status,
                "priority": project.priority,
                "due_date": project.due_date,
                "progress_percentage": project.progress_percentage,
                "task_count": len(project.tasks) if project.tasks else 0,
                "created_at": project.created_at
            }
            for project in projects
        ]
    }