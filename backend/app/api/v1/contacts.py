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
from ...models.contact import Contact
from ..dependencies import get_current_active_user

templates = Jinja2Templates(directory="app/templates")

router = APIRouter()


# Pydantic schemas for request/response
class ContactBase(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    job_position: Optional[str] = None
    organization: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    linkedin_url: Optional[str] = None
    website: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None


class ContactCreate(ContactBase):
    pass


class ContactUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    job_position: Optional[str] = None
    organization: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    linkedin_url: Optional[str] = None
    website: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None


class ContactResponse(ContactBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_contacted: Optional[datetime] = None
    full_address: str

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


class ContactList(BaseModel):
    contacts: List[ContactResponse]
    total: int
    page: int
    per_page: int
    total_pages: int


@router.get("")
async def list_contacts(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by name, email, or organization"),
    organization: Optional[str] = Query(None, description="Filter by organization"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get paginated list of contacts for current user"""

    # Build base query
    query = select(Contact).where(Contact.user_id == current_user.id)

    # Apply search filter
    if search:
        search_filter = or_(
            Contact.name.ilike(f"%{search}%"),
            Contact.email.ilike(f"%{search}%"),
            Contact.organization.ilike(f"%{search}%"),
            Contact.job_position.ilike(f"%{search}%")
        )
        query = query.where(search_filter)

    # Apply organization filter
    if organization:
        query = query.where(Contact.organization.ilike(f"%{organization}%"))

    # Apply tags filter
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        # PostgreSQL JSON contains operation
        for tag in tag_list:
            query = query.where(Contact.tags.op('@>')([tag]))

    # Apply sorting
    sort_column = getattr(Contact, sort_by, Contact.created_at)
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

    # Execute query
    result = await db.execute(query)
    contacts = result.scalars().all()

    # Calculate pagination info
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
            "partials/contacts_list.html",
            {
                "request": request,
                "contacts": contacts,
                "pagination": pagination
            }
        )
    else:
        # Return JSON for API clients
        return ContactList(
            contacts=[ContactResponse.model_validate(contact) for contact in contacts],
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages
        )


@router.post("", response_model=ContactResponse, status_code=status.HTTP_201_CREATED)
async def create_contact(
    contact_data: ContactCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new contact"""

    # Check if contact with same email already exists for this user
    if contact_data.email:
        existing_query = select(Contact).where(
            and_(
                Contact.user_id == current_user.id,
                Contact.email == contact_data.email
            )
        )
        existing_result = await db.execute(existing_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Contact with this email already exists"
            )

    # Create new contact
    contact = Contact(
        user_id=current_user.id,
        **contact_data.model_dump()
    )

    db.add(contact)
    await db.commit()
    await db.refresh(contact)

    return ContactResponse.model_validate(contact)


@router.get("/{contact_id}", response_model=ContactResponse)
async def get_contact(
    contact_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific contact by ID"""

    query = select(Contact).where(
        and_(
            Contact.id == contact_id,
            Contact.user_id == current_user.id
        )
    )
    result = await db.execute(query)
    contact = result.scalar_one_or_none()

    if not contact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact not found"
        )

    return ContactResponse.model_validate(contact)


@router.put("/{contact_id}", response_model=ContactResponse)
async def update_contact(
    contact_id: int,
    contact_data: ContactUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a contact"""

    # Get existing contact
    query = select(Contact).where(
        and_(
            Contact.id == contact_id,
            Contact.user_id == current_user.id
        )
    )
    result = await db.execute(query)
    contact = result.scalar_one_or_none()

    if not contact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact not found"
        )

    # Check for email conflicts if email is being updated
    if contact_data.email and contact_data.email != contact.email:
        existing_query = select(Contact).where(
            and_(
                Contact.user_id == current_user.id,
                Contact.email == contact_data.email,
                Contact.id != contact_id
            )
        )
        existing_result = await db.execute(existing_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Contact with this email already exists"
            )

    # Update fields
    update_data = contact_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(contact, field, value)

    await db.commit()
    await db.refresh(contact)

    return ContactResponse.model_validate(contact)


@router.delete("/{contact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_contact(
    contact_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a contact"""

    query = select(Contact).where(
        and_(
            Contact.id == contact_id,
            Contact.user_id == current_user.id
        )
    )
    result = await db.execute(query)
    contact = result.scalar_one_or_none()

    if not contact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact not found"
        )

    await db.delete(contact)
    await db.commit()


@router.post("/{contact_id}/mark-contacted")
async def mark_contacted(
    contact_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Mark contact as contacted (updates last_contacted timestamp)"""

    query = select(Contact).where(
        and_(
            Contact.id == contact_id,
            Contact.user_id == current_user.id
        )
    )
    result = await db.execute(query)
    contact = result.scalar_one_or_none()

    if not contact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact not found"
        )

    contact.last_contacted = datetime.utcnow()
    await db.commit()

    return {"message": "Contact marked as contacted", "last_contacted": contact.last_contacted}


@router.get("/export/csv")
async def export_contacts_csv(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Export contacts as CSV"""
    from fastapi.responses import StreamingResponse
    import csv
    import io

    # Get all contacts for user
    query = select(Contact).where(Contact.user_id == current_user.id)
    result = await db.execute(query)
    contacts = result.scalars().all()

    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        'Name', 'Email', 'Phone', 'Job Position', 'Organization',
        'City', 'State', 'Country', 'LinkedIn', 'Website',
        'Created At', 'Last Contacted', 'Notes'
    ])

    # Write data
    for contact in contacts:
        writer.writerow([
            contact.name,
            contact.email or '',
            contact.phone or '',
            contact.job_position or '',
            contact.organization or '',
            contact.city or '',
            contact.state or '',
            contact.country or '',
            contact.linkedin_url or '',
            contact.website or '',
            contact.created_at.strftime('%Y-%m-%d %H:%M:%S') if contact.created_at else '',
            contact.last_contacted.strftime('%Y-%m-%d %H:%M:%S') if contact.last_contacted else '',
            contact.notes or ''
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=contacts.csv"}
    )