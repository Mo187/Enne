from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Form, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel, EmailStr
from typing import Optional

from ...core.database import get_db
from ...core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    generate_email_verification_token,
    verify_email_verification_token
)
from ...core.config import settings
from ...models.user import User
from ..dependencies import get_current_user, get_current_active_user

router = APIRouter()


# Pydantic models for request/response
class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str
    company: Optional[str] = None
    job_title: Optional[str] = None
    phone: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    is_active: bool
    is_verified: bool
    company: Optional[str] = None
    job_title: Optional[str] = None
    phone: Optional[str] = None
    theme: str
    language: str
    preferred_ai_model: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordReset(BaseModel):
    token: str
    new_password: str


@router.post("/register", response_model=Token)
async def register(
    response: Response,
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""

    # Check if user already exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        email=user_data.email,
        name=user_data.name,
        password_hash=hashed_password,
        company=user_data.company,
        job_title=user_data.job_title,
        phone=user_data.phone,
        is_active=True,
        is_verified=False  # Will be verified via email
    )

    try:
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        subject=db_user.id, expires_delta=access_token_expires
    )

    # Set HTTP-only cookie for server-side authentication
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=settings.access_token_expire_minutes * 60,
        samesite="lax",
        secure=False  # Set to True in production with HTTPS
    )

    # TODO: Send verification email
    # verification_token = generate_email_verification_token(db_user.email)
    # send_verification_email(db_user.email, verification_token)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.model_validate(db_user)
    }


@router.post("/login", response_model=Token)
async def login(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Authenticate user and return token"""

    # Find user by email
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        subject=user.id, expires_delta=access_token_expires
    )

    # Set HTTP-only cookie for server-side authentication
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=settings.access_token_expire_minutes * 60,
        samesite="lax",
        secure=False  # Set to True in production with HTTPS
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.model_validate(user)
    }


@router.post("/logout")
async def logout(response: Response):
    """Logout user by clearing authentication cookie"""
    response.delete_cookie(key="access_token")
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information"""
    return UserResponse.model_validate(current_user)


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    name: Optional[str] = Form(None),
    company: Optional[str] = Form(None),
    job_title: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    theme: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    preferred_ai_model: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user information"""

    # Update fields if provided
    if name is not None:
        current_user.name = name
    if company is not None:
        current_user.company = company
    if job_title is not None:
        current_user.job_title = job_title
    if phone is not None:
        current_user.phone = phone
    if theme in ["light", "dark"]:
        current_user.theme = theme
    if language is not None:
        current_user.language = language
    if preferred_ai_model in ["claude", "openai", "gemini"]:
        current_user.preferred_ai_model = preferred_ai_model

    try:
        await db.commit()
        await db.refresh(current_user)
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not update user"
        )

    return UserResponse.model_validate(current_user)


@router.post("/verify-email")
async def verify_email(
    token: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Verify user email with token"""

    email = verify_email_verification_token(token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token"
        )

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if user.is_verified:
        return {"message": "Email already verified"}

    user.is_verified = True
    await db.commit()

    return {"message": "Email verified successfully"}


@router.post("/forgot-password")
async def forgot_password(
    request: PasswordResetRequest,
    db: AsyncSession = Depends(get_db)
):
    """Request password reset"""

    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    # Always return success to prevent email enumeration
    if not user:
        return {"message": "If the email exists, a reset link has been sent"}

    # TODO: Send password reset email
    # reset_token = generate_password_reset_token(user.email)
    # send_password_reset_email(user.email, reset_token)

    return {"message": "If the email exists, a reset link has been sent"}


@router.post("/reset-password")
async def reset_password(
    request: PasswordReset,
    db: AsyncSession = Depends(get_db)
):
    """Reset password with token"""

    from ...core.security import verify_password_reset_token

    email = verify_password_reset_token(request.token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reset token"
        )

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Update password
    user.password_hash = get_password_hash(request.new_password)
    await db.commit()

    return {"message": "Password reset successfully"}