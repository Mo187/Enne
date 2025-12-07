from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from typing import Optional
import structlog
import time

from .core.config import settings
from .core.database import create_tables, get_db
from .core.startup import initialize_mcp_integrations, cleanup_mcp_integrations
from .core.security import verify_token
from .models.user import User
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Security scheme for page authentication
security_scheme = HTTPBearer(auto_error=False)


async def get_current_user_from_request(
    request: Request,
    db: AsyncSession = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> Optional[User]:
    """
    Get current user from request for page routes.
    Checks Authorization header (for API-like requests) or falls back to cookie.
    Returns None if not authenticated instead of raising exception.
    """
    token = None

    # Try to get token from Authorization header
    if credentials:
        token = credentials.credentials
    # Fallback: try to get from cookie
    elif "access_token" in request.cookies:
        token = request.cookies.get("access_token")

    if not token:
        return None

    # Verify token
    user_id = verify_token(token)
    if user_id is None:
        return None

    # Get user from database
    try:
        result = await db.execute(
            select(User).where(User.id == int(user_id))
        )
        user = result.scalar_one_or_none()

        if user and user.is_active:
            return user
    except Exception as e:
        logger.error("Error fetching user", error=str(e), user_id=user_id)

    return None




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting up CRM application", version=settings.version)

    # Create database tables
    await create_tables()
    logger.info("Database tables created/verified")

    # Initialize MCP integrations
    await initialize_mcp_integrations()

    yield

    # Shutdown
    logger.info("Shutting down CRM application")

    # Cleanup MCP integrations
    await cleanup_mcp_integrations()


# FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    debug=settings.debug,
    lifespan=lifespan
)

# Security middleware - allow Railway and production hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost", "*.railway.app", "*.up.railway.app", "*"]  # * allows all in production
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    # Log slow requests
    if process_time > 1.0:
        logger.warning(
            "Slow request detected",
            path=request.url.path,
            method=request.method,
            process_time=process_time
        )

    return response


# Static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.version,
        "app": settings.app_name
    }


# Root endpoint - serve the main application
@app.get("/")
async def root(request: Request, user: Optional[User] = Depends(get_current_user_from_request)):
    """Root endpoint serving the main dashboard"""
    # Require authentication for dashboard
    if user is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        "pages/dashboard.html",
        {"request": request, "title": "Dashboard", "user": user}
    )


# AI Assistant page
@app.get("/assistant")
async def assistant_page(request: Request, user: Optional[User] = Depends(get_current_user_from_request)):
    """AI Assistant chat interface"""
    if user is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        "pages/assistant.html",
        {"request": request, "title": "AI Assistant", "user": user}
    )


# Authentication pages
@app.get("/login")
async def login_page(request: Request):
    """Login page"""
    return templates.TemplateResponse(
        "pages/login.html",
        {"request": request, "title": "Login"}
    )


@app.get("/register")
async def register_page(request: Request):
    """Registration page"""
    return templates.TemplateResponse(
        "pages/register.html",
        {"request": request, "title": "Register"}
    )


@app.get("/logout")
async def logout_page(request: Request):
    """Logout endpoint - clears cookie and redirects to login"""
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response


# Protected pages
@app.get("/contacts")
async def contacts_page(request: Request, user: Optional[User] = Depends(get_current_user_from_request)):
    """Contacts management page"""
    if user is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        "pages/contacts.html",
        {"request": request, "title": "Contacts", "user": user}
    )


@app.get("/organizations")
async def organizations_page(request: Request, user: Optional[User] = Depends(get_current_user_from_request)):
    """Organizations management page"""
    if user is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        "pages/organizations.html",
        {"request": request, "title": "Organizations", "user": user}
    )


@app.get("/projects")
async def projects_page(request: Request, user: Optional[User] = Depends(get_current_user_from_request)):
    """Projects management page"""
    if user is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        "pages/projects.html",
        {"request": request, "title": "Projects", "user": user}
    )


@app.get("/tasks")
async def tasks_page(request: Request, user: Optional[User] = Depends(get_current_user_from_request)):
    """Tasks management page"""
    if user is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        "pages/tasks.html",
        {"request": request, "title": "Tasks", "user": user}
    )


@app.get("/calendar")
async def calendar_page(request: Request, user: Optional[User] = Depends(get_current_user_from_request)):
    """Calendar page"""
    if user is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        "pages/calendar.html",
        {"request": request, "title": "Calendar", "user": user}
    )


@app.get("/settings")
async def settings_page(request: Request, user: Optional[User] = Depends(get_current_user_from_request)):
    """Settings page"""
    if user is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        "pages/settings.html",
        {"request": request, "title": "Settings", "user": user}
    )


# Include API routes
from .api.v1 import router as api_router
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )