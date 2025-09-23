from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import structlog
import time

from .core.config import settings
from .core.database import create_tables


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting up CRM application", version=settings.version)

    # Create database tables
    await create_tables()
    logger.info("Database tables created/verified")

    yield

    # Shutdown
    logger.info("Shutting down CRM application")


# FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    debug=settings.debug,
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
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
async def root(request: Request):
    """Root endpoint serving the main dashboard"""
    return templates.TemplateResponse(
        "pages/dashboard.html",
        {"request": request, "title": "Dashboard"}
    )


# AI Assistant page
@app.get("/assistant")
async def assistant_page(request: Request):
    """AI Assistant chat interface"""
    return templates.TemplateResponse(
        "pages/assistant.html",
        {"request": request, "title": "AI Assistant"}
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
    """Logout endpoint - just redirects to login"""
    return templates.TemplateResponse(
        "pages/login.html",
        {"request": request, "title": "Login", "logout": True}
    )


# Protected pages
@app.get("/contacts")
async def contacts_page(request: Request):
    """Contacts management page"""
    return templates.TemplateResponse(
        "pages/contacts.html",
        {"request": request, "title": "Contacts"}
    )


@app.get("/organizations")
async def organizations_page(request: Request):
    """Organizations management page"""
    return templates.TemplateResponse(
        "pages/organizations.html",
        {"request": request, "title": "Organizations"}
    )


@app.get("/projects")
async def projects_page(request: Request):
    """Projects management page"""
    return templates.TemplateResponse(
        "pages/projects.html",
        {"request": request, "title": "Projects"}
    )


@app.get("/tasks")
async def tasks_page(request: Request):
    """Tasks management page"""
    return templates.TemplateResponse(
        "pages/tasks.html",
        {"request": request, "title": "Tasks"}
    )


@app.get("/calendar")
async def calendar_page(request: Request):
    """Calendar page"""
    return templates.TemplateResponse(
        "pages/calendar.html",
        {"request": request, "title": "Calendar"}
    )


@app.get("/settings")
async def settings_page(request: Request):
    """Settings page"""
    return templates.TemplateResponse(
        "pages/settings.html",
        {"request": request, "title": "Settings"}
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