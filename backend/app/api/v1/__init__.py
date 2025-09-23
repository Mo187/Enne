from fastapi import APIRouter
from .auth import router as auth_router
from .contacts import router as contacts_router
from .organizations import router as organizations_router
from .projects import router as projects_router
from .tasks import router as tasks_router
from .chat import router as chat_router

router = APIRouter()

# Include all API routers
router.include_router(auth_router, prefix="/auth", tags=["authentication"])
router.include_router(contacts_router, prefix="/contacts", tags=["contacts"])
router.include_router(organizations_router, prefix="/organizations", tags=["organizations"])
router.include_router(projects_router, prefix="/projects", tags=["projects"])
router.include_router(tasks_router, prefix="/tasks", tags=["tasks"])
router.include_router(chat_router, prefix="/chat", tags=["ai-chat"])