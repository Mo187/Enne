from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Profile information
    company = Column(String(255), nullable=True)
    job_title = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    timezone = Column(String(50), default="UTC")

    # Preferences
    theme = Column(String(20), default="light")  # light, dark
    language = Column(String(10), default="en")

    # AI Preferences
    preferred_ai_model = Column(String(50), default="claude")  # claude, openai, gemini
    ai_context_length = Column(Integer, default=4000)

    # Relationships
    contacts = relationship("Contact", back_populates="user", cascade="all, delete-orphan")
    organizations = relationship("Organization", back_populates="user", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")
    integrations = relationship("Integration", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

    # Many-to-many relationships
    assigned_projects = relationship(
        "Project",
        secondary="project_assignees",
        back_populates="assignees"
    )

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', name='{self.name}')>"