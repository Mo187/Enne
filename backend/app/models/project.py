from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


# Association table for many-to-many relationship between projects and users
project_assignees = Table(
    'project_assignees',
    Base.metadata,
    Column('project_id', Integer, ForeignKey('projects.id'), primary_key=True),
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True)
)


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # Project owner
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)

    # Basic Information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    status = Column(String(50), default="planned", index=True)  # planned, in_progress, completed, on_hold, cancelled

    # Project Management
    priority = Column(String(20), default="medium")  # low, medium, high, urgent
    progress_percentage = Column(Integer, default=0)  # 0-100

    # Dates
    start_date = Column(DateTime(timezone=True), nullable=True)
    due_date = Column(DateTime(timezone=True), nullable=True, index=True)
    completed_date = Column(DateTime(timezone=True), nullable=True)

    # Budget and resources
    estimated_hours = Column(Integer, nullable=True)
    actual_hours = Column(Integer, default=0)
    budget = Column(String(50), nullable=True)  # Store as string for flexibility

    # Notes and tags
    notes = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # Array of strings for tagging

    # Metadata for AI context
    meta_data = Column(JSON, nullable=True)  # Store AI-derived information

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="projects")
    organization = relationship("Organization", back_populates="projects")
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")

    # Many-to-many relationship with users (assignees)
    assignees = relationship(
        "User",
        secondary=project_assignees,
        back_populates="assigned_projects"
    )

    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}', status='{self.status}')>"

    @property
    def is_overdue(self):
        """Check if project is overdue"""
        if not self.due_date or self.status in ["completed", "cancelled"]:
            return False
        return self.due_date < func.now()

    @property
    def days_until_due(self):
        """Calculate days until due date"""
        if not self.due_date:
            return None
        from datetime import datetime
        today = datetime.now(self.due_date.tzinfo)
        delta = self.due_date - today
        return delta.days

    @property
    def completion_rate(self):
        """Calculate task completion rate"""
        if not self.tasks:
            return 0
        completed_tasks = sum(1 for task in self.tasks if task.status == "completed")
        return (completed_tasks / len(self.tasks)) * 100

    def to_ai_context(self):
        """Convert project to AI-friendly context string"""
        context_parts = []

        context_parts.append(f"Project: {self.name}")
        context_parts.append(f"Status: {self.status}")

        if self.organization:
            context_parts.append(f"Organization: {self.organization.name}")

        if self.priority:
            context_parts.append(f"Priority: {self.priority}")

        if self.due_date:
            context_parts.append(f"Due: {self.due_date.strftime('%Y-%m-%d')}")

        if self.assignees:
            assignee_names = [assignee.name for assignee in self.assignees]
            context_parts.append(f"Assignees: {', '.join(assignee_names)}")

        task_count = len(self.tasks) if self.tasks else 0
        context_parts.append(f"Tasks: {task_count}")

        return " | ".join(context_parts)