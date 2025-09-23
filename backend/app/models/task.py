from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from ..core.database import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)

    # Basic Information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    status = Column(String(50), default="pending", index=True)  # pending, in_progress, completed, blocked, cancelled

    # Task Management
    priority = Column(String(20), default="medium")  # low, medium, high, urgent
    assignee = Column(String(255), nullable=True)  # Can be user name or external person
    estimated_hours = Column(Integer, nullable=True)
    actual_hours = Column(Integer, default=0)

    # Dates
    start_date = Column(DateTime(timezone=True), nullable=True)
    due_date = Column(DateTime(timezone=True), nullable=True, index=True)
    completed_date = Column(DateTime(timezone=True), nullable=True)

    # Dependencies and ordering
    parent_task_id = Column(Integer, ForeignKey("tasks.id"), nullable=True)
    sort_order = Column(Integer, default=0)

    # Notes and tags
    notes = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # Array of strings for tagging

    # Metadata for AI context
    meta_data = Column(JSON, nullable=True)  # Store AI-derived information

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    project = relationship("Project", back_populates="tasks")

    # Self-referential relationship for subtasks
    parent_task = relationship("Task", remote_side=[id], backref=backref("subtasks", lazy="selectin"))

    def __repr__(self):
        return f"<Task(id={self.id}, name='{self.name}', status='{self.status}')>"

    @property
    def is_overdue(self):
        """Check if task is overdue"""
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
    def is_blocked(self):
        """Check if task is blocked"""
        return self.status == "blocked"

    @property
    def has_subtasks(self):
        """Check if task has subtasks"""
        return len(self.subtasks) > 0 if self.subtasks else False

    @property
    def subtask_completion_rate(self):
        """Calculate subtask completion rate"""
        if not self.subtasks:
            return 100  # No subtasks means 100% complete
        completed_subtasks = sum(1 for subtask in self.subtasks if subtask.status == "completed")
        return (completed_subtasks / len(self.subtasks)) * 100

    def to_ai_context(self):
        """Convert task to AI-friendly context string"""
        context_parts = []

        context_parts.append(f"Task: {self.name}")
        context_parts.append(f"Status: {self.status}")

        if self.project:
            context_parts.append(f"Project: {self.project.name}")

        if self.priority:
            context_parts.append(f"Priority: {self.priority}")

        if self.assignee:
            context_parts.append(f"Assignee: {self.assignee}")

        if self.due_date:
            context_parts.append(f"Due: {self.due_date.strftime('%Y-%m-%d')}")

        if self.estimated_hours:
            context_parts.append(f"Estimated: {self.estimated_hours}h")

        if self.subtasks:
            context_parts.append(f"Subtasks: {len(self.subtasks)}")

        return " | ".join(context_parts)