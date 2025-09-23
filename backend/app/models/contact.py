from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class Contact(Base):
    __tablename__ = "contacts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Basic Information
    name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), nullable=True, index=True)
    phone = Column(String(50), nullable=True)
    job_position = Column(String(255), nullable=True)
    organization = Column(String(255), nullable=True, index=True)

    # Additional Information
    notes = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # Array of strings for tagging

    # Social/Professional Links
    linkedin_url = Column(String(500), nullable=True)
    website = Column(String(500), nullable=True)

    # Address Information
    address_line1 = Column(String(255), nullable=True)
    address_line2 = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)

    # Metadata for AI context
    meta_data = Column(JSON, nullable=True)  # Store AI-derived information

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_contacted = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="contacts")

    def __repr__(self):
        return f"<Contact(id={self.id}, name='{self.name}', email='{self.email}')>"

    @property
    def full_address(self):
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

    def to_ai_context(self):
        """Convert contact to AI-friendly context string"""
        context_parts = []

        if self.name:
            context_parts.append(f"Name: {self.name}")
        if self.job_position and self.organization:
            context_parts.append(f"Position: {self.job_position} at {self.organization}")
        elif self.organization:
            context_parts.append(f"Organization: {self.organization}")
        elif self.job_position:
            context_parts.append(f"Position: {self.job_position}")

        if self.email:
            context_parts.append(f"Email: {self.email}")
        if self.phone:
            context_parts.append(f"Phone: {self.phone}")

        return " | ".join(context_parts)