from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Basic Information
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    industry = Column(String(100), nullable=True, index=True)

    # Contact Information
    website = Column(String(500), nullable=True)
    phone = Column(String(50), nullable=True)
    email = Column(String(255), nullable=True)

    # Address Information
    address_line1 = Column(String(255), nullable=True)
    address_line2 = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)

    # Business Information
    company_size = Column(String(50), nullable=True)  # "1-10", "11-50", "51-200", etc.
    annual_revenue = Column(String(50), nullable=True)  # "< $1M", "$1M-$10M", etc.
    founded_year = Column(Integer, nullable=True)

    # Social/Professional Links
    linkedin_url = Column(String(500), nullable=True)
    twitter_url = Column(String(500), nullable=True)

    # Status and relationship
    relationship_status = Column(String(50), default="prospect")  # prospect, client, partner, vendor
    priority = Column(String(20), default="medium")  # low, medium, high

    # Notes and tags
    notes = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # Array of strings for tagging

    # Metadata for AI context
    meta_data = Column(JSON, nullable=True)  # Store AI-derived information

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_interaction = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="organizations")
    projects = relationship("Project", back_populates="organization")

    def __repr__(self):
        return f"<Organization(id={self.id}, name='{self.name}', industry='{self.industry}')>"

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
        """Convert organization to AI-friendly context string"""
        context_parts = []

        context_parts.append(f"Organization: {self.name}")

        if self.industry:
            context_parts.append(f"Industry: {self.industry}")
        if self.company_size:
            context_parts.append(f"Size: {self.company_size}")
        if self.relationship_status:
            context_parts.append(f"Relationship: {self.relationship_status}")

        if self.website:
            context_parts.append(f"Website: {self.website}")
        if self.email:
            context_parts.append(f"Email: {self.email}")

        return " | ".join(context_parts)