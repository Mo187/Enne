from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class Integration(Base):
    __tablename__ = "integrations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Integration Details
    service_type = Column(String(50), nullable=False, index=True)  # microsoft365, google_workspace, etc.
    service_name = Column(String(100), nullable=False)  # Human-readable name
    is_active = Column(Boolean, default=True)

    # Authentication Data (encrypted)
    access_token = Column(Text, nullable=True)  # Encrypted
    refresh_token = Column(Text, nullable=True)  # Encrypted
    token_expires_at = Column(DateTime(timezone=True), nullable=True)

    # OAuth Configuration
    client_id = Column(String(255), nullable=True)
    scopes = Column(JSON, nullable=True)  # Array of granted scopes

    # Sync Configuration
    sync_enabled = Column(Boolean, default=True)
    sync_calendars = Column(Boolean, default=True)
    sync_contacts = Column(Boolean, default=False)  # Optional
    sync_emails = Column(Boolean, default=False)  # Optional
    sync_files = Column(Boolean, default=False)  # Optional

    # Sync Status
    last_sync_at = Column(DateTime(timezone=True), nullable=True)
    sync_status = Column(String(50), default="pending")  # pending, syncing, success, error
    sync_error_message = Column(Text, nullable=True)

    # Webhook Configuration
    webhook_url = Column(String(500), nullable=True)
    webhook_secret = Column(String(255), nullable=True)

    # Settings and preferences
    settings = Column(JSON, nullable=True)  # Service-specific settings
    meta_data = Column(JSON, nullable=True)  # Additional metadata

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    connected_at = Column(DateTime(timezone=True), nullable=True)
    last_error_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="integrations")

    def __repr__(self):
        return f"<Integration(id={self.id}, service='{self.service_type}', user_id={self.user_id})>"

    @property
    def is_token_expired(self):
        """Check if access token is expired"""
        if not self.token_expires_at:
            return False
        from datetime import datetime, timezone, timedelta
        # Consider token expired if it expires within 5 minutes (buffer)
        buffer_time = timedelta(minutes=5)
        expiry_threshold = datetime.now(timezone.utc) + buffer_time
        return self.token_expires_at <= expiry_threshold

    @property
    def needs_refresh(self):
        """Check if token needs refresh (expires within 1 hour)"""
        if not self.token_expires_at or not self.refresh_token:
            return False
        from datetime import datetime, timezone, timedelta
        threshold = datetime.now(timezone.utc) + timedelta(hours=1)
        return self.token_expires_at <= threshold

    @property
    def is_connected(self):
        """Check if integration is connected and usable"""
        return (
            self.is_active and
            self.access_token and
            not self.is_token_expired
        )

    @property
    def is_healthy(self):
        """Check if integration is healthy and working"""
        return (
            self.is_active and
            self.sync_status in ["success", "syncing"] and
            not self.is_token_expired
        )

    def get_display_name(self):
        """Get human-readable display name"""
        service_names = {
            "microsoft365": "Microsoft 365",
            "google_workspace": "Google Workspace",
            "outlook": "Outlook",
            "gmail": "Gmail",
            "teams": "Microsoft Teams",
            "onedrive": "OneDrive",
            "sharepoint": "SharePoint",
            "google_drive": "Google Drive",
            "google_calendar": "Google Calendar"
        }
        return service_names.get(self.service_type, self.service_name or self.service_type.title())

    def to_ai_context(self):
        """Convert integration to AI-friendly context string"""
        context_parts = []

        context_parts.append(f"Integration: {self.get_display_name()}")
        context_parts.append(f"Status: {'Active' if self.is_active else 'Inactive'}")

        if self.sync_enabled:
            sync_types = []
            if self.sync_calendars:
                sync_types.append("Calendar")
            if self.sync_contacts:
                sync_types.append("Contacts")
            if self.sync_emails:
                sync_types.append("Email")
            if self.sync_files:
                sync_types.append("Files")

            if sync_types:
                context_parts.append(f"Syncing: {', '.join(sync_types)}")

        if self.last_sync_at:
            context_parts.append(f"Last Sync: {self.last_sync_at.strftime('%Y-%m-%d %H:%M')}")

        return " | ".join(context_parts)