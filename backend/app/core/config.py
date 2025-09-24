from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # App Configuration
    app_name: str = "AI-Assisted CRM"
    debug: bool = False
    version: str = "1.0.0"

    # Database Configuration
    database_url: str = "postgresql+asyncpg://crm_user:crm_password@localhost/crm_db"

    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"

    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # AI Services
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_ai_key: Optional[str] = None

    # OAuth Credentials
    microsoft_client_id: Optional[str] = None
    microsoft_client_secret: Optional[str] = None
    microsoft_tenant_id: Optional[str] = None
    microsoft_redirect_uri: Optional[str] = None
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None

    # MCP Integration
    mcp_microsoft365_url: str = "http://host.docker.internal:8001"
    mcp_timeout_seconds: int = 30
    mcp_retry_attempts: int = 3
    mcp_connection_pool_size: int = 10

    # External URLs
    frontend_url: str = "http://localhost:8000"
    backend_url: str = "http://localhost:8000"

    # Email Configuration (for notifications)
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None

    # Rate Limiting
    rate_limit_requests_per_minute: int = 100

    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()