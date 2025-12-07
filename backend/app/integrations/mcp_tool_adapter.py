"""
MCP Tool Adapter for Microsoft 365 Integration

This module provides a bridge between MCP Microsoft 365 tools and the CRM's
BaseTool interface, allowing seamless integration of external MCP tools
with the existing tool registry and execution framework.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json
import re
import structlog

from ..services.tool_interface import BaseTool, ToolResult, ToolSchema, ToolType
from ..models.integration import Integration
from .mcp_client import MCPMicrosoft365Client, MCPToolResult
from ..core.database import get_db
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


# Tool parameter schemas for validation
MCP_TOOL_SCHEMAS = {
    "outlook_search_emails": {
        "required": [],
        "optional": ["query", "folder", "limit", "from_address", "subject", "has_attachments"],
        "defaults": {"limit": 10, "folder": "inbox"},
        "types": {"limit": int, "has_attachments": bool}
    },
    "outlook_get_email": {
        "required": [],  # email_id is optional - MCP server fetches most recent if None
        "optional": ["email_id", "include_attachments"],
        "defaults": {},
        "friendly_name": "email"
    },
    "outlook_send_email": {
        "required": ["to_recipients", "subject", "body"],
        "optional": ["cc_recipients", "bcc_recipients", "importance"],
        "defaults": {"importance": "normal"},
        "error_hints": {
            "to_recipients": "Who should I send this email to?",
            "subject": "What should the email subject be?",
            "body": "What would you like to say in the email?"
        }
    },
    "outlook_create_draft": {
        "required": ["subject", "body"],
        "optional": ["to_recipients", "cc_recipients", "bcc_recipients"],
        "error_hints": {
            "to_recipients": "Who is this draft for?",
            "subject": "What should the subject line be?",
            "body": "What would you like the draft to say?"
        }
    },
    "outlook_reply_email": {
        "required": ["email_id", "body"],
        "optional": ["reply_all"],
        "defaults": {"reply_all": False},
        "error_hints": {
            "email_id": "Which email would you like to reply to?",
            "body": "What would you like to say in your reply?"
        }
    },
    "outlook_get_calendar_events": {
        "required": [],
        "optional": ["days", "limit", "start_date", "end_date"],
        "defaults": {"days": 7, "limit": 20},
        "types": {"days": int, "limit": int}
    },
    "outlook_create_calendar_event": {
        "required": ["subject", "start_time", "end_time"],
        "optional": ["location", "body", "attendees", "is_online_meeting"],
        "defaults": {"is_online_meeting": False},
        "error_hints": {
            "subject": "What should the meeting be called?",
            "start_time": "When should the meeting start?",
            "end_time": "When should the meeting end?"
        }
    },
    "onedrive_list_files": {
        "required": [],
        "optional": ["path", "limit"],
        "defaults": {"path": "/", "limit": 50}
    },
    "onedrive_search_files": {
        "required": ["query"],
        "optional": ["limit"],
        "defaults": {"limit": 20},
        "error_hints": {
            "query": "What would you like to search for in OneDrive?"
        }
    },
    "sharepoint_list_sites": {
        "required": [],
        "optional": ["limit"],
        "defaults": {"limit": 50}
    },
    "sharepoint_search_documents": {
        "required": ["query"],
        "optional": ["site_id", "limit"],
        "defaults": {"limit": 20},
        "error_hints": {
            "query": "What would you like to search for in SharePoint?"
        }
    },
    "teams_list_teams": {
        "required": [],
        "optional": ["limit"],
        "defaults": {"limit": 50}
    },
    "teams_list_channels": {
        "required": ["team_id"],
        "optional": [],
        "error_hints": {
            "team_id": "Which team's channels would you like to see?"
        }
    },
    "teams_send_message": {
        "required": ["team_id", "channel_id", "message"],
        "optional": [],
        "error_hints": {
            "team_id": "Which team should I send this to?",
            "channel_id": "Which channel should I post in?",
            "message": "What would you like to say?"
        }
    }
}

# User-friendly error message mappings
ERROR_MESSAGE_MAPPINGS = {
    "401": "Your Microsoft 365 session has expired. Please reconnect in Settings → Integrations.",
    "403": "You don't have permission to perform this action in Microsoft 365.",
    "404": "The requested item couldn't be found. It may have been deleted or moved.",
    "429": "Microsoft 365 is rate limiting requests. Please wait a moment and try again.",
    "500": "Microsoft 365 is experiencing issues. Please try again in a few minutes.",
    "503": "Microsoft 365 service is temporarily unavailable. Please try again later.",
    "ECONNREFUSED": "Unable to connect to Microsoft 365. Please check your internet connection.",
    "ETIMEDOUT": "The request to Microsoft 365 timed out. Please try again.",
    "invalid_grant": "Your Microsoft 365 authorization has expired. Please reconnect your account.",
    "token_expired": "Your session has expired. Please reconnect your Microsoft 365 account.",
    "no_emails_found": "No emails found matching your search criteria.",
    "mailbox_not_found": "Couldn't access your mailbox. Please check your Microsoft 365 connection.",
}


class MCPMicrosoft365ToolAdapter(BaseTool):
    """
    Adapter that wraps MCP Microsoft 365 tools to work with CRM's BaseTool interface.

    This adapter handles:
    - Converting between ToolResult and MCP TextContent formats
    - Managing user authentication tokens
    - Parameter mapping and validation
    - Error handling and logging
    - User-friendly error messages
    """

    def __init__(self, mcp_tool_name: str, tool_type: ToolType, mcp_client: MCPMicrosoft365Client, description: str = None, schema: Dict[str, Any] = None):
        super().__init__(mcp_tool_name, tool_type)
        self.mcp_tool_name = mcp_tool_name
        self.mcp_client = mcp_client
        self.tool_description = description or f"Microsoft 365 {mcp_tool_name} tool"
        self.tool_schema = schema or {}
        self.logger = logger.bind(tool=mcp_tool_name, component="mcp_adapter")
        # Get validation schema for this tool
        self.validation_schema = MCP_TOOL_SCHEMAS.get(mcp_tool_name, {})

    def _validate_parameters(self, parameters: Dict[str, Any]) -> Optional[ToolResult]:
        """
        Validate parameters before calling MCP tool.

        Args:
            parameters: Tool parameters to validate

        Returns:
            ToolResult with error if validation fails, None if valid
        """
        schema = self.validation_schema
        if not schema:
            return None  # No schema defined, skip validation

        # Check required parameters
        required = schema.get("required", [])
        missing = []
        for param in required:
            if param not in parameters or parameters[param] is None or parameters[param] == "":
                missing.append(param)

        if missing:
            # Build user-friendly error message
            error_hints = schema.get("error_hints", {})
            if len(missing) == 1 and missing[0] in error_hints:
                error_msg = error_hints[missing[0]]
            elif len(missing) == 1:
                friendly_name = schema.get("friendly_name", missing[0].replace("_", " "))
                error_msg = f"I need the {friendly_name} to complete this action."
                if "error_hint" in schema:
                    error_msg = schema["error_hint"]
            else:
                missing_friendly = [p.replace("_", " ") for p in missing]
                error_msg = f"I need more information: {', '.join(missing_friendly)}."

            self.logger.warning(
                "Parameter validation failed - missing required parameters",
                tool_name=self.mcp_tool_name,
                missing_params=missing
            )

            return ToolResult(
                success=False,
                error=error_msg,
                requires_clarification=True,
                clarification_type="missing_parameters",
                clarification_data={
                    "tool": self.mcp_tool_name,
                    "missing_params": missing,
                    "hints": {p: error_hints.get(p, f"Please provide {p}") for p in missing}
                }
            )

        # Type validation
        types = schema.get("types", {})
        for param, expected_type in types.items():
            if param in parameters and parameters[param] is not None:
                value = parameters[param]
                if expected_type == int and not isinstance(value, int):
                    try:
                        parameters[param] = int(value)
                    except (ValueError, TypeError):
                        return ToolResult(
                            success=False,
                            error=f"Invalid value for {param.replace('_', ' ')}. Please provide a number.",
                            requires_clarification=True,
                            clarification_type="invalid_parameter_type",
                            clarification_data={"param": param, "expected": "number"}
                        )
                elif expected_type == bool and not isinstance(value, bool):
                    if isinstance(value, str):
                        parameters[param] = value.lower() in ("true", "yes", "1")

        # Apply defaults
        defaults = schema.get("defaults", {})
        for param, default_value in defaults.items():
            if param not in parameters or parameters[param] is None:
                parameters[param] = default_value

        return None  # Validation passed

    def _format_user_friendly_error(self, error: str, tool_name: str = None) -> str:
        """
        Convert cryptic error messages to user-friendly messages.

        Args:
            error: Original error message
            tool_name: Name of the tool that failed

        Returns:
            User-friendly error message
        """
        if not error:
            return "An unexpected error occurred. Please try again."

        error_lower = error.lower()

        # Check for known error patterns
        for pattern, friendly_message in ERROR_MESSAGE_MAPPINGS.items():
            if pattern.lower() in error_lower:
                return friendly_message

        # Check for HTTP status codes in error
        status_match = re.search(r'\b(4\d{2}|5\d{2})\b', error)
        if status_match:
            status = status_match.group(1)
            if status in ERROR_MESSAGE_MAPPINGS:
                return ERROR_MESSAGE_MAPPINGS[status]

        # Check for common error patterns
        if "authentication" in error_lower or "unauthorized" in error_lower:
            return "Your Microsoft 365 session has expired. Please reconnect in Settings → Integrations."
        elif "permission" in error_lower or "forbidden" in error_lower:
            return "You don't have permission to perform this action in Microsoft 365."
        elif "not found" in error_lower:
            return "The requested item couldn't be found. It may have been deleted or moved."
        elif "timeout" in error_lower:
            return "The request took too long. Please try again."
        elif "network" in error_lower or "connection" in error_lower:
            return "Unable to connect to Microsoft 365. Please check your internet connection."
        elif "rate limit" in error_lower or "too many requests" in error_lower:
            return "Too many requests. Please wait a moment and try again."
        elif "invalid" in error_lower and "email" in error_lower:
            return "The email address appears to be invalid. Please check and try again."
        elif "mailbox" in error_lower:
            return "There was an issue accessing your mailbox. Please try again."

        # For unrecognized errors, provide a cleaner message
        # Remove technical details but keep the essence
        if len(error) > 200:
            # Long error - truncate and simplify
            return f"Microsoft 365 error: {error[:100]}... Please try again or contact support if this persists."

        # Return a cleaned version of the error
        return f"Microsoft 365 couldn't complete this action: {error}"

    async def execute(
        self,
        parameters: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> ToolResult:
        """
        Execute the MCP tool with CRM integration and automatic token management.

        Args:
            parameters: Tool-specific parameters from the CRM
            user_context: CRM user context including user ID and database session

        Returns:
            ToolResult with execution results
        """
        user_id = user_context.get("user_id")
        user = user_context.get("user")

        if not user_id and not user:
            return ToolResult(
                success=False,
                error="User context required for Microsoft 365 integration"
            )

        # Extract user ID
        if user and hasattr(user, 'id'):
            user_id = user.id
        elif isinstance(user_id, int):
            pass  # user_id is already an integer
        else:
            return ToolResult(
                success=False,
                error="Invalid user ID in context"
            )

        try:
            # Validate parameters before calling MCP tool
            validation_error = self._validate_parameters(parameters)
            if validation_error:
                self.logger.info(
                    "Parameter validation failed",
                    tool_name=self.mcp_tool_name,
                    user_id=user_id,
                    error=validation_error.error
                )
                return validation_error

            # Get user's Microsoft 365 integration with token validation
            integration = await self._get_integration_with_validation(user_id, user_context.get("db"))

            if not integration:
                return ToolResult(
                    success=False,
                    error="Microsoft 365 not connected. Please connect your Microsoft 365 account in Settings.",
                    requires_clarification=True,
                    clarification_type="authentication_required",
                    clarification_data={
                        "service": "microsoft365",
                        "action": "connect",
                        "url": "/settings#integrations"
                    }
                )

            # Check if token is expired or about to expire, refresh if needed
            if integration.is_token_expired:
                self.logger.warning("Token expired, attempting refresh before execution", user_id=user_id)
                integration = await self._refresh_integration_tokens(integration, user_context.get("db"))
                if not integration or integration.is_token_expired:
                    return ToolResult(
                        success=False,
                        error="Microsoft 365 authentication expired and refresh failed. Please reconnect your account.",
                        requires_clarification=True,
                        clarification_type="authentication_required",
                        clarification_data={
                            "service": "microsoft365",
                            "action": "reconnect",
                            "url": "/settings#integrations"
                        }
                    )
            elif integration.needs_refresh:
                # Proactive refresh if expires within 1 hour
                self.logger.info("Token expires soon, refreshing proactively", user_id=user_id)
                try:
                    integration = await self._refresh_integration_tokens(integration, user_context.get("db"))
                except Exception as refresh_error:
                    # Log but continue with existing token - it's not expired yet
                    self.logger.warning(
                        "Proactive token refresh failed, continuing with current token",
                        user_id=user_id,
                        error=str(refresh_error)
                    )

            # Convert CRM parameters to MCP format
            mcp_arguments = self._convert_to_mcp_format(parameters, user_id)

            self.logger.info(
                "Executing MCP tool",
                tool_name=self.mcp_tool_name,
                user_id=user_id,
                parameters=list(mcp_arguments.keys()),
                token_expires_in_minutes=int((integration.token_expires_at - datetime.now(timezone.utc)).total_seconds() / 60) if integration.token_expires_at else None
            )

            # Execute MCP tool with retry on auth failure
            max_retries = 2
            mcp_result = None

            for attempt in range(max_retries):
                try:
                    mcp_result = await self.mcp_client.call_tool(
                        tool_name=self.mcp_tool_name,
                        arguments=mcp_arguments,
                        access_token=integration.access_token,
                        refresh_token=integration.refresh_token,
                        expires_at=integration.token_expires_at.isoformat() if integration.token_expires_at else None,
                        user_id=str(user_id)
                    )

                    # If authentication failed and we have retries left, refresh and retry
                    if mcp_result.requires_auth and attempt < max_retries - 1:
                        self.logger.info(
                            "Authentication failed, refreshing token and retrying",
                            user_id=user_id,
                            attempt=attempt + 1,
                            max_retries=max_retries
                        )

                        integration = await self._refresh_integration_tokens(integration, user_context.get("db"))

                        if not integration or integration.is_token_expired:
                            return ToolResult(
                                success=False,
                                error="Microsoft 365 authentication expired and refresh failed. Please reconnect your account.",
                                requires_clarification=True,
                                clarification_type="authentication_required",
                                clarification_data={
                                    "service": "microsoft365",
                                    "action": "reconnect",
                                    "url": "/settings#integrations"
                                }
                            )

                        # Continue to retry with refreshed token
                        continue
                    else:
                        # Success or final failure
                        break

                except Exception as tool_error:
                    if attempt < max_retries - 1:
                        self.logger.warning(
                            "MCP tool call failed, retrying",
                            user_id=user_id,
                            attempt=attempt + 1,
                            error=str(tool_error)
                        )
                        continue
                    else:
                        raise

            if not mcp_result:
                return ToolResult(
                    success=False,
                    error="Microsoft 365 tool execution failed after retries"
                )

            # Convert MCP result to CRM ToolResult
            return self._convert_to_tool_result(mcp_result)

        except Exception as e:
            self.logger.error(
                "MCP tool execution failed",
                tool_name=self.mcp_tool_name,
                error=str(e),
                user_id=user_id
            )
            # Use friendly error formatter
            friendly_error = self._format_user_friendly_error(str(e), self.mcp_tool_name)
            return ToolResult(
                success=False,
                error=friendly_error
            )

    def get_schema(self) -> ToolSchema:
        """Get the tool's schema definition."""
        return ToolSchema(
            name=self.name,
            description=self.tool_description,
            tool_type=self.tool_type,
            parameters=self.tool_schema,
            examples=[],
            requires_auth=True,
            async_execution=True
        )

    def _convert_to_mcp_format(self, parameters: Dict[str, Any], user_id: int) -> Dict[str, Any]:
        """
        Convert CRM parameters to MCP format.

        Args:
            parameters: CRM tool parameters
            user_id: User ID for MCP user context

        Returns:
            MCP-compatible arguments
        """
        mcp_args = parameters.copy()

        # DEBUG LOGGING
        self.logger.info(
            "Converting parameters to MCP format",
            tool_name=self.mcp_tool_name,
            crm_parameters=parameters
        )

        # Add user_id for user-based tools (if not stateless) - use consistent format
        if self.mcp_tool_name not in ["extract_document_content_stateless", "onedrive_upload_document", "sharepoint_upload_document_stateless"]:
            mcp_args["user_id"] = f"crm_user_{user_id}"

        # Handle special parameter mappings
        parameter_mappings = {
            "search_query": "query",
            "email_subject": "subject",
            "email_body": "body",
            "email_to": "to_recipients",  # MCP server expects to_recipients (array)
            "email_cc": "cc_recipients",  # MCP server expects cc_recipients (array)
            "email_bcc": "bcc_recipients",  # MCP server expects bcc_recipients (array)
            "meeting_subject": "subject",
            "meeting_start": "start_time",
            "meeting_end": "end_time",
            "file_path": "path",
            "folder_name": "folder",
            "document_name": "name"
        }

        # Apply parameter mappings
        for crm_param, mcp_param in parameter_mappings.items():
            if crm_param in mcp_args:
                mcp_args[mcp_param] = mcp_args.pop(crm_param)

        # DEBUG LOGGING
        self.logger.info(
            "Converted to MCP format",
            tool_name=self.mcp_tool_name,
            mcp_parameters=mcp_args
        )

        return mcp_args

    def _convert_to_tool_result(self, mcp_result: MCPToolResult) -> ToolResult:
        """
        Convert MCP result to CRM ToolResult.

        Args:
            mcp_result: Result from MCP tool execution

        Returns:
            CRM ToolResult object
        """
        # Determine total count if the data is a list
        total_count = None
        if mcp_result.data and isinstance(mcp_result.data, list):
            total_count = len(mcp_result.data)
        elif mcp_result.data and isinstance(mcp_result.data, dict):
            # Check for common count fields
            if "total" in mcp_result.data:
                total_count = mcp_result.data["total"]
            elif "count" in mcp_result.data:
                total_count = mcp_result.data["count"]

        # Create enhanced message with execution time
        message = mcp_result.message
        if mcp_result.execution_time and mcp_result.success:
            time_info = f" (completed in {mcp_result.execution_time:.2f}s)"
            message = f"{message}{time_info}" if message else f"Operation completed{time_info}"

        # Extract actual data based on tool type for better chat handler processing
        result_data = mcp_result.data
        if mcp_result.success and isinstance(mcp_result.data, dict):
            # For email search results, extract the emails array
            if self.mcp_tool_name == "outlook_search_emails" and "emails" in mcp_result.data:
                result_data = mcp_result.data["emails"]
                # If emails is empty but operation was successful, still return empty array
                if not result_data:
                    result_data = []
            # For single email retrieval, extract the email dict
            elif self.mcp_tool_name == "outlook_get_email" and "email" in mcp_result.data:
                result_data = mcp_result.data["email"]

                # CRITICAL: Verify body_content is present for email retrieval
                if isinstance(result_data, dict):
                    has_body = "body_content" in result_data
                    body_length = len(result_data.get("body_content", "")) if has_body else 0
                    self.logger.info(
                        "Email body content verification",
                        tool_name=self.mcp_tool_name,
                        has_body_content=has_body,
                        body_length=body_length,
                        subject=result_data.get("subject", "unknown"),
                        body_preview=result_data.get("body_content", "")[:100] if has_body else "NO BODY"
                    )

                    if not has_body or body_length == 0:
                        self.logger.warning(
                            "Email retrieved without body content!",
                            tool_name=self.mcp_tool_name,
                            email_id=result_data.get("id", "unknown"),
                            subject=result_data.get("subject", "unknown")
                        )
            # For other structured responses, extract relevant data arrays
            elif "items" in mcp_result.data:
                result_data = mcp_result.data["items"]
            elif "results" in mcp_result.data:
                result_data = mcp_result.data["results"]

        # Format error message to be user-friendly
        formatted_error = None
        if mcp_result.error:
            formatted_error = self._format_user_friendly_error(mcp_result.error, self.mcp_tool_name)

        return ToolResult(
            success=mcp_result.success,
            data=result_data,
            message=message,
            error=formatted_error,
            total_count=total_count,
            requires_clarification=mcp_result.requires_auth,
            clarification_type="authentication_required" if mcp_result.requires_auth else None,
            clarification_data={
                "service": "microsoft365",
                "tool": self.mcp_tool_name
            } if mcp_result.requires_auth else None
        )

    async def _get_integration_with_validation(self, user_id: int, db: AsyncSession = None) -> Optional[Integration]:
        """
        Get user's Microsoft 365 integration with validation.

        Args:
            user_id: User ID
            db: Database session (optional)

        Returns:
            Integration object or None if not found/invalid
        """
        try:
            # Use provided session or get a new one
            if db is None:
                async with get_db() as db:
                    return await self._fetch_integration(user_id, db)
            else:
                return await self._fetch_integration(user_id, db)

        except Exception as e:
            self.logger.error("Failed to get user integration", user_id=user_id, error=str(e))
            return None

    async def _fetch_integration(self, user_id: int, db: AsyncSession) -> Optional[Integration]:
        """Fetch integration from database."""
        query = select(Integration).where(
            Integration.user_id == user_id,
            Integration.service_type == "microsoft365",
            Integration.is_active == True
        )
        result = await db.execute(query)
        integration = result.scalar_one_or_none()

        if not integration or not integration.access_token:
            self.logger.warning("No valid Microsoft 365 integration found", user_id=user_id)
            return None

        self.logger.info(
            "Integration fetched",
            user_id=user_id,
            is_expired=integration.is_token_expired,
            needs_refresh=integration.needs_refresh,
            expires_at=integration.token_expires_at.isoformat() if integration.token_expires_at else "unknown"
        )

        return integration

    async def _get_user_tokens(self, user_id: int, db: AsyncSession = None) -> Optional[Dict[str, str]]:
        """
        Get user's Microsoft 365 tokens from the database (legacy method).

        Args:
            user_id: User ID
            db: Database session (optional)

        Returns:
            Dictionary with access_token, refresh_token, expires_at
        """
        integration = await self._get_integration_with_validation(user_id, db)
        if not integration:
            return None

        return {
            "access_token": integration.access_token,
            "refresh_token": integration.refresh_token,
            "expires_at": integration.token_expires_at.isoformat() if integration.token_expires_at else None
        }

    async def _fetch_tokens(self, user_id: int, db: AsyncSession) -> Optional[Dict[str, str]]:
        """Fetch tokens from database (legacy method)."""
        integration = await self._fetch_integration(user_id, db)
        if not integration:
            return None

        return {
            "access_token": integration.access_token,
            "refresh_token": integration.refresh_token,
            "expires_at": integration.token_expires_at.isoformat() if integration.token_expires_at else None
        }

    async def _refresh_integration_tokens(self, integration: Integration, db: AsyncSession = None) -> Optional[Integration]:
        """
        Refresh integration tokens using MCP client.

        Args:
            integration: Integration object with current tokens
            db: Database session (optional)

        Returns:
            Updated Integration object or None if refresh failed
        """
        try:
            self.logger.info("Refreshing Microsoft 365 tokens", user_id=integration.user_id)

            # Refresh tokens via MCP client
            new_tokens = await self.mcp_client.refresh_access_token(
                refresh_token=integration.refresh_token,
                user_id=str(integration.user_id)
            )

            if not new_tokens or "access_token" not in new_tokens:
                self.logger.error("Token refresh failed - no access token returned", user_id=integration.user_id)
                return None

            # Update integration in database
            if db is None:
                async with get_db() as db:
                    return await self._update_integration_tokens(integration.user_id, new_tokens, db)
            else:
                return await self._update_integration_tokens(integration.user_id, new_tokens, db)

        except Exception as e:
            self.logger.error("Token refresh failed", user_id=integration.user_id, error=str(e))
            return None

    async def _update_integration_tokens(self, user_id: int, tokens: Dict[str, Any], db: AsyncSession) -> Optional[Integration]:
        """Update integration tokens in database and return updated integration."""
        query = select(Integration).where(
            Integration.user_id == user_id,
            Integration.service_type == "microsoft365"
        )
        result = await db.execute(query)
        integration = result.scalar_one_or_none()

        if integration:
            integration.access_token = tokens.get("access_token")
            if "refresh_token" in tokens:
                integration.refresh_token = tokens["refresh_token"]
            if "expires_at" in tokens:
                # Handle both datetime and string formats
                expires_at = tokens["expires_at"]
                if isinstance(expires_at, str):
                    integration.token_expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                else:
                    integration.token_expires_at = expires_at
            integration.updated_at = datetime.now(timezone.utc)

            await db.commit()
            await db.refresh(integration)

            self.logger.info(
                "Tokens updated successfully",
                user_id=user_id,
                new_expires_at=integration.token_expires_at.isoformat() if integration.token_expires_at else "unknown"
            )

            return integration

        return None

    async def _refresh_user_tokens(self, user_id: int, refresh_token: str, db: AsyncSession = None) -> Optional[Dict[str, str]]:
        """
        Refresh user's Microsoft 365 tokens.

        Args:
            user_id: User ID
            refresh_token: Current refresh token
            db: Database session (optional)

        Returns:
            Dictionary with new tokens or None if refresh failed
        """
        try:
            # Refresh tokens via MCP client
            new_tokens = await self.mcp_client.refresh_access_token(refresh_token, str(user_id))

            # Update tokens in database
            if db is None:
                async with get_db() as db:
                    await self._update_tokens(user_id, new_tokens, db)
            else:
                await self._update_tokens(user_id, new_tokens, db)

            return new_tokens

        except Exception as e:
            self.logger.error("Token refresh failed", user_id=user_id, error=str(e))
            return None

    async def _update_tokens(self, user_id: int, tokens: Dict[str, Any], db: AsyncSession):
        """Update tokens in database."""
        query = select(Integration).where(
            Integration.user_id == user_id,
            Integration.service_type == "microsoft365"
        )
        result = await db.execute(query)
        integration = result.scalar_one_or_none()

        if integration:
            integration.access_token = tokens.get("access_token")
            if "refresh_token" in tokens:
                integration.refresh_token = tokens["refresh_token"]
            if "expires_at" in tokens:
                integration.token_expires_at = datetime.fromisoformat(tokens["expires_at"].replace("Z", "+00:00"))
            integration.updated_at = datetime.now(timezone.utc)

            await db.commit()


class MCPToolRegistrar:
    """
    Helper class to register multiple MCP tools with the CRM tool registry.
    """

    def __init__(self, mcp_client: MCPMicrosoft365Client):
        self.mcp_client = mcp_client
        self.logger = logger.bind(component="mcp_registrar")

    async def register_all_tools(self, tool_registry) -> int:
        """
        Register all available MCP Microsoft 365 tools with the CRM tool registry.

        Args:
            tool_registry: CRM ToolRegistry instance

        Returns:
            Number of tools registered
        """
        try:
            # Get available tools from MCP server
            available_tools = await self.mcp_client.get_available_tools()

            registered_count = 0

            for tool_info in available_tools:
                tool_name = tool_info.get("name")
                description = tool_info.get("description", "")
                schema = tool_info.get("inputSchema", {})

                if tool_name:
                    # Determine tool type based on name prefix
                    tool_type = self._determine_tool_type(tool_name)

                    # Create and register adapter
                    adapter = MCPMicrosoft365ToolAdapter(
                        mcp_tool_name=tool_name,
                        tool_type=tool_type,
                        mcp_client=self.mcp_client,
                        description=description,
                        schema=schema
                    )

                    tool_registry.register_tool(adapter)
                    registered_count += 1

                    self.logger.info(
                        "Registered MCP tool",
                        tool_name=tool_name,
                        tool_type=tool_type.value
                    )

            self.logger.info(f"Registered {registered_count} MCP Microsoft 365 tools")
            return registered_count

        except Exception as e:
            self.logger.error("Failed to register MCP tools", error=str(e))
            return 0

    def _determine_tool_type(self, tool_name: str) -> ToolType:
        """Determine CRM tool type based on MCP tool name."""
        if tool_name.startswith("outlook_") and ("email" in tool_name or "draft" in tool_name):
            return ToolType.EMAIL
        elif tool_name.startswith("outlook_") and "calendar" in tool_name:
            return ToolType.CALENDAR
        elif tool_name.startswith(("teams_", "outlook_")) and ("meeting" in tool_name or "calendar" in tool_name):
            return ToolType.CALENDAR
        elif tool_name.startswith(("sharepoint_", "onedrive_")):
            return ToolType.STORAGE
        elif tool_name.startswith("teams_") and "meeting" not in tool_name:
            return ToolType.COMMUNICATION
        elif "extract_document" in tool_name:
            return ToolType.EXPORT
        else:
            return ToolType.COMMUNICATION  # Default for Microsoft 365 tools