"""
MCP Tool Adapter for Microsoft 365 Integration

This module provides a bridge between MCP Microsoft 365 tools and the CRM's
BaseTool interface, allowing seamless integration of external MCP tools
with the existing tool registry and execution framework.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json
import structlog

from ..services.tool_interface import BaseTool, ToolResult, ToolSchema, ToolType
from ..models.integration import Integration
from .mcp_client import MCPMicrosoft365Client, MCPToolResult
from ..core.database import get_db
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


class MCPMicrosoft365ToolAdapter(BaseTool):
    """
    Adapter that wraps MCP Microsoft 365 tools to work with CRM's BaseTool interface.

    This adapter handles:
    - Converting between ToolResult and MCP TextContent formats
    - Managing user authentication tokens
    - Parameter mapping and validation
    - Error handling and logging
    """

    def __init__(self, mcp_tool_name: str, tool_type: ToolType, mcp_client: MCPMicrosoft365Client, description: str = None, schema: Dict[str, Any] = None):
        super().__init__(mcp_tool_name, tool_type)
        self.mcp_tool_name = mcp_tool_name
        self.mcp_client = mcp_client
        self.tool_description = description or f"Microsoft 365 {mcp_tool_name} tool"
        self.tool_schema = schema or {}
        self.logger = logger.bind(tool=mcp_tool_name, component="mcp_adapter")

    async def execute(
        self,
        parameters: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> ToolResult:
        """
        Execute the MCP tool with CRM integration.

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
            # Get user's Microsoft 365 integration
            tokens = await self._get_user_tokens(user_id, user_context.get("db"))

            if not tokens:
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

            # Convert CRM parameters to MCP format
            mcp_arguments = self._convert_to_mcp_format(parameters, user_id)

            self.logger.info(
                "Executing MCP tool",
                tool_name=self.mcp_tool_name,
                user_id=user_id,
                parameters=list(mcp_arguments.keys())
            )

            # Execute MCP tool
            mcp_result = await self.mcp_client.call_tool(
                tool_name=self.mcp_tool_name,
                arguments=mcp_arguments,
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                expires_at=tokens["expires_at"],
                user_id=str(user_id)
            )

            # Handle token refresh if needed
            if mcp_result.requires_auth:
                self.logger.info("Attempting token refresh", user_id=user_id)
                refreshed = await self._refresh_user_tokens(user_id, tokens["refresh_token"], user_context.get("db"))

                if refreshed:
                    # Retry with new tokens
                    mcp_result = await self.mcp_client.call_tool(
                        tool_name=self.mcp_tool_name,
                        arguments=mcp_arguments,
                        access_token=refreshed["access_token"],
                        refresh_token=refreshed["refresh_token"],
                        expires_at=refreshed["expires_at"],
                        user_id=str(user_id)
                    )
                else:
                    return ToolResult(
                        success=False,
                        error="Microsoft 365 authentication expired. Please reconnect your account.",
                        requires_clarification=True,
                        clarification_type="authentication_required",
                        clarification_data={
                            "service": "microsoft365",
                            "action": "reconnect",
                            "url": "/settings#integrations"
                        }
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
            return ToolResult(
                success=False,
                error=f"Microsoft 365 integration error: {str(e)}"
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

        # Add user_id for user-based tools (if not stateless) - use consistent format
        if self.mcp_tool_name not in ["extract_document_content_stateless", "onedrive_upload_document", "sharepoint_upload_document_stateless"]:
            mcp_args["user_id"] = f"crm_user_{user_id}"

        # Handle special parameter mappings
        parameter_mappings = {
            "search_query": "query",
            "email_subject": "subject",
            "email_body": "body",
            "email_to": "to",
            "email_cc": "cc",
            "email_bcc": "bcc",
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

        return ToolResult(
            success=mcp_result.success,
            data=mcp_result.data,
            message=message,
            error=mcp_result.error,
            total_count=total_count,
            requires_clarification=mcp_result.requires_auth,
            clarification_type="authentication_required" if mcp_result.requires_auth else None,
            clarification_data={
                "service": "microsoft365",
                "tool": self.mcp_tool_name
            } if mcp_result.requires_auth else None
        )

    async def _get_user_tokens(self, user_id: int, db: AsyncSession = None) -> Optional[Dict[str, str]]:
        """
        Get user's Microsoft 365 tokens from the database.

        Args:
            user_id: User ID
            db: Database session (optional)

        Returns:
            Dictionary with access_token, refresh_token, expires_at
        """
        try:
            # Use provided session or get a new one
            if db is None:
                async with get_db() as db:
                    return await self._fetch_tokens(user_id, db)
            else:
                return await self._fetch_tokens(user_id, db)

        except Exception as e:
            self.logger.error("Failed to get user tokens", user_id=user_id, error=str(e))
            return None

    async def _fetch_tokens(self, user_id: int, db: AsyncSession) -> Optional[Dict[str, str]]:
        """Fetch tokens from database."""
        query = select(Integration).where(
            Integration.user_id == user_id,
            Integration.service_type == "microsoft365",
            Integration.is_active == True
        )
        result = await db.execute(query)
        integration = result.scalar_one_or_none()

        if not integration or not integration.access_token:
            return None

        return {
            "access_token": integration.access_token,
            "refresh_token": integration.refresh_token,
            "expires_at": integration.token_expires_at.isoformat() if integration.token_expires_at else None
        }

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