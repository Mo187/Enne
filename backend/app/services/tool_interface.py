"""
Tool Interface for CRM - MCP Integration Ready

This module provides a standardized interface for tools that can be used by the AI assistant.
Designed to be compatible with MCP (Model Context Protocol) for future integrations
with Microsoft 365, Google Workspace, and other external services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()


class ToolType(Enum):
    """Types of tools available to the AI assistant"""
    DATABASE = "database"  # Internal CRM database operations
    CALENDAR = "calendar"  # Calendar integrations (Outlook, Google)
    EMAIL = "email"       # Email operations
    STORAGE = "storage"   # File storage (OneDrive, Google Drive)
    COMMUNICATION = "communication"  # Teams, Slack, etc.
    EXPORT = "export"     # Data export operations


@dataclass
class ToolResult:
    """Standardized tool execution result"""
    success: bool
    data: Optional[Union[Dict, List, str, int]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    total_count: Optional[int] = None
    requires_clarification: bool = False
    clarification_type: Optional[str] = None
    clarification_data: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        result = {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "error": self.error,
        }

        if self.total_count is not None:
            result["total_count"] = self.total_count

        if self.requires_clarification:
            result["requires_clarification"] = True
            result["clarification_type"] = self.clarification_type
            result["clarification_data"] = self.clarification_data

        return {k: v for k, v in result.items() if v is not None}


@dataclass
class ToolSchema:
    """Schema definition for a tool - MCP compatible"""
    name: str
    description: str
    tool_type: ToolType
    parameters: Dict[str, Any]  # JSON Schema for parameters
    examples: List[Dict[str, Any]]  # Example usages
    requires_auth: bool = False
    async_execution: bool = True


class BaseTool(ABC):
    """
    Abstract base class for all CRM tools.

    This interface is designed to be compatible with MCP (Model Context Protocol)
    and can be extended for external integrations.
    """

    def __init__(self, name: str, tool_type: ToolType):
        self.name = name
        self.tool_type = tool_type
        self.logger = logger.bind(tool=name)

    @abstractmethod
    async def execute(
        self,
        parameters: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> ToolResult:
        """
        Execute the tool with given parameters

        Args:
            parameters: Tool-specific parameters
            user_context: Context about the current user

        Returns:
            ToolResult with execution results
        """
        pass

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """
        Get the tool's schema definition

        Returns:
            ToolSchema describing this tool
        """
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate input parameters against schema

        Args:
            parameters: Parameters to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            schema = self.get_schema()
            # Basic validation - can be enhanced with jsonschema library
            required_params = schema.parameters.get("required", [])

            for param in required_params:
                if param not in parameters:
                    self.logger.warning("Missing required parameter", param=param)
                    return False

            return True

        except Exception as e:
            self.logger.error("Parameter validation error", error=str(e))
            return False


class ToolRegistry:
    """
    Registry for managing available tools.

    This will be extended to support MCP tool registration
    for external integrations like Microsoft 365 and Google Workspace.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self.logger = logger.bind(component="tool_registry")

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool"""
        self._tools[tool.name] = tool
        self.logger.info("Tool registered", tool_name=tool.name, tool_type=tool.tool_type.value)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def list_tools(self, tool_type: Optional[ToolType] = None) -> List[BaseTool]:
        """List all tools, optionally filtered by type"""
        if tool_type:
            return [tool for tool in self._tools.values() if tool.tool_type == tool_type]
        return list(self._tools.values())

    def get_schemas(self) -> List[ToolSchema]:
        """Get schemas for all registered tools - MCP compatible"""
        return [tool.get_schema() for tool in self._tools.values()]

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> ToolResult:
        """
        Execute a tool by name

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            user_context: User context

        Returns:
            ToolResult with execution results
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found"
            )

        if not tool.validate_parameters(parameters):
            return ToolResult(
                success=False,
                error=f"Invalid parameters for tool '{tool_name}'"
            )

        try:
            result = await tool.execute(parameters, user_context)
            self.logger.info("Tool executed", tool_name=tool_name, success=result.success)
            return result

        except Exception as e:
            self.logger.error("Tool execution failed", tool_name=tool_name, error=str(e))
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )


# Global tool registry instance
tool_registry = ToolRegistry()


# MCP Integration Preparation
class MCPToolAdapter:
    """
    Adapter for MCP (Model Context Protocol) tool integration.

    This will be used to integrate external tools from Microsoft 365,
    Google Workspace, and other MCP-compatible services.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.logger = logger.bind(component="mcp_adapter")

    async def register_mcp_tool(
        self,
        mcp_tool_spec: Dict[str, Any],
        execution_handler: callable
    ) -> None:
        """
        Register an MCP tool with the local registry

        Args:
            mcp_tool_spec: MCP tool specification
            execution_handler: Function to handle tool execution
        """
        # This will be implemented when MCP integrations are added
        # For now, this is a placeholder for the architecture
        pass

    def convert_to_mcp_format(self, tool_result: ToolResult) -> Dict[str, Any]:
        """Convert internal ToolResult to MCP-compatible format"""
        return {
            "content": [
                {
                    "type": "text" if isinstance(tool_result.data, str) else "json",
                    "text": tool_result.message or str(tool_result.data)
                }
            ],
            "isError": not tool_result.success
        }


# Future MCP integrations will be added here:
# - Microsoft365CalendarTool
# - GoogleWorkspaceEmailTool
# - OneDriveStorageTool
# - TeamsMessagingTool