"""
Application startup configuration for MCP integrations and tool registration.
"""

import structlog
from ..services.tool_interface import tool_registry
from ..integrations.mcp_client import mcp_client
from ..integrations.mcp_tool_adapter import MCPToolRegistrar

logger = structlog.get_logger()


async def initialize_mcp_integrations():
    """
    Initialize MCP integrations and register tools with the CRM tool registry.

    This function should be called during application startup to:
    1. Connect to MCP servers
    2. Register MCP tools with the CRM tool registry
    3. Set up monitoring and health checks
    """

    logger.info("Initializing MCP integrations...")

    try:
        # Initialize Microsoft 365 MCP integration
        await initialize_microsoft365_mcp()

        logger.info("MCP integrations initialized successfully")

    except Exception as e:
        logger.error("Failed to initialize MCP integrations", error=str(e))
        # Don't raise the exception - we want the app to start even if MCP is unavailable
        # The integration will be retried when users try to use Microsoft 365 features


async def initialize_microsoft365_mcp():
    """Initialize Microsoft 365 MCP integration and register tools."""

    try:
        # Check if MCP server is available
        is_healthy = await mcp_client.health_check()

        if not is_healthy:
            logger.warning("Microsoft 365 MCP server is not available - tools will not be registered")
            return

        # Create tool registrar
        registrar = MCPToolRegistrar(mcp_client)

        # Register all Microsoft 365 tools with the CRM tool registry
        registered_count = await registrar.register_all_tools(tool_registry)

        logger.info(
            "Microsoft 365 MCP tools registered",
            count=registered_count,
            server_url=mcp_client.base_url
        )

        # Log available tools for debugging
        tools = tool_registry.list_tools()
        ms365_tools = [tool for tool in tools if tool.name.startswith(("outlook_", "sharepoint_", "onedrive_", "teams_", "authenticate", "extract_document"))]

        if ms365_tools:
            logger.info(
                "Available Microsoft 365 tools",
                tools=[tool.name for tool in ms365_tools]
            )

    except Exception as e:
        logger.error("Failed to initialize Microsoft 365 MCP", error=str(e))
        raise


async def cleanup_mcp_integrations():
    """
    Cleanup MCP integrations during application shutdown.

    This function should be called during application shutdown to:
    1. Close MCP client connections
    2. Clean up resources
    """

    logger.info("Cleaning up MCP integrations...")

    try:
        # Close MCP client
        await mcp_client.close()

        logger.info("MCP integrations cleaned up successfully")

    except Exception as e:
        logger.error("Failed to cleanup MCP integrations", error=str(e))