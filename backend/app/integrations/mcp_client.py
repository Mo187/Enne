"""
MCP Microsoft 365 HTTP Client for CRM Integration

This module provides HTTP client functionality to communicate with the
Microsoft 365 MCP server running in HTTP bridge mode.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..core.config import settings

logger = structlog.get_logger()


class MCPClientError(Exception):
    """Base exception for MCP client errors"""
    pass


class MCPServerUnavailableError(MCPClientError):
    """Raised when MCP server is not available"""
    pass


class MCPAuthenticationError(MCPClientError):
    """Raised when authentication with MCP server fails"""
    pass


class TokenRefreshError(MCPClientError):
    """Raised when token refresh fails"""
    pass


@dataclass
class MCPToolResult:
    """Result from MCP tool execution"""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    tool_name: Optional[str] = None
    execution_time: Optional[float] = None
    requires_auth: bool = False

    @classmethod
    def from_mcp_response(cls, response_data: Dict, tool_name: str, execution_time: float = None) -> 'MCPToolResult':
        """Create MCPToolResult from MCP server response"""
        try:
            # MCP responses are in result.content[0].text as JSON
            if "result" in response_data and "content" in response_data["result"]:
                content = response_data["result"]["content"][0]["text"]
                data = json.loads(content)

                return cls(
                    success=data.get("success", False),
                    data=data,
                    message=data.get("message"),
                    error=data.get("error"),
                    tool_name=tool_name,
                    execution_time=execution_time
                )
            else:
                return cls(
                    success=False,
                    error="Invalid MCP response format",
                    tool_name=tool_name,
                    execution_time=execution_time
                )
        except Exception as e:
            return cls(
                success=False,
                error=f"Failed to parse MCP response: {str(e)}",
                tool_name=tool_name,
                execution_time=execution_time
            )


class MCPMicrosoft365Client:
    """
    HTTP client for communicating with Microsoft 365 MCP server.

    Provides stateless authentication and tool execution with
    comprehensive error handling and monitoring.
    """

    def __init__(self, base_url: str = None, timeout: float = None):
        self.base_url = base_url or settings.mcp_microsoft365_url
        self.timeout = timeout or settings.mcp_timeout_seconds
        self.client = None
        self.logger = logger.bind(component="mcp_client")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                headers={"Content-Type": "application/json"}
            )
        return self.client

    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def health_check(self) -> bool:
        """Check if MCP server is available"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning("MCP server health check failed", error=str(e))
            return False

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from MCP server"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/tools")
            response.raise_for_status()

            data = response.json()
            return data.get("tools", [])

        except Exception as e:
            self.logger.error("Failed to get available tools", error=str(e))
            raise MCPClientError(f"Failed to get available tools: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException))
    )
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        expires_at: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> MCPToolResult:
        """
        Call MCP tool with stateless authentication.

        Args:
            tool_name: Name of the MCP tool to execute
            arguments: Tool-specific arguments
            access_token: Microsoft access token for stateless mode
            refresh_token: Microsoft refresh token for token refresh
            expires_at: Token expiration timestamp
            user_id: User identifier for logging and tracking

        Returns:
            MCPToolResult with execution results
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Prepare payload for stateless mode
            payload = {
                "arguments": arguments.copy()
            }

            # Add authentication tokens if provided
            if access_token:
                payload.update({
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "expires_at": expires_at,
                    "token_type": "Bearer"
                })
            elif user_id:
                # For tools that require user_id (like authenticate)
                payload["arguments"]["user_id"] = user_id

            client = await self._get_client()

            self.logger.info(
                "Calling MCP tool",
                tool_name=tool_name,
                user_id=user_id,
                has_token=bool(access_token)
            )

            # Make HTTP request to MCP server
            response = await client.post(
                f"{self.base_url}/tools/{tool_name}",
                json=payload
            )

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            if response.status_code == 200:
                result_data = response.json()
                result = MCPToolResult.from_mcp_response(result_data, tool_name, execution_time)

                self.logger.info(
                    "MCP tool executed successfully",
                    tool_name=tool_name,
                    success=result.success,
                    execution_time=execution_time,
                    user_id=user_id
                )

                return result

            elif response.status_code == 401:
                self.logger.warning("MCP authentication failed", tool_name=tool_name, user_id=user_id)
                return MCPToolResult(
                    success=False,
                    error="Authentication required",
                    tool_name=tool_name,
                    execution_time=execution_time,
                    requires_auth=True
                )

            else:
                error_msg = f"MCP server error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("detail", error_msg)
                except:
                    pass

                self.logger.error(
                    "MCP tool execution failed",
                    tool_name=tool_name,
                    status_code=response.status_code,
                    error=error_msg,
                    user_id=user_id
                )

                return MCPToolResult(
                    success=False,
                    error=error_msg,
                    tool_name=tool_name,
                    execution_time=execution_time
                )

        except httpx.RequestError as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.error("MCP request failed", tool_name=tool_name, error=str(e), user_id=user_id)
            raise MCPServerUnavailableError(f"MCP server request failed: {e}")

        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.error("Unexpected MCP error", tool_name=tool_name, error=str(e), user_id=user_id)
            return MCPToolResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                tool_name=tool_name,
                execution_time=execution_time
            )

    async def get_auth_url(self, user_id: str, redirect_uri: str = None) -> str:
        """
        Get Microsoft 365 OAuth authorization URL.

        Args:
            user_id: User identifier for state parameter
            redirect_uri: Optional custom redirect URI

        Returns:
            OAuth authorization URL
        """
        try:
            arguments = {"user_id": user_id}
            if redirect_uri:
                arguments["redirect_uri"] = redirect_uri

            result = await self.call_tool("authenticate", arguments, user_id=user_id)

            if result.success and result.data:
                # Handle both nested and flat structures
                if isinstance(result.data, dict):
                    # Try nested structure first
                    if "data" in result.data and "auth_url" in result.data["data"]:
                        return result.data["data"]["auth_url"]
                    # Fallback to flat structure
                    return result.data.get("auth_url")

            raise MCPAuthenticationError(result.error or "Failed to get auth URL")

        except Exception as e:
            self.logger.error("Failed to get auth URL", user_id=user_id, error=str(e))
            raise MCPAuthenticationError(f"Failed to get auth URL: {e}")

    async def handle_auth_callback(self, code: str, state: str) -> Dict[str, Any]:
        """
        Handle OAuth callback and exchange code for tokens.

        Args:
            code: Authorization code from OAuth callback
            state: State parameter (should be user_id)

        Returns:
            Dictionary containing access token, refresh token, and expiration
        """
        try:
            client = await self._get_client()

            response = await client.post(
                f"{self.base_url}/auth/callback",
                json={"code": code, "state": state}
            )

            response.raise_for_status()

            token_data = response.json()

            self.logger.info("OAuth callback handled successfully", user_id=state)

            return token_data

        except Exception as e:
            self.logger.error("OAuth callback failed", state=state, error=str(e))
            raise MCPAuthenticationError(f"OAuth callback failed: {e}")

    async def refresh_access_token(self, refresh_token: str, user_id: str) -> Dict[str, Any]:
        """
        Refresh Microsoft access token.

        Args:
            refresh_token: Microsoft refresh token
            user_id: User identifier for logging

        Returns:
            Dictionary containing new access token and expiration
        """
        try:
            client = await self._get_client()

            response = await client.post(
                f"{self.base_url}/auth/refresh",
                json={"refresh_token": refresh_token, "user_id": user_id}
            )

            response.raise_for_status()

            token_data = response.json()

            self.logger.info("Token refreshed successfully", user_id=user_id)

            return token_data

        except Exception as e:
            self.logger.error("Token refresh failed", user_id=user_id, error=str(e))
            raise TokenRefreshError(f"Token refresh failed: {e}")

    async def call_tool_safe(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        access_token: Optional[str] = None,
        **kwargs
    ) -> MCPToolResult:
        """
        Call MCP tool with comprehensive error handling and fallback.

        Returns a result even if the MCP server is unavailable.
        """
        try:
            # Check server health first
            if not await self.health_check():
                return MCPToolResult(
                    success=False,
                    error="Microsoft 365 integration temporarily unavailable",
                    tool_name=tool_name,
                    requires_auth=False
                )

            return await self.call_tool(tool_name, arguments, access_token, **kwargs)

        except MCPServerUnavailableError:
            return MCPToolResult(
                success=False,
                error="Microsoft 365 integration temporarily unavailable",
                tool_name=tool_name,
                requires_auth=False
            )
        except MCPAuthenticationError:
            return MCPToolResult(
                success=False,
                error="Microsoft 365 authentication required",
                tool_name=tool_name,
                requires_auth=True
            )
        except Exception as e:
            self.logger.error("Unexpected error in safe tool call", tool_name=tool_name, error=str(e))
            return MCPToolResult(
                success=False,
                error="An unexpected error occurred",
                tool_name=tool_name,
                requires_auth=False
            )

    def __del__(self):
        """Cleanup on garbage collection"""
        if self.client:
            # Note: This might not work in all async contexts
            # Better to explicitly call close()
            pass


# Global client instance for dependency injection
mcp_client = MCPMicrosoft365Client()


async def get_mcp_client() -> MCPMicrosoft365Client:
    """Dependency injection function for FastAPI"""
    return mcp_client


async def cleanup_mcp_client():
    """Cleanup function for application shutdown"""
    await mcp_client.close()