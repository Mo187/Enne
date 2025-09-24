"""
Microsoft 365 Integration API endpoints for OAuth authentication and management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Dict, Any, Optional
from pydantic import BaseModel
import secrets
import structlog
import urllib.parse
from datetime import datetime, timezone, timedelta

from ...core.database import get_db
from ...models.user import User
from ...models.integration import Integration
from ..dependencies import get_current_active_user
from ...integrations.mcp_client import get_mcp_client, MCPMicrosoft365Client
from ...integrations.mcp_tool_adapter import MCPToolRegistrar
from ...services.tool_interface import tool_registry
from ...core.config import settings

logger = structlog.get_logger()

router = APIRouter(tags=["Integrations"])


# Pydantic models for request/response
class AuthURLResponse(BaseModel):
    auth_url: str
    state: str


class CallbackRequest(BaseModel):
    code: str
    state: str


class IntegrationStatus(BaseModel):
    connected: bool
    service: str
    connected_at: Optional[datetime] = None
    last_sync: Optional[datetime] = None
    error: Optional[str] = None
    available_tools: Optional[list] = None


class IntegrationsListResponse(BaseModel):
    integrations: list[IntegrationStatus]


# Temporary storage for OAuth state validation
# In production, this should be stored in Redis or database
oauth_states: Dict[str, Dict[str, Any]] = {}


def cleanup_expired_states():
    """Clean up expired OAuth states (older than 30 minutes)"""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
    expired_states = [
        state for state, data in oauth_states.items()
        if data.get("created_at", datetime.min.replace(tzinfo=timezone.utc)) < cutoff
    ]
    for state in expired_states:
        oauth_states.pop(state, None)


@router.get("/")
async def list_integrations(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get list of all integrations for current user"""

    query = select(Integration).where(Integration.user_id == current_user.id)
    result = await db.execute(query)
    integrations = result.scalars().all()

    integration_statuses = []
    for integration in integrations:
        status_info = IntegrationStatus(
            connected=integration.is_active and not integration.is_token_expired,
            service=integration.service_type,
            connected_at=integration.connected_at,
            last_sync=integration.last_sync_at,
            error=integration.sync_error_message if integration.sync_status == "error" else None
        )
        integration_statuses.append(status_info)

    return IntegrationsListResponse(integrations=integration_statuses)


@router.get("/ms365/auth")
async def start_ms365_auth(
    redirect_uri: Optional[str] = Query(None, description="Custom redirect URI"),
    current_user: User = Depends(get_current_active_user),
    mcp_client: MCPMicrosoft365Client = Depends(get_mcp_client)
):
    """
    Initiate Microsoft 365 OAuth authentication flow.

    Returns the authorization URL that the user should visit to grant permissions.
    """

    try:
        # Clean up expired states
        cleanup_expired_states()

        # Get authorization URL from MCP server
        auth_url = await mcp_client.get_auth_url(
            user_id=f"crm_user_{current_user.id}",
            redirect_uri=redirect_uri or settings.microsoft_redirect_uri
        )

        # Extract state parameter from the MCP auth URL
        parsed_url = urllib.parse.urlparse(auth_url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        mcp_state = query_params.get('state', [None])[0]

        if not mcp_state:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="MCP server did not provide state parameter"
            )

        # Store MCP state with user info for validation
        oauth_states[mcp_state] = {
            "user_id": current_user.id,
            "created_at": datetime.now(timezone.utc),
            "redirect_uri": redirect_uri
        }

        logger.info(
            "Microsoft 365 auth initiated",
            user_id=current_user.id,
            state=mcp_state
        )

        return AuthURLResponse(auth_url=auth_url, state=mcp_state)

    except Exception as e:
        logger.error(
            "Failed to initiate Microsoft 365 auth",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate authentication: {str(e)}"
        )


@router.post("/ms365/callback")
async def ms365_oauth_callback(
    callback_data: CallbackRequest,
    db: AsyncSession = Depends(get_db),
    mcp_client: MCPMicrosoft365Client = Depends(get_mcp_client)
):
    """
    Handle Microsoft 365 OAuth callback with authorization code.

    Exchanges the authorization code for access and refresh tokens,
    then stores them securely in the database.
    """

    try:
        # Validate state parameter
        state_data = oauth_states.get(callback_data.state)
        if not state_data:
            logger.warning("Invalid OAuth state", state=callback_data.state)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired authentication state"
            )

        user_id = state_data["user_id"]

        # Additional validation: ensure MCP state format is correct
        if callback_data.state.startswith("crm_user_"):
            try:
                user_id_from_state = int(callback_data.state.replace("crm_user_", ""))
                # Verify the user ID from state matches our stored user ID
                if user_id != user_id_from_state:
                    logger.warning(
                        "State user ID mismatch",
                        stored_user_id=user_id,
                        state_user_id=user_id_from_state,
                        state=callback_data.state
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="State validation failed"
                    )
            except ValueError:
                logger.warning("Invalid MCP state format", state=callback_data.state)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid state format"
                )
        else:
            logger.warning("State missing MCP prefix", state=callback_data.state)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid state format"
            )

        # Remove used state
        oauth_states.pop(callback_data.state, None)

        # Exchange authorization code for tokens via MCP server
        token_data = await mcp_client.handle_auth_callback(
            code=callback_data.code,
            state=callback_data.state  # Use the original MCP state directly
        )

        if not token_data.get("access_token"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to obtain access token from Microsoft"
            )

        # Check if integration already exists
        query = select(Integration).where(
            and_(
                Integration.user_id == user_id,
                Integration.service_type == "microsoft365"
            )
        )
        result = await db.execute(query)
        integration = result.scalar_one_or_none()

        if integration:
            # Update existing integration
            integration.access_token = token_data["access_token"]
            integration.refresh_token = token_data.get("refresh_token")

            if token_data.get("expires_at"):
                integration.token_expires_at = datetime.fromisoformat(
                    token_data["expires_at"].replace("Z", "+00:00")
                )
            elif token_data.get("expires_in"):
                # expires_in is in seconds
                integration.token_expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=int(token_data["expires_in"])
                )

            integration.is_active = True
            integration.connected_at = datetime.now(timezone.utc)
            integration.sync_status = "success"
            integration.sync_error_message = None
            integration.updated_at = datetime.now(timezone.utc)

        else:
            # Create new integration
            expires_at = None
            if token_data.get("expires_at"):
                expires_at = datetime.fromisoformat(
                    token_data["expires_at"].replace("Z", "+00:00")
                )
            elif token_data.get("expires_in"):
                expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=int(token_data["expires_in"])
                )

            integration = Integration(
                user_id=user_id,
                service_type="microsoft365",
                service_name="Microsoft 365",
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                token_expires_at=expires_at,
                client_id=settings.microsoft_client_id,
                scopes=token_data.get("scope", "").split(" ") if token_data.get("scope") else [],
                is_active=True,
                sync_enabled=True,
                sync_calendars=True,
                sync_emails=True,
                sync_files=True,
                connected_at=datetime.now(timezone.utc),
                sync_status="success",
                created_at=datetime.now(timezone.utc)
            )

            db.add(integration)

        await db.commit()
        await db.refresh(integration)

        # Register MCP tools with the tool registry for immediate availability
        try:
            registrar = MCPToolRegistrar(mcp_client)
            registered_count = await registrar.register_all_tools(tool_registry)

            logger.info(
                "MCP tools registered successfully",
                user_id=user_id,
                registered_tools=registered_count
            )
        except Exception as e:
            logger.warning(
                "Failed to register MCP tools but authentication succeeded",
                user_id=user_id,
                error=str(e)
            )
            # Don't fail the entire operation if tool registration fails

        logger.info(
            "Microsoft 365 integration connected successfully",
            user_id=user_id,
            integration_id=integration.id
        )

        return {
            "success": True,
            "message": "Microsoft 365 account connected successfully",
            "integration": {
                "id": integration.id,
                "service": integration.service_type,
                "connected_at": integration.connected_at.isoformat(),
                "status": "connected"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Microsoft 365 OAuth callback failed",
            error=str(e),
            code=callback_data.code[:10] + "..." if callback_data.code else None
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete authentication: {str(e)}"
        )


@router.get("/ms365/callback")
async def ms365_oauth_callback_get(
    request: Request,
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    mcp_client: MCPMicrosoft365Client = Depends(get_mcp_client)
):
    """
    Handle Microsoft 365 OAuth callback via GET request (browser redirect).

    This endpoint handles the browser redirect from Microsoft after the user
    grants or denies permissions.
    """

    if error:
        logger.warning(
            "Microsoft 365 OAuth error",
            error=error,
            description=error_description
        )

        # Redirect to frontend with error
        frontend_url = f"{settings.frontend_url}/settings?integration=ms365&error={error}"
        return RedirectResponse(url=frontend_url)

    if not code or not state:
        logger.warning("Missing code or state in OAuth callback")
        frontend_url = f"{settings.frontend_url}/settings?integration=ms365&error=missing_parameters"
        return RedirectResponse(url=frontend_url)

    try:
        # Use the POST callback logic
        callback_data = CallbackRequest(code=code, state=state)
        result = await ms365_oauth_callback(callback_data, db, mcp_client)

        # Redirect to frontend with success
        frontend_url = f"{settings.frontend_url}/settings?integration=ms365&success=true"
        return RedirectResponse(url=frontend_url)

    except HTTPException as e:
        logger.error("OAuth callback failed", error=str(e.detail))
        frontend_url = f"{settings.frontend_url}/settings?integration=ms365&error=authentication_failed"
        return RedirectResponse(url=frontend_url)


@router.get("/ms365/status")
async def get_ms365_status(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    mcp_client: MCPMicrosoft365Client = Depends(get_mcp_client)
):
    """
    Get Microsoft 365 integration status and available tools.
    """

    try:
        # Get integration from database
        query = select(Integration).where(
            and_(
                Integration.user_id == current_user.id,
                Integration.service_type == "microsoft365"
            )
        )
        result = await db.execute(query)
        integration = result.scalar_one_or_none()

        if not integration:
            return IntegrationStatus(
                connected=False,
                service="microsoft365"
            )

        # Check if tokens are valid
        is_connected = (
            integration.is_active and
            integration.access_token and
            not integration.is_token_expired
        )

        status_info = IntegrationStatus(
            connected=is_connected,
            service=integration.service_type,
            connected_at=integration.connected_at,
            last_sync=integration.last_sync_at,
            error=integration.sync_error_message if integration.sync_status == "error" else None
        )

        # If connected, get available tools from MCP server
        if is_connected:
            try:
                tools = await mcp_client.get_available_tools()
                status_info.available_tools = [
                    {
                        "name": tool.get("name"),
                        "description": tool.get("description")
                    }
                    for tool in tools
                    if tool.get("name", "").startswith(("outlook_", "sharepoint_", "onedrive_", "teams_"))
                ]
            except Exception as e:
                logger.warning("Failed to get available tools", error=str(e))
                status_info.error = "Failed to check available tools"

        return status_info

    except Exception as e:
        logger.error(
            "Failed to get Microsoft 365 status",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check integration status"
        )


@router.delete("/ms365")
async def disconnect_ms365(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Disconnect Microsoft 365 integration.

    This removes the stored tokens and deactivates the integration.
    """

    try:
        # Find the integration
        query = select(Integration).where(
            and_(
                Integration.user_id == current_user.id,
                Integration.service_type == "microsoft365"
            )
        )
        result = await db.execute(query)
        integration = result.scalar_one_or_none()

        if not integration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Microsoft 365 integration not found"
            )

        # Deactivate and clear tokens
        integration.is_active = False
        integration.access_token = None
        integration.refresh_token = None
        integration.token_expires_at = None
        integration.sync_status = "disconnected"
        integration.updated_at = datetime.now(timezone.utc)

        await db.commit()

        logger.info(
            "Microsoft 365 integration disconnected",
            user_id=current_user.id,
            integration_id=integration.id
        )

        return {
            "success": True,
            "message": "Microsoft 365 account disconnected successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to disconnect Microsoft 365",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disconnect integration"
        )


@router.post("/ms365/test")
async def test_ms365_connection(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    mcp_client: MCPMicrosoft365Client = Depends(get_mcp_client)
):
    """
    Test Microsoft 365 integration by making a simple API call.
    """

    try:
        # Get integration
        query = select(Integration).where(
            and_(
                Integration.user_id == current_user.id,
                Integration.service_type == "microsoft365",
                Integration.is_active == True
            )
        )
        result = await db.execute(query)
        integration = result.scalar_one_or_none()

        if not integration or not integration.access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Microsoft 365 not connected"
            )

        # Test connection by listing email folders
        test_result = await mcp_client.call_tool_safe(
            tool_name="outlook_list_folders",
            arguments={"user_id": f"crm_user_{current_user.id}"},
            access_token=integration.access_token,
            refresh_token=integration.refresh_token,
            expires_at=integration.token_expires_at.isoformat() if integration.token_expires_at else None,
            user_id=str(current_user.id)
        )

        if test_result.success:
            # Additional validation: check if the response contains actual data
            # This helps detect false positives where HTTP succeeds but tool fails
            response_data = test_result.data or {}
            tool_success = response_data.get("success", False) if isinstance(response_data, dict) else True

            if tool_success:
                # Update sync status
                integration.sync_status = "success"
                integration.last_sync_at = datetime.now(timezone.utc)
                integration.sync_error_message = None
                await db.commit()

                # Extract folder count from response for validation
                folders_count = 0
                if isinstance(response_data, dict) and "folders" in response_data:
                    folders_count = len(response_data.get("folders", []))
                elif isinstance(test_result.data, list):
                    folders_count = len(test_result.data)

                return {
                    "success": True,
                    "message": "Microsoft 365 connection is working",
                    "test_result": {
                        "folders_found": folders_count,
                        "tool_response": response_data
                    }
                }
            else:
                # Tool execution failed even though HTTP succeeded
                error_msg = response_data.get("error", "Tool execution failed") if isinstance(response_data, dict) else "Unknown tool error"
                integration.sync_status = "error"
                integration.sync_error_message = f"Tool execution failed: {error_msg}"
                await db.commit()

                return {
                    "success": False,
                    "message": "Microsoft 365 connection test failed",
                    "error": error_msg
                }
        else:
            # Update sync status with error
            integration.sync_status = "error"
            integration.sync_error_message = test_result.error
            await db.commit()

            return {
                "success": False,
                "message": "Microsoft 365 connection test failed",
                "error": test_result.error
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Microsoft 365 connection test failed",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test connection"
        )