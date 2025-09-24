# MCP (Model Context Protocol) Integration Plan for CRM

## Executive Summary

This document outlines the comprehensive plan to integrate Microsoft 365 MCP tools (email, calendar, SharePoint, OneDrive) with our AI-powered CRM system. The user has already built and configured a functional MS 365 MCP server registered on Azure with all necessary permissions.

## Current State Analysis

### 1. Existing CRM Architecture

#### Tool Execution Flow
```
User Message → LLM Command Parser → API Action → execute_api_action() → Database Operations → AI Response
```

#### Key Components
- **AI Service** (`/backend/app/services/ai_service.py`): Multi-provider AI support (Claude, OpenAI, Gemini)
- **Chat API** (`/backend/app/api/v1/chat.py`): Contains hardcoded `execute_api_action` for database operations
- **LLM Command Parser** (`/backend/app/services/llm_command_parser.py`): NLP to structured actions
- **Tool Interface** (`/backend/app/services/tool_interface.py`): MCP-ready architecture with `BaseTool`, `ToolRegistry`, `MCPToolAdapter`
- **Database Tools** (`/backend/app/services/database_tools.py`): Implements tool interface but NOT connected to chat flow

### 2. MCP Foundation Already Created

We have established an MCP-ready foundation with:

```python
# tool_interface.py structure
- ToolType (enum): DATABASE, CALENDAR, EMAIL, STORAGE, COMMUNICATION, EXPORT
- ToolResult (dataclass): Standardized result format
- ToolSchema (dataclass): MCP-compatible schema definition
- BaseTool (ABC): Abstract interface for all tools
- ToolRegistry: Manages tool registration and execution
- MCPToolAdapter: Placeholder for MCP integration (NOT implemented)
```

**Critical Gap**: This foundation exists but is NOT connected to the main chat flow.

### 3. External MS 365 MCP Server

User has already built:
- Fully configured MS 365 MCP server
- Registered on Azure with all permissions
- Delegated Graph API access
- Supports: Outlook email, calendar, SharePoint, OneDrive
- Works with MCP-compatible LLMs

## MCP Protocol Understanding

### What is MCP?

Model Context Protocol (MCP) is a standardized protocol enabling LLMs to securely access external data and functionality through:
- **Resources**: Read-only data endpoints
- **Tools**: Executable functions with side effects
- **Prompts**: Interaction templates

### Communication Flow
```
LLM → MCP Client → Transport (HTTP/WebSocket) → MCP Server → External Service (MS 365)
```

### Transport Options (2025)
- **Streamable HTTP**: Recommended for production (bi-directional, single endpoint)
- **SSE**: Legacy, deprecated as of March 2025
- **WebSocket**: Real-time bidirectional
- **Stdio**: Local development

## Integration Architecture Decision

### Option Analysis

#### Option A: External MCP Server Integration (RECOMMENDED)
**Use the existing MS 365 MCP server**

```
FastAPI CRM ↔ HTTP/MCP ↔ External MS 365 MCP Server ↔ Microsoft Graph API
```

**Pros:**
- ✅ Leverage existing tested server
- ✅ No duplicate Azure configuration
- ✅ Separation of concerns
- ✅ Independent scaling
- ✅ Reusable across applications

**Cons:**
- ❌ Network latency (10-50ms)
- ❌ Additional service to maintain

#### Option B: Internal Implementation
**Rebuild MS 365 tools in CRM**

**Pros:**
- ✅ Single codebase
- ✅ No network overhead

**Cons:**
- ❌ Duplicate work
- ❌ Reconfigure Azure
- ❌ Lose existing investment

**Decision: Use Option A - External MCP Server**

## Implementation Plan

### Phase 1: MCP Client Infrastructure

#### 1.1 Dependencies
```txt
# requirements.txt additions
pydantic-ai[mcp]==0.1.0
mcp==0.1.0
fastmcp==0.2.0
httpx==0.24.1
```

#### 1.2 MCP Manager Service
Create `/backend/app/services/mcp_manager.py`:

```python
from typing import Dict, Optional
from pydantic_ai.mcp import MCPServerStreamableHTTP
from ..models.user import User
from ..core.config import settings
import structlog

logger = structlog.get_logger()

class MCPManager:
    """Manages MCP server connections for users"""

    def __init__(self):
        self.ms365_servers: Dict[int, MCPServerStreamableHTTP] = {}
        self.ms365_server_url = settings.MS365_MCP_SERVER_URL

    async def get_user_ms365_server(self, user: User) -> Optional[MCPServerStreamableHTTP]:
        """Get or create MS 365 MCP server connection for user"""
        try:
            # Get user's MS 365 integration
            integration = await self.get_user_integration(user.id, "ms365")
            if not integration or not integration.access_token:
                return None

            # Create or reuse connection
            if user.id not in self.ms365_servers:
                headers = {
                    "Authorization": f"Bearer {integration.access_token}",
                    "X-User-ID": str(user.id)
                }

                self.ms365_servers[user.id] = MCPServerStreamableHTTP(
                    self.ms365_server_url,
                    http_client_kwargs={"headers": headers}
                )

            return self.ms365_servers[user.id]

        except Exception as e:
            logger.error("Failed to get MS365 MCP server", error=str(e))
            return None

mcp_manager = MCPManager()
```

### Phase 2: Connect Tool Systems

#### 2.1 Enhance execute_api_action

Modify `/backend/app/api/v1/chat.py`:

```python
async def execute_api_action(action, user, db):
    method = action.get("method")

    # Check if it's a database action (existing logic)
    if method in ["GET", "POST", "PUT", "DELETE"]:
        # [Keep existing database operation code]

    # NEW: Check if it's an MCP tool call
    elif method == "MCP_TOOL":
        tool_name = action.get("tool_name")
        parameters = action.get("parameters", {})

        # Try MS 365 tools
        ms365_server = await mcp_manager.get_user_ms365_server(user)
        if ms365_server:
            async with ms365_server:
                tools = await ms365_server.list_tools()
                if any(t.name == tool_name for t in tools):
                    result = await ms365_server.call_tool(tool_name, parameters)
                    return {
                        "success": True,
                        "result": result,
                        "message": f"Executed MS 365 tool: {tool_name}"
                    }

        # Fallback to internal tools
        return await tool_registry.execute_tool(tool_name, parameters, {"user": user})
```

### Phase 3: LLM Command Parser Enhancement

Add MS 365 intents to `/backend/app/services/llm_command_parser.py`:

```python
# New intents
self.available_intents.extend([
    "send_email",
    "read_emails",
    "check_calendar",
    "create_calendar_event",
    "list_sharepoint_files",
    "read_onedrive_file"
])

# In generate_api_action()
elif intent == "send_email":
    return {
        "method": "MCP_TOOL",
        "tool_name": "send_email",
        "parameters": {
            "to": entities.get("recipient"),
            "subject": entities.get("subject"),
            "body": entities.get("body")
        },
        "description": "Send email via Outlook"
    }
```

### Phase 4: User Integration Management

#### 4.1 Integration Model

Create `/backend/app/models/integration.py`:

```python
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base

class Integration(Base):
    __tablename__ = "integrations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    service = Column(String(50), nullable=False)  # "ms365"
    access_token = Column(Text)
    refresh_token = Column(Text)
    token_expires_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="integrations")
```

#### 4.2 OAuth Flow

Implement MS 365 OAuth in `/backend/app/api/v1/integrations.py`:

```python
@router.get("/ms365/auth")
async def start_ms365_auth(current_user: User = Depends(get_current_active_user)):
    """Initiate MS 365 OAuth flow"""
    auth_url = f"https://login.microsoftonline.com/{settings.MS365_TENANT_ID}/oauth2/v2.0/authorize"
    params = {
        "client_id": settings.MS365_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": settings.MS365_REDIRECT_URI,
        "scope": "Mail.ReadWrite Calendars.ReadWrite Files.ReadWrite",
        "state": generate_state_token(current_user.id)
    }
    return {"auth_url": f"{auth_url}?{urlencode(params)}"}

@router.post("/ms365/callback")
async def ms365_oauth_callback(code: str, state: str, current_user: User):
    """Handle MS 365 OAuth callback"""
    # Exchange code for tokens
    # Store in Integration model
    # Return success
```

### Phase 5: Configuration

#### 5.1 Environment Variables

Add to `.env`:
```env
# MS 365 MCP Configuration
MS365_MCP_SERVER_URL=http://localhost:8001/mcp
MS365_CLIENT_ID=your_azure_app_id
MS365_CLIENT_SECRET=your_secret
MS365_TENANT_ID=your_tenant_id
MS365_REDIRECT_URI=http://localhost:8000/api/v1/integrations/ms365/callback
```

#### 5.2 Config Updates

Add to `/backend/app/core/config.py`:
```python
# MCP Settings
MS365_MCP_SERVER_URL: str = os.getenv("MS365_MCP_SERVER_URL")
MS365_CLIENT_ID: str = os.getenv("MS365_CLIENT_ID")
MS365_CLIENT_SECRET: str = os.getenv("MS365_CLIENT_SECRET")
MS365_TENANT_ID: str = os.getenv("MS365_TENANT_ID")
MS365_REDIRECT_URI: str = os.getenv("MS365_REDIRECT_URI")
```

### Phase 6: Frontend Integration

#### 6.1 Settings Page

Add to `/backend/app/templates/pages/settings.html`:

```html
<div class="tab-pane" id="integrations">
    <h3>Microsoft 365 Integration</h3>
    <div id="ms365-status">
        <!-- Dynamic status display -->
    </div>
    <button onclick="connectMS365()" class="btn btn-primary">
        Connect Microsoft 365
    </button>

    <div id="ms365-tools" class="hidden">
        <h4>Available Tools:</h4>
        <ul id="tools-list"></ul>
    </div>
</div>

<script>
async function connectMS365() {
    const response = await fetch('/api/v1/integrations/ms365/auth', {
        headers: {'Authorization': 'Bearer ' + localStorage.getItem('access_token')}
    });
    const data = await response.json();
    window.location.href = data.auth_url;
}
</script>
```

### Phase 7: Testing & Monitoring

#### 7.1 Health Check Endpoint

Add to `/backend/app/api/v1/chat.py`:

```python
@router.get("/mcp/status")
async def check_mcp_status(current_user: User = Depends(get_current_active_user)):
    """Check MCP server connections"""
    ms365_server = await mcp_manager.get_user_ms365_server(current_user)

    if ms365_server:
        try:
            async with ms365_server:
                tools = await ms365_server.list_tools()
                return {
                    "ms365": {
                        "connected": True,
                        "tools_count": len(tools),
                        "tools": [t.name for t in tools]
                    }
                }
        except Exception as e:
            return {"ms365": {"connected": False, "error": str(e)}}

    return {"ms365": {"connected": False}}
```

## Testing Scenarios

1. **Email Operations**
   - "Send an email to john@company.com about our meeting"
   - "Show me emails from the last 3 days"
   - "Find emails about the quarterly report"

2. **Calendar Operations**
   - "What's on my calendar tomorrow?"
   - "Schedule a meeting with the team next Monday at 2pm"
   - "Show me this week's meetings"

3. **File Operations**
   - "Find the Q4 report in SharePoint"
   - "List my recent OneDrive files"
   - "Share the budget spreadsheet with the finance team"

## Security Considerations

1. **Token Management**
   - Secure storage of OAuth tokens (encrypted in database)
   - Automatic token refresh before expiration
   - User-specific token isolation

2. **Audit Logging**
   - Log all MCP tool usage
   - Track user actions for compliance
   - Monitor for unusual activity

3. **Error Handling**
   - Graceful degradation if MCP server unavailable
   - Clear error messages to users
   - Automatic retry with exponential backoff

## Performance Optimization

1. **Connection Pooling**
   - Reuse MCP server connections
   - Implement connection timeout
   - Health checks every 60 seconds

2. **Tool Caching**
   - Cache tool schemas for 5 minutes
   - Invalidate on error
   - Background refresh

3. **Async Operations**
   - All MCP calls are async
   - Parallel tool execution where possible
   - Non-blocking UI updates

## Migration Path

### Week 1: Foundation
- Set up MCP client infrastructure
- Implement basic connection management
- Test with simple tools

### Week 2: Integration
- Connect to MS 365 MCP server
- Implement OAuth flow
- Add user settings UI

### Week 3: Enhancement
- Add all MS 365 tools
- Implement error handling
- Add monitoring

### Week 4: Production
- Performance optimization
- Security hardening
- User documentation

## Success Metrics

1. **Functional Metrics**
   - 100% of MS 365 tools accessible
   - < 100ms additional latency for MCP calls
   - 99.9% availability

2. **User Metrics**
   - 80% of users connect MS 365 within first week
   - 90% satisfaction with integration
   - 50% reduction in manual data entry

3. **Technical Metrics**
   - Zero security incidents
   - < 1% error rate on MCP calls
   - Automatic token refresh success rate > 99%

## Rollback Plan

If issues arise:
1. Feature flag to disable MCP integration
2. Revert to database-only operations
3. Preserve user data and settings
4. Gradual re-enablement after fixes

## Next Steps

1. Review and approve this plan
2. Analyze MS 365 MCP server codebase
3. Begin Phase 1 implementation
4. Set up development environment
5. Create integration tests

---

*Document Version: 1.0*
*Last Updated: September 23, 2025*
*Status: Ready for Implementation*