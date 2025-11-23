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

## Implementation Status - IN PRODUCTION WITH FIXES ⚠️

**Status**: **FUNCTIONAL WITH RECENT CRITICAL FIXES**
**Initial Implementation**: September 24, 2025
**Major Fixes Applied**: September 30, 2025

### What Was Actually Built

The Microsoft 365 MCP integration has been **successfully implemented** using a **stateless HTTP-based architecture**. Here's what was delivered:

#### ✅ **Core Architecture Implemented**
```
CRM Frontend → FastAPI Backend → HTTP Bridge → MCP Server → Microsoft Graph API
```

### Critical Issues Discovered & Fixed (September 30, 2025)

After initial implementation, extensive testing revealed **fundamental architectural bugs** requiring complete rewrites of core components:

#### 🐛 **Issue 1: Multiple `$filter` Parameters Breaking Email Search**
**Problem**: Email search by sender returned ALL inbox emails instead of filtered results.

**Root Cause**:
```python
# BROKEN CODE: Multiple $filter params added separately
search_params.append(f"$filter=from/emailAddress/address eq 'user@email.com'")  # Line 300
search_params.append(f"$filter=receivedDateTime ge ...")  # Line 314 - OVERWRITES!
# Result: Microsoft Graph only processes LAST $filter, ignoring sender filter
```

**Solution**: Complete rewrite of filter management (`outlook.py:267-312`)
```python
# NEW: Single filter_conditions list, combined into ONE $filter
filter_conditions = []
if "@" in query:
    filter_conditions.append(f"from/emailAddress/address eq '{query}'")
if "from_date" in args:
    filter_conditions.append(f"receivedDateTime ge ...")
if unread_only:
    filter_conditions.append("isRead eq false")

# Combine into SINGLE $filter parameter
combined_filter = " and ".join(filter_conditions)
search_params.append(f"$filter={combined_filter}")
```

**Impact**: Email filtering now works correctly. "momoagoumar@gmail.com" returns ONLY emails from that address.

#### 🐛 **Issue 2: Microsoft Graph API Constraints**
**Problems**:
- `$search` + `$orderby` = 400 error "SearchWithOrderBy not supported"
- `contains()` + `$orderby` = 400 error "InefficientFilter: too complex"

**Solutions**:
```python
# Skip $orderby when using incompatible filters
using_search = any("$search" in param for param in search_params)
using_contains = any("contains(" in param for param in search_params)
if not using_search and not using_contains:
    search_params.append("$orderby=receivedDateTime desc")
```

**Added Debug Logging**:
```python
logger.info(f"Microsoft Graph API URL: {url}")
logger.info(f"Combined filter: {combined_filter}")
logger.info(f"API returned {len(emails)} emails")
```

#### 🐛 **Issue 3: AI Hallucinating Email Content**
**Problem**: AI fabricated complete email body content, making up subjects, dates, and details not in actual emails.

**Root Causes**:
1. **Data Boundaries Unclear**: AI received previews (~200 chars) but claimed to have full content
2. **HTML Emails**: Body content with HTML tags confused the AI
3. **Weak Warnings**: System prompts didn't strongly prevent hallucination

**Solutions**:

**A. Explicit Data Type Labels** (`chat.py:1211`)
```python
# BEFORE: "Preview: [text]"
# AFTER: "Body Preview (TRUNCATED - ~200 chars only): [text]"
```

**B. Strong Context Warnings** (`chat.py:1230-1236`)
```python
CRITICAL CONTEXT FOR EMAIL LISTS:
- You are seeing {len(result_data)} email(s) with PREVIEW TEXT ONLY (~200 characters)
- You DO NOT have full email body content for these search results
- NEVER claim to know full email content when you only have previews
- NEVER make up or extrapolate content beyond the preview shown
```

**C. HTML Stripping** (`outlook.py:437-460`)
```python
def _strip_html(self, html_content: str) -> str:
    text = re.sub(r'<[^<]+?>', '', html_content)
    text = unescape(text)
    return text.strip()
```

**D. Extreme Anti-Hallucination Boundaries** (`chat.py:1264-1295`)
```python
==== EMAIL DATA START ====
{json structure}

==== BODY CONTENT START ====
{actual body or [EMPTY - THIS EMAIL HAS NO BODY TEXT]}
==== BODY CONTENT END ====

==== EMAIL DATA END ====

CRITICAL ANTI-HALLUCINATION INSTRUCTIONS:
1. Text between markers is COMPLETE content
2. If [EMPTY], there is NO body - do not make up content
3. DO NOT add information not present
4. If you add ANY info not in body, you are FAILING
```

**E. System Prompt Clarity** (`ai_service.py:325-342`)
```
Two types of email data you will see:

1. EMAIL SEARCH RESULTS (lists):
   - Body Preview (~200 chars ONLY)
   - YOU DO NOT HAVE FULL BODY CONTENT
   - NEVER extrapolate beyond preview

2. SINGLE EMAIL RETRIEVAL (individual):
   - FULL BODY CONTENT (complete text)
   - You HAVE complete content
```

#### 🐛 **Issue 4: Context Loss in Follow-Up Queries**
**Problem**: User asks "What's my recent email?" (AI shows it), then "What's the content?" (AI tries to execute tool again → auth error).

**Root Cause**: Command parser processed each message independently without conversation history.

**Solution**: Follow-Up Detection System (`llm_command_parser.py:1177-1222`)
```python
def _is_follow_up_question(self, text: str, conversation_history: List[Dict]) -> bool:
    """Detect follow-up questions about previous data."""
    follow_up_indicators = [
        "that email", "the email", "tell me more", "what does it say",
        "show me the content", "the body", "about that", "about it"
    ]

    has_indicator = any(indicator in text.lower() for indicator in follow_up_indicators)

    # Check if conversation history mentions emails
    if has_indicator and conversation_history:
        recent_messages = conversation_history[-4:]
        for msg in recent_messages:
            if any(ind in msg.get("content", "").lower()
                   for ind in ["email", "subject:", "from:", "inbox"]):
                return True
    return False

# In parse_command():
if conversation_history and self._is_follow_up_question(text, conversation_history):
    return {
        "intent": "answer_from_context",
        "entities": {"question": text},
        "confidence": 0.95
    }
```

**Chat Handler** (`chat.py:1161-1169`)
```python
if parsed_command.get("intent") == "answer_from_context":
    user_message = f"""User asked: "{chat_data.message}"

IMPORTANT: This is a follow-up question about data from the previous conversation.
Look at the conversation history above to find the relevant information.
The data was already retrieved - DO NOT say you need to retrieve it again.
"""
```

**Impact**: Natural conversations work. No redundant tool calls. No auth errors.

#### 🛡️ **Issue 5: Catastrophic Error Handling**
**Problem**: When tools failed, AI was prompted to "suggest how to fix it" → AI fabricated fake data as "workaround".

**Solution**: Strict Error Handling (`chat.py:1290-1315`)
```python
# BEFORE: "Please help me understand what went wrong and suggest how to fix it"
# ↑ ENCOURAGED HALLUCINATION!

# AFTER:
CRITICAL ERROR - TOOL EXECUTION FAILED:
Error: {error_message}

STRICT INSTRUCTIONS:
- DO NOT make up fake data
- DO NOT fabricate email content
- DO NOT pretend operation succeeded
- Be apologetic but HONEST

ABSOLUTELY FORBIDDEN:
- Inventing fake data
- Making up email subjects, senders, or content
```

#### 🔒 **Defensive Measures Added**

**Client-Side Filtering Fallback** (`outlook.py:362-371`)
```python
# Verify Microsoft Graph API actually filtered correctly
if query and "@" in query:
    original_count = len(emails)
    emails = [email for email in emails
              if email.get("from", {}).get("emailAddress", {}).get("address", "").lower() == query.lower()]
    if len(emails) != original_count:
        logger.warning(f"Client-side filtering: {original_count} -> {len(emails)} emails")
```

**Impact**: Guarantees correct results even if Graph API filtering fails.

### Current Approach & Philosophy

#### **Stateless HTTP Architecture**
- Tokens passed per-request (no server-side session state)
- User context: `crm_user_{id}` format for consistency
- MCP Server on `localhost:8001` with HTTP bridge

#### **Defense-in-Depth for Data Integrity**
1. **Server-side filtering** (Microsoft Graph API)
2. **Client-side verification** (fallback if API fails)
3. **Explicit data labeling** (TRUNCATED vs FULL CONTENT)
4. **Strong boundary markers** (`==== START ====` / `==== END ====`)
5. **Multiple layers of anti-hallucination prompts**

#### **Conversation Context Management**
- Parser receives `conversation_history`
- Follow-up detection prevents redundant tool calls
- AI maintains awareness of previous queries
- Natural multi-turn conversations supported

#### **Error Philosophy**
- **Fail loudly**: Errors reported honestly, never masked
- **No fabrication**: AI never makes up data to "work around" errors
- **User trust**: Better to admit failure than provide false information

#### ✅ **Key Components Built**

**1. Database Layer** (`/backend/app/models/integration.py`)
- Complete Integration model with token storage
- User relationships and connection status tracking
- Token expiration and refresh logic

**2. OAuth Integration** (`/backend/app/api/v1/integrations.py`)
- Full Microsoft 365 OAuth 2.0 flow
- Token exchange and storage
- Connection status API endpoints
- Automatic tool registration after authentication

**3. MCP Client** (`/backend/app/integrations/mcp_client.py`)
- HTTP-based MCP communication
- Stateless token passing
- Tool execution with authentication
- Error handling and retry logic

**4. Tool Integration** (`/backend/app/integrations/mcp_tool_adapter.py`)
- MCP tool registration system
- Parameter conversion and validation
- User context management

**5. AI Assistant Integration** (`/backend/app/services/ai_service.py`)
- Dynamic capability reporting based on connected integrations
- Context-aware system prompts
- Microsoft 365 feature awareness

**6. Settings UI** (`/backend/app/templates/pages/settings.html`)
- OAuth connection flow
- Real-time connection status
- Integration management interface

#### ✅ **Critical Fixes Applied**

**1. Stateless Token Architecture**
- **Problem**: MCP tools were looking for stored tokens instead of using passed tokens
- **Solution**: Modified `src/tools/outlook.py` and `src/tools/sharepoint.py` to check stateless tokens first
- **Result**: `stateless mode: True` in logs, tools work with passed authentication

**2. OAuth Token Return**
- **Problem**: MCP server returned success message but not actual tokens
- **Solution**: Modified `http_bridge.py` OAuth callback to return complete token data
- **Result**: CRM can store and use real Microsoft tokens

**3. User ID Format Consistency**
- **Problem**: Components used different user ID formats (`"1"` vs `"crm_user_1"`)
- **Solution**: Standardized all components to use `f"crm_user_{user_id}"` format
- **Result**: Token lookup works correctly across all components

**4. Search Query Handling**
- **Problem**: Empty search queries were passed as `"None"` to Microsoft Graph API
- **Solution**: Added proper None/empty validation in email search
- **Result**: Generic email retrieval works when no specific search term provided

#### ✅ **Features Working**

**Authentication:**
- ✅ Microsoft 365 OAuth 2.0 flow
- ✅ Token storage and refresh
- ✅ Connection status display in Settings
- ✅ Automatic disconnection handling

**AI Assistant Capabilities:**
- ✅ Dynamic feature detection ("I can search your emails...")
- ✅ Natural language email searching
- ✅ Calendar event retrieval
- ✅ SharePoint and OneDrive file access
- ✅ Teams integration

**Microsoft 365 Tools Available:**
- ✅ `outlook_search_emails` - Search email messages
- ✅ `outlook_list_folders` - List mail folders
- ✅ `outlook_get_calendar_events` - Get calendar events
- ✅ `sharepoint_list_sites` - List SharePoint sites
- ✅ `onedrive_list_files` - List OneDrive files
- ✅ `teams_list_teams` - List Teams

#### ✅ **End-to-End Flow Working**

1. **Settings → Connect Microsoft 365** → OAuth flow → Tokens stored ✅
2. **Settings shows "Connected" status** → Real connection validation ✅
3. **AI Assistant gains capabilities** → Context-aware prompts ✅
4. **Natural language commands work** → "search my emails" → Tool execution ✅
5. **Stateless authentication** → Tokens passed per request ✅

### Current Architecture Details

#### **HTTP Bridge Pattern**
- **MCP Server**: Runs on `localhost:8001` with HTTP bridge
- **Communication**: REST API instead of stdio transport
- **Token Passing**: Stateless - tokens included in each request
- **User Context**: `crm_user_{id}` format for consistency

#### **Database Schema**
```sql
integrations (
    id, user_id, service_type, service_name,
    access_token, refresh_token, token_expires_at,
    is_active, connected_at, sync_status,
    sync_calendars, sync_emails, sync_files
)
```

#### **API Endpoints**
- `GET /api/v1/integrations/ms365/auth` - Start OAuth flow
- `GET/POST /api/v1/integrations/ms365/callback` - Handle OAuth callback
- `GET /api/v1/integrations/ms365/status` - Check connection status
- `POST /api/v1/integrations/ms365/test` - Test connection
- `DELETE /api/v1/integrations/ms365` - Disconnect integration

### Production Deployment Notes

#### **Environment Variables Required**
```env
# Microsoft 365 OAuth
MICROSOFT_CLIENT_ID=your_azure_app_id
MICROSOFT_CLIENT_SECRET=your_azure_secret
MICROSOFT_TENANT_ID=your_tenant_id
MICROSOFT_REDIRECT_URI=http://localhost:8000/api/v1/integrations/ms365/callback

# MCP Server
MCP_MICROSOFT365_URL=http://host.docker.internal:8001
MCP_TIMEOUT_SECONDS=30
```

#### **Services Required**
1. **CRM Backend** (FastAPI) - Port 8000
2. **MCP Server** (HTTP Bridge) - Port 8001
3. **PostgreSQL Database** - For token storage
4. **Redis** (optional) - For caching

#### **Security Considerations**
- ✅ Tokens encrypted in database
- ✅ HTTPS required in production
- ✅ State parameter validation
- ✅ Token expiration handling
- ✅ Scope limitation
- ✅ User isolation (tokens are user-specific)

### Future Enhancements

#### **Phase 1 - Advanced Features**
- Email sending and drafts
- Calendar event creation
- File upload to OneDrive/SharePoint
- Teams message posting

#### **Phase 2 - Optimization**
- Connection pooling
- Response caching
- Bulk operations
- Webhook notifications

#### **Phase 3 - Multi-Tenant**
- Organization-level integrations
- Admin management interface
- Usage analytics
- Rate limiting per org

---

## Current Status Summary

**Overall Status**: 🟡 **FUNCTIONAL WITH CAVEATS**

**What Works**:
- ✅ OAuth authentication flow
- ✅ Email search by sender (with fixes)
- ✅ Single email retrieval
- ✅ Follow-up conversation context
- ✅ Unread email filtering
- ✅ HTML content stripping
- ✅ Client-side filtering fallback
- ✅ Debug logging for troubleshooting

**Known Limitations**:
- ⚠️ AI may still occasionally hallucinate despite multiple safeguards
- ⚠️ Microsoft Graph API constraints require careful query construction
- ⚠️ No connection pooling yet (new connection per request)
- ⚠️ Limited to email operations (calendar/files need testing)

**Production Readiness**: 🔴 **NOT RECOMMENDED**
- Core functionality works but requires extensive user acceptance testing
- Anti-hallucination measures are defensive layers, not guarantees
- More real-world testing needed to validate reliability

**Next Steps**:
1. Extended user testing with diverse email scenarios
2. Implement connection pooling for performance
3. Add comprehensive error tracking and monitoring
4. Expand to calendar and file operations
5. Consider LLM fine-tuning to reduce hallucination tendency

---

*Document Version: 3.0*
*Last Updated: September 30, 2025*
*Status: **ALPHA - ACTIVE DEVELOPMENT**

## Testing Commands

Try these commands with the AI Assistant:

**Email Operations:**
- "Can you search my emails?"
- "Show me emails from last week"
- "Find emails about meetings"

**Calendar Operations:**
- "What's on my calendar today?"
- "Show me this week's meetings"

**File Operations:**
- "List my OneDrive files"
- "What SharePoint sites do I have access to?"