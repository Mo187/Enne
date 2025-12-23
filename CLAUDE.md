# AI-Assisted CRM Requirements

## Critical Development Rules - DO NOT VIOLATE
- ***DONT EVER SAY SOMETHING IS COMPLETED OR FULLY DEVELOPED WHEN IN FACT ITS NOT! WE NEED TO BE SURE ITS REALLY DONE.***
- **NEVER create mock data or simplified components** unless explicitly requested !
- **NEVER replace complex components with simplified versions** - fix the actual problem !
- **ALWAYS work with existing codebase** - no creating alternatives !
- **ALWAYS fix root causes** - no workarounds !
- **FOLLOW existing technical structure and coding conventions**

- **Claude Code is running in WSL on Windows 11.**

## Project Overview
Enterprise CRM with integrated AI chat assistant (primarily Claude API, with support for OpenAI, Gemini). Organizations can use their own API keys.

## Core Features

### 1. AI Assistant Chat
- **Primary interface**: Chat-based interaction for all CRUD operations
- **Natural language processing** for commands.
- **Context-aware** responses based on user data
- **Multi-LLM support** (Claude PRIMARY especially because of support for MCP integrations, OpenAI, Gemini secondary)

### 2. Data Models

#### Contacts
- Fields: `name`, `email`, `phone`, `job_position`, `organization`, `created_at`, `updated_at`
- CRUD via chat commands and through Frontend also.
- User-specific data (each user has their own contacts)
* The Assistant should be able to create contacts for you based on your conversation and if you ask them to. Adding the contact to your Contacts list alongside their information like : Job position, email, Phone number, Organisation/Role IF provided by the user, or store only whatever was provided by the user for now.

#### Organizations
- Fields: `name`, `description`, `industry`, `created_at`, `updated_at`
- CRUD via chat commands and through Frontend also.
- User-specific data

The app will allow each user to create organisations they follow, and also can be created by asking the AI Assistant to do it. 

#### Projects
- Fields: `name`, `status` (Planned/In Progress/Completed), `organization_id`, `assignees[]`, `due_date`, `created_at`, `updated_at`
- Has many Tasks (one-to-many relationship)
- CRUD via chat commands

The user should be able to create Projects through the frontend directly or through the AI chat. 
Projects will display: Name, Status (Planned, In progress, Completed etc.), Organisation, Assignees (their names; one or many), Due Date. 

Inside each project (clicking on them to view them) Should display more information about the Project and more importantly each tasks associated with that Project. The tasks will also have : Task name, Status, Priority, Assignee, Project, Due Date , Actions (Edit/Delete etc...). A button to add new tasks under that Project.

Projects should show dates including last updated date in case of updates.

#### Tasks
- Fields: `name`, `status`, `priority`, `assignee`, `project_id`, `due_date`, `created_at`, `updated_at`
- Belongs to Project
- CRUD via chat commands

A list of all your Tasks from all projects with their information and dates.

### 3. Calendar Integration
- **Sync with**: Outlook/Teams, Google Calendar
- **AI capabilities**: Create reminders (with notifications), events, send email notifications
- **Display**: Monthly/weekly/daily views
- **OAuth integration** required

* Display a calendar integrated with Outlook/Teams or Google Calendar for that user that show their activities and meetigns at a glance. Users will have to connect all their required accounts properly.

The user should be able to manipulate the Calendar with the AI. THE AI CHAT SHOULD BE ABLE TO ALSO CREATE REMINDERS, EVENTS THAT ARE TRACKED AND CAN SEND EMAIL NOTIFICATIONS.

### 4. MCP Integrations
Required integrations:
- Microsoft 365 (Outlook, OneDrive, SharePoint, Teams) - PRIMARY
- Google Workspace - Secondary
- Connection management in Settings → Integrations

## Technical Stack

### Backend
- **Language**: Python
- **Framework**: Flask with async support (or FastAPI)
- **Database**: PostgreSQL
- **Authentication**: JWT tokens
- **API Design**: RESTful + WebSocket for real-time chat

### Frontend
- **Layout**: 
  - Left sidebar navigation
  - Main content area (chat or data views)
  - Company logo (top-left)
  - User account menu (top-right)
- **Pages**:
  - `/assistant` - AI chat interface
  - `/contacts` - Contact management
  - `/organizations` - Organization management
  - `/projects` - Project list and details
  - `/tasks` - Task overview
  - `/calendar` - Calendar view
  - `/settings` - Profile and Integrations tabs
  - `/feedback` - User feedback form

### Security Requirements
- Secure API key storage (encrypted)
- Row-level security for multi-tenant data
- OAuth 2.0 for third-party integrations
- Rate limiting on API endpoints
- Input validation and sanitization

## Database Schema (PostgreSQL)

```sql
-- Core tables structure - INITIAL DESIGN
users (id, email, password_hash, name, created_at)
contacts (id, user_id, name, email, phone, job_position, organization, metadata)
organizations (id, user_id, name, description, industry, metadata)
projects (id, user_id, name, status, organization_id, due_date, metadata)
tasks (id, project_id, name, status, priority, assignee, due_date)
project_assignees (project_id, user_id)
integrations (id, user_id, service_type, credentials_encrypted, refresh_token)
```

## API Endpoints Structure

```
POST   /api/chat          - Process AI chat messages
GET    /api/contacts      - List user contacts
POST   /api/contacts      - Create contact
PUT    /api/contacts/{id} - Update contact
DELETE /api/contacts/{id} - Delete contact
[Similar CRUD for organizations, projects, tasks]
POST   /api/calendar/sync - Sync external calendars
POST   /api/integrations/connect - Connect external service
```

## Development Principles
- **DRY**: Create reusable components and services
- **KISS**: Simple, clear implementations
- **YAGNI**: Build only requested features
- **SOLID**: Separate concerns, dependency injection
- SHORT, CLEAN AND EFFICIENT CODE.

## Implementation Priority
1. Authentication system
2. Database models and migrations
3. Basic CRUD APIs
4. AI chat integration
5. Frontend UI
6. MCP integrations
7. Calendar integration

## Edge Cases to Consider
- API key expiration/rotation
- Concurrent user edits
- Large dataset pagination
- Chat context overflow
- Integration authentication failures
- Offline functionality requirements

---

## Implementation Status & Architecture Notes

### Conversation Memory (Database-Backed)
- **Models**: `Conversation`, `ConversationMessage`, `EntityTracking` in `backend/app/models/conversation.py`
- **Service**: `PersistentConversationMemory` in `backend/app/services/conversation_memory.py`
- **Features**: Cross-session persistence, entity tracking with positions, smart pruning, tiktoken token counting

### LLM Command Parser
- **File**: `backend/app/services/llm_command_parser.py`
- **Features**: Semantic intent detection, synonym support, `contact_name` entity for updates
- **Intents**: CRUD for contacts/orgs/projects/tasks + MS365 email/calendar/files
- **Prompt Version**: `v3.0_optimized` (~70 lines, 73% smaller than v2.x)

### MCP Integration (Microsoft 365)
- **Adapter**: `backend/app/integrations/mcp_tool_adapter.py`
- **Client**: `backend/app/integrations/mcp_client.py`
- **Server**: `365mcp/` (separate process)
- **Tools**: outlook_*, sharepoint_*, onedrive_*, teams_*

### Extended Thinking (Claude)
- **File**: `backend/app/services/ai_service.py`
- **Triggers**: Low confidence (<0.7), clarifications pending, long conversations (>20 msgs)
- **SDK**: Requires anthropic>=0.49.0

### Recent Fixes (Dec 2024)
| Fix | Description | File |
|-----|-------------|------|
| Follow-up entity updates | Structured offer tracking for "add more info" after entity creation | `chat.py`, `conversation_memory.py` |
| Pre-parse entity intercept | Intercepts affirmative + data responses to pending offers | `chat.py` |
| Field extraction helper | Extracts email/phone from user messages | `chat.py` |
| Pre-resolved ID updates | Contact updates use entity ID from offer tracking | `chat.py` |
| Context-aware parsing | LLM parser receives recent entities for smarter intent detection | `llm_command_parser.py` |
| Entity cards UI | Moved cards outside chat bubble for proper sizing | `assistant.html`, `modern.css` |
| Document content extraction | Added `onedrive_get_document_content` and `sharepoint_get_document_content` tools | `365mcp/src/tools/sharepoint.py` |
| Teams meeting organizer | Fixed organizer recognition, auto-filter from attendees | `365mcp/src/tools/sharepoint.py` |

### Key Files
| File | Purpose |
|------|---------|
| `backend/app/api/v1/chat.py` | Main chat endpoint, action execution, entity tracking |
| `backend/app/services/ai_service.py` | AI providers, system prompt, extended thinking |
| `backend/app/services/llm_command_parser.py` | Intent/entity extraction from natural language |
| `backend/app/services/conversation_memory.py` | Persistent memory, entity tracking, pending offer tracking |
| `backend/app/integrations/mcp_tool_adapter.py` | MCP tool schemas and validation |
| `365mcp/` | Microsoft 365 MCP server (separate process) |

### MCP Server (365mcp/)
- **Location**: `/365mcp/` - separate Python process
- **Documentation**: See [`365mcp/DEVELOPER_GUIDE.md`](365mcp/DEVELOPER_GUIDE.md) for comprehensive docs
- **Run**: `python 365mcp/run_server.py` (stdio) or `python 365mcp/http_bridge.py` (HTTP)
- **Tools (25 total)**:
  - **Outlook (6)**: `outlook_search_emails`, `outlook_get_email`, `outlook_send_email`, `outlook_create_draft`, `outlook_list_folders`, `outlook_get_calendar_events`
  - **SharePoint (8)**: `sharepoint_list_sites`, `sharepoint_list_documents`, `sharepoint_get_document`, `sharepoint_search_documents`, `sharepoint_upload_document`, `sharepoint_create_folder`, `sharepoint_share_document`, `sharepoint_get_document_content`
  - **OneDrive (3)**: `onedrive_list_files`, `onedrive_get_document_content`, `onedrive_upload_document`
  - **Teams (6)**: `teams_list_teams`, `teams_list_files`, `teams_create_meeting`, `teams_get_meeting_attendees`, `teams_add_meeting_attendees`, `teams_get_meeting_notes`
  - **Auth (2)**: `authenticate`, `auth_callback`
- **Key Files**:
  - `365mcp/src/server.py` - Main MCP server (~290 lines)
  - `365mcp/src/tools/outlook.py` - Outlook/email operations (~918 lines)
  - `365mcp/src/tools/sharepoint.py` - SharePoint/OneDrive/Teams (~2794 lines)
  - `365mcp/src/auth/oauth_handler.py` - OAuth2 token management
  - `365mcp/src/security.py` - Rate limiting, input validation
  - `365mcp/http_bridge.py` - FastAPI HTTP wrapper for testing

### Email Configuration
**IMPORTANT**: Sending emails via MS365 uses **Microsoft Graph API**, NOT SMTP.
- The SMTP settings in `.env` (SMTP_SERVER, SMTP_PORT, etc.) are for **local notifications only**
- For MS365 email sending:
  1. Configure MS365 OAuth in `.env`: `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET`, `MICROSOFT_TENANT_ID`
  2. Run MCP server: `python 365mcp/http_bridge.py`
  3. User must connect MS365 account in Settings → Integrations
  4. OAuth scopes required: `Mail.ReadWrite`, `Mail.Send`, `Calendars.ReadWrite`, + 12 more (see [`365mcp/DEVELOPER_GUIDE.md`](365mcp/DEVELOPER_GUIDE.md#oauth2-scopes))

---

## Railway Deployment (Dec 2024)

### Architecture
- **CRM Backend**: `backend/` - FastAPI app with PostgreSQL + Redis
- **MCP Server**: Separate repo (`365mcp`) - Microsoft 365 integration server
- **Communication**: Private networking via `mcp-server.railway.internal:8001`

### Deployment Files

| File | Purpose |
|------|---------|
| `backend/Dockerfile` | CRM container - runs migrations on startup |
| `backend/railway.toml` | Railway config with healthcheck at `/health` |
| `backend/.env.local.example` | Local dev environment template |

### Key Configuration Changes

**`backend/app/core/config.py`**:
- Added `@field_validator` to convert `postgresql://` → `postgresql+asyncpg://` (Railway compatibility)
- Multi-env support: `.env` for production, `.env.local` for local (overrides)

**`backend/app/main.py`**:
- TrustedHostMiddleware allows `*.railway.app`, `*.up.railway.app`

### Environment Variables (Railway)

**CRM Backend Service:**
```
DATABASE_URL=<from Railway PostgreSQL addon>
REDIS_URL=<from Railway Redis addon>
MCP_MICROSOFT365_URL=http://mcp-server.railway.internal:8080
FRONTEND_URL=https://your-app.up.railway.app
BACKEND_URL=https://your-app.up.railway.app
MICROSOFT_REDIRECT_URI=https://your-app.up.railway.app/api/v1/integrations/ms365/callback
```

**MCP Server Service:**
```
PORT=8080
MICROSOFT_REDIRECT_URI=https://your-crm.up.railway.app/api/v1/integrations/ms365/callback
FRONTEND_URL=https://your-crm.up.railway.app
```

### Local Development
1. Copy `.env.local.example` to `.env.local`
2. `.env.local` uses localhost URLs and overrides `.env`
3. Run MCP server: `python 365mcp/http_bridge.py`
4. Run CRM: `uvicorn app.main:app --reload`

### Common Deployment Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Healthcheck fails | DATABASE_URL format mismatch | `field_validator` auto-converts `postgresql://` to `postgresql+asyncpg://` |
| 400 Bad Request on health | TrustedHostMiddleware blocking | Added `*.railway.app` to allowed_hosts |
| SSL error to MCP | Wrong protocol | Use `http://` not `https://` for internal networking |
| OAuth redirect mismatch | Localhost URI in production | Set `MICROSOFT_REDIRECT_URI` to production URL on MCP server |

### Azure App Registration
Redirect URIs to register:
- `http://localhost:8000/api/v1/integrations/ms365/callback` (local)
- `https://your-app.up.railway.app/api/v1/integrations/ms365/callback` (production)