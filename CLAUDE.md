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