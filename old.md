### Critical Rules - DO NOT VIOLATE

- **NEVER create mock data or simplified components** unless explicitly told to do so.
- **NEVER replace existing complex components with simplified versions** - always fix the actual problem!
- **ALWAYS work with the existing codebase** - do not create new alternatives!
- **ALWAYS find and fix the root cause** of issues instead of creating workarounds !
- **When debugging issues, focus on fixing the existing implementation, not replacing it**
- **When something doesn't work, debug and fix it - don't start over with a simple version**
- **ALWAYS FOLLOW the existing technical structure & Coding conventions from the Codebase**

### AI Assisted CRM. ###

An ai assisted CRM that lets you chat with the AI (LLM) - using differnet LLMs of your choice (OpenAI chat gpt, Gemini etc...) but mostly Claude AI (API) as *primary*. The CRM will be an application by Corporate entities and Organisations, so it should be built with that in mind ! We will allow Organisations to connect their own API keys if they want.

We are building this in Mind as if we were remaking Claude Ai chat with easy MCP integrations just like regular Claude Desktop.

The CRM will consist of the following features and components:

1. Contacts :

* The Assistant should be able to create contacts for you based on your conversation and if you ask them to. Adding the contact to your Contacts list (stored for that user in the db) alonside their information like : Job position, email, Phone number, Organisation/Role IF provided by the user, or store only whatever was provided by the user for now. The LLM (AI) should have CRUD functionality for this in with natural chat processing.

Ability to edit or delete the Contact(s).

2. Organisations

* The app will allow each user to create organisations they follow, and also can be created by asking the AI Assistant to do it. 

Each user will have their own list of Orgs just like Contacts. The LLM (AI) should have CRUD functionality for this in with natural chat processing.

Ability to edit or delete the Organisation.

3. Projects 

The user should be able to create Projects through the frontend directly or through the AI chat. 
Projects will display: Name (project name) , Status (Planned, In progress, Completed etc.), Organisation, Assignees (one or many), Due Date. The LLM (AI) should have CRUD functionality for this in with natural chat processing.

Inside each project (clicking on them to view them) Should display more information about the Project and more importantly each tasks associated with that Project. The tasks will also have : Task name, Status, Priority, Assignee, Project, Due Date , Actions (Edit/Delete etc...). A button to add new tasks under that Project.

Projects should show dates including last updated date in case of updates.

Ability to edit or delete the project.

4. Tasks 

A list of all your Tasks from all projects with their information and dates.

5. Calendar

* Display a calendar integrated with Outlook/Teams or Google Calendar for that user that show their activities and meetigns at a glance. Users will have to connect all their required accounts properly.

The user should be able to manipulate the Calendar with the AI. THE AI CHAT SHOULD BE ABLE TO ALSO CREATE REMINDERS, EVENTS THAT ARE TRACKED AND CAN SEND EMAIL NOTIFICATIONS.



### Tech/Developement Stack and Frameworks ###

Python (FastAPI or Flask (Async))
PostgreSQL
Some Robust and very secure Auth method.
BUILT WITH MCP integrations in mind and easy integrations with MS 365, Google, etc. Users will have to connect all their required accounts properly.
Built with Security in mind.
Built with speed, performance and concurrent use.

# database models and ERP #

Based on all this documents requirements and inc

### Coding practices to follow ### :
- Clean, efficient, SHORT CODE with DRY (Dont Repeat Yourself) practices.
- Develop with reusable components if possbile.
- Follow KISS, YAGNI practices.

### Authentication ###
Need to decide on this (token based maybe? JWT?) whatever works well. We need to plan and decide what will work.

### Frontend UI ###

* Modern beautiful UI.
* Company Logo Top Left.
* Login andd Account logged in top Right.
* Should look like most AI Chat frontends (like Claude AI chat for example), That shows a different greeting with the User's name each time they login.

Side bar with :

* Assistant : Where users can chat with the LLM (AI assistant) and do all sorts of actions.
* Contacts
* Organizations
* Projects
* Calendar
* Tasks
* Settings : Will have a Profile tab (showing User profile pic and their basic info) and an Integration tab (where MCP integrations services can be connected etc). Users will have to connect all their required accounts on this page.

* Feedback


### MCP Integrations with LLM ###

- MS365 services and tools (Outlook, Onedrive, Sharepoint, Teams)
- Google
