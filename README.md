# AI-Assisted CRM

A modern, AI-powered Customer Relationship Management system built with FastAPI, HTMX, and TailwindCSS.

## 🚀 Features

- **AI-Powered Assistant**: Natural language interface for all CRM operations using Claude, OpenAI, and Gemini
- **Real-time Communication**: WebSocket-based chat for instant responses
- **Modern UI**: Responsive design with TailwindCSS and DaisyUI components
- **Dynamic Interactions**: HTMX for seamless user experience without complex JavaScript
- **Comprehensive Data Management**: Contacts, Organizations, Projects, and Tasks
- **Calendar Integration**: Microsoft 365 and Google Workspace sync
- **Secure Authentication**: JWT-based authentication with encrypted API key storage

## 🛠 Tech Stack

### Backend
- **FastAPI**: High-performance async Python framework
- **PostgreSQL**: Robust relational database
- **Redis**: Caching and session management
- **SQLAlchemy**: Async ORM
- **Alembic**: Database migrations
- **Celery**: Background task processing

### Frontend
- **HTMX**: Dynamic interactions without JavaScript complexity
- **Alpine.js**: Minimal client-side interactivity
- **Jinja2**: Server-side templating
- **TailwindCSS**: Utility-first CSS framework
- **DaisyUI**: Beautiful Tailwind components

### AI & Integrations
- **Anthropic Claude**: Primary AI model
- **OpenAI GPT**: Secondary AI support
- **Google Gemini**: Additional AI option
- **Microsoft 365**: Calendar, OneDrive, Teams integration
- **Google Workspace**: Calendar, Drive, Gmail integration

## 📋 Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose
- Git

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd crm
```

### 2. Start Database Services

```bash
cd docker
docker-compose up -d postgres redis
```

### 3. Set up Python Environment

```bash
cd ../backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp ../.env.example ../.env
# Edit .env with your configuration
```

### 5. Initialize Database

```bash
# Run database migrations
alembic upgrade head

# Test the setup
python test_setup.py
```

### 6. Start the Application

```bash
uvicorn app.main:app --reload
```

Visit http://localhost:8000 to access your CRM!

## 📚 Important URLs

- **Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **AI Assistant**: http://localhost:8000/assistant

## 🔧 Development Commands

### Database Operations

```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Running Tests

```bash
# Setup verification
python test_setup.py

# Run API tests (when implemented)
pytest tests/

# Load testing (when implemented)
locust -f tests/load_test.py
```

### Docker Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📁 Project Structure

```
crm/
├── backend/
│   ├── app/
│   │   ├── models/          # SQLAlchemy models
│   │   ├── api/v1/          # API endpoints
│   │   ├── services/        # Business logic
│   │   ├── core/            # Configuration & database
│   │   ├── templates/       # Jinja2 templates
│   │   └── static/          # CSS, JS, images
│   ├── alembic/             # Database migrations
│   ├── tests/               # Test files
│   └── requirements.txt     # Python dependencies
├── docker/
│   ├── docker-compose.yml   # Development services
│   └── Dockerfile           # Application container
├── docs/
│   ├── DEVELOPMENT.md       # Development progress
│   └── API.md               # API documentation
├── scripts/
│   └── setup_dev.sh         # Development setup script
├── CLAUDE.md                # Project requirements
└── README.md                # This file
```

## 🔐 Environment Variables

Create a `.env` file with the following variables:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://crm_user:crm_password@localhost:5432/crm_db
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-super-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI APIs
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key
GOOGLE_AI_KEY=your-google-ai-api-key

# OAuth (optional)
MICROSOFT_CLIENT_ID=your-microsoft-client-id
MICROSOFT_CLIENT_SECRET=your-microsoft-client-secret
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

## 🤖 AI Assistant Usage

The AI assistant can help you with:

```
> Add contact John Smith from Acme Corp with email john@acme.com
> Create project "Q1 Marketing Campaign" for Acme Corp
> Show me all overdue tasks
> Schedule meeting with John next Tuesday at 2 PM
> Export all contacts from the tech industry
```

## 🧪 Testing

```bash
# Basic setup test
python backend/test_setup.py

# API tests (when implemented)
cd backend && pytest

# Load testing (when implemented)
locust -f backend/tests/load_test.py --host=http://localhost:8000
```

## 📈 Development Progress

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for current development status and milestones.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support, please check:
1. [DEVELOPMENT.md](docs/DEVELOPMENT.md) for current status
2. GitHub Issues for known problems
3. API documentation at `/docs` when running

---

**Built with ❤️ using FastAPI, HTMX, and AI assistance**