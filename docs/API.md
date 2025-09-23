# AI-Assisted CRM API Documentation

Base URL: `http://localhost:8000/api/v1`

## Authentication

All API endpoints (except auth endpoints) require a Bearer token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### Get Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your@email.com&password=yourpassword"
```

## Authentication Endpoints

### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "name": "John Doe",
  "password": "securepassword",
  "company": "Acme Corp",
  "job_title": "Developer"
}
```

### Login
```http
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=securepassword
```

### Get Current User
```http
GET /api/v1/auth/me
Authorization: Bearer <token>
```

## Contacts API

### List Contacts
```http
GET /api/v1/contacts?page=1&per_page=20&search=john&organization=acme
Authorization: Bearer <token>
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20, max: 100)
- `search`: Search by name, email, or organization
- `organization`: Filter by organization
- `tags`: Filter by tags (comma-separated)
- `sort_by`: Sort field (default: created_at)
- `sort_order`: asc or desc (default: desc)

### Create Contact
```http
POST /api/v1/contacts
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "John Smith",
  "email": "john@example.com",
  "phone": "+1-555-0123",
  "job_position": "Software Engineer",
  "organization": "Tech Corp",
  "notes": "Met at conference",
  "tags": ["prospect", "developer"],
  "linkedin_url": "https://linkedin.com/in/johnsmith",
  "city": "San Francisco",
  "state": "CA",
  "country": "USA"
}
```

### Get Contact
```http
GET /api/v1/contacts/{contact_id}
Authorization: Bearer <token>
```

### Update Contact
```http
PUT /api/v1/contacts/{contact_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "John Smith Jr.",
  "phone": "+1-555-0124"
}
```

### Delete Contact
```http
DELETE /api/v1/contacts/{contact_id}
Authorization: Bearer <token>
```

### Mark Contact as Contacted
```http
POST /api/v1/contacts/{contact_id}/mark-contacted
Authorization: Bearer <token>
```

### Export Contacts to CSV
```http
GET /api/v1/contacts/export/csv
Authorization: Bearer <token>
```

## Organizations API

### List Organizations
```http
GET /api/v1/organizations?page=1&per_page=20&industry=technology
Authorization: Bearer <token>
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20, max: 100)
- `search`: Search by name, industry, or description
- `industry`: Filter by industry
- `relationship_status`: Filter by relationship status
- `priority`: Filter by priority
- `company_size`: Filter by company size
- `tags`: Filter by tags (comma-separated)
- `sort_by`: Sort field (default: created_at)
- `sort_order`: asc or desc (default: desc)

### Create Organization
```http
POST /api/v1/organizations
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Tech Corp",
  "description": "Leading technology company",
  "industry": "Technology",
  "website": "https://techcorp.com",
  "email": "contact@techcorp.com",
  "phone": "+1-555-0100",
  "company_size": "51-200",
  "annual_revenue": "$10M-$50M",
  "founded_year": 2010,
  "relationship_status": "client",
  "priority": "high",
  "linkedin_url": "https://linkedin.com/company/techcorp",
  "city": "San Francisco",
  "state": "CA",
  "country": "USA",
  "tags": ["technology", "client"]
}
```

### Get Organization
```http
GET /api/v1/organizations/{organization_id}
Authorization: Bearer <token>
```

### Update Organization
```http
PUT /api/v1/organizations/{organization_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "relationship_status": "partner",
  "priority": "urgent"
}
```

### Delete Organization
```http
DELETE /api/v1/organizations/{organization_id}
Authorization: Bearer <token>
```

### Get Organization Statistics
```http
GET /api/v1/organizations/stats
Authorization: Bearer <token>
```

### Mark Organization Interaction
```http
POST /api/v1/organizations/{organization_id}/mark-interaction
Authorization: Bearer <token>
```

### Get Organization Projects
```http
GET /api/v1/organizations/{organization_id}/projects
Authorization: Bearer <token>
```

## Error Responses

All endpoints return standard HTTP status codes with JSON error messages:

```json
{
  "detail": "Error message"
}
```

**Common Status Codes:**
- `200`: Success
- `201`: Created
- `204`: No Content (successful deletion)
- `400`: Bad Request (validation error)
- `401`: Unauthorized (invalid/missing token)
- `404`: Not Found
- `422`: Unprocessable Entity (validation error)
- `500`: Internal Server Error

## Rate Limiting

API requests are limited to 100 requests per minute per user.

## Pagination Response Format

```json
{
  "contacts": [...],
  "total": 150,
  "page": 1,
  "per_page": 20,
  "total_pages": 8
}
```

## Testing with curl

### Complete Example: Create and Manage Contact

1. **Register User:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "name": "Test User",
    "password": "testpassword123"
  }'
```

2. **Login to get token:**
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=testpassword123" \
  | jq -r '.access_token')
```

3. **Create contact:**
```bash
curl -X POST http://localhost:8000/api/v1/contacts \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Smith",
    "email": "john@example.com",
    "organization": "Tech Corp"
  }'
```

4. **List contacts:**
```bash
curl -X GET "http://localhost:8000/api/v1/contacts?search=john" \
  -H "Authorization: Bearer $TOKEN"
```

## Interactive API Documentation

When the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive documentation where you can test all endpoints directly in your browser.