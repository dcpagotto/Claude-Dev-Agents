# Example: Authentication System Workflow

This example shows how the agent team collaborates to build a complete authentication system for a Django project.

## User Request

```
"Build a JWT authentication system with login, registration, and password reset"
```

## Phase 1: Tech-Lead Analysis

The main agent invokes `tech-lead-orchestrator` to analyze the request.

```
Main Agent: "I'll use the tech-lead-orchestrator to analyze this request."
[Invokes tech-lead-orchestrator]
```

### Tech-Lead Returns

```markdown
### Task Analysis
- User needs JWT-based authentication with three flows: login, registration, password reset
- Django project detected (manage.py, settings.py, requirements.txt with Django)
- Django REST Framework already installed

### SubAgent Assignments
Task 1: Design auth database schema → AGENT: @agent-django-orm-expert
Task 2: Implement auth backend logic → AGENT: @agent-django-backend-expert
Task 3: Create auth API endpoints → AGENT: @agent-django-api-developer
Task 4: Security review → AGENT: @agent-code-reviewer

### Execution Order
- **Sequential**: Task 1 → Task 2 → Task 3 → Task 4

### Available Agents for This Project
- django-orm-expert: Database schema for User model and tokens
- django-backend-expert: Auth service layer, JWT handling, password hashing
- django-api-developer: REST endpoints for login/register/reset
- code-reviewer: Security audit of auth implementation

### Instructions to Main Agent
- Start with django-orm-expert for the User model and token schema
- Pass schema output to django-backend-expert for service implementation
- Pass service details to django-api-developer for API endpoints
- Finish with code-reviewer for security validation
```

## Phase 2: Approval Gate

The main agent presents the plan to the user and waits for approval before proceeding.

```
Main Agent: "The tech-lead recommends a 4-step approach:
1. Database schema design (django-orm-expert)
2. Backend auth logic (django-backend-expert)
3. API endpoints (django-api-developer)
4. Security review (code-reviewer)

Shall I proceed?"

User: "Yes, go ahead."
```

## Phase 3: Execution

### Step 1: Database Schema (django-orm-expert)

```
Main Agent → django-orm-expert:
"Design the database schema for JWT authentication with:
- Custom User model with email-based login
- Token storage for refresh tokens
- Password reset token model"
```

**Agent Returns:**
```markdown
## Task Completed: Auth Database Schema

### Models Created
- `CustomUser` extending AbstractUser with email as USERNAME_FIELD
- `RefreshToken` with user FK, token, expires_at, is_revoked
- `PasswordResetToken` with user FK, token, expires_at, is_used

### Migrations
- 0001_custom_user.py
- 0002_auth_tokens.py

### Next specialist needs:
- User model location: accounts/models.py
- Token models support revocation and expiry checking
- Custom manager includes `create_user` and `create_superuser`
```

### Step 2: Backend Logic (django-backend-expert)

```
Main Agent → django-backend-expert:
"Implement authentication service layer using:
- User model at accounts/models.py (CustomUser with email login)
- RefreshToken and PasswordResetToken models
- JWT token generation and validation
- Password hashing and reset flow"
```

**Agent Returns:**
```markdown
## Task Completed: Auth Service Layer

### Services Created
- `AuthService` in accounts/services.py
  - `register(email, password, **kwargs)` → creates user + sends welcome email
  - `login(email, password)` → validates credentials, returns access + refresh tokens
  - `refresh_token(refresh_token)` → validates and rotates refresh token
  - `logout(refresh_token)` → revokes refresh token
  - `request_password_reset(email)` → generates reset token, sends email
  - `reset_password(token, new_password)` → validates token and updates password

### Security Measures
- Passwords hashed with Argon2
- JWT access tokens expire in 15 minutes
- Refresh tokens expire in 7 days with rotation
- Password reset tokens single-use, expire in 1 hour

### Next specialist needs:
- AuthService at accounts/services.py with all methods documented
- JWT settings in settings.py under AUTH_CONFIG
- Email templates at templates/emails/
```

### Step 3: API Endpoints (django-api-developer)

```
Main Agent → django-api-developer:
"Create DRF API endpoints for authentication using:
- AuthService at accounts/services.py
- Endpoints needed: POST /auth/register, POST /auth/login,
  POST /auth/refresh, POST /auth/logout,
  POST /auth/password-reset, POST /auth/password-reset/confirm"
```

**Agent Returns:**
```markdown
## Task Completed: Auth API Endpoints

### Endpoints Created
- POST /api/auth/register/ → RegisterSerializer → 201 Created
- POST /api/auth/login/ → LoginSerializer → 200 OK with tokens
- POST /api/auth/refresh/ → RefreshSerializer → 200 OK with new tokens
- POST /api/auth/logout/ → 204 No Content (revokes refresh token)
- POST /api/auth/password-reset/ → 200 OK (sends reset email)
- POST /api/auth/password-reset/confirm/ → 200 OK (resets password)

### Files Created
- accounts/api/serializers.py - Request/response serializers
- accounts/api/views.py - DRF APIViews
- accounts/api/urls.py - URL routing

### Next specialist needs:
- Full API surface ready for security review
- All endpoints use AuthService (no direct model access)
- Rate limiting applied to login and password reset
```

### Step 4: Security Review (code-reviewer)

```
Main Agent → code-reviewer:
"Review the authentication implementation for security:
- accounts/models.py, accounts/services.py
- accounts/api/serializers.py, views.py, urls.py
- Focus on: auth bypass, token security, injection, rate limiting"
```

**Agent Returns:**
```markdown
## Code Review - Auth Implementation

### Executive Summary
| Metric | Result |
|--------|--------|
| Overall Assessment | Good |
| Security Score | B+ |
| Maintainability | A |

### Minor Suggestions
- Add account lockout after 5 failed login attempts
- Consider adding TOTP/2FA support as future enhancement
- Add logging for failed authentication attempts

### Positive Highlights
- Proper password hashing with Argon2
- Refresh token rotation prevents replay attacks
- Rate limiting on sensitive endpoints
- No raw SQL, all queries through ORM

### Action Checklist
- [ ] Add failed login attempt tracking
- [ ] Add audit logging for auth events
```

## Final Result

The main agent summarizes the completed work to the user:

```
Main Agent: "Authentication system is complete:

- Custom User model with email-based login
- JWT authentication with access/refresh tokens
- 6 API endpoints (register, login, refresh, logout, password reset)
- Security review passed with minor suggestions

Files created:
- accounts/models.py (User + token models)
- accounts/services.py (AuthService)
- accounts/api/ (serializers, views, urls)
- 2 database migrations

Recommended follow-ups:
- Add failed login attempt tracking
- Consider 2FA support"
```
