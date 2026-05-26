# rgChat-backend

Minimal FastAPI backend implementing JWT auth, REST endpoints for conversations/messages, and a WebSocket streaming flow backed by mock LLM adapters.

Now uses PostgreSQL with SQLModel ORM and async SQLAlchemy.

## Quick Start (without database - testing only)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Access the API docs at: http://localhost:8000/docs

**Note:** Without a database, endpoints will fail. To enable full functionality, complete the setup below.

## Production Setup (with PostgreSQL)

### 1. Start PostgreSQL via Docker
```bash
docker-compose up -d postgres
```

Verify it's running:
```bash
docker ps  # Should show postgres container
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Copy environment file
```bash
cp .env.example .env
# Update .env DATABASE_URL if needed (default works with docker-compose)
```

### 4. Run the server
```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Tables are auto-created on startup.

## API Endpoints

**Auth**
- `POST /api/v1/auth/register` — Register and get JWT token
- `POST /api/v1/auth/login` — Login and get JWT token

**Conversations**
- `POST /api/v1/chats/` — Create conversation
- `GET /api/v1/chats/` — List conversations
- `GET /api/v1/chats/{conversation_id}` — Get conversation details

**Messages**
- `POST /api/v1/messages/` — Send message
- `GET /api/v1/messages/{conversation_id}` — Get conversation messages

**WebSocket**
- `WS /api/v1/ws/stream?token=<JWT>` — Streaming chat responses

## WebSocket Usage Example

```javascript
const token = "your-jwt-token";
const conversationId = "conv-123";
const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/stream?token=${token}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "token") {
    console.log("Token:", data.token); // Stream token from LLM
  } else if (data.type === "done") {
    console.log("Reply:", data.message); // Full response
  }
};

ws.send(JSON.stringify({
  type: "message",
  conversation_id: conversationId,
  content: "Hello",
  llm: "mock"
}));
```