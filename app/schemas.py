from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    # Users can log in with either their username or email.
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    username: str
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class AuthResponse(BaseModel):
    """Returned by signup and login: user details plus tokens."""
    user: "UserOut"
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class Message(BaseModel):
    detail: str


# ---------- Chat schemas ----------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    # Optional: continue an existing conversation. Omit to start a new one.
    conversation_id: int | None = None
    # Optional: pick a model. Defaults applied server-side.
    model: str | None = None


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    model: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    conversation_id: int
    reply: str
    model: str


class ConversationOut(BaseModel):
    id: int
    title: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationDetail(ConversationOut):
    messages: list[MessageOut] = []


class ChatErrorResponse(BaseModel):
    """Returned (HTTP 502/503/504/429) when the LLM provider fails.
    `user_message` is echoed back so the frontend can restore the input field."""
    detail: str
    user_message: str