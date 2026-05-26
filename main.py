from contextlib import asynccontextmanager

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base, engine, get_db
from app.dependencies import bearer_scheme, get_current_user
from app.routers import chat as chat_router
from app.models import TokenBlocklist, User
from app.schemas import (
    AuthResponse,
    Message,
    RefreshRequest,
    Token,
    UserCreate,
    UserLogin,
    UserOut,
)
from app.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables on startup.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="FastAPI JWT Auth + Chat (async)", version="1.1.0", lifespan=lifespan)

# Allow the frontend (different origin) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the chat endpoints (/chat, /chat/conversations, ...).
app.include_router(chat_router.router)


@app.post("/auth/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user and log them in immediately (returns user + tokens)."""
    result = await db.execute(
        select(User).where(
            or_(User.email == payload.email, User.username == payload.username)
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email or username already registered",
        )

    user = User(
        email=payload.email,
        username=payload.username,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    subject = str(user.id)
    return AuthResponse(
        user=user,
        access_token=create_access_token(subject),
        refresh_token=create_refresh_token(subject),
    )


@app.post("/auth/login", response_model=AuthResponse)
async def login(payload: UserLogin, db: AsyncSession = Depends(get_db)):
    """Authenticate a user and return user details + access + refresh tokens."""
    result = await db.execute(
        select(User).where(
            or_(User.username == payload.username, User.email == payload.username)
        )
    )
    user = result.scalar_one_or_none()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    subject = str(user.id)
    return AuthResponse(
        user=user,
        access_token=create_access_token(subject),
        refresh_token=create_refresh_token(subject),
    )


@app.post("/auth/refresh", response_model=Token)
async def refresh(payload: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Exchange a valid refresh token for a new pair of tokens."""
    try:
        data = decode_token(payload.refresh_token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    if data.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    jti = data.get("jti")
    if jti:
        result = await db.execute(
            select(TokenBlocklist).where(TokenBlocklist.jti == jti)
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has been revoked",
            )
        # Rotate: revoke the old refresh token so it can't be reused.
        db.add(TokenBlocklist(jti=jti))
        await db.commit()

    subject = data["sub"]
    return Token(
        access_token=create_access_token(subject),
        refresh_token=create_refresh_token(subject),
    )


@app.post("/auth/logout", response_model=Message)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Revoke the current access token by adding its JTI to the blocklist."""
    payload = decode_token(credentials.credentials)
    jti = payload.get("jti")
    if jti:
        result = await db.execute(
            select(TokenBlocklist).where(TokenBlocklist.jti == jti)
        )
        if not result.scalar_one_or_none():
            db.add(TokenBlocklist(jti=jti))
            await db.commit()
    return Message(detail="Successfully logged out")


@app.get("/auth/me", response_model=UserOut)
async def read_me(current_user: User = Depends(get_current_user)):
    """Return the currently authenticated user. Example of a protected route."""
    return current_user


@app.get("/")
async def root():
    return {"message": "FastAPI JWT Auth API (async). See /docs for documentation."}