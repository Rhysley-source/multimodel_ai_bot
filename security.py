import uuid
from datetime import datetime, timezone

import jwt
from passlib.context import CryptContext

from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---------- Password hashing ----------

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# ---------- JWT tokens ----------

def _create_token(subject: str, token_type: str, expires_delta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,            # the user id
        "type": token_type,        # "access" or "refresh"
        "iat": now,
        "exp": now + expires_delta,
        "jti": str(uuid.uuid4()),  # unique id so individual tokens can be revoked
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_access_token(subject: str) -> str:
    return _create_token(subject, "access", settings.access_token_expires)


def create_refresh_token(subject: str) -> str:
    return _create_token(subject, "refresh", settings.refresh_token_expires)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT. Raises jwt exceptions on failure."""
    return jwt.decode(
        token,
        settings.SECRET_KEY,
        algorithms=[settings.ALGORITHM],
    )