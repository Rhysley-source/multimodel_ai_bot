import os
from datetime import timedelta

from dotenv import load_dotenv

# Load variables from a .env file in the project root (if present).
load_dotenv()


class Settings:
    """Application settings loaded from environment variables / .env file."""

    # IMPORTANT: In production, set SECRET_KEY via the environment / .env file.
    # Generate one with: openssl rand -hex 32
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "CHANGE_ME_dev_only_secret_key_do_not_use_in_production",
    )

    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")

    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(
        os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7")
    )

    # Async driver expected, e.g.
    #   postgresql+asyncpg://user:pass@localhost:5432/dbname
    #   sqlite+aiosqlite:///./auth.db
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite+aiosqlite:///./auth.db"
    )

    @property
    def access_token_expires(self) -> timedelta:
        return timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)

    @property
    def refresh_token_expires(self) -> timedelta:
        return timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)


settings = Settings()