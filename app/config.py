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

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str | None = os.getenv("LOG_FILE", "logs/app.log")

    # Max previous messages sent to LLM for context (excludes current user message)
    HISTORY_LIMIT: int = int(os.getenv("HISTORY_LIMIT", "10"))

    # System prompt — sets the LLM's persona and behaviour
    SYSTEM_PROMPT: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful, friendly AI assistant. Answer clearly and concisely.",
    )

    # LLM API keys
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY:    str | None = os.getenv("OPENAI_API_KEY", "").strip() or None
    GROK_API_KEY:      str | None = os.getenv("GROK_API_KEY", "").strip() or None
    DEEPSEEK_API_KEY:  str | None = os.getenv("DEEPSEEK_API_KEY", "").strip() or None
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-mini")

    # Comma-separated list of allowed CORS origins.
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
        ).split(",")
        if o.strip()
    ]

    @property
    def access_token_expires(self) -> timedelta:
        return timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)

    @property
    def refresh_token_expires(self) -> timedelta:
        return timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)


settings = Settings()