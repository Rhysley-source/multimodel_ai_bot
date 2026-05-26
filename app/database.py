from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from app.config import settings


async def ensure_database_exists() -> None:
    """Create the target database if it does not already exist.
    Only runs for PostgreSQL connections; SQLite is skipped.
    """
    url = settings.DATABASE_URL
    if not url.startswith("postgresql"):
        return  # SQLite creates the file automatically

    from sqlalchemy import text

    # Parse the database name from the URL (last path segment).
    db_name = url.rsplit("/", 1)[-1]

    # Build an admin URL pointing at the default 'postgres' maintenance DB.
    admin_url = url.rsplit("/", 1)[0] + "/postgres"
    admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with admin_engine.connect() as conn:
            result = await conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            )
            if not result.scalar():
                await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                print(f"[startup] Database '{db_name}' created.")
            else:
                print(f"[startup] Database '{db_name}' already exists.")
    finally:
        await admin_engine.dispose()


engine = create_async_engine(settings.DATABASE_URL, echo=False)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

Base = declarative_base()


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields an async database session."""
    async with AsyncSessionLocal() as session:
        yield session