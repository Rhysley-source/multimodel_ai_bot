from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from app.config import settings


async def ensure_database_exists() -> None:
    """Create the target database if it does not already exist.
    Only runs for PostgreSQL; SQLite is skipped.
    If the user lacks CREATEDB privilege, logs a warning and continues
    (the DB may already exist and be usable).
    """
    url = settings.DATABASE_URL
    if not url.startswith("postgresql"):
        return

    from sqlalchemy import exc, text

    db_name = url.rsplit("/", 1)[-1]
    admin_url = url.rsplit("/", 1)[0] + "/postgres"
    admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with admin_engine.connect() as conn:
            result = await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            )
            if result.scalar():
                print(f"[startup] Database '{db_name}' already exists.")
                return
            await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
            print(f"[startup] Database '{db_name}' created.")
    except exc.ProgrammingError as e:
        if "permission denied" in str(e).lower():
            print(
                f"[startup] WARNING: No CREATEDB privilege — skipping auto-create.\n"
                f"          Create the database manually:\n"
                f"          psql -U postgres -c 'CREATE DATABASE \"{db_name}\";'\n"
                f"          Continuing startup (DB may already exist)..."
            )
        else:
            raise
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