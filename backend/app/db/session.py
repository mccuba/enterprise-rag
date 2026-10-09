from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.db.models import Base


def _ensure_sqlite_dir(url: str) -> None:
    if "sqlite" in url and "///" in url:
        path = url.split("///", 1)[1]
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def create_engine_and_session(settings: Settings):
    _ensure_sqlite_dir(settings.database_url)
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return engine, session_factory


async def init_db(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session(session_factory) -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        yield session
