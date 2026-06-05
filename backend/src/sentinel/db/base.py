"""SQLAlchemy engine, session factory, and declarative base."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from sentinel.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: Engine | None = None
_sessionmaker: sessionmaker[Session] | None = None


def get_engine(database_url: str | None = None) -> Engine:
    """Return a process-wide SQLAlchemy engine, creating it on first use."""
    global _engine
    if database_url is not None:
        # Explicit URL (e.g. tests) — build a dedicated, uncached engine.
        return create_engine(database_url, future=True, pool_pre_ping=True)
    if _engine is None:
        _engine = create_engine(
            get_settings().database_url, future=True, pool_pre_ping=True
        )
    return _engine


def get_sessionmaker(engine: Engine | None = None) -> sessionmaker[Session]:
    """Return a cached session factory bound to the default engine."""
    global _sessionmaker
    if engine is not None:
        return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    if _sessionmaker is None:
        _sessionmaker = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False
        )
    return _sessionmaker


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""
    factory = get_sessionmaker(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
