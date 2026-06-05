"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from sentinel.db.base import get_sessionmaker


def get_db() -> Iterator[Session]:
    """Yield a database session for the duration of a request."""
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()
