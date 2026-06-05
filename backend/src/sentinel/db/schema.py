"""Schema bootstrap helpers.

``init_db`` enables the PostGIS extension and creates every table. It is the
baseline used by the ``sentinel init-db`` CLI command and by integration tests.
Incremental schema changes after the baseline are managed with Alembic
(see ``migrations/``).
"""

from __future__ import annotations

from sqlalchemy import Engine, text

from sentinel.db import models  # noqa: F401  (register models on Base.metadata)
from sentinel.db.base import Base


def enable_postgis(engine: Engine) -> None:
    """Enable the PostGIS extension (idempotent)."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))


def init_db(engine: Engine, *, drop: bool = False) -> None:
    """Create the PostGIS extension and all tables.

    Args:
        engine: Target database engine.
        drop: If ``True``, drop existing tables first (test convenience).
    """
    enable_postgis(engine)
    if drop:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
