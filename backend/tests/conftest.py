"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from sentinel.geo import BBox


@pytest.fixture(autouse=True)
def _fake_aws_credentials() -> Iterator[None]:
    """Ensure boto3/moto never reach real AWS during tests."""
    saved = {k: os.environ.get(k) for k in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SECURITY_TOKEN",
        "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION",
    )}
    os.environ.update(
        AWS_ACCESS_KEY_ID="testing",
        AWS_SECRET_ACCESS_KEY="testing",
        AWS_SECURITY_TOKEN="testing",
        AWS_SESSION_TOKEN="testing",
        AWS_DEFAULT_REGION="us-east-1",
    )
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def sample_bbox() -> BBox:
    """A small bbox over the Sierra Nevada (fire-prone) for tests."""
    return BBox(-120.5, 38.5, -120.0, 39.0)
