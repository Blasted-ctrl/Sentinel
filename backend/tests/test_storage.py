"""Tests for object storage (S3 mocked with moto)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from moto import mock_aws

from sentinel.ingest.storage import ObjectStorage


@pytest.fixture
def storage() -> Iterator[ObjectStorage]:
    with mock_aws():
        # endpoint_url=None so moto intercepts at the botocore layer.
        yield ObjectStorage(
            "sentinel-test",
            endpoint_url=None,
            access_key="testing",
            secret_key="testing",
            region="us-east-1",
        )


def test_ensure_bucket_is_idempotent(storage: ObjectStorage) -> None:
    storage.ensure_bucket()
    storage.ensure_bucket()  # second call must not raise
    assert not storage.exists("missing-key")


def test_put_and_exists(storage: ObjectStorage) -> None:
    storage.ensure_bucket()
    key = storage.put_bytes(
        "tiles/a/thumb.jpg", b"\xff\xd8\xff data", content_type="image/jpeg"
    )
    assert key == "tiles/a/thumb.jpg"
    assert storage.exists(key)

    body = storage.client.get_object(Bucket="sentinel-test", Key=key)["Body"].read()
    assert body == b"\xff\xd8\xff data"
