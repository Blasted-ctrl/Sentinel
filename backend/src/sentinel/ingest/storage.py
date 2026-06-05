"""Object storage abstraction over S3 / MinIO via boto3."""

from __future__ import annotations

from typing import TYPE_CHECKING

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from sentinel.config import Settings, get_settings

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


class ObjectStorage:
    """Thin wrapper around an S3-compatible bucket.

    Works against real AWS S3 (``endpoint_url=None``) or a local MinIO
    container (``endpoint_url=http://localhost:9000``).
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
    ) -> None:
        self.bucket = bucket
        self._endpoint_url = endpoint_url
        self._access_key = access_key
        self._secret_key = secret_key
        self._region = region
        self._cached: S3Client | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> ObjectStorage:
        s = settings or get_settings()
        return cls(
            s.s3_bucket,
            endpoint_url=s.s3_endpoint_url,
            access_key=s.s3_access_key,
            secret_key=s.s3_secret_key,
            region=s.s3_region,
        )

    @property
    def client(self) -> S3Client:
        if self._cached is None:
            self._cached = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                region_name=self._region,
                config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
            )
        return self._cached

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not already exist (idempotent)."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> str:
        """Upload ``data`` under ``key`` and return the key."""
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
        )
        return key

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError:
            return False
        return True
