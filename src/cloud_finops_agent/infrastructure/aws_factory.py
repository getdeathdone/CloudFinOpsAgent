"""Async AWS client factory backed by aioboto3."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import aioboto3
import structlog

from cloud_finops_agent.config.settings import Settings, get_settings

logger = structlog.get_logger(__name__)

AWSServiceName = Literal["s3", "ec2", "cloudwatch", "iam"]


class AsyncAWSClientFactory:
    """Creates configured aioboto3 clients for AWS or LocalStack."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._session = aioboto3.Session(region_name=self._settings.AWS_REGION)

    @asynccontextmanager
    async def client(self, service_name: AWSServiceName) -> AsyncIterator[Any]:
        """Yield an async aioboto3 client configured from application settings."""

        kwargs = self._settings.get_aws_client_kwargs()
        logger.debug(
            "creating_aws_client",
            service_name=service_name,
            environment=self._settings.ENVIRONMENT,
            endpoint_url=kwargs.get("endpoint_url"),
            region_name=kwargs.get("region_name"),
        )
        async with self._session.client(service_name, **kwargs) as client:
            yield client

    @asynccontextmanager
    async def s3_client(self) -> AsyncIterator[Any]:
        """Yield an async S3 client."""

        async with self.client("s3") as client:
            yield client

    @asynccontextmanager
    async def ec2_client(self) -> AsyncIterator[Any]:
        """Yield an async EC2 client."""

        async with self.client("ec2") as client:
            yield client

    @asynccontextmanager
    async def cloudwatch_client(self) -> AsyncIterator[Any]:
        """Yield an async CloudWatch client."""

        async with self.client("cloudwatch") as client:
            yield client


def get_aws_client_factory(settings: Settings | None = None) -> AsyncAWSClientFactory:
    """Return a configured AWS client factory."""

    return AsyncAWSClientFactory(settings=settings)
