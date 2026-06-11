"""Application settings loaded from environment variables and .env files."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Any

from pydantic import AnyUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Supported runtime environments."""

    DEV = "dev"
    PROD = "prod"


class Settings(BaseSettings):
    """Typed application configuration shared by agents, tools, and scripts."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    ENVIRONMENT: Environment = Field(default=Environment.DEV)
    AWS_REGION: str = Field(default="us-east-1")
    LOCALSTACK_ENDPOINT_URL: AnyUrl = Field(default="http://localhost:4566")
    QDRANT_URL: AnyUrl = Field(default="http://localhost:6333")
    QDRANT_COLLECTION: str = Field(default="aws_finops_rules")
    LANGFUSE_PUBLIC_KEY: SecretStr | None = Field(default=None)
    LANGFUSE_SECRET_KEY: SecretStr | None = Field(default=None)
    LANGFUSE_HOST: AnyUrl = Field(default="http://localhost:3000")
    LLM_MODEL_NAME: str = Field(default="gpt-4o-mini")
    OPENAI_API_KEY: SecretStr | None = Field(default=None)

    @field_validator("ENVIRONMENT", mode="before")
    @classmethod
    def normalize_environment(cls, value: object) -> object:
        """Accept common local aliases as dev mode."""

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"local", "development"}:
                return Environment.DEV
            if normalized in {"production"}:
                return Environment.PROD
            return normalized
        return value

    @property
    def is_dev(self) -> bool:
        """Return whether the app should target LocalStack-compatible endpoints."""

        return self.ENVIRONMENT == Environment.DEV

    def get_aws_client_kwargs(self) -> dict[str, Any]:
        """Return kwargs used by boto3/aioboto3 clients.

        In dev mode the clients are redirected to LocalStack with deterministic fake
        credentials. In prod mode only the region is passed so the standard AWS credential
        provider chain can be used.
        """

        if self.is_dev:
            return {
                "endpoint_url": str(self.LOCALSTACK_ENDPOINT_URL),
                "aws_access_key_id": "test",
                "aws_secret_access_key": "test",
                "region_name": self.AWS_REGION,
            }
        return {"region_name": self.AWS_REGION}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
