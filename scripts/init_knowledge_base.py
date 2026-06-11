"""Initialize the Qdrant knowledge base with baseline AWS FinOps rules."""

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cloud_finops_agent.config.settings import get_settings  # noqa: E402

logger = structlog.get_logger(__name__)

DEFAULT_VECTOR_SIZE = 384


@dataclass(frozen=True)
class FinOpsRule:
    """Seed FinOps rule inserted into Qdrant."""

    rule_id: str
    title: str
    text: str
    category: str
    source: str = "AWS Well-Architected Framework"


class AsyncEmbedder(Protocol):
    """Protocol for asynchronous text embedders."""

    @property
    def vector_size(self) -> int:
        """Return embedding vector size."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""


class DeterministicHashEmbedder:
    """Small dependency-free fallback embedder for local development without API keys."""

    vector_size = DEFAULT_VECTOR_SIZE

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create deterministic normalized vectors from SHA-256 token hashes."""

        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.vector_size
        tokens = [token for token in text.lower().replace(".", " ").replace(",", " ").split() if token]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % self.vector_size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OpenAIAsyncEmbedder:
    """OpenAI embedder wrapper used when OPENAI_API_KEY is configured."""

    vector_size = 1536

    def __init__(self) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with OpenAI via LangChain's async API."""

        return await self._embedder.aembed_documents(texts)


def build_seed_rules() -> list[FinOpsRule]:
    """Return baseline FinOps optimization rules for the local knowledge base."""

    return [
        FinOpsRule(
            rule_id="ec2-idle-cpu-7d",
            title="Stop or downsize idle EC2 instances",
            category="compute",
            text=(
                "EC2 instances with CPU Utilization below 5% for 7 consecutive days are "
                "considered idle and should be stopped, scheduled, or downsized after owner validation."
            ),
        ),
        FinOpsRule(
            rule_id="s3-versioning-lifecycle",
            title="Add lifecycle policies for versioned S3 buckets",
            category="storage",
            text=(
                "S3 buckets with Versioning enabled but without a Lifecycle Policy for deleting old "
                "noncurrent versions generate hidden storage costs and should receive expiration rules."
            ),
        ),
        FinOpsRule(
            rule_id="s3-storage-class-transition",
            title="Transition cold S3 data to cheaper storage classes",
            category="storage",
            text=(
                "S3 objects that are rarely accessed should be transitioned from Standard storage to "
                "Standard-IA, One Zone-IA, Glacier Instant Retrieval, or Glacier Flexible Retrieval "
                "based on access and recovery requirements."
            ),
        ),
        FinOpsRule(
            rule_id="ec2-rightsize-overprovisioned",
            title="Rightsize overprovisioned EC2 instances",
            category="compute",
            text=(
                "EC2 instances with consistently low CPU, network, and memory utilization should be "
                "rightsized to a smaller instance family or size to reduce monthly compute spend."
            ),
        ),
    ]


def select_embedder() -> AsyncEmbedder:
    """Select OpenAI embeddings when configured, otherwise use deterministic local vectors."""

    settings = get_settings()
    if settings.OPENAI_API_KEY is not None and settings.OPENAI_API_KEY.get_secret_value():
        logger.info("using_openai_embeddings", model="text-embedding-3-small")
        return OpenAIAsyncEmbedder()

    logger.info("using_deterministic_hash_embeddings", vector_size=DEFAULT_VECTOR_SIZE)
    return DeterministicHashEmbedder()


async def recreate_collection(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """Create a fresh Qdrant collection for AWS FinOps rules."""

    exists = await client.collection_exists(collection_name)
    if exists:
        await client.delete_collection(collection_name)

    await client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


async def initialize_knowledge_base() -> None:
    """Create the Qdrant collection and upload seed FinOps rules."""

    settings = get_settings()
    embedder = select_embedder()
    rules = build_seed_rules()
    texts = [rule.text for rule in rules]
    vectors = await embedder.embed_documents(texts)
    collection_name = settings.QDRANT_COLLECTION

    client = AsyncQdrantClient(url=str(settings.QDRANT_URL))
    try:
        await recreate_collection(
            client=client,
            collection_name=collection_name,
            vector_size=embedder.vector_size,
        )
        points = [
            qdrant_models.PointStruct(
                id=index + 1,
                vector=vector,
                payload={
                    "rule_id": rule.rule_id,
                    "title": rule.title,
                    "text": rule.text,
                    "category": rule.category,
                    "source": rule.source,
                },
            )
            for index, (rule, vector) in enumerate(zip(rules, vectors, strict=True))
        ]
        await client.upsert(collection_name=collection_name, points=points)
    except Exception:
        logger.exception(
            "knowledge_base_initialization_failed",
            collection_name=collection_name,
            qdrant_url=str(settings.QDRANT_URL),
        )
        raise
    finally:
        await client.close()

    logger.info(
        "knowledge_base_initialized",
        collection_name=collection_name,
        rule_count=len(rules),
        vector_size=embedder.vector_size,
    )


def main() -> None:
    """Script entrypoint."""

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )
    asyncio.run(initialize_knowledge_base())


if __name__ == "__main__":
    main()
