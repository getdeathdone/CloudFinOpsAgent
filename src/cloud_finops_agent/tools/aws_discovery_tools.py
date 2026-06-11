"""LangChain tools used by Agent-Discovery to collect AWS infrastructure data."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import structlog
from botocore.exceptions import ClientError
from langchain_core.tools import tool
from pydantic import TypeAdapter, ValidationError

from cloud_finops_agent.config.settings import get_settings
from cloud_finops_agent.infrastructure.aws_factory import get_aws_client_factory
from cloud_finops_agent.models.aws_resources import EC2InstanceModel, S3BucketModel

logger = structlog.get_logger(__name__)

EC2_MONTHLY_COST_BY_TYPE: dict[str, Decimal] = {
    "t2.micro": Decimal("9.50"),
    "t2.small": Decimal("19.00"),
    "t2.medium": Decimal("38.00"),
    "t3.nano": Decimal("5.00"),
    "t3.micro": Decimal("10.00"),
    "t3.small": Decimal("20.00"),
    "t3.medium": Decimal("40.00"),
    "t3.large": Decimal("80.00"),
    "m5.large": Decimal("70.00"),
    "m5.xlarge": Decimal("140.00"),
}

DEFAULT_EC2_MONTHLY_COST = Decimal("25.00")
S3_STORAGE_COST_PER_GB_MONTH = Decimal("0.023")
EC2_LIST_ADAPTER: TypeAdapter[list[EC2InstanceModel]] = TypeAdapter(list[EC2InstanceModel])
S3_LIST_ADAPTER: TypeAdapter[list[S3BucketModel]] = TypeAdapter(list[S3BucketModel])


def _estimate_ec2_monthly_cost(instance_type: str) -> Decimal:
    """Return a deterministic local FinOps estimate for an EC2 instance type."""

    return EC2_MONTHLY_COST_BY_TYPE.get(instance_type, DEFAULT_EC2_MONTHLY_COST)


def _estimate_s3_monthly_cost(size_bytes: int) -> Decimal:
    """Return an approximate S3 standard storage monthly cost."""

    size_gb = Decimal(size_bytes) / Decimal(1024**3)
    return (size_gb * S3_STORAGE_COST_PER_GB_MONTH).quantize(Decimal("0.0001"))


async def _get_ec2_cpu_average_percent(
    *,
    cloudwatch_client: Any,
    instance_id: str,
    now: datetime | None = None,
) -> float | None:
    """Fetch average EC2 CPU utilization for the last seven days."""

    end_time = now or datetime.now(UTC)
    start_time = end_time - timedelta(days=7)
    try:
        response = await cloudwatch_client.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName="CPUUtilization",
            Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=86400,
            Statistics=["Average"],
        )
    except ClientError as exc:
        logger.warning(
            "cloudwatch_cpu_metric_failed",
            instance_id=instance_id,
            error_code=exc.response.get("Error", {}).get("Code"),
            error_message=exc.response.get("Error", {}).get("Message"),
        )
        return None

    datapoints = response.get("Datapoints", [])
    averages = [float(point["Average"]) for point in datapoints if "Average" in point]
    if not averages:
        return None
    return round(sum(averages) / len(averages), 4)


def _extract_ec2_instances(reservations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten EC2 reservations into raw instance dictionaries."""

    instances: list[dict[str, Any]] = []
    for reservation in reservations:
        instances.extend(reservation.get("Instances", []))
    return instances


@tool
async def discover_ec2_instances() -> list[dict[str, Any]]:
    """Discover EC2 instances and enrich them with cost and CPU utilization data."""

    settings = get_settings()
    factory = get_aws_client_factory(settings)
    discovered: list[EC2InstanceModel] = []

    try:
        async with factory.ec2_client() as ec2_client, factory.cloudwatch_client() as cloudwatch_client:
            paginator = ec2_client.get_paginator("describe_instances")
            async for page in paginator.paginate():
                raw_instances = _extract_ec2_instances(page.get("Reservations", []))
                for raw_instance in raw_instances:
                    instance_id = str(raw_instance["InstanceId"])
                    instance_type = str(raw_instance.get("InstanceType", "unknown"))
                    cpu_average = await _get_ec2_cpu_average_percent(
                        cloudwatch_client=cloudwatch_client,
                        instance_id=instance_id,
                    )
                    placement = raw_instance.get("Placement", {})
                    model = EC2InstanceModel(
                        resource_id=instance_id,
                        instance_id=instance_id,
                        instance_type=instance_type,
                        state=raw_instance.get("State"),
                        launch_time=raw_instance.get("LaunchTime", datetime.now(UTC)),
                        tags=raw_instance.get("Tags", []),
                        estimated_monthly_cost=_estimate_ec2_monthly_cost(instance_type),
                        availability_zone=placement.get("AvailabilityZone"),
                        vpc_id=raw_instance.get("VpcId"),
                        subnet_id=raw_instance.get("SubnetId"),
                        private_ip_address=raw_instance.get("PrivateIpAddress"),
                        public_ip_address=raw_instance.get("PublicIpAddress"),
                        cpu_average_percent=cpu_average,
                    )
                    discovered.append(model)
    except ClientError as exc:
        logger.exception(
            "discover_ec2_instances_aws_api_failed",
            error_code=exc.response.get("Error", {}).get("Code"),
            error_message=exc.response.get("Error", {}).get("Message"),
        )
        raise
    except ValidationError:
        logger.exception("discover_ec2_instances_validation_failed")
        raise

    logger.info("discover_ec2_instances_completed", count=len(discovered))
    return EC2_LIST_ADAPTER.dump_python(discovered, mode="json")


async def _get_bucket_region(*, s3_client: Any, bucket_name: str, default_region: str) -> str:
    """Fetch and normalize S3 bucket region."""

    try:
        response = await s3_client.get_bucket_location(Bucket=bucket_name)
    except ClientError as exc:
        logger.warning(
            "s3_bucket_region_failed",
            bucket_name=bucket_name,
            error_code=exc.response.get("Error", {}).get("Code"),
        )
        return default_region

    location = response.get("LocationConstraint")
    if location in {None, ""}:
        return "us-east-1"
    return str(location)


async def _get_bucket_versioning_enabled(*, s3_client: Any, bucket_name: str) -> bool:
    """Return whether S3 bucket versioning is enabled."""

    try:
        response = await s3_client.get_bucket_versioning(Bucket=bucket_name)
    except ClientError as exc:
        logger.warning(
            "s3_bucket_versioning_failed",
            bucket_name=bucket_name,
            error_code=exc.response.get("Error", {}).get("Code"),
        )
        return False
    return response.get("Status") == "Enabled"


async def _get_bucket_encryption_enabled(*, s3_client: Any, bucket_name: str) -> bool:
    """Return whether S3 bucket default encryption is configured."""

    try:
        await s3_client.get_bucket_encryption(Bucket=bucket_name)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"ServerSideEncryptionConfigurationNotFoundError", "NoSuchBucket"}:
            return False
        logger.warning("s3_bucket_encryption_failed", bucket_name=bucket_name, error_code=error_code)
        return False
    return True


async def _get_bucket_public_access_blocked(*, s3_client: Any, bucket_name: str) -> bool:
    """Return whether all S3 public access block switches are enabled."""

    try:
        response = await s3_client.get_public_access_block(Bucket=bucket_name)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"NoSuchPublicAccessBlockConfiguration", "NoSuchBucket"}:
            return False
        logger.warning(
            "s3_bucket_public_access_block_failed",
            bucket_name=bucket_name,
            error_code=error_code,
        )
        return False

    config = response.get("PublicAccessBlockConfiguration", {})
    return all(
        bool(config.get(key))
        for key in (
            "BlockPublicAcls",
            "IgnorePublicAcls",
            "BlockPublicPolicy",
            "RestrictPublicBuckets",
        )
    )


async def _get_bucket_tags(*, s3_client: Any, bucket_name: str) -> dict[str, str]:
    """Fetch and normalize S3 bucket tags."""

    try:
        response = await s3_client.get_bucket_tagging(Bucket=bucket_name)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"NoSuchTagSet", "NoSuchBucket"}:
            return {}
        logger.warning("s3_bucket_tags_failed", bucket_name=bucket_name, error_code=error_code)
        return {}

    return {
        str(tag["Key"]): str(tag["Value"])
        for tag in response.get("TagSet", [])
        if "Key" in tag and "Value" in tag
    }


async def _get_bucket_object_stats(*, s3_client: Any, bucket_name: str) -> tuple[int, int, dict[str, int]]:
    """Calculate object count, total size, and storage class breakdown for a bucket."""

    object_count = 0
    size_bytes = 0
    storage_class_breakdown: dict[str, int] = {}

    paginator = s3_client.get_paginator("list_objects_v2")
    async for page in paginator.paginate(Bucket=bucket_name):
        for item in page.get("Contents", []):
            object_count += 1
            object_size = int(item.get("Size", 0))
            size_bytes += object_size
            storage_class = str(item.get("StorageClass", "STANDARD"))
            storage_class_breakdown[storage_class] = (
                storage_class_breakdown.get(storage_class, 0) + object_size
            )

    return object_count, size_bytes, storage_class_breakdown


@tool
async def discover_s3_buckets() -> list[dict[str, Any]]:
    """Discover S3 buckets and enrich them with storage, versioning, and cost data."""

    settings = get_settings()
    factory = get_aws_client_factory(settings)
    discovered: list[S3BucketModel] = []

    try:
        async with factory.s3_client() as s3_client:
            response = await s3_client.list_buckets()
            for raw_bucket in response.get("Buckets", []):
                bucket_name = str(raw_bucket["Name"])
                object_count, size_bytes, storage_class_breakdown = await _get_bucket_object_stats(
                    s3_client=s3_client,
                    bucket_name=bucket_name,
                )
                model = S3BucketModel(
                    resource_id=bucket_name,
                    bucket_name=bucket_name,
                    launch_time=raw_bucket.get("CreationDate", datetime.now(UTC)),
                    tags=await _get_bucket_tags(s3_client=s3_client, bucket_name=bucket_name),
                    estimated_monthly_cost=_estimate_s3_monthly_cost(size_bytes),
                    region=await _get_bucket_region(
                        s3_client=s3_client,
                        bucket_name=bucket_name,
                        default_region=settings.AWS_REGION,
                    ),
                    size_bytes=size_bytes,
                    object_count=object_count,
                    versioning_enabled=await _get_bucket_versioning_enabled(
                        s3_client=s3_client,
                        bucket_name=bucket_name,
                    ),
                    encryption_enabled=await _get_bucket_encryption_enabled(
                        s3_client=s3_client,
                        bucket_name=bucket_name,
                    ),
                    public_access_blocked=await _get_bucket_public_access_blocked(
                        s3_client=s3_client,
                        bucket_name=bucket_name,
                    ),
                    storage_class_breakdown=storage_class_breakdown,
                )
                discovered.append(model)
    except ClientError as exc:
        logger.exception(
            "discover_s3_buckets_aws_api_failed",
            error_code=exc.response.get("Error", {}).get("Code"),
            error_message=exc.response.get("Error", {}).get("Message"),
        )
        raise
    except ValidationError:
        logger.exception("discover_s3_buckets_validation_failed")
        raise

    logger.info("discover_s3_buckets_completed", count=len(discovered))
    return S3_LIST_ADAPTER.dump_python(discovered, mode="json")
