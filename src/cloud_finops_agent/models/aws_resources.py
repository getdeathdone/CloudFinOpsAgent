"""Pydantic contracts for AWS resources discovered by the Discovery agent."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class AWSResourceType(StrEnum):
    """Supported AWS resource families in the FinOps audit scope."""

    S3_BUCKET = "s3_bucket"
    EC2_INSTANCE = "ec2_instance"
    CLOUDWATCH_METRIC = "cloudwatch_metric"


class EC2InstanceState(StrEnum):
    """Normalized EC2 instance states returned by AWS APIs."""

    PENDING = "pending"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting-down"
    TERMINATED = "terminated"
    STOPPING = "stopping"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class CloudWatchStatistic(StrEnum):
    """CloudWatch statistics used for metric datapoints."""

    AVERAGE = "Average"
    MINIMUM = "Minimum"
    MAXIMUM = "Maximum"
    SUM = "Sum"
    SAMPLE_COUNT = "SampleCount"


class CloudWatchDatapointModel(BaseModel):
    """Single normalized CloudWatch datapoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: datetime = Field(description="Datapoint timestamp in UTC.")
    value: float = Field(description="Metric value for the selected statistic.")
    unit: str | None = Field(default=None, description="CloudWatch unit name, if provided.")

    @field_validator("timestamp")
    @classmethod
    def ensure_utc_timestamp(cls, value: datetime) -> datetime:
        """Normalize naive and timezone-aware datetimes to UTC."""

        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class AWSResourceModel(BaseModel):
    """Base contract shared by all discovered AWS resources."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    resource_id: str = Field(
        validation_alias=AliasChoices("resource_id", "ResourceId", "id"),
        min_length=1,
        description="Stable AWS resource identifier used across graph nodes.",
    )
    launch_time: datetime = Field(
        validation_alias=AliasChoices("launch_time", "LaunchTime", "CreationDate"),
        description="Creation or first-observed timestamp normalized to UTC.",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Normalized AWS tags represented as a string dictionary.",
    )
    estimated_monthly_cost: Decimal = Field(
        default=Decimal("0"),
        validation_alias=AliasChoices("estimated_monthly_cost", "EstimatedMonthlyCost"),
        ge=Decimal("0"),
        description="Estimated monthly cost in USD.",
    )

    @field_validator("launch_time")
    @classmethod
    def ensure_utc_launch_time(cls, value: datetime) -> datetime:
        """Normalize resource timestamps to UTC for deterministic comparisons."""

        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: Any) -> dict[str, str]:
        """Accept AWS tag lists and normalize them into a plain dictionary."""

        if value is None:
            return {}
        if isinstance(value, dict):
            return {str(key): str(tag_value) for key, tag_value in value.items()}
        if isinstance(value, list):
            normalized: dict[str, str] = {}
            for item in value:
                if isinstance(item, dict) and "Key" in item and "Value" in item:
                    normalized[str(item["Key"])] = str(item["Value"])
                elif isinstance(item, dict) and "key" in item and "value" in item:
                    normalized[str(item["key"])] = str(item["value"])
                else:
                    raise ValueError("AWS tag lists must contain Key/Value dictionaries.")
            return normalized
        raise TypeError("tags must be a dictionary or an AWS-style list of tag objects.")

    @field_validator("estimated_monthly_cost")
    @classmethod
    def quantize_estimated_monthly_cost(cls, value: Decimal) -> Decimal:
        """Normalize cost precision for deterministic downstream analysis."""

        return value.quantize(Decimal("0.0001"))


class S3BucketModel(AWSResourceModel):
    """Discovered S3 bucket with storage and lifecycle metadata."""

    resource_type: AWSResourceType = Field(default=AWSResourceType.S3_BUCKET, frozen=True)
    bucket_name: str = Field(
        validation_alias=AliasChoices("bucket_name", "Name", "BucketName"),
        min_length=3,
        description="S3 bucket name.",
    )
    region: str = Field(default="us-east-1", min_length=1, description="Bucket region.")
    size_bytes: int = Field(default=0, ge=0, description="Total bucket size in bytes.")
    object_count: int = Field(default=0, ge=0, description="Number of objects in the bucket.")
    versioning_enabled: bool = Field(default=False, description="Whether bucket versioning is enabled.")
    encryption_enabled: bool = Field(default=False, description="Whether default encryption is enabled.")
    public_access_blocked: bool = Field(
        default=True,
        description="Whether public access is blocked by bucket-level controls.",
    )
    storage_class_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Storage class to byte-size mapping.",
    )

    @field_validator("resource_id", mode="before")
    @classmethod
    def default_resource_id_from_bucket_name(cls, value: Any) -> Any:
        """Keep explicit resource identifiers intact."""

        return value


class EC2InstanceModel(AWSResourceModel):
    """Discovered EC2 instance with sizing, state, and utilization context."""

    resource_type: AWSResourceType = Field(default=AWSResourceType.EC2_INSTANCE, frozen=True)
    instance_id: str = Field(
        validation_alias=AliasChoices("instance_id", "InstanceId"),
        min_length=1,
        description="EC2 instance identifier.",
    )
    instance_type: str = Field(
        validation_alias=AliasChoices("instance_type", "InstanceType"),
        min_length=1,
        description="EC2 instance type, for example t3.micro.",
    )
    state: EC2InstanceState = Field(
        default=EC2InstanceState.UNKNOWN,
        validation_alias=AliasChoices("state", "State"),
        description="Normalized EC2 lifecycle state.",
    )
    availability_zone: str | None = Field(
        default=None,
        validation_alias=AliasChoices("availability_zone", "AvailabilityZone"),
        description="Availability zone where the instance runs.",
    )
    vpc_id: str | None = Field(default=None, validation_alias=AliasChoices("vpc_id", "VpcId"))
    subnet_id: str | None = Field(default=None, validation_alias=AliasChoices("subnet_id", "SubnetId"))
    private_ip_address: str | None = Field(
        default=None,
        validation_alias=AliasChoices("private_ip_address", "PrivateIpAddress"),
    )
    public_ip_address: str | None = Field(
        default=None,
        validation_alias=AliasChoices("public_ip_address", "PublicIpAddress"),
    )
    cpu_average_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Average CPU utilization percentage over the discovery window.",
    )
    network_in_bytes: int | None = Field(default=None, ge=0)
    network_out_bytes: int | None = Field(default=None, ge=0)

    @field_validator("state", mode="before")
    @classmethod
    def normalize_state(cls, value: Any) -> EC2InstanceState | str:
        """Accept raw EC2 State objects and normalize to the state name."""

        if isinstance(value, dict):
            name = value.get("Name") or value.get("name")
            return str(name).lower() if name else EC2InstanceState.UNKNOWN
        if value is None:
            return EC2InstanceState.UNKNOWN
        return str(value).lower()


class CloudWatchMetricModel(AWSResourceModel):
    """Discovered CloudWatch metric bound to an AWS resource or namespace."""

    resource_type: AWSResourceType = Field(default=AWSResourceType.CLOUDWATCH_METRIC, frozen=True)
    namespace: str = Field(
        validation_alias=AliasChoices("namespace", "Namespace"),
        min_length=1,
        description="CloudWatch metric namespace.",
    )
    metric_name: str = Field(
        validation_alias=AliasChoices("metric_name", "MetricName"),
        min_length=1,
        description="CloudWatch metric name.",
    )
    dimensions: dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("dimensions", "Dimensions"),
        description="Metric dimensions normalized to a string dictionary.",
    )
    statistic: CloudWatchStatistic = Field(default=CloudWatchStatistic.AVERAGE)
    period_seconds: int = Field(default=300, gt=0, description="Metric period in seconds.")
    unit: str | None = Field(default=None, description="CloudWatch unit name.")
    datapoints: list[CloudWatchDatapointModel] = Field(default_factory=list)

    @field_validator("dimensions", mode="before")
    @classmethod
    def normalize_dimensions(cls, value: Any) -> dict[str, str]:
        """Accept CloudWatch dimension lists and normalize them into a dictionary."""

        if value is None:
            return {}
        if isinstance(value, dict):
            return {str(key): str(dimension_value) for key, dimension_value in value.items()}
        if isinstance(value, list):
            normalized: dict[str, str] = {}
            for item in value:
                if isinstance(item, dict) and "Name" in item and "Value" in item:
                    normalized[str(item["Name"])] = str(item["Value"])
                elif isinstance(item, dict) and "name" in item and "value" in item:
                    normalized[str(item["name"])] = str(item["value"])
                else:
                    raise ValueError("CloudWatch dimensions must contain Name/Value dictionaries.")
            return normalized
        raise TypeError("dimensions must be a dictionary or a CloudWatch-style dimension list.")


class AWSInfrastructureSnapshot(BaseModel):
    """Point-in-time inventory collected by the Discovery agent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    collected_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the inventory snapshot was collected.",
    )
    account_id: str | None = Field(default=None, description="AWS account identifier if available.")
    region: str = Field(default="us-east-1", min_length=1)
    s3_buckets: list[S3BucketModel] = Field(default_factory=list)
    ec2_instances: list[EC2InstanceModel] = Field(default_factory=list)
    cloudwatch_metrics: list[CloudWatchMetricModel] = Field(default_factory=list)

    @property
    def total_estimated_monthly_cost(self) -> Decimal:
        """Return the aggregated monthly cost for all discovered billable resources."""

        resources: list[AWSResourceModel] = [*self.s3_buckets, *self.ec2_instances]
        return sum((resource.estimated_monthly_cost for resource in resources), Decimal("0"))

    @property
    def resource_count(self) -> int:
        """Return the total number of resources and metrics in the snapshot."""

        return len(self.s3_buckets) + len(self.ec2_instances) + len(self.cloudwatch_metrics)

    @field_validator("collected_at")
    @classmethod
    def ensure_utc_collected_at(cls, value: datetime) -> datetime:
        """Normalize snapshot timestamps to UTC."""

        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
