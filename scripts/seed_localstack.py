"""Seed LocalStack with sample AWS resources for the Cloud FinOps audit."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta

import boto3
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LOCALSTACK_ENDPOINT_URL = os.getenv("LOCALSTACK_ENDPOINT_URL", "http://localhost:4566")
AWS_KWARGS = {
    "endpoint_url": LOCALSTACK_ENDPOINT_URL,
    "aws_access_key_id": "test",
    "aws_secret_access_key": "test",
    "region_name": AWS_REGION,
}


def client(service_name: str):
    """Create a boto3 client configured for LocalStack."""

    return boto3.client(service_name, **AWS_KWARGS)


def create_bucket_if_missing(bucket_name: str) -> None:
    """Create an S3 bucket unless it already exists."""

    s3 = client("s3")
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"S3 bucket already exists: {bucket_name}")
        return
    except ClientError:
        pass

    if AWS_REGION == "us-east-1":
        s3.create_bucket(Bucket=bucket_name)
    else:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
        )
    print(f"Created S3 bucket: {bucket_name}")


def seed_s3() -> None:
    """Create sample S3 buckets and objects."""

    s3 = client("s3")
    empty_bucket = "finops-empty-untagged-bucket"
    large_bucket = "finops-versioned-large-bucket"

    create_bucket_if_missing(empty_bucket)
    create_bucket_if_missing(large_bucket)

    s3.put_bucket_versioning(
        Bucket=large_bucket,
        VersioningConfiguration={"Status": "Enabled"},
    )
    s3.put_bucket_tagging(
        Bucket=large_bucket,
        Tagging={
            "TagSet": [
                {"Key": "Environment", "Value": "dev"},
                {"Key": "Owner", "Value": "platform-team"},
                {"Key": "CostCenter", "Value": "sandbox"},
            ]
        },
    )

    payload = b"x" * 1024 * 1024
    for index in range(8):
        s3.put_object(
            Bucket=large_bucket,
            Key=f"raw-logs/year=2026/month=06/object-{index}.bin",
            Body=payload,
            StorageClass="STANDARD",
        )
    print(f"Seeded S3 objects in {large_bucket}")


def seed_ec2() -> list[str]:
    """Create sample EC2 instances in LocalStack."""

    ec2 = client("ec2")
    response = ec2.run_instances(
        ImageId="ami-12345678",
        InstanceType="t3.micro",
        MinCount=1,
        MaxCount=2,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Environment", "Value": "dev"},
                    {"Key": "Owner", "Value": "finops-demo"},
                    {"Key": "Workload", "Value": "idle-sandbox"},
                ],
            }
        ],
    )
    instance_ids = [instance["InstanceId"] for instance in response["Instances"]]
    print(f"Created EC2 instances: {', '.join(instance_ids)}")
    return instance_ids


def seed_cloudwatch_cpu(instance_ids: list[str]) -> None:
    """Publish low CPU metrics so the Analyst can detect idle EC2 resources."""

    cloudwatch = client("cloudwatch")
    now = datetime.now(UTC)
    for instance_id in instance_ids:
        metric_data = [
            {
                "MetricName": "CPUUtilization",
                "Dimensions": [{"Name": "InstanceId", "Value": instance_id}],
                "Timestamp": now - timedelta(days=day),
                "Value": 1.5,
                "Unit": "Percent",
            }
            for day in range(7)
        ]
        cloudwatch.put_metric_data(
            Namespace="AWS/EC2",
            MetricData=metric_data,
        )
    print("Seeded CloudWatch CPUUtilization metrics")


def main() -> None:
    """Seed LocalStack resources for an end-to-end demo."""

    print(f"Seeding LocalStack at {LOCALSTACK_ENDPOINT_URL}")
    seed_s3()
    instance_ids = seed_ec2()
    time.sleep(1)
    seed_cloudwatch_cpu(instance_ids)
    print("LocalStack seed completed")


if __name__ == "__main__":
    main()
