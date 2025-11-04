"""AWS Lambda to start the trading EC2 instance."""
import os

import boto3

REGION = os.getenv("AWS_REGION", "ap-northeast-2")
INSTANCE_ID = os.getenv("INSTANCE_ID", "i-0abcd1234ef56789")

ec2 = boto3.client("ec2", region_name=REGION)


def lambda_handler(event, context):  # pragma: no cover - AWS runtime
    ec2.start_instances(InstanceIds=[INSTANCE_ID])
    return {"status": "EC2 started", "instance_id": INSTANCE_ID}
