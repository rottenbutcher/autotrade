"""AWS Lambda to stop the trading EC2 instance."""
import os

import boto3

REGION = os.getenv("AWS_REGION", "ap-northeast-2")
INSTANCE_ID = os.getenv("INSTANCE_ID", "i-0abcd1234ef56789")

ec2 = boto3.client("ec2", region_name=REGION)


def lambda_handler(event, context):  # pragma: no cover - AWS runtime
    ec2.stop_instances(InstanceIds=[INSTANCE_ID])
    return {"status": "EC2 stopped", "instance_id": INSTANCE_ID}
