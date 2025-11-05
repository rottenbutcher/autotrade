"""Utility to stop the EC2 instance after work is complete."""
from __future__ import annotations

import json
import os

import boto3

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def _load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            "AWS automation config not found. Please create aws_automation/config.json"
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def stop_instance() -> None:
    cfg = _load_config()
    client = boto3.client("ec2", region_name=cfg["REGION"])
    instance_id = cfg["INSTANCE_ID"]
    print("🛑 Stopping EC2 instance to save cost…")
    try:
        client.stop_instances(InstanceIds=[instance_id])
        print("✅ Instance stop command sent.")
    except Exception as exc:  # pragma: no cover - depends on AWS env
        print(f"⚠️ Failed to stop instance {instance_id}: {exc}")


if __name__ == "__main__":
    stop_instance()
