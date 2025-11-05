"""Utilities for archiving artifacts to S3."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Iterable, Optional

import boto3

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

_CACHED_CONFIG: Optional[dict] = None
_S3_CLIENT: Optional["boto3.client"] = None


def _load_config() -> dict:
    """Return the cached AWS automation config."""

    global _CACHED_CONFIG
    if _CACHED_CONFIG is None:
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(
                "AWS automation config not found. Please create aws_automation/config.json"
            )
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            _CACHED_CONFIG = json.load(handle)
    return _CACHED_CONFIG


def _get_client():
    """Return a cached boto3 S3 client."""

    global _S3_CLIENT
    if _S3_CLIENT is None:
        cfg = _load_config()
        _S3_CLIENT = boto3.client("s3", region_name=cfg["REGION"])
    return _S3_CLIENT


def _iter_files(path: str) -> Iterable[str]:
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for filename in files:
                yield os.path.join(root, filename)
    elif os.path.isfile(path):
        yield path
    else:
        print(f"ℹ️ Skipping missing path: {path}")


def upload_file(local_path: str, s3_path: str) -> None:
    """Upload a single file to the configured S3 bucket."""

    cfg = _load_config()
    client = _get_client()
    if not os.path.exists(local_path):
        print(f"ℹ️ Skipping missing file: {local_path}")
        return
    try:
        client.upload_file(local_path, cfg["S3_BUCKET"], s3_path)
        print(f"✅ Uploaded: {local_path} → s3://{cfg['S3_BUCKET']}/{s3_path}")
    except Exception as exc:  # pragma: no cover - depends on AWS env
        print(f"⚠️ Failed to upload {local_path}: {exc}")


def backup_all() -> None:
    """Upload the configured artifacts to their dated backup prefix."""

    cfg = _load_config()
    prefix = f"backups/{datetime.now().strftime('%Y%m%d')}"

    for path in cfg.get("BACKUP_PATHS", []):
        for file_path in _iter_files(path):
            rel_path = os.path.relpath(file_path, start=os.getcwd())
            upload_file(file_path, f"{prefix}/{rel_path}")


if __name__ == "__main__":
    backup_all()
