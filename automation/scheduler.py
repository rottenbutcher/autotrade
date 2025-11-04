"""Blocking scheduler orchestrating the automated trading workflow."""
from __future__ import annotations

import json
import os
import subprocess
import traceback
from datetime import datetime

import boto3
from apscheduler.schedulers.blocking import BlockingScheduler

from aws_automation.ec2_self_shutdown import stop_instance as stop_ec2_instance
from aws_automation.s3_backup import backup_all, upload_file
from utils.backtester import backtest, send_email_report


def run(script: str, *args: str) -> None:
    """Execute a python script with optional CLI arguments."""

    command = ["python", script, *args]
    print(f"▶ {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"⚠️ Error running {' '.join(command)}: {exc}")
        traceback.print_exc()
        raise


def pretrained_exists() -> bool:
    """Check for the pretrained encoder locally or in S3."""

    local_path = os.path.join("models", "dual_mae_encoder.pth")
    if os.path.exists(local_path):
        print("✅ Found local pretrained model.")
        return True

    try:
        with open("aws_automation/config.json", "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        s3 = boto3.client("s3", region_name=cfg["REGION"])
        objs = s3.list_objects_v2(
            Bucket=cfg["S3_BUCKET"], Prefix="backups/latest/dual_mae_encoder.pth"
        )
        if "Contents" in objs and objs.get("KeyCount", 0) > 0:
            print("✅ Found pretrained model in S3. Downloading...")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(
                cfg["S3_BUCKET"], "backups/latest/dual_mae_encoder.pth", local_path
            )
            return True
    except FileNotFoundError:
        print("⚠️ AWS config missing; skipping S3 pretrained lookup.")
    except Exception as exc:  # pragma: no cover - depends on AWS env
        print(f"⚠️ Error checking S3: {exc}")

    print("❌ No pretrained model found.")
    return False


def daily_update() -> None:
    print(f"=== 🚀 Daily Update Started at {datetime.now()} ===")

    try:
        run("data_pipeline/data_loader.py")

        if not pretrained_exists():
            print("🧠 No pretrained model detected → Running full pretraining once...")
            run("train/train_dual_mae.py")
            try:
                upload_file("models/dual_mae_encoder.pth", "backups/latest/dual_mae_encoder.pth")
            except Exception as exc:  # pragma: no cover - depends on AWS env
                print(f"⚠️ Failed to upload pretrained model to 'latest': {exc}")
        else:
            print("⚡ Pretrained model exists → Skipping pretrain and running fine-tuning only.")

        run("train/finetune_dual_predictor.py")
        run("bot/upbit_bot.py", "trade")

        if datetime.now().weekday() == 6:
            total, sharpe, image_path = backtest()
            if total is not None and sharpe is not None:
                recipient = os.getenv("REPORT_EMAIL_TO", "your_email@gmail.com")
                send_email_report(recipient, total, sharpe, image_path)
    finally:
        try:
            backup_all()
        except Exception as exc:  # pragma: no cover - depends on AWS env
            print(f"⚠️ Backup failed: {exc}")
        stop_ec2_instance()
        print("✅ Daily pipeline complete. Shutdown initiated.")


def trade_morning() -> None:
    run("bot/upbit_bot.py", "trade")


def sell_evening() -> None:
    run("bot/upbit_bot.py", "sell")


def weekly_report() -> None:
    total, sharpe, image_path = backtest()
    if total is not None and sharpe is not None:
        recipient = os.getenv("REPORT_EMAIL_TO", "your_email@gmail.com")
        send_email_report(recipient, total, sharpe, image_path)


def start() -> None:
    scheduler = BlockingScheduler()
    scheduler.add_job(daily_update, "cron", hour=8, minute=59)
    scheduler.add_job(sell_evening, "cron", hour=8, minute=50)
    print("📆 Scheduler started. Waiting for events…")
    scheduler.start()


if __name__ == "__main__":
    start()
