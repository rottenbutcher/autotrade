"""Blocking scheduler orchestrating the automated trading workflow."""
from __future__ import annotations

import os
import subprocess
import traceback
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler

from utils.backtester import backtest, send_email_report


def _run_script(script: str, *args: str) -> None:
    command = ["python", script, *args]
    print(f"▶ {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"⚠️ Error running {' '.join(command)}: {exc}")
        traceback.print_exc()


def daily_update() -> None:
    print(f"\n=== {datetime.now()} Daily Update ===")
    _run_script("data_pipeline/data_loader.py")
    _run_script("train/train_dual_mae.py")
    _run_script("train/finetune_dual_predictor.py")
    print("✅ Retrain complete")


def trade_morning() -> None:
    _run_script("bot/upbit_bot.py", "trade")


def sell_evening() -> None:
    _run_script("bot/upbit_bot.py", "sell")


def weekly_report() -> None:
    total, sharpe, image_path = backtest()
    if total is not None and sharpe is not None:
        recipient = os.getenv("REPORT_EMAIL_TO", "your_email@gmail.com")
        send_email_report(recipient, total, sharpe, image_path)


def start() -> None:
    scheduler = BlockingScheduler()
    scheduler.add_job(daily_update, "cron", hour=8, minute=59)
    scheduler.add_job(trade_morning, "cron", hour=9, minute=0)
    scheduler.add_job(sell_evening, "cron", hour=8, minute=50)
    scheduler.add_job(weekly_report, "cron", day_of_week="sun", hour=18)
    print("📆 Scheduler started. Waiting for events…")
    scheduler.start()


if __name__ == "__main__":
    start()
