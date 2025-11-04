"""Backtesting utilities, visualization, and email reporting for the trading system."""
from __future__ import annotations

import os
import smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency guard
    plt = None

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency guard
    wandb = None

TRADES_PATH = os.path.join(os.path.dirname(__file__), "trade_log.csv")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


def _init_wandb(name: str):
    if wandb is None:
        return None
    mode = os.getenv("WANDB_MODE", "offline")
    try:
        return wandb.init(project="BTC-AutoTrader", name=name, mode=mode, reinit=True)
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Failed to init wandb: {exc}")
        return None


def _plot_equity_curves(df: pd.DataFrame) -> Optional[str]:
    if plt is None:
        print("⚠️ matplotlib not installed; skipping chart generation.")
        return None

    os.makedirs(REPORTS_DIR, exist_ok=True)
    today = datetime.today().strftime("%Y%m%d")
    base_path = os.path.join(REPORTS_DIR, f"weekly_report_{today}.png")

    plt.figure(figsize=(10, 5))
    plt.plot(df["cum_return"], label="Cumulative Return", linewidth=2)
    plt.title("BTC Auto-Trader Performance")
    plt.xlabel("Trades")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(base_path, bbox_inches="tight")
    plt.close()

    hist_path = base_path.replace(".png", "_hist.png")
    plt.figure(figsize=(8, 5))
    plt.hist(df["return"], bins=30, alpha=0.7)
    plt.title("Distribution of Trade Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(hist_path, bbox_inches="tight")
    plt.close()

    return base_path


def backtest(trades_path: str = TRADES_PATH) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if not os.path.exists(trades_path):
        print("❌ No trade log found.")
        return None, None, None

    df = pd.read_csv(trades_path)
    if df.empty or "return" not in df.columns:
        print("ℹ️ Trade log missing return data; skipping backtest.")
        return None, None, None

    returns = pd.to_numeric(df["return"], errors="coerce").dropna()
    if returns.empty:
        print("ℹ️ No realized returns available yet.")
        return None, None, None

    df = df.loc[returns.index].copy()
    df["return"] = returns
    df["cum_return"] = (1 + df["return"]).cumprod()

    total_return = df["cum_return"].iloc[-1] - 1
    sharpe = np.nan
    if returns.std(ddof=0) > 0:
        sharpe = returns.mean() / returns.std(ddof=0) * np.sqrt(252)

    chart_path = _plot_equity_curves(df)

    run = _init_wandb("weekly_backtest")
    metrics = {"total_return": total_return, "sharpe": sharpe}
    if run is not None:
        run.log(metrics)
        if chart_path and os.path.exists(chart_path):
            histogram_path = chart_path.replace(".png", "_hist.png")
            images = {"equity_curve": wandb.Image(chart_path)}
            if os.path.exists(histogram_path):
                images["return_histogram"] = wandb.Image(histogram_path)
            run.log(images)
        run.finish()

    print(f"📈 Backtest - Total Return: {total_return:.2%}, Sharpe: {sharpe:.2f}")
    return total_return, sharpe, chart_path


def send_email_report(to_email: str, total: float, sharpe: float, img_path: Optional[str]) -> None:
    user = os.getenv("REPORT_EMAIL_USER", "your_email@gmail.com")
    password = os.getenv("REPORT_EMAIL_PASS", "app_password")
    smtp_host = os.getenv("REPORT_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("REPORT_SMTP_PORT", "587"))

    message = MIMEMultipart()
    message["Subject"] = f"BTC Auto-Trader Weekly Report ({datetime.today().strftime('%Y-%m-%d')})"
    message["From"] = user
    message["To"] = to_email

    body = (
        "📈 Weekly BTC Auto-Trader Performance Report\n\n"
        f"Total Return: {total:.2%}\n"
        f"Sharpe Ratio: {sharpe:.2f}\n\n"
        "Please find the attached performance graph."
    )
    message.attach(MIMEText(body, "plain"))

    if img_path and os.path.exists(img_path):
        try:
            with open(img_path, "rb") as file:
                image = MIMEImage(file.read())
            image.add_header("Content-Disposition", "attachment", filename=os.path.basename(img_path))
            message.attach(image)
        except Exception as exc:  # pragma: no cover
            print(f"⚠️ Failed to attach report image: {exc}")

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(message)
        print(f"📧 Weekly report email sent to {to_email}.")
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Failed to send email report: {exc}")


if __name__ == "__main__":
    total, sharpe, image_path = backtest()
    if total is not None and sharpe is not None:
        send_email_report(os.getenv("REPORT_EMAIL_TO", "your_email@gmail.com"), total, sharpe, image_path)
