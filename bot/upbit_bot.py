"""Upbit trading bot leveraging the Dual-Stream MAE predictor."""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
import torch

from models.dual_mae import DualStreamMAE
from train.finetune_dual_predictor import BTCClassifier, FineTuneConfig

try:
    import pyupbit
except ImportError:  # pragma: no cover - optional dependency at runtime
    pyupbit = None

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency guard
    wandb = None

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "dataset.csv")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "dual_mae_encoder.pth")
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "btc_predictor.pth")
TRADE_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "utils", "trade_log.csv")
UPBIT_MARKET = "KRW-BTC"


def _init_wandb(name: str):
    if wandb is None:
        return None
    mode = os.getenv("WANDB_MODE", "offline")
    try:
        return wandb.init(project="BTC-AutoTrader", name=name, mode=mode, reinit=True)
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Failed to init wandb: {exc}")
        return None


def _load_models(device: torch.device) -> tuple[DualStreamMAE, BTCClassifier, float, int]:
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder weights not found at {ENCODER_PATH}. Run pretraining first.")
    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(f"Predictor weights not found at {PREDICTOR_PATH}. Run fine-tuning first.")

    encoder = DualStreamMAE()
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    encoder.to(device)
    encoder.eval()

    classifier = BTCClassifier().to(device)
    state = torch.load(PREDICTOR_PATH, map_location=device)
    classifier.load_state_dict(state["classifier"] if "classifier" in state else state)
    classifier.eval()

    config = FineTuneConfig()
    return encoder, classifier, config.threshold, config.seq_len


def _get_client() -> Optional["pyupbit.Upbit"]:
    if pyupbit is None:
        print("⚠️ pyupbit not installed. Skipping live trade execution.")
        return None
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    if not access or not secret:
        print("⚠️ Upbit API keys not set. Skipping live trade execution.")
        return None
    try:
        return pyupbit.Upbit(access, secret)
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Failed to create Upbit client: {exc}")
        return None


def _load_latest_window(seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(DATA_PATH, index_col=0)
    if len(df) < seq_len:
        raise ValueError("Dataset shorter than required sequence length.")
    btc = torch.tensor(df[["BTC-USD"]].values[-seq_len:], dtype=torch.float32, device=device)
    stock = torch.tensor(
        df[["^GSPC", "^IXIC", "GC=F", "DX-Y.NYB", "KRW=X"]].values[-seq_len:],
        dtype=torch.float32,
        device=device,
    )
    return btc.unsqueeze(0), stock.unsqueeze(0)


def _append_trade(row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
    row_df = pd.DataFrame([row])
    if os.path.exists(TRADE_LOG_PATH):
        row_df.to_csv(TRADE_LOG_PATH, mode="a", header=False, index=False)
    else:
        row_df.to_csv(TRADE_LOG_PATH, index=False)


def _get_current_price() -> Optional[float]:
    if pyupbit is None:
        return None
    try:
        price = pyupbit.get_current_price(UPBIT_MARKET)
        return float(price) if price is not None else None
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Failed to fetch current price: {exc}")
        return None


def _log_to_wandb(run, payload: Dict[str, Any]) -> None:
    if run is None:
        return
    try:
        run.log(payload)
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Failed to log to wandb: {exc}")


def trade_morning() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, classifier, threshold, seq_len = _load_models(device)

    try:
        btc_window, stock_window = _load_latest_window(seq_len, device)
    except Exception as exc:
        print(f"⚠️ Failed to prepare input window: {exc}")
        return
    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            latent = encoder.encode(btc_window, stock_window)
            probability = classifier(latent).item()

    current_price = _get_current_price()
    run = _init_wandb(f"trade_{datetime.now().strftime('%Y%m%d')}")

    action = "hold"
    executed = False
    amount = 0.0

    if probability > threshold:
        action = "buy"
        client = _get_client()
        if client is not None:
            try:
                krw_balance = float(client.get_balance("KRW"))
                order_size = max(0.0, krw_balance * 0.95)
                if order_size >= 5000:
                    response = client.buy_market_order(UPBIT_MARKET, order_size)
                    amount = float(response.get("volume", 0.0)) if isinstance(response, dict) else 0.0
                    executed = True
                else:
                    print("⚠️ KRW balance too low for trading.")
            except Exception as exc:  # pragma: no cover
                print(f"⚠️ Buy order failed: {exc}")
        else:
            print("ℹ️ Simulating buy (no client available).")
    else:
        print("ℹ️ Holding position based on probability threshold.")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "probability_up": probability,
        "threshold": threshold,
        "price": current_price,
        "amount": amount,
        "executed": int(executed),
    }
    _append_trade({**payload, "return": ""})
    _log_to_wandb(run, payload)

    if run is not None:
        run.finish()


def sell_all() -> None:
    run = _init_wandb(f"sell_{datetime.now().strftime('%Y%m%d')}")
    client = _get_client()
    executed = False
    amount = 0.0
    price = _get_current_price()

    if client is not None:
        try:
            btc_balance = float(client.get_balance("BTC"))
            if btc_balance > 1e-6:
                response = client.sell_market_order(UPBIT_MARKET, btc_balance)
                amount = btc_balance
                executed = True
            else:
                print("ℹ️ No BTC to sell.")
        except Exception as exc:  # pragma: no cover
            print(f"⚠️ Sell order failed: {exc}")
    else:
        print("ℹ️ Simulating sell (no client available).")

    return_pct: Optional[float] = None
    if os.path.exists(TRADE_LOG_PATH):
        history = pd.read_csv(TRADE_LOG_PATH)
        buys = history[history["action"] == "buy"]
        if not buys.empty and price is not None:
            last_buy = buys.iloc[-1]
            buy_price = float(last_buy.get("price", 0.0) or 0.0)
            if buy_price:
                return_pct = (price - buy_price) / buy_price

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "sell",
        "probability_up": "",
        "threshold": "",
        "price": price,
        "amount": amount,
        "executed": int(executed),
        "return": return_pct if return_pct is not None else "",
    }
    _append_trade(payload)
    log_payload = {k: v for k, v in payload.items() if k != "return"}
    log_payload["return"] = return_pct
    _log_to_wandb(run, log_payload)

    if run is not None:
        run.finish()


def main(action: str) -> None:
    if action == "trade":
        trade_morning()
    elif action == "sell":
        sell_all()
    else:
        raise ValueError(f"Unsupported action: {action}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Stream MAE Upbit trading bot")
    parser.add_argument("action", choices=["trade", "sell"], nargs="?", default="trade")
    args = parser.parse_args()
    main(args.action)
