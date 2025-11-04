import os
from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency guard
    wandb = None


ASSETS: List[str] = ["^GSPC", "^IXIC", "GC=F", "DX-Y.NYB", "KRW=X", "BTC-USD"]
DATA_DIR = os.path.join(os.path.dirname(__file__))
RAW_PATH = os.path.join(DATA_DIR, "dataset_raw.csv")
NORM_PATH = os.path.join(DATA_DIR, "dataset.csv")


def fetch(symbol: str) -> pd.DataFrame:
    """Download daily close prices for the given ticker."""
    try:
        df = yf.download(
            symbol,
            start="2020-01-01",
            end=datetime.today().strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()
        return df[["Close"]].rename(columns={"Close": symbol})
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"⚠️ {symbol} download failed: {exc}")
        return pd.DataFrame()


def _init_wandb():
    if wandb is None:
        return None
    mode = os.getenv("WANDB_MODE", "offline")
    try:
        return wandb.init(project="BTC-AutoTrader", name="data_loader", mode=mode)
    except Exception as exc:  # pragma: no cover - wandb optional
        print(f"⚠️ Failed to init wandb: {exc}")
        return None


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    run = _init_wandb()

    print("📡 Downloading market data...")
    frames = []
    for symbol in ASSETS:
        data = fetch(symbol)
        if data.empty:
            print(f"⚠️ No data downloaded for {symbol}")
            continue
        frames.append(data)

    if not frames:
        raise RuntimeError("No market data downloaded. Check connectivity or ticker symbols.")

    print("🔄 Merging and aligning datasets...")
    merged = pd.concat(frames, axis=1).interpolate("time").fillna(method="bfill").fillna(method="ffill")
    merged.index.name = "Date"

    normed = (merged - merged.mean()) / merged.std(ddof=0).replace(0, pd.NA)
    normed = normed.fillna(0)

    print("💾 Saving dataset...")
    merged.to_csv(RAW_PATH)
    normed.to_csv(NORM_PATH)

    if run is not None:
        run.log({
            "rows": len(normed),
            "last_date": str(normed.index[-1]),
        })
        run.finish()

    print(f"✅ dataset.csv updated at {datetime.now()}")


if __name__ == "__main__":
    main()
