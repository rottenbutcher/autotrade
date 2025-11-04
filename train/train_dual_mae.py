"""Pretraining script for the Dual-Stream MAE."""
from __future__ import annotations

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.dual_mae import DualStreamMAE

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency guard
    wandb = None

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "dual_mae_encoder.pth")
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4


def load_dataset(path: str = DATA_PATH, seq_len: int = SEQ_LEN) -> TensorDataset:
    df = pd.read_csv(path, index_col=0)
    if df.empty or len(df) <= seq_len:
        raise ValueError("Dataset is empty or shorter than sequence length.")

    btc = df[["BTC-USD"]].values
    stock = df[["^GSPC", "^IXIC", "GC=F", "DX-Y.NYB", "KRW=X"]].values

    xb, xs = [], []
    for idx in range(len(df) - seq_len):
        xb.append(btc[idx : idx + seq_len])
        xs.append(stock[idx : idx + seq_len])

    btc_tensor = torch.tensor(xb, dtype=torch.float32)
    stock_tensor = torch.tensor(xs, dtype=torch.float32)
    return TensorDataset(btc_tensor, stock_tensor)


def _init_wandb():
    if wandb is None:
        return None
    try:
        return wandb.init(project="BTC-AutoTrader", name="pretrain_dual_MAE")
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Failed to init wandb: {exc}")
        return None


def main() -> None:
    dataset = load_dataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamMAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    run = _init_wandb()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for btc_batch, stock_batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            btc_batch = btc_batch.to(device)
            stock_batch = stock_batch.to(device)

            loss, _ = model(btc_batch, stock_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(dataloader))
        if run is not None:
            run.log({"pretrain_loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch}: {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    if run is not None:
        run.finish()
    print(f"✅ Saved encoder weights to {MODEL_PATH}")


if __name__ == "__main__":
    main()
