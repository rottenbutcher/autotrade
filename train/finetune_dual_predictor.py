"""Fine-tuning predictor using the pretrained Dual-Stream MAE encoder."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.dual_mae import DualStreamMAE

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency guard
    wandb = None

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "dataset.csv")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "dual_mae_encoder.pth")
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "btc_predictor.pth")
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
THRESHOLD = 0.65


@dataclass
class FineTuneConfig:
    seq_len: int = SEQ_LEN
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    learning_rate: float = LR
    threshold: float = THRESHOLD
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BTCClassifier(nn.Module):
    def __init__(self, encoder_dim: int = 128) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


def load_dataset(path: str = DATA_PATH, seq_len: int = SEQ_LEN) -> Tuple[TensorDataset, TensorDataset]:
    df = pd.read_csv(path, index_col=0)
    if df.empty or len(df) <= seq_len:
        raise ValueError("Dataset is empty or shorter than sequence length.")

    btc = df[["BTC-USD"]].values
    stock = df[["^GSPC", "^IXIC", "GC=F", "DX-Y.NYB", "KRW=X"]].values

    xb, xs, y = [], [], []
    for idx in range(len(df) - seq_len - 1):
        xb.append(btc[idx : idx + seq_len])
        xs.append(stock[idx : idx + seq_len])
        today = btc[idx + seq_len - 1]
        tomorrow = btc[idx + seq_len]
        y.append([1.0 if tomorrow > today else 0.0])

    xb_tensor = torch.tensor(np.array(xb), dtype=torch.float32)
    xs_tensor = torch.tensor(np.array(xs), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)

    split = int(len(xb_tensor) * 0.8)
    train_ds = TensorDataset(
        xb_tensor[:split],
        xs_tensor[:split],
        y_tensor[:split],
    )
    val_ds = TensorDataset(
        xb_tensor[split:],
        xs_tensor[split:],
        y_tensor[split:],
    )
    return train_ds, val_ds


def _init_wandb():
    if wandb is None:
        return None
    try:
        return wandb.init(project="BTC-AutoTrader", name="finetune_dual_predictor")
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Failed to init wandb: {exc}")
        return None


def _load_encoder(device: torch.device) -> DualStreamMAE:
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder weights not found at {ENCODER_PATH}. Run pretraining first.")
    model = DualStreamMAE()
    state = torch.load(ENCODER_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def train_epoch(
    encoder: DualStreamMAE,
    classifier: BTCClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    classifier.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for xb_batch, xs_batch, y_batch in dataloader:
        xb_batch = xb_batch.to(device)
        xs_batch = xs_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            latent = encoder.encode(xb_batch, xs_batch)
        preds = classifier(latent)
        loss = criterion(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb_batch.size(0)
        predicted = (preds >= 0.5).float()
        total_correct += (predicted == y_batch).sum().item()
        total_samples += xb_batch.size(0)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


def evaluate(
    encoder: DualStreamMAE,
    classifier: BTCClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for xb_batch, xs_batch, y_batch in dataloader:
            xb_batch = xb_batch.to(device)
            xs_batch = xs_batch.to(device)
            y_batch = y_batch.to(device)

            latent = encoder.encode(xb_batch, xs_batch)
            preds = classifier(latent)
            loss = criterion(preds, y_batch)

            total_loss += loss.item() * xb_batch.size(0)
            predicted = (preds >= 0.5).float()
            total_correct += (predicted == y_batch).sum().item()
            total_samples += xb_batch.size(0)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


def main() -> None:
    config = FineTuneConfig()
    train_ds, val_ds = load_dataset(seq_len=config.seq_len)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    device = config.device
    encoder = _load_encoder(device)
    classifier = BTCClassifier().to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.learning_rate)

    run = _init_wandb()
    os.makedirs(os.path.dirname(PREDICTOR_PATH), exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_epoch(encoder, classifier, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(encoder, classifier, val_loader, criterion, device)

        if run is not None:
            run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            })

        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"classifier": classifier.state_dict()}, PREDICTOR_PATH)

    if run is not None:
        run.finish()
    print(f"✅ Saved predictor to {PREDICTOR_PATH}")


if __name__ == "__main__":
    main()
