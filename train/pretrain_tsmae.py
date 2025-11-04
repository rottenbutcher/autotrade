"""Pretraining script for the Time-Series Masked Autoencoder."""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data_pipeline.data_loader import EmptyDatasetError, OUTPUT_PATH, prepare_dataset
from models.ts_mae import TS_MAE

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional at runtime
    wandb = None


LOGGER = logging.getLogger(__name__)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 200
DEFAULT_WINDOW = 30
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "tsmae_encoder.pth")
MODEL_PATH = os.path.abspath(MODEL_PATH)


@dataclass
class TrainingConfig:
    dataset_path: str = OUTPUT_PATH
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    window_size: int = DEFAULT_WINDOW
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    mask_ratio: float = 0.4
    project: str = "ts-mae-bitcoin"
    run_name: Optional[str] = None


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int):
        if window_size <= 1:
            raise ValueError("window_size must be greater than 1")
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self) -> int:
        return max(0, self.data.shape[0] - self.window_size)

    def __getitem__(self, idx: int) -> torch.Tensor:
        window = self.data[idx : idx + self.window_size]
        return window


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def _init_wandb(config: TrainingConfig) -> Optional["wandb.sdk.wandb_run.Run"]:
    if wandb is None:
        LOGGER.warning("wandb is not installed; skipping experiment logging.")
        return None
    try:
        run = wandb.init(
            project=config.project,
            name=config.run_name or "ts-mae-pretrain",
            config={
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "window_size": config.window_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "mask_ratio": config.mask_ratio,
            },
        )
        return run
    except Exception as err:  # pragma: no cover - runtime fallback
        LOGGER.warning("Failed to initialize wandb: %s", err)
        return None


def _ensure_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        LOGGER.info("Dataset not found at %s; generating a new one.", path)
        return prepare_dataset()
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if df.empty:
        raise EmptyDatasetError("Loaded dataset is empty.")
    return df


def train(config: TrainingConfig) -> None:
    LOGGER.info("Loading dataset from %s", config.dataset_path)
    df = _ensure_dataset(config.dataset_path)
    data = df.values
    num_features = data.shape[1]
    dataset = TimeSeriesDataset(data, window_size=config.window_size)
    if len(dataset) == 0:
        raise EmptyDatasetError("Not enough data to create training windows.")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    if len(dataloader) == 0:
        raise EmptyDatasetError(
            "Batch size is larger than available samples. Reduce --batch-size or increase data."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    model = TS_MAE(input_dim=num_features, mask_ratio=config.mask_ratio)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    run = _init_wandb(config)

    model.train()
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(dataloader))
        LOGGER.info("Epoch %d/%d - Loss: %.6f", epoch, config.epochs, avg_loss)
        if run is not None:
            run.log({"pretrain_loss": avg_loss, "epoch": epoch})

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "embedding_state_dict": model.embedding.state_dict(),
            "input_dim": num_features,
            "window_size": config.window_size,
        },
        MODEL_PATH,
    )
    LOGGER.info("Saved encoder weights to %s", MODEL_PATH)

    if run is not None:
        run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the TS-MAE model")
    parser.add_argument("--dataset", type=str, default=OUTPUT_PATH, help="Path to dataset.csv")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--run-name", type=str, default=None, help="Optional wandb run name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _setup_logging(args.verbose)
    config = TrainingConfig(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        window_size=args.window_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio,
        run_name=args.run_name,
    )
    try:
        train(config)
    except EmptyDatasetError as err:
        LOGGER.error("Training aborted: %s", err)
        raise SystemExit(1) from err
    except Exception as exc:  # pragma: no cover - runtime safeguard
        LOGGER.exception("Unexpected error during pretraining: %s", exc)
        raise SystemExit(1) from exc
