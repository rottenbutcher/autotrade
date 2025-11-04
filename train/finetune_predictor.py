"""Fine-tuning script for Bitcoin direction prediction using the pretrained TS-MAE."""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from data_pipeline.data_loader import EmptyDatasetError, OUTPUT_PATH, prepare_dataset
from models.ts_mae import TS_MAE

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency at runtime
    wandb = None


LOGGER = logging.getLogger(__name__)
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_WINDOW = 30
MODEL_ARTIFACT = os.path.join(os.path.dirname(__file__), "..", "models", "tsmae_encoder.pth")
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "btc_predictor.pth")
PREDICTOR_PATH = os.path.abspath(PREDICTOR_PATH)


@dataclass
class FineTuneConfig:
    dataset_path: str = OUTPUT_PATH
    encoder_path: str = MODEL_ARTIFACT
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    window_size: int = DEFAULT_WINDOW
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    threshold: float = 0.65
    project: str = "ts-mae-bitcoin"
    run_name: Optional[str] = None


class BTCWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, target: np.ndarray, window_size: int):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self) -> int:
        return max(0, self.data.shape[0] - self.window_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.data[idx : idx + self.window_size]
        current_price = self.target[idx + self.window_size - 1]
        next_price = self.target[idx + self.window_size]
        label = torch.tensor(1.0 if next_price > current_price else 0.0, dtype=torch.float32)
        return window, label.unsqueeze(0)


class BTCClassifier(nn.Module):
    def __init__(self, encoder_dim: int = 128):
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


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def _init_wandb(config: FineTuneConfig) -> Optional["wandb.sdk.wandb_run.Run"]:
    if wandb is None:
        LOGGER.warning("wandb is not installed; skipping logging.")
        return None
    try:
        run = wandb.init(
            project=config.project,
            name=config.run_name or "btc-direction-finetune",
            config={
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "window_size": config.window_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "threshold": config.threshold,
            },
        )
        return run
    except Exception as err:  # pragma: no cover
        LOGGER.warning("Failed to initialize wandb: %s", err)
        return None


def _ensure_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        LOGGER.info("Dataset not found at %s; creating a new one.", path)
        return prepare_dataset()
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if df.empty:
        raise EmptyDatasetError("Loaded dataset is empty.")
    return df


def _load_pretrained_encoder(path: str, input_dim: int, mask_ratio: float) -> TS_MAE:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pretrained encoder weights not found at {path}. Please run pretraining first."
        )
    checkpoint = torch.load(path, map_location="cpu")
    model = TS_MAE(input_dim=input_dim, mask_ratio=mask_ratio)
    encoder_state = checkpoint.get("encoder_state_dict")
    embedding_state = checkpoint.get("embedding_state_dict")
    if encoder_state is None or embedding_state is None:
        raise KeyError("Checkpoint missing encoder or embedding weights.")
    model.encoder.load_state_dict(encoder_state)
    model.embedding.load_state_dict(embedding_state)
    model.mask_ratio = 0.0  # Disable masking during inference for stability
    return model


def _train_epoch(
    encoder: TS_MAE,
    classifier: BTCClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    run: Optional["wandb.sdk.wandb_run.Run"],
    epoch: int,
) -> float:
    encoder.eval()
    classifier.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for features, label in dataloader:
        features = features.to(device)
        label = label.to(device)
        with torch.no_grad():
            latent = encoder.encoder(encoder.embedding(features)).mean(dim=1)
        preds = classifier(latent)
        loss = criterion(preds, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        predicted = (preds >= 0.5).float()
        total_correct += (predicted == label).sum().item()
        total_samples += features.size(0)

    accuracy = total_correct / max(1, total_samples)
    avg_loss = total_loss / max(1, total_samples)
    if run is not None:
        run.log({"train_loss": avg_loss, "train_accuracy": accuracy, "epoch": epoch})
    LOGGER.info("Train Epoch %d - Loss: %.6f - Accuracy: %.4f", epoch, avg_loss, accuracy)
    return accuracy


def _evaluate(
    encoder: TS_MAE,
    classifier: BTCClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    run: Optional["wandb.sdk.wandb_run.Run"],
    epoch: int,
) -> Tuple[float, float]:
    encoder.eval()
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for features, label in dataloader:
            features = features.to(device)
            label = label.to(device)
            latent = encoder.encoder(encoder.embedding(features)).mean(dim=1)
            preds = classifier(latent)
            loss = criterion(preds, label)
            total_loss += loss.item() * features.size(0)
            predicted = (preds >= 0.5).float()
            total_correct += (predicted == label).sum().item()
            total_samples += features.size(0)

    accuracy = total_correct / max(1, total_samples)
    avg_loss = total_loss / max(1, total_samples)
    if run is not None:
        run.log({"val_loss": avg_loss, "val_accuracy": accuracy, "epoch": epoch})
    LOGGER.info("Validation Epoch %d - Loss: %.6f - Accuracy: %.4f", epoch, avg_loss, accuracy)
    return avg_loss, accuracy


def train(config: FineTuneConfig) -> None:
    df = _ensure_dataset(config.dataset_path)
    if "btc_usd_close" not in df.columns:
        raise KeyError("Dataset must contain btc_usd_close column for labeling.")
    data = df.values
    target = df["btc_usd_close"].values

    dataset = BTCWindowDataset(data, target, window_size=config.window_size)
    if len(dataset) == 0:
        raise EmptyDatasetError("Not enough data to create training samples.")

    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, shuffle=False)

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise EmptyDatasetError(
            "Insufficient data for training/validation. Adjust window size or batch size."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    encoder = _load_pretrained_encoder(config.encoder_path, input_dim=data.shape[1], mask_ratio=0.0)
    encoder.to(device)

    classifier = BTCClassifier(encoder_dim=encoder.embedding.out_features).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    run = _init_wandb(config)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, config.epochs + 1):
        _train_epoch(encoder, classifier, train_loader, criterion, optimizer, device, run, epoch)
        val_loss, val_acc = _evaluate(encoder, classifier, val_loader, criterion, device, run, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "encoder_state_dict": encoder.encoder.state_dict(),
                "embedding_state_dict": encoder.embedding.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "input_dim": data.shape[1],
                "window_size": config.window_size,
                "threshold": config.threshold,
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    os.makedirs(os.path.dirname(PREDICTOR_PATH), exist_ok=True)
    torch.save(best_state, PREDICTOR_PATH)
    LOGGER.info("Saved fine-tuned model to %s", PREDICTOR_PATH)

    if run is not None:
        run.log({"best_val_loss": best_val_loss})
        run.finish()


def load_model(device: Optional[torch.device] = None) -> Tuple[TS_MAE, BTCClassifier, float, int]:
    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(
            f"Predictor file not found at {PREDICTOR_PATH}. Run finetuning first."
        )
    checkpoint = torch.load(PREDICTOR_PATH, map_location=device or "cpu")
    input_dim = checkpoint["input_dim"]
    window_size = checkpoint.get("window_size", DEFAULT_WINDOW)
    threshold = checkpoint.get("threshold", 0.65)
    encoder = TS_MAE(input_dim=input_dim, mask_ratio=0.0)
    encoder.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.embedding.load_state_dict(checkpoint["embedding_state_dict"])
    classifier = BTCClassifier(encoder_dim=encoder.embedding.out_features)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    classifier.to(device)
    encoder.eval()
    classifier.eval()
    return encoder, classifier, threshold, window_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BTC direction predictor")
    parser.add_argument("--dataset", type=str, default=OUTPUT_PATH)
    parser.add_argument("--encoder", type=str, default=MODEL_ARTIFACT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _setup_logging(args.verbose)
    config = FineTuneConfig(
        dataset_path=args.dataset,
        encoder_path=args.encoder,
        batch_size=args.batch_size,
        epochs=args.epochs,
        window_size=args.window_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        run_name=args.run_name,
    )
    try:
        train(config)
    except EmptyDatasetError as err:
        LOGGER.error("Fine-tuning aborted: %s", err)
        raise SystemExit(1) from err
    except Exception as exc:  # pragma: no cover - runtime safeguard
        LOGGER.exception("Unexpected error during fine-tuning: %s", exc)
        raise SystemExit(1) from exc
