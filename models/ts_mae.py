"""Time-Series Masked Autoencoder implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def _generate_mask(shape: Tuple[int, int], mask_ratio: float, device: torch.device) -> torch.Tensor:
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in [0.0, 1.0)")
    mask = torch.rand(shape, device=device)
    return mask < mask_ratio


@dataclass
class TSMAEOutput:
    loss: torch.Tensor
    representation: torch.Tensor
    reconstruction: torch.Tensor


class TS_MAE(nn.Module):
    def __init__(self, input_dim: int = 8, embed_dim: int = 128, num_layers: int = 6, mask_ratio: float = 0.4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.GRU(embed_dim, input_dim, batch_first=True)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> TSMAEOutput:
        if x.dim() != 3:
            raise ValueError("Input tensor must have shape [batch, time, features]")
        batch, time, _ = x.shape
        device = x.device
        mask = _generate_mask((batch, time), self.mask_ratio, device)
        x_masked = x.clone()
        x_masked[mask.unsqueeze(-1).expand_as(x_masked)] = 0.0
        embedded = self.embedding(x_masked)
        encoded = self.encoder(embedded)
        recon, _ = self.decoder(encoded)
        loss = torch.mean((recon - x) ** 2)
        representation = encoded.mean(dim=1)
        return TSMAEOutput(loss=loss, representation=representation, reconstruction=recon)
