"""Dual-stream Masked Autoencoder for joint Bitcoin and macro features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def _sinusoidal_position_encoding(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


@dataclass
class DualMAEConfig:
    btc_dim: int = 1
    stock_dim: int = 5
    embed_dim: int = 128
    num_layers: int = 4
    feedforward_dim: int = 256
    num_heads: int = 8
    mask_ratio: float = 0.4
    max_seq_len: int = 512


class DualStreamMAE(nn.Module):
    """Dual-stream MAE with cross attention between BTC and macro streams."""

    def __init__(self, config: DualMAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or DualMAEConfig()
        c = self.config

        self.btc_embed = nn.Linear(c.btc_dim, c.embed_dim)
        self.stock_embed = nn.Linear(c.stock_dim, c.embed_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=c.embed_dim,
            nhead=c.num_heads,
            dim_feedforward=c.feedforward_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.btc_encoder = TransformerEncoder(encoder_layer, num_layers=c.num_layers)
        self.stock_encoder = TransformerEncoder(encoder_layer, num_layers=c.num_layers)

        self.cross_attn_btc = nn.MultiheadAttention(embed_dim=c.embed_dim, num_heads=c.num_heads, batch_first=True)
        self.cross_attn_stock = nn.MultiheadAttention(embed_dim=c.embed_dim, num_heads=c.num_heads, batch_first=True)

        self.btc_decoder = nn.Sequential(
            nn.Linear(c.embed_dim, c.embed_dim),
            nn.ReLU(),
            nn.Linear(c.embed_dim, c.btc_dim),
        )
        self.stock_decoder = nn.Sequential(
            nn.Linear(c.embed_dim, c.embed_dim),
            nn.ReLU(),
            nn.Linear(c.embed_dim, c.stock_dim),
        )

        pe = _sinusoidal_position_encoding(c.max_seq_len, c.embed_dim)
        self.register_buffer("positional_encoding", pe, persistent=False)

        self.mask_ratio = c.mask_ratio

    def _prepare_inputs(self, xb: torch.Tensor, xs: torch.Tensor, apply_mask: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if xb.dim() != 3 or xs.dim() != 3:
            raise ValueError("Input tensors must be 3D (batch, seq, features)")
        if xb.size(0) != xs.size(0) or xb.size(1) != xs.size(1):
            raise ValueError("BTC and macro streams must have matching batch and sequence dimensions")

        if apply_mask:
            btc_mask = torch.rand_like(xb[..., 0]) < self.mask_ratio
            stock_mask = torch.rand_like(xs[..., 0]) < self.mask_ratio
            xb = xb.clone()
            xs = xs.clone()
            xb[btc_mask] = 0
            xs[stock_mask] = 0
        return xb, xs

    def _encode(self, xb: torch.Tensor, xs: torch.Tensor, apply_mask: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xb, xs = self._prepare_inputs(xb, xs, apply_mask)
        seq_len = xb.size(1)
        pos = self.positional_encoding[:seq_len].unsqueeze(0)

        btc_embed = self.btc_embed(xb) + pos
        stock_embed = self.stock_embed(xs) + pos

        btc_encoded = self.btc_encoder(btc_embed)
        stock_encoded = self.stock_encoder(stock_embed)

        btc_cross, _ = self.cross_attn_btc(btc_encoded, stock_encoded, stock_encoded)
        stock_cross, _ = self.cross_attn_stock(stock_encoded, btc_encoded, btc_encoded)

        btc_fused = btc_encoded + btc_cross
        stock_fused = stock_encoded + stock_cross
        joint = btc_fused + stock_fused
        return btc_fused, stock_fused, joint

    def forward(self, xb: torch.Tensor, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        btc_fused, stock_fused, joint = self._encode(xb, xs, apply_mask=True)

        btc_recon = self.btc_decoder(btc_fused)
        stock_recon = self.stock_decoder(stock_fused)

        btc_loss = nn.functional.mse_loss(btc_recon, xb, reduction="mean")
        stock_loss = nn.functional.mse_loss(stock_recon, xs, reduction="mean")
        loss = btc_loss + stock_loss
        latent = joint.mean(dim=1)
        return loss, latent

    @torch.no_grad()
    def encode(self, xb: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """Return latent representation without applying masking."""
        _, _, joint = self._encode(xb, xs, apply_mask=False)
        return joint.mean(dim=1)
