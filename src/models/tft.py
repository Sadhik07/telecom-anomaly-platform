"""A compact Temporal Fusion Transformer for multi-horizon KPI forecasting.

This is a faithful-but-lightweight TFT: variable selection via gated residual
networks (GRN), an LSTM encoder for local processing, interpretable
multi-head attention over the encoded sequence, and multi-horizon output heads.
Kept small enough to train on CPU for demo purposes while preserving the
architecture's defining components.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CONFIG


class GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated Residual Network — the core TFT building block."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.elu = nn.ELU()
        self.glu = GLU(dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.fc2(self.elu(self.fc1(x)))
        h = self.drop(h)
        return self.norm(x + self.glu(h))


class TemporalFusionTransformer(nn.Module):
    def __init__(self, n_features: int, cfg=CONFIG):
        super().__init__()
        t = cfg.tft
        self.input_len = t.input_len
        self.horizons = list(t.horizons)
        self.embed = nn.Linear(n_features, t.hidden)
        self.var_select = GRN(t.hidden, t.dropout)
        self.encoder = nn.LSTM(t.hidden, t.hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(t.hidden, t.heads, dropout=t.dropout, batch_first=True)
        self.post_attn = GRN(t.hidden, t.dropout)
        # one linear head per forecast horizon (predicts target KPI)
        self.heads = nn.ModuleList([nn.Linear(t.hidden, 1) for _ in self.horizons])

    def forward(self, x):  # x: (B, input_len, n_features)
        h = self.var_select(self.embed(x))
        enc, _ = self.encoder(h)
        attn_out, attn_w = self.attn(enc, enc, enc, need_weights=True)
        z = self.post_attn(enc + attn_out)
        ctx = z[:, -1, :]  # last-step context summarizes the window
        preds = torch.cat([head(ctx) for head in self.heads], dim=-1)  # (B, n_horizons)
        return preds, attn_w


def quantile_loss(pred, target, q=0.5):
    e = target - pred
    return torch.max((q - 1) * e, q * e).mean()


if __name__ == "__main__":
    m = TemporalFusionTransformer(n_features=4)
    x = torch.randn(8, CONFIG.tft.input_len, 4)
    y, w = m(x)
    print("pred shape:", y.shape, "| attn shape:", w.shape)
    print("params:", sum(p.numel() for p in m.parameters()))
