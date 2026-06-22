"""Train the Temporal Fusion Transformer on synthetic fleet telemetry.

Builds sliding windows per site, normalizes features, trains the multi-horizon
forecaster with quantile loss, and writes the checkpoint + normalization stats
to artifacts/ for ONNX export and serving.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import CONFIG
from src.data.synthetic_kpi import generate_fleet
from src.models.tft import TemporalFusionTransformer, quantile_loss

KPIS = list(CONFIG.data.kpis)
TARGET = "latency_ms"


def build_windows(fleet, cfg=CONFIG):
    xs, ys = [], []
    target_idx = KPIS.index(TARGET)
    L, H = cfg.tft.input_len, max(cfg.tft.horizons)
    for _, g in fleet.groupby("site_id", sort=True):
        arr = g.sort_values("step")[KPIS].to_numpy(dtype=np.float32)
        for i in range(0, len(arr) - L - H, 6):  # stride 6 (30 min) to limit size
            window = arr[i : i + L]
            future = arr[i + L : i + L + H, target_idx]
            ys.append([future[h - 1] for h in cfg.tft.horizons])
            xs.append(window)
    X = np.asarray(xs, dtype=np.float32)
    Y = np.asarray(ys, dtype=np.float32)
    return X, Y


def normalize(X, stats=None):
    if stats is None:
        mu = X.reshape(-1, X.shape[-1]).mean(0)
        sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
        stats = {"mu": mu.tolist(), "sd": sd.tolist()}
    mu = np.asarray(stats["mu"], dtype=np.float32)
    sd = np.asarray(stats["sd"], dtype=np.float32)
    return (X - mu) / sd, stats


def main(cfg=CONFIG):
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    print("Generating synthetic fleet telemetry ...")
    fleet = generate_fleet(cfg)
    X, Y = build_windows(fleet, cfg)
    print(f"windows: X={X.shape} Y={Y.shape}")
    Xn, stats = normalize(X)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(Y))
    dl = DataLoader(ds, batch_size=cfg.tft.batch_size, shuffle=True)

    model = TemporalFusionTransformer(n_features=len(KPIS), cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.tft.lr)

    model.train()
    for epoch in range(cfg.tft.epochs):
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred, _ = model(xb)
            loss = quantile_loss(pred, yb, q=0.5)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f"epoch {epoch + 1}/{cfg.tft.epochs}  loss={total / len(ds):.4f}")

    torch.save(model.state_dict(), os.path.join(cfg.artifacts_dir, "tft.pt"))
    with open(os.path.join(cfg.artifacts_dir, "norm_stats.json"), "w") as f:
        json.dump({"stats": stats, "kpis": KPIS, "target": TARGET}, f, indent=2)
    print(f"saved checkpoint -> {cfg.artifacts_dir}/tft.pt")


if __name__ == "__main__":
    main()
