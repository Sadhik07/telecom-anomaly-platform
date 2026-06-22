"""Synthetic multivariate KPI telemetry for distributed cell sites.

Proprietary network telemetry can't be shipped in a public repo, so this module
generates realistic streaming KPI data: daily + weekly seasonality, slow drift,
correlated noise, and injected fault episodes (latency spikes, throughput
collapse, packet-loss bursts) that propagate to topological neighbors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG


def _seasonal_profile(n: int, period: int, amp: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n)
    return amp * np.sin(2 * np.pi * (t / period) + phase)


def generate_site_series(site_id: int, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """One site's KPI history with seasonality, drift and correlated noise."""
    day = 288  # 5-min samples per 24h cycle

    base_latency = rng.uniform(18, 35)
    latency = (
        base_latency
        + _seasonal_profile(n, day, amp=6, phase=rng.uniform(0, 6))
        + _seasonal_profile(n, day * 7, amp=3)
        + rng.normal(0, 1.5, n)
    )
    throughput = (
        rng.uniform(180, 420)
        - _seasonal_profile(n, day, amp=60, phase=2.5)  # busy-hour dips
        + rng.normal(0, 12, n)
    )
    jitter = np.clip(0.25 * latency + rng.normal(0, 1.0, n), 0.1, None)
    packet_loss = np.clip(
        0.05 + 0.002 * np.maximum(latency - base_latency, 0) + rng.exponential(0.05, n),
        0,
        100,
    )

    df = pd.DataFrame(
        {
            "site_id": site_id,
            "step": np.arange(n),
            "latency_ms": latency,
            "jitter_ms": jitter,
            "packet_loss_pct": packet_loss,
            "throughput_mbps": np.clip(throughput, 5, None),
            "is_fault": 0,
        }
    )
    return df


def inject_fault(df: pd.DataFrame, start: int, length: int, severity: float) -> None:
    """Mutate a site frame in place with a fault episode (positional slicing)."""
    n = len(df)
    end = min(start + length, n)
    k = end - start
    if k <= 0:
        return
    ramp = np.linspace(0.3, 1.0, k) * severity
    col = {c: df.columns.get_loc(c) for c in
           ("latency_ms", "jitter_ms", "packet_loss_pct", "throughput_mbps", "is_fault")}
    df.iloc[start:end, col["latency_ms"]] = df.iloc[start:end, col["latency_ms"]].to_numpy() + 40 * ramp
    df.iloc[start:end, col["jitter_ms"]] = df.iloc[start:end, col["jitter_ms"]].to_numpy() + 12 * ramp
    df.iloc[start:end, col["packet_loss_pct"]] = np.clip(
        df.iloc[start:end, col["packet_loss_pct"]].to_numpy() + 18 * ramp, 0, 100)
    df.iloc[start:end, col["throughput_mbps"]] = df.iloc[start:end, col["throughput_mbps"]].to_numpy() * (1 - 0.6 * ramp)
    df.iloc[start:end, col["is_fault"]] = 1


def generate_fleet(cfg=CONFIG) -> pd.DataFrame:
    """Generate the full multi-site telemetry table with labeled fault episodes."""
    rng = np.random.default_rng(cfg.data.seed)
    n = int(cfg.data.days * 24 * 60 / cfg.data.freq_minutes)
    frames = []
    for site in range(cfg.data.n_sites):
        s = generate_site_series(site, n, rng)
        # ~12% of sites experience at least one fault episode
        if rng.random() < 0.12:
            n_episodes = rng.integers(1, 3)
            for _ in range(n_episodes):
                start = int(rng.integers(cfg.tft.input_len, n - 80))
                length = int(rng.integers(20, 72))
                inject_fault(s, start, length, severity=rng.uniform(0.6, 1.4))
        frames.append(s)
    fleet = pd.concat(frames, ignore_index=True)
    return fleet


if __name__ == "__main__":
    df = generate_fleet()
    print(df.head())
    print(f"\nRows: {len(df):,} | Sites: {df.site_id.nunique()} | "
          f"Fault samples: {int(df.is_fault.sum()):,} "
          f"({100 * df.is_fault.mean():.2f}%)")
