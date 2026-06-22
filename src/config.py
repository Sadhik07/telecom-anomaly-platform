"""Central configuration for the telecom anomaly + outage forecasting platform."""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    n_sites: int = 500            # number of simulated distributed cell sites
    freq_minutes: int = 5         # telemetry sampling interval
    days: int = 14               # history length to synthesize
    seed: int = 7
    # KPIs emitted per site
    kpis: tuple = ("latency_ms", "jitter_ms", "packet_loss_pct", "throughput_mbps")


@dataclass
class AnomalyConfig:
    stl_period: int = 288         # 24h of 5-min samples (288 = 1 day) for STL seasonality
    z_window: int = 144           # rolling window for adaptive z-score (~12h)
    z_threshold: float = 3.5      # flag points beyond this many robust std-devs
    min_run: int = 3              # require this many consecutive exceedances (faults are sustained)


@dataclass
class TFTConfig:
    input_len: int = 48           # 4h lookback (48 * 5min)
    horizons: tuple = (3, 12, 36, 72)  # 15m, 1h, 3h, 6h ahead (in 5-min steps)
    hidden: int = 64
    heads: int = 4
    dropout: float = 0.1
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3


@dataclass
class GNNConfig:
    hidden: int = 32
    layers: int = 2
    dropout: float = 0.1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    tft: TFTConfig = field(default_factory=TFTConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    artifacts_dir: str = "artifacts"


CONFIG = Config()
