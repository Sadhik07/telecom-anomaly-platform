"""STL decomposition + adaptive robust z-score anomaly flagging.

Seasonal-Trend decomposition (statsmodels STL) removes daily/weekly structure,
then a rolling median/MAD z-score on the residual flags points whose deviation
exceeds an adaptive threshold. MAD is used instead of std so a fault episode
inside the window doesn't inflate the threshold and mask itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG

try:
    from statsmodels.tsa.seasonal import STL
    _HAS_STL = True
except Exception:  # pragma: no cover - optional dep fallback
    _HAS_STL = False


def _rolling_mad_z(resid: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(resid)
    med = s.rolling(window, min_periods=window // 4, center=False).median()
    mad = (s - med).abs().rolling(window, min_periods=window // 4, center=False).median()
    # 1.4826 scales MAD to be consistent with std for normal data
    robust_std = 1.4826 * mad.replace(0, np.nan)
    z = (s - med) / robust_std
    return z.fillna(0).to_numpy()


def detect_series(values: np.ndarray, cfg=CONFIG) -> dict:
    """Return residual, z-score and boolean anomaly mask for one KPI series."""
    values = np.asarray(values, dtype=float)
    if _HAS_STL and len(values) >= 2 * cfg.anomaly.stl_period:
        stl = STL(values, period=cfg.anomaly.stl_period, robust=True)
        res = stl.fit()
        resid = res.resid
        seasonal = res.seasonal
        trend = res.trend
    else:
        # lightweight fallback: rolling-median detrend
        s = pd.Series(values)
        trend = s.rolling(cfg.anomaly.stl_period, min_periods=1, center=True).median().to_numpy()
        resid = values - trend
        seasonal = np.zeros_like(values)

    z = _rolling_mad_z(resid, cfg.anomaly.z_window)
    raw = np.abs(z) > cfg.anomaly.z_threshold
    mask = _persistence_filter(raw, cfg.anomaly.min_run)
    return {"trend": trend, "seasonal": seasonal, "resid": resid, "z": z, "anomaly": mask}


def _persistence_filter(raw: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only exceedances that persist for >= min_run consecutive samples.

    Sustained fault episodes survive; isolated noise spikes are dropped, which
    is what separates a degradation from a transient blip.
    """
    if min_run <= 1:
        return raw
    out = np.zeros_like(raw)
    run = 0
    for i, v in enumerate(raw):
        if v:
            run += 1
        else:
            run = 0
        if run >= min_run:
            out[i - min_run + 1 : i + 1] = True
    return out


def detect_fleet(fleet: pd.DataFrame, kpi: str = "latency_ms", cfg=CONFIG) -> pd.DataFrame:
    """Run detection per site for a chosen KPI; returns fleet with z + anomaly columns."""
    out = []
    for site_id, g in fleet.groupby("site_id", sort=True):
        g = g.sort_values("step").copy()
        r = detect_series(g[kpi].to_numpy(), cfg)
        g["z"] = r["z"]
        g["anomaly"] = r["anomaly"].astype(int)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def precision_recall(flagged: pd.DataFrame) -> dict:
    tp = int(((flagged.anomaly == 1) & (flagged.is_fault == 1)).sum())
    fp = int(((flagged.anomaly == 1) & (flagged.is_fault == 0)).sum())
    fn = int(((flagged.anomaly == 0) & (flagged.is_fault == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


if __name__ == "__main__":
    from src.data.synthetic_kpi import generate_fleet

    fleet = generate_fleet()
    flagged = detect_fleet(fleet, "latency_ms")
    print(precision_recall(flagged))
