"""FastAPI inference microservice.

Endpoints:
  GET  /health
  POST /forecast      -> multi-horizon KPI forecast for a single site window
  POST /detect        -> anomaly z-scores + mask for a KPI series
  POST /root-cause    -> rank upstream nodes by how well they explain site anomalies

Serves the ONNX FP16 graph via onnxruntime when available, falling back to the
PyTorch checkpoint otherwise.
"""
from __future__ import annotations

import json
import os
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import CONFIG
from src.features.anomaly import detect_series

app = FastAPI(title="Telecom Anomaly & Outage Forecasting API", version="1.0.0")

_session = None
_torch_model = None
_stats = None


def _load():
    global _session, _torch_model, _stats
    stats_path = os.path.join(CONFIG.artifacts_dir, "norm_stats.json")
    if os.path.exists(stats_path):
        _stats = json.load(open(stats_path))["stats"]
    onnx_fp16 = os.path.join(CONFIG.artifacts_dir, "tft_fp16.onnx")
    onnx_fp32 = os.path.join(CONFIG.artifacts_dir, "tft.onnx")
    try:
        import onnxruntime as ort

        path = onnx_fp16 if os.path.exists(onnx_fp16) else onnx_fp32
        if os.path.exists(path):
            _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            return
    except Exception:
        pass
    # torch fallback
    import torch
    from src.models.tft import TemporalFusionTransformer

    ckpt = os.path.join(CONFIG.artifacts_dir, "tft.pt")
    if os.path.exists(ckpt):
        m = TemporalFusionTransformer(n_features=len(CONFIG.data.kpis))
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.eval()
        _torch_model = m


@app.on_event("startup")
def startup():
    _load()


class Window(BaseModel):
    window: List[List[float]]  # shape (input_len, n_features)


class Series(BaseModel):
    values: List[float]


class SiteAnomalies(BaseModel):
    anomaly_by_site: dict  # {site_id: bool}
    n_sites: int = 50


def _normalize(x: np.ndarray) -> np.ndarray:
    if _stats is None:
        return x
    mu = np.asarray(_stats["mu"], dtype=np.float32)
    sd = np.asarray(_stats["sd"], dtype=np.float32)
    return (x - mu) / sd


@app.get("/health")
def health():
    backend = "onnxruntime" if _session else ("torch" if _torch_model else "none")
    return {"status": "ok", "backend": backend}


@app.post("/forecast")
def forecast(w: Window):
    x = np.asarray(w.window, dtype=np.float32)[None, ...]
    x = _normalize(x)
    horizons = list(CONFIG.tft.horizons)
    if _session is not None:
        out = _session.run(None, {_session.get_inputs()[0].name: x})[0]
        pred = out[0].tolist()
    elif _torch_model is not None:
        import torch

        with torch.no_grad():
            pred = _torch_model(torch.from_numpy(x))[0][0].tolist()
    else:
        pred = [float("nan")] * len(horizons)
    labels = ["15m", "1h", "3h", "6h"][: len(horizons)]
    return {"horizons": labels, "forecast_latency_ms": pred}


@app.post("/detect")
def detect(s: Series):
    r = detect_series(np.asarray(s.values, dtype=float))
    return {
        "z": [round(float(v), 3) for v in r["z"]],
        "anomaly": [int(v) for v in r["anomaly"]],
        "n_flagged": int(r["anomaly"].sum()),
    }


@app.post("/root-cause")
def root_cause(payload: SiteAnomalies):
    from src.graph.topology import build_topology, localize_root_cause

    g = build_topology(payload.n_sites)
    anomaly = {int(k): bool(v) for k, v in payload.anomaly_by_site.items()}
    ranking = localize_root_cause(g, anomaly)
    return {"ranking": ranking[:10]}
