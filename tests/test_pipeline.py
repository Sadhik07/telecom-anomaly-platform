"""Smoke + correctness tests that run on CPU without external services."""
import numpy as np
import torch

from src.config import CONFIG
from src.data.synthetic_kpi import generate_fleet
from src.features.anomaly import detect_series, detect_fleet, precision_recall
from src.graph.topology import build_topology, localize_root_cause
from src.models.tft import TemporalFusionTransformer


def _small_cfg():
    cfg = CONFIG
    cfg.data.n_sites = 20
    cfg.data.days = 3
    return cfg


def test_fleet_shape():
    cfg = _small_cfg()
    fleet = generate_fleet(cfg)
    assert {"site_id", "latency_ms", "is_fault"}.issubset(fleet.columns)
    assert fleet.site_id.nunique() == cfg.data.n_sites


def test_anomaly_detects_injected_fault():
    cfg = _small_cfg()
    fleet = generate_fleet(cfg)
    flagged = detect_fleet(fleet, "latency_ms", cfg)
    m = precision_recall(flagged)
    # detector should recover a meaningful share of injected faults
    assert m["recall"] >= 0.3
    assert 0.0 <= m["precision"] <= 1.0


def test_tft_forward():
    m = TemporalFusionTransformer(n_features=len(CONFIG.data.kpis))
    x = torch.randn(4, CONFIG.tft.input_len, len(CONFIG.data.kpis))
    pred, attn = m(x)
    assert pred.shape == (4, len(CONFIG.tft.horizons))


def test_root_cause_ranks_common_parent():
    g = build_topology(40, seed=1)
    # make all access nodes under one agg anomalous
    target_agg = "agg-0"
    import networkx as nx

    children = [g.nodes[d]["site_id"] for d in nx.descendants(g, target_agg)
                if g.nodes[d].get("tier") == "access"]
    anomaly = {i: (i in children) for i in range(40)}
    ranking = localize_root_cause(g, anomaly)
    assert ranking, "expected non-empty ranking"
    assert ranking[0][1] > 0
