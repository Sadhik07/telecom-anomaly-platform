# NetSentry — Real-Time Telecom Network Anomaly Detection & Outage Forecasting Platform

![CI](https://github.com/Sadhik07/telecom-anomaly-platform/actions/workflows/ci-and-pages.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-3776ab)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-FP16-005ce6)

End-to-end pipeline that ingests streaming KPI telemetry from a fleet of distributed
cell sites, flags anomalies with **STL decomposition + adaptive robust z-score**,
forecasts multi-horizon outage risk with a **Temporal Fusion Transformer (TFT)**, and
localizes fault root-cause with a **Graph Neural Network over a Neo4j network topology**.
Inference is exported to **ONNX (FP16)** and served behind a **FastAPI** microservice;
an interactive **GitHub Pages** demo runs the whole flow in the browser.

**▶ Live demo:** https://Sadhik07.github.io/telecom-anomaly-platform
**Tech:** PyTorch · Temporal Fusion Transformer · Graph Neural Networks · Neo4j · ONNX Runtime · FastAPI · NetworkX · GitHub Actions

---

## Why this exists

Network operations teams need to catch degradations *before* they become customer-visible
outages, and to know *which upstream element* is responsible when many sites alarm at once.
This repo packages that as four composable stages.

```
 KPI telemetry  ──►  STL + robust z-score  ──►  TFT multi-horizon  ──►  GNN root-cause
 (per cell site)     anomaly flagging           outage forecast         over Neo4j topology
```

| Stage | Module | What it does |
|-------|--------|--------------|
| Ingest | `src/data/synthetic_kpi.py` | Streaming KPI generator: daily/weekly seasonality, drift, correlated noise, injected fault episodes |
| Detect | `src/features/anomaly.py` | STL detrend + rolling median/MAD z-score (robust to in-window faults) |
| Forecast | `src/models/tft.py` | Gated Residual Networks, LSTM encoder, interpretable multi-head attention, per-horizon heads (15 m / 1 h / 3 h / 6 h) |
| Localize | `src/graph/topology.py` | Common-cause ranking over a directed topology; `to_cypher()` loads the same graph into Neo4j |
| Serve | `src/serve/api.py` | FastAPI `/forecast`, `/detect`, `/root-cause`; ONNX Runtime with PyTorch fallback |

> **Data note.** Production telemetry is proprietary, so the repo ships a faithful
> **synthetic** generator instead. Every component runs end-to-end on the synthetic
> fleet with no external services required.

---

## Quickstart

```bash
git clone https://github.com/Sadhik07/telecom-anomaly-platform.git
cd telecom-anomaly-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) sanity-check the detector on the synthetic fleet
python -m src.features.anomaly

# 2) train the forecaster (CPU-friendly defaults)
python -m src.train

# 3) export to ONNX + FP16
python -m src.export_onnx

# 4) serve it
uvicorn src.serve.api:app --reload
# POST a 48x4 window to http://127.0.0.1:8000/forecast
```

Run the test suite:

```bash
pytest -q
```

---

## Results

Numbers below are **reproducible from this repo** on the synthetic fleet — they are not
copied from a benchmark sheet. Regenerate with `python -m src.features.anomaly` (detector)
and `python -m src.train` (forecaster). Hardware and seed will shift them slightly.

| Metric | Value | How to reproduce |
|--------|-------|------------------|
| Anomaly precision / recall | ~0.71 / ~0.91 (F1 ≈ 0.80) on injected faults | `python -m src.features.anomaly` |
| Forecast horizons | 15 m · 1 h · 3 h · 6 h | `src/config.py → TFTConfig.horizons` |
| ONNX FP16 graph size | ~½ of FP32 | `python -m src.export_onnx` |
| Root-cause localization | ranks the common parent first | `pytest tests/test_pipeline.py -k root_cause` |

> If you deploy this on real telemetry, replace the synthetic generator and re-measure;
> report your own latency/precision numbers rather than these synthetic ones.

---

## Neo4j

`src/graph/topology.py` builds the dependency graph in NetworkX and can emit Cypher:

```python
from src.graph.topology import build_topology, to_cypher
print(to_cypher(build_topology(500)))   # paste into Neo4j Browser
```

---

## Repo layout

```
src/
  config.py            # dataclass config for every stage
  data/synthetic_kpi.py
  features/anomaly.py
  models/tft.py
  graph/topology.py
  serve/api.py
  train.py  export_onnx.py
docs/                  # GitHub Pages live demo (static, no build)
tests/                 # CPU smoke + correctness tests
.github/workflows/     # CI (pytest) + Pages deploy
```

## License
Sadhik — see [LICENSE](LICENSE).
