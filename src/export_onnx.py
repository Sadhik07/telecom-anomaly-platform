"""Export the trained TFT to ONNX and produce an FP16-optimized graph.

Run after train.py. Produces:
  artifacts/tft.onnx       (FP32)
  artifacts/tft_fp16.onnx  (FP16, ~2x smaller, lower-latency on supported HW)
"""
from __future__ import annotations

import os

import torch

from src.config import CONFIG
from src.models.tft import TemporalFusionTransformer

KPIS = list(CONFIG.data.kpis)


def export(cfg=CONFIG):
    ckpt = os.path.join(cfg.artifacts_dir, "tft.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError("Train first: python -m src.train")

    model = TemporalFusionTransformer(n_features=len(KPIS), cfg=cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    # attention weights are an interpretability output; export only the forecast
    class ForecastOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)[0]

    wrapped = ForecastOnly(model)
    dummy = torch.randn(1, cfg.tft.input_len, len(KPIS))
    onnx_path = os.path.join(cfg.artifacts_dir, "tft.onnx")
    export_kwargs = dict(
        input_names=["kpi_window"],
        output_names=["forecast"],
        dynamic_axes={"kpi_window": {0: "batch"}, "forecast": {0: "batch"}},
        opset_version=17,
    )
    try:
        # legacy TorchScript exporter — no onnxscript dependency
        torch.onnx.export(wrapped, dummy, onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        # older torch without the dynamo kwarg
        torch.onnx.export(wrapped, dummy, onnx_path, **export_kwargs)
    print(f"exported {onnx_path}")

    try:
        import onnx
        from onnxconverter_common import float16

        m = onnx.load(onnx_path)
        m16 = float16.convert_float_to_float16(m, keep_io_types=True)
        fp16_path = os.path.join(cfg.artifacts_dir, "tft_fp16.onnx")
        onnx.save(m16, fp16_path)
        print(f"exported {fp16_path}")
    except Exception as e:  # pragma: no cover
        print(f"[skip fp16] install onnx + onnxconverter-common to enable: {e}")


if __name__ == "__main__":
    export()
