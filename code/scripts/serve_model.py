#!/usr/bin/env python3
"""serve_model.py — FastAPI model serving endpoint for MVGT-Net inference.

Exposes a trained MVGT-Net checkpoint as a REST API for online inference.

Endpoints:
    GET  /health           — health check (returns model metadata)
    GET  /info             — model card (input/output schema, version)
    POST /predict          — single-record prediction
    POST /predict/batch    — batch prediction (up to 64 records per call)
    GET  /metrics          — Prometheus-style metrics (request count, latency)

Usage:
    # Start the server (after training a checkpoint):
    python3 scripts/serve_model.py \\
        --checkpoint 10_mvgtnet_code/checkpoints/Economy_Trade/best.pt \\
        --config 10_mvgtnet_code/configs/default.yaml \\
        --host 0.0.0.0 --port 8000

    # Query the server:
    curl -X POST http://localhost:8000/predict \\
        -H "Content-Type: application/json" \\
        -d '{"text": "Q3 GDP growth slowed to 2.1%", "history": [1.2, 1.5, 1.8, 1.7]}'

    # Or use the OpenAPI docs at http://localhost:8000/docs

Requirements:
    pip install fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.9.2

Note: This is a reference implementation. For production, add:
  - Authentication (API key / OAuth2)
  - Rate limiting
  - Request validation
  - Logging / tracing (OpenTelemetry)
  - Horizontal scaling (behind a load balancer)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# FastAPI is optional — defer import so --help works without it
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Request / response schemas (Pydantic v2)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = ""
    history: list[float] = []
    metadata: dict[str, Any] = {}


class PredictResponse(BaseModel):
    prediction: list[float]
    uncertainty: list[float] | None = None
    model_version: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    records: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_latency_ms: float


# ---------------------------------------------------------------------------
# Model wrapper (loads checkpoint lazily)
# ---------------------------------------------------------------------------

class ModelServer:
    """Wraps a trained MVGT-Net checkpoint for inference."""

    def __init__(self, checkpoint_path: str, config_path: str):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.device = None
        self.model_version = "4.0.0"
        self.request_count = 0
        self.total_latency_ms = 0.0
        self._load()

    def _load(self) -> None:
        import torch
        # Load config
        import yaml
        with open(self.config_path) as fh:
            self.config = yaml.safe_load(fh)

        # Load checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model_version = ckpt.get("model_version", "4.0.0")

        # Try to load the MVGT-Net model
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from mvgt_net.st_llm_plus import STLLMPlus
            self.model = STLLMPlus(self.config)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            elif "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            print(f"[ModelServer] Loaded checkpoint from {self.checkpoint_path}")
            print(f"[ModelServer] Device: {self.device}")
        except Exception as e:
            print(f"[ModelServer] WARNING: Could not load full model ({e}). Serving in stub mode.")
            self.model = None

    def predict(self, req: PredictRequest) -> PredictResponse:
        import torch
        start = time.time()
        self.request_count += 1

        if self.model is not None:
            # Real inference path
            with torch.no_grad():
                # This is a simplified path — the real implementation would
                # build the full graph + text encoding from the request
                batch = {
                    "text": [req.text],
                    "history": torch.tensor([req.history], dtype=torch.float32).to(self.device),
                }
                output = self.model(batch)
                prediction = output["prediction"][0].cpu().tolist()
                uncertainty = output.get("uncertainty", [None])[0]
                uncertainty = uncertainty.cpu().tolist() if uncertainty is not None else None
        else:
            # Stub mode (checkpoint not loadable)
            horizon = self.config.get("model", {}).get("prediction_horizon", 12)
            prediction = [0.0] * horizon
            uncertainty = [0.0] * horizon

        latency_ms = (time.time() - start) * 1000
        self.total_latency_ms += latency_ms
        return PredictResponse(
            prediction=prediction,
            uncertainty=uncertainty,
            model_version=self.model_version,
            latency_ms=round(latency_ms, 2),
        )


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(server: ModelServer) -> "FastAPI":
    app = FastAPI(
        title="ST-LLM-Plus / MVGT-Net Inference API",
        version="4.0.0",
        description="REST API for online time-series forecasting with MVGT-Net.",
    )

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model_version": server.model_version,
            "device": str(server.device) if server.device else "cpu",
            "checkpoint": server.checkpoint_path,
        }

    @app.get("/info")
    def info():
        return {
            "model": "MVGT-Net (ST-LLM-Plus)",
            "version": "4.0.0",
            "paper": "Liu et al., IEEE TKDE 2025, doi:10.1109/TKDE.2025.3570705",
            "input_schema": {
                "text": "string (text description of the time series context)",
                "history": "array of floats (historical values)",
                "metadata": "object (optional domain-specific metadata)",
            },
            "output_schema": {
                "prediction": "array of floats (forecasted values)",
                "uncertainty": "array of floats (conformal prediction interval widths)",
            },
            "limits": {
                "max_batch_size": 64,
                "max_history_length": 720,
            },
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if len(req.history) > 720:
            raise HTTPException(400, "history length exceeds 720")
        return server.predict(req)

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(req: BatchPredictRequest):
        if len(req.records) > 64:
            raise HTTPException(400, "batch size exceeds 64")
        start = time.time()
        results = [server.predict(r) for r in req.records]
        return BatchPredictResponse(
            results=results,
            total_latency_ms=round((time.time() - start) * 1000, 2),
        )

    @app.get("/metrics")
    def metrics():
        return {
            "request_count": server.request_count,
            "total_latency_ms": round(server.total_latency_ms, 2),
            "avg_latency_ms": round(server.total_latency_ms / max(server.request_count, 1), 2),
        }

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Serve MVGT-Net via FastAPI.")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if not FASTAPI_AVAILABLE:
        print("ERROR: fastapi + uvicorn not installed.", file=sys.stderr)
        print("Install with: pip install fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.9.2", file=sys.stderr)
        return 1

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    server = ModelServer(args.checkpoint, args.config)
    app = create_app(server)
    print(f"\n[serve_model] Starting server on {args.host}:{args.port}")
    print(f"[serve_model] Docs: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
