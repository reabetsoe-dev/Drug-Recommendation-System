"""
Backend API that proxies frontend requests to the AI service.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import settings


app = FastAPI(title="AIDrugReview Backend API", version="1.0.0")


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "backend",
        "status": "running",
        "health": f"{settings.backend_base_url}/health",
        "docs": f"{settings.backend_base_url}/docs",
        "predict_endpoint": f"{settings.backend_base_url}/predict",
        "ai_service_url": settings.ai_service_base_url,
    }


class PredictRequest(BaseModel):
    text: Optional[str] = None
    condition: Optional[str] = None
    top_n: int = Field(default=5, ge=1, le=25)
    min_reviews: int = Field(default=5, ge=1, le=200)


def call_ai_service(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{settings.ai_service_base_url}/analyze",
            json=payload,
            timeout=settings.request_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"AI service is unreachable at {settings.ai_service_base_url}. "
                "Start the AI service terminal command and retry."
            ),
        ) from exc

    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        if 400 <= response.status_code < 500:
            raise HTTPException(status_code=response.status_code, detail=detail)
        raise HTTPException(status_code=response.status_code, detail=f"AI service error: {detail}")

    return response.json()


@app.get("/health")
def health() -> Dict[str, Any]:
    ai_status: Dict[str, Any] = {"status": "unknown", "ready": False}
    backend_status = "ok"

    try:
        response = requests.get(
            f"{settings.ai_service_base_url}/health",
            timeout=10,
        )
        if response.ok:
            ai_status = response.json()
        else:
            backend_status = "degraded"
            ai_status = {"status": "error", "ready": False, "error": response.text}
    except requests.RequestException as exc:
        backend_status = "degraded"
        ai_status = {
            "status": "error",
            "ready": False,
            "error": (
                f"Cannot reach AI service at {settings.ai_service_base_url}. "
                "Start the AI service terminal command."
            ),
            "exception": str(exc),
        }

    return {
        "service": "backend",
        "status": backend_status,
        "backend_url": settings.backend_base_url,
        "ai_service_url": settings.ai_service_base_url,
        "ai_service": ai_status,
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> Dict[str, Any]:
    if not payload.text and not payload.condition:
        raise HTTPException(
            status_code=400,
            detail="Provide review text or a condition.",
        )

    request_payload = {
        "text": payload.text,
        "condition": payload.condition,
        "top_n": payload.top_n,
        "min_reviews": payload.min_reviews,
    }
    return call_ai_service(request_payload)
