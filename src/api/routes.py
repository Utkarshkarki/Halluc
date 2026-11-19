"""API Route definitions for Phase 6 (/api/v1).

Exposes:
- GET  /api/v1/health   (Liveness check)
- GET  /api/v1/ready    (Readiness check)
- GET  /api/v1/models   (Model & configuration metadata)
- POST /api/v1/analyze  (Claim-level hallucination detection)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    HealthResponse,
    ModelsMetadataResponse,
    ReadyResponse,
)
from .service import PipelineService, get_pipeline_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Hallucination Detection"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description="Returns HTTP 200 if the API service process is active.",
)
def health(
    service: PipelineService = Depends(get_pipeline_service),
) -> HealthResponse:
    return service.get_health()


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness Probe",
    description="Returns HTTP 200 if all ML core inference models are loaded and ready to serve traffic.",
)
def ready(
    service: PipelineService = Depends(get_pipeline_service),
) -> ReadyResponse:
    res = service.get_ready()
    if not res.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline models are still initializing or unavailable.",
        )
    return res


@router.get(
    "/models",
    response_model=ModelsMetadataResponse,
    summary="Models and Configuration Metadata",
    description="Returns metadata about the active NLI model, embedding model, thresholds, and execution device.",
)
def models(
    service: PipelineService = Depends(get_pipeline_service),
) -> ModelsMetadataResponse:
    return service.get_models_metadata()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze Response for Hallucinations",
    description=(
        "Decomposes the input response into atomic claims, retrieves relevant evidence passages "
        "from the reference context, runs NLI inference, aggregates verdicts using a contradiction-first "
        "policy, and computes the response-level hallucination risk."
    ),
    status_code=status.HTTP_200_OK,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Validation or Unprocessable Entity Error",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal Server Error",
        },
        503: {
            "model": ErrorResponse,
            "description": "Service or Subsystems Unavailable",
        },
    },
)
def analyze(
    raw_request: Request,
    request: AnalyzeRequest,
    service: PipelineService = Depends(get_pipeline_service),
) -> AnalyzeResponse:
    req_id = getattr(raw_request.state, "request_id", None)
    try:
        return service.analyze(request, request_id=req_id)
    except HTTPException:
        raise
    except ValueError as val_err:
        raise HTTPException(
            status_code=422,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        logger.exception("Analysis pipeline error for request %s", req_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis pipeline failed during execution.",
        ) from exc
