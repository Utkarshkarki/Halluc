"""FastAPI Application Entrypoint for Phase 6.

Configures:
- Application metadata and OpenAPI documentation (/docs, /redoc)
- Lifespan event for eager ML model warm-up
- CORS middleware for Lovable / frontend web client integration
- API v1 routing (/api/v1/health, /ready, /models, /analyze)
- Clean global error handlers
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import router as api_v1_router
from .service import get_pipeline_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("haluc_api")

REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


def get_cors_configuration() -> tuple[list[str], bool]:
    """Parse CORS allowed origins from environment variable or use safe defaults."""
    env_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "").strip()
    if not env_origins:
        return DEFAULT_CORS_ORIGINS, True

    if env_origins.startswith("[") and env_origins.endswith("]"):
        try:
            origins = json.loads(env_origins)
            return [str(o).strip() for o in origins if str(o).strip()], True
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    origins = [o.strip() for o in env_origins.split(",") if o.strip()]
    if "*" in origins:
        # Per CORS specification: Wildcard origin cannot be used with credentials
        return ["*"], False
    return origins, True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager to eagerly warm up ML models on startup."""
    logger.info("Starting up Hallucination Detection API server...")
    try:
        service = get_pipeline_service()
        service.warmup()
        logger.info("Pipeline models successfully loaded and ready.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Warmup encountered an error (will retry lazily on first request): %s",
            exc,
        )
    yield
    logger.info("Shutting down Hallucination Detection API server...")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="Claim-Level LLM Hallucination Detection API",
        description=(
            "Production REST API wrapping the independently audited Phase 1–5 "
            "claim-level hallucination detection pipeline. Decomposes LLM responses into atomic "
            "propositions, retrieves grounded evidence via FAISS vector search, runs NLI "
            "verification, and scores response-level hallucination risk."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # Request ID Middleware
    # -----------------------------------------------------------------------
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:
        client_req_id = request.headers.get("X-Request-ID")
        if client_req_id and REQUEST_ID_REGEX.match(client_req_id):
            req_id = client_req_id
        else:
            req_id = f"req_{uuid.uuid4().hex[:12]}"

        request.state.request_id = req_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

    # -----------------------------------------------------------------------
    # CORS Middleware (securely configured with explicit allowed origins)
    # -----------------------------------------------------------------------
    allowed_origins, allow_creds = get_cors_configuration()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_creds,
        allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Routers
    # -----------------------------------------------------------------------
    app.include_router(api_v1_router)

    # -----------------------------------------------------------------------
    # Root & Diagnostic Endpoints
    # -----------------------------------------------------------------------
    @app.get("/", tags=["General"])
    def root() -> dict[str, str]:
        return {
            "name": "Claim-Level LLM Hallucination Detection API",
            "version": "0.1.0",
            "docs_url": "/docs",
            "health_url": "/api/v1/health",
            "ready_url": "/api/v1/ready",
        }

    # -----------------------------------------------------------------------
    # Exception Handlers
    # -----------------------------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:12]}"
        errors = [
            {"loc": err.get("loc"), "msg": err.get("msg"), "type": err.get("type")}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": "Unprocessable Entity",
                "detail": errors,
                "request_id": req_id,
                "status_code": 422,
            },
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:12]}"
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Error",
                "detail": exc.detail,
                "request_id": req_id,
                "status_code": exc.status_code,
            },
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:12]}"
        logger.error(
            "Unhandled exception for request %s: %s",
            req_id,
            exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": "An internal pipeline error occurred during processing.",
                "request_id": req_id,
                "status_code": 500,
            },
            headers={"X-Request-ID": req_id},
        )

    return app


app = create_app()
