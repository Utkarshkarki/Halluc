"""Automated tests for Phase 6 FastAPI endpoints.

Covers:
- Root and diagnostic endpoints (/)
- Liveness (/api/v1/health)
- Readiness (/api/v1/ready)
- Metadata (/api/v1/models)
- CORS headers
- Input validation & error responses (HTTP 422)
- Real end-to-end inference verification for:
  - Supported claims
  - Contradicted claims
  - Insufficient evidence claims
  - Multi-passage reference context
  - Provenance & custom tracking IDs
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.service import get_pipeline_service


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient with eager service warm-up."""
    service = get_pipeline_service()
    service.warmup()
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Diagnostics & Liveness / Readiness
# ---------------------------------------------------------------------------

def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "0.1.0"
    assert "/docs" in data["docs_url"]


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert data["version"] == "0.1.0"


def test_ready_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["components"]["nli"] is True
    assert data["components"]["embeddings"] is True
    assert data["components"]["faiss"] is True
    assert data["components"]["claim_extractor"] is True
    assert data["device"] in ("cuda", "cpu")


def test_models_metadata_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == "v1.0.0"
    assert "DeBERTa-v3" in data["nli_model"]
    assert "bge-small-en" in data["embedding_model"]
    assert data["embedding_dimension"] == 384
    assert data["support_threshold"] == 0.50
    assert data["contradiction_threshold"] == 0.40
    assert data["risk_thresholds"]["low_max"] == 0.20
    assert data["risk_thresholds"]["medium_max"] == 0.50


def test_cors_headers(client: TestClient) -> None:
    # Explicitly allowed origin must be echoed and allow credentials
    origin = "http://localhost:3000"
    response = client.get(
        "/api/v1/health",
        headers={"Origin": origin},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_disallowed_origin_rejected(client: TestClient) -> None:
    # Disallowed origin must NOT receive access-control-allow-origin
    disallowed = "http://evil-attacker.com"
    response = client.get(
        "/api/v1/health",
        headers={"Origin": disallowed},
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_options(client: TestClient) -> None:
    # Preflight OPTIONS request for an allowed origin
    origin = "http://localhost:3000"
    response = client.options(
        "/api/v1/analyze",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin
    assert "POST" in response.headers.get("access-control-allow-methods", "")



# ---------------------------------------------------------------------------
# Input Validation & Bad Request Handling
# ---------------------------------------------------------------------------

def test_analyze_validation_missing_response(client: TestClient) -> None:
    payload = {
        "reference_context": "The Eiffel Tower is located in Paris.",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_analyze_validation_empty_response(client: TestClient) -> None:
    payload = {
        "response": "   ",
        "reference_context": "The Eiffel Tower is located in Paris.",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_validation_missing_context(client: TestClient) -> None:
    payload = {
        "response": "The Eiffel Tower is in Paris.",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_validation_empty_context(client: TestClient) -> None:
    payload = {
        "response": "The Eiffel Tower is in Paris.",
        "reference_context": "   ",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_analyze_validation_empty_list_context(client: TestClient) -> None:
    payload = {
        "response": "The Eiffel Tower is in Paris.",
        "reference_context": ["", "  "],
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Real Inference Endpoints Verification
# ---------------------------------------------------------------------------

def test_analyze_supported_claim_real_inference(client: TestClient) -> None:
    payload = {
        "response": "The Eiffel Tower is located in Paris. Its construction was finished in 1889.",
        "reference_context": "The Eiffel Tower is located in Paris, France. Construction was completed on March 31, 1889.",
        "sample_id": "test_api_supported",
        "question": "Where is the Eiffel Tower located and when was it built?",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["sample_id"] == "test_api_supported"
    assert data["risk_level"] == "LOW"
    assert data["hallucination_risk"] == 0.0
    assert data["support_rate"] == 1.0
    assert data["evidence_coverage"] == 1.0
    assert data["abstention_rate"] == 0.0
    assert data["total_factual_claims"] >= 1

    # Check verified claim details
    claims = data["verified_claims"]
    assert len(claims) >= 1
    for c in claims:
        assert c["claim_id"] != "", "Verified claim must have a non-empty claim_id"
        assert c["verdict"] == "SUPPORTED"
        assert c["confidence"] >= 0.50
        assert len(c["evidence"]) >= 1
        top_ev = c["evidence"][0]
        assert top_ev["rank"] == 1
        assert top_ev["nli_verdict"] == "SUPPORTED"
        assert "SUPPORTED" in top_ev["nli_probabilities"]

    assert "total" in data["latency_ms"]
    assert data["metadata"]["question"] == "Where is the Eiffel Tower located and when was it built?"


def test_analyze_contradicted_claim_real_inference(client: TestClient) -> None:
    payload = {
        "response": "The Eiffel Tower is located in Berlin, Germany.",
        "reference_context": "The Eiffel Tower is located in Paris, France.",
        "sample_id": "test_api_contradicted",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["sample_id"] == "test_api_contradicted"
    assert data["risk_level"] == "HIGH"
    assert data["hallucination_risk"] == 1.0
    assert data["support_rate"] == 0.0
    assert data["evidence_coverage"] == 1.0  # Decisive contradictory evidence was retrieved
    assert data["abstention_rate"] == 0.0

    claims = data["verified_claims"]
    assert len(claims) >= 1
    assert claims[0]["verdict"] == "CONTRADICTED"
    assert claims[0]["evidence"][0]["nli_verdict"] == "CONTRADICTED"


def test_analyze_insufficient_evidence_real_inference(client: TestClient) -> None:
    payload = {
        "response": "Undersea miners discovered an ancient obelisk near the Mariana Trench.",
        "reference_context": "The Eiffel Tower is located in Paris, France.",
        "sample_id": "test_api_insufficient",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["sample_id"] == "test_api_insufficient"
    assert data["risk_level"] == "LOW"
    assert data["hallucination_risk"] == 0.0
    assert data["support_rate"] == 0.0
    assert data["evidence_coverage"] == 0.0
    assert data["abstention_rate"] == 1.0

    claims = data["verified_claims"]
    assert len(claims) >= 1
    assert claims[0]["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_analyze_multi_passage_context_and_top_k(client: TestClient) -> None:
    payload = {
        "response": "Jupiter is the largest planet in our solar system.",
        "reference_context": [
            "Mercury is the smallest planet in the Solar System.",
            "Venus has a thick, toxic atmosphere of carbon dioxide.",
            "Earth is the third planet from the Sun.",
            "Jupiter is the largest planet in the Solar System, more than twice as massive as all other planets combined.",
        ],
        "top_k": 3,
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["risk_level"] == "LOW"
    assert data["support_rate"] == 1.0
    claims = data["verified_claims"]
    assert len(claims) >= 1
    assert claims[0]["verdict"] == "SUPPORTED"
    # Should have retrieved up to top_k passages
    assert len(claims[0]["evidence"]) <= 3
    # Top passage should match Jupiter
    assert "Jupiter" in claims[0]["evidence"][0]["text"]


# ---------------------------------------------------------------------------
# Request ID & Error Contract Tests
# ---------------------------------------------------------------------------

def test_request_id_auto_generated(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    req_id = response.headers.get("X-Request-ID")
    assert req_id is not None
    assert req_id.startswith("req_")


def test_request_id_client_provided(client: TestClient) -> None:
    custom_id = "client-trace-id-12345"
    response = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id


def test_request_id_malformed_sanitized(client: TestClient) -> None:
    # Malformed ID with CRLF / forbidden characters must be replaced with safe UUID
    malformed_id = "bad\r\nid<script>"
    response = client.get("/api/v1/health", headers={"X-Request-ID": malformed_id})
    assert response.status_code == 200
    req_id = response.headers.get("X-Request-ID")
    assert req_id != malformed_id
    assert req_id.startswith("req_")


def test_analyze_validation_oversized_context(client: TestClient) -> None:
    # Context exceeding 200,000 characters must return 422 before model inference
    payload = {
        "response": "The Eiffel Tower is in Paris.",
        "reference_context": "x" * 200001,
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Unprocessable Entity"
    assert "request_id" in data


def test_ready_uninitialized_returns_503() -> None:
    from unittest.mock import MagicMock
    from src.api.routes import router
    from fastapi import FastAPI
    from src.api.schemas import ReadyResponse

    mock_service = MagicMock()
    mock_service.get_ready.return_value = ReadyResponse(
        ready=False,
        components={"nli": False, "embeddings": False, "faiss": False, "claim_extractor": False},
        device="uninitialized",
    )

    test_app = FastAPI()
    test_app.include_router(router)
    from src.api.routes import get_pipeline_service
    test_app.dependency_overrides[get_pipeline_service] = lambda: mock_service

    with TestClient(test_app) as tc:
        resp = tc.get("/api/v1/ready")
        assert resp.status_code == 503
        assert "initializing or unavailable" in resp.text


def test_zero_verifiable_claims(client: TestClient) -> None:
    # Response containing purely conversational/opinion text with 0 factual claims
    payload = {
        "response": "In my opinion, this is the most terrible movie ever made. Please avoid it! Hope this helps.",
        "reference_context": "The Eiffel Tower is located in Paris, France.",
        "sample_id": "zero_claim_test",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_factual_claims"] == 0
    assert data["verified_claims"] == []
    assert data["hallucination_risk"] == 0.0
    assert data["support_rate"] == 0.0
    assert data["evidence_coverage"] == 0.0
    assert data["abstention_rate"] == 0.0
    assert len(data["excluded_claims"]) >= 1

