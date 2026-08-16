import pytest
from fastapi.testclient import TestClient

from src.api.server import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)


def test_rag_endpoint_requires_query():
    response = client.post(
        "/api/rag",
        json={},
    )

    # FastAPI should reject an invalid request body.
    assert response.status_code in (400, 422)


def test_rag_endpoint_with_query():
    response = client.post(
        "/api/rag",
        json={
            "query": "What are the best places to visit in Goa?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)

    assert "answer" in data
    assert "confidence" in data
    assert "chunks" in data


def test_documents_endpoint():
    response = client.get("/api/documents")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)


def test_benchmark_endpoint():
    response = client.get("/api/benchmark")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)