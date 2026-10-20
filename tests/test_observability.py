import json
import logging
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from app.middleware.request_context import request_context_middleware
from app.models.generation import GenerateRequest
from app.services.observability import (
    log_event, logger, request_id_context, trace_operation,
)


@pytest.fixture
def events(caplog):
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="ai_platform")
    try:
        yield lambda: [
            json.loads(record.getMessage())
            for record in caplog.records if record.name == "ai_platform"
        ]
    finally:
        logger.removeHandler(caplog.handler)


def test_logs_are_json_and_sensitive_fields_are_excluded(events):
    log_event(
        "example", status="success", prompt="PRIVATE_PROMPT",
        authorization="PRIVATE_TOKEN", password="PRIVATE_PASSWORD",
        messages=[{"content": "PRIVATE_MESSAGE"}], error="PRIVATE_EXCEPTION",
    )

    record, = events()
    assert record["event"] == "example"
    assert record["status"] == "success"
    assert "timestamp" in record
    assert "PRIVATE_" not in json.dumps(record)


def test_successful_operation_records_duration(events):
    with trace_operation("model_inference"):
        pass

    record, = events()
    assert record["event"] == "operation_completed"
    assert record["operation"] == "model_inference"
    assert record["status"] == "success"
    assert record["duration_ms"] >= 0


def test_failed_operation_logs_type_without_exception_contents(events):
    with pytest.raises(RuntimeError, match="PRIVATE_SECRET"):
        with trace_operation("model_inference"):
            raise RuntimeError("PRIVATE_SECRET")

    failed, completed = events()
    assert failed["event"] == "operation_failed"
    assert failed["error_type"] == "RuntimeError"
    assert completed["status"] == "error"
    assert completed["duration_ms"] >= 0
    assert "PRIVATE_SECRET" not in json.dumps(events())


@pytest.fixture
def observed_client():
    app = FastAPI()
    app.middleware("http")(request_context_middleware)

    @app.post("/work/{item_id}")
    def work(item_id: str, body: GenerateRequest, request: Request):
        assert request_id_context.get() == request.state.request_id
        with trace_operation("model_inference"):
            if item_id == "fail":
                raise RuntimeError("PRIVATE_EXCEPTION_SECRET")
            if item_id == "unavailable":
                raise HTTPException(status_code=503, detail="Unavailable")
        return {"ok": True}

    with TestClient(app) as client:
        yield client


def test_http_and_model_logs_share_request_id_without_payloads(observed_client, events):
    response = observed_client.post(
        "/work/PRIVATE_PATH?api_key=PRIVATE_QUERY",
        json={"prompt": "PRIVATE_PROMPT"},
        headers={"Authorization": "Bearer PRIVATE_TOKEN", "Cookie": "PRIVATE_COOKIE"},
    )

    assert response.status_code == 200
    operation, request = events()
    assert operation["event"] == "operation_completed"
    assert request["event"] == "http_request_completed"
    assert request["status_code"] == 200
    assert request["duration_ms"] >= 0
    assert request["path"] == "/work/{item_id}"
    assert request["method"] == "POST"
    assert operation["request_id"] == request["request_id"] == response.headers["X-Request-ID"]
    assert "PRIVATE_" not in json.dumps(events())
    assert request_id_context.get() is None


@pytest.mark.parametrize("path, payload, expected_status", [
    ("/work/fail", {"prompt": "PRIVATE_PROMPT"}, 500),
    ("/work/unavailable", {"prompt": "PRIVATE_PROMPT"}, 503),
    ("/work/ok", {"prompt": "PRIVATE_PROMPT", "max_new_tokens": 0}, 422),
    ("/PRIVATE_UNKNOWN_PATH", {"prompt": "PRIVATE_PROMPT"}, 404),
])
def test_http_failure_logs_status_and_safe_metadata(observed_client, events, path, payload, expected_status):
    response = observed_client.post(path, json=payload)

    assert response.status_code == expected_status
    failed, completed = events()[-2:]
    assert failed["event"] == "http_request_failed"
    assert completed["event"] == "http_request_completed"
    assert failed["status_code"] == completed["status_code"] == expected_status
    assert completed["duration_ms"] >= 0
    assert all(record["request_id"] == response.headers["X-Request-ID"] for record in events())
    assert "PRIVATE_" not in json.dumps(events())
    if expected_status == 500:
        assert failed["error_type"] == "RuntimeError"
        assert response.json() == {"detail": "Internal server error"}
    assert request_id_context.get() is None


def test_generate_inference_failure_is_traced_without_prompt_or_exception(events):
    from app import main

    with (
        patch.object(main.model_gateway.model, "generate", side_effect=RuntimeError("PRIVATE_SECRET")),
        TestClient(main.app) as client,
    ):
        response = client.post("/generate", json={"prompt": "PRIVATE_PROMPT"})

    assert response.status_code == 503
    records = events()
    assert [record["event"] for record in records] == [
        "operation_failed", "operation_completed",
        "http_request_failed", "http_request_completed",
    ]
    assert records[0]["operation"] == "model_inference"
    assert records[0]["error_type"] == "RuntimeError"
    assert records[1]["duration_ms"] >= 0
    assert all(record["request_id"] == response.headers["X-Request-ID"] for record in records)
    assert "PRIVATE_" not in json.dumps(records)