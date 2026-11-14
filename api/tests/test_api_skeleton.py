"""T201: the app factory, /api/v1/health, the OpenAPI docs and the error shape."""

import pytest
from flask import Flask
from flask.testing import FlaskClient

from cfi_api import create_app


@pytest.fixture
def client() -> FlaskClient:
    return create_app().test_client()


def test_health_says_ok_and_has_no_model_yet(client: FlaskClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.content_type == "application/json"
    assert response.get_json() == {"status": "ok", "model_version": None}


def test_health_reports_the_loaded_models_version(reference_model: str) -> None:
    app = create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "v1"})
    assert app.test_client().get("/api/v1/health").get_json()["model_version"] == "v1"


def test_the_openapi_document_lists_health(client: FlaskClient) -> None:
    spec = client.get("/api/v1/openapi.json").get_json()
    assert spec["openapi"] == "3.1.0"
    assert spec["info"] == {"title": "Customer Fit Intelligence API", "version": "v1"}
    get = spec["paths"]["/api/v1/health"]["get"]
    assert get["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Health"
    }
    error = spec["components"]["schemas"]["Error"]
    assert error["required"] == ["code", "message"]
    assert set(error["properties"]) == {"code", "message", "errors"}


def test_the_docs_page_serves_swagger_ui(client: FlaskClient) -> None:
    response = client.get("/api/v1/docs")
    assert response.status_code == 200
    assert response.content_type.startswith("text/html")
    assert b"swagger-ui" in response.data


@pytest.mark.parametrize(
    ("method", "path", "status", "code"),
    [
        ("get", "/api/v1/nowhere", 404, "not_found"),
        ("post", "/api/v1/health", 405, "method_not_allowed"),
    ],
)
def test_errors_are_code_and_message(
    client: FlaskClient, method: str, path: str, status: int, code: str
) -> None:
    response = getattr(client, method)(path)
    assert response.status_code == status
    body = response.get_json()
    assert set(body) == {"code", "message"}
    assert body["code"] == code
    assert body["message"]


def _app_with_a_crash() -> Flask:
    app = create_app()

    @app.get("/api/v1/crash")
    def crash() -> str:
        raise RuntimeError("a bug in a view")

    return app


def test_a_crash_is_a_500_without_its_details() -> None:
    response = _app_with_a_crash().test_client().get("/api/v1/crash")
    assert response.status_code == 500
    body = response.get_json()
    assert body["code"] == "internal_server_error"
    assert "a bug in a view" not in response.get_data(as_text=True)
