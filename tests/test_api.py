"""Тести. Справжній Gemini API підмінюється заглушкою, тому ключ і мережа не потрібні."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.main import app, get_generator
from app.podcast import PodcastGenerator
from app.schemas import Line, Script

SCRIPT = Script(
    title="Як працюють трансформери",
    summary="Пояснюємо механізм уваги.",
    lines=[
        Line(speaker="Олена", text="Вітаю! Сьогодні говоримо про трансформери."),
        Line(speaker="Віталій", text="Так, це архітектура, на якій працюють сучасні LLM."),
    ],
)


@pytest.fixture
def gen():
    mock = MagicMock(spec=PodcastGenerator)
    mock.text_model = "gemini-test"
    mock.last_tokens = 1234
    app.dependency_overrides[get_generator] = lambda: mock
    yield mock
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ---------- REST API ----------

def test_script_from_pdf(client, gen):
    gen.write_script.return_value = SCRIPT
    res = client.post(
        "/api/script",
        files={"pdf": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"minutes": "5"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["script"]["title"] == SCRIPT.title
    assert body["tokens"] == 1234
    gen.write_script.assert_called_once_with(b"%PDF-1.4 fake", 5)


def test_script_rejects_non_pdf(client, gen):
    res = client.post("/api/script", files={"pdf": ("a.txt", b"hi", "text/plain")})
    assert res.status_code == 415


def test_script_minutes_validation(client, gen):
    res = client.post(
        "/api/script",
        files={"pdf": ("p.pdf", b"%PDF", "application/pdf")},
        data={"minutes": "60"},
    )
    assert res.status_code == 422


def test_gemini_error_is_mapped(client, gen):
    gen.write_script.side_effect = genai_errors.ClientError(
        429, {"error": {"code": 429, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
    )
    res = client.post("/api/script", files={"pdf": ("p.pdf", b"%PDF", "application/pdf")})
    assert res.status_code == 429
    assert "quota exceeded" in res.json()["detail"]


def test_missing_api_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    get_generator.cache_clear()
    res = client.post("/api/script", files={"pdf": ("p.pdf", b"%PDF", "application/pdf")})
    assert res.status_code == 503
    assert "GEMINI_API_KEY" in res.json()["detail"]


# ---------- логіка генератора ----------

def _generator_with_fake_client():
    gen = PodcastGenerator(api_key="fake")
    gen.client = MagicMock()
    return gen


def test_write_script_sends_pdf_and_schema():
    gen = _generator_with_fake_client()
    gen.client.models.generate_content.return_value = MagicMock(
        parsed=None, text=SCRIPT.model_dump_json(), usage_metadata=None
    )

    result = gen.write_script(b"%PDF", minutes=2)

    kwargs = gen.client.models.generate_content.call_args.kwargs
    pdf_part = kwargs["contents"][0]
    assert pdf_part.inline_data.mime_type == "application/pdf"
    assert kwargs["config"].response_schema is Script
    assert "2 хв" in kwargs["contents"][1]
    assert result == SCRIPT