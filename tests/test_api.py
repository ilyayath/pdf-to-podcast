"""Тести. Справжній Gemini API підмінюється заглушкою, тому ключ і мережа не потрібні."""

import io
import wave
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.main import app, get_generator
from app.podcast import (
    PAUSE_SECONDS,
    PodcastGenerator,
    chunk_lines,
    pcm_to_mp3,
    pcm_to_wav,
    to_tts_contents,
)
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
    mock.last_model = "gemini-test"
    mock.last_tokens = 1234
    app.dependency_overrides[get_generator] = lambda: mock
    yield mock
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ---------- REST API ----------

def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Подкаст" in res.text


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
    gen.write_script.assert_called_once_with(b"%PDF-1.4 fake", 5, None)


def test_script_passes_focus(client, gen):
    gen.write_script.return_value = SCRIPT
    client.post(
        "/api/script",
        files={"pdf": ("book.pdf", b"%PDF", "application/pdf")},
        data={"focus": "розділ 3, ПІД-регулятор"},
    )
    gen.write_script.assert_called_once_with(b"%PDF", 3, "розділ 3, ПІД-регулятор")


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


def test_audio_returns_mp3_by_default(client, gen):
    gen.synthesize.return_value = b"ID3fake"
    res = client.post("/api/audio", json=SCRIPT.model_dump(mode="json"))
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/mpeg"
    assert res.content == b"ID3fake"
    gen.synthesize.assert_called_once_with(SCRIPT, "mp3")


def test_audio_wav_format(client, gen):
    gen.synthesize.return_value = b"RIFF....WAVE"
    res = client.post("/api/audio?format=wav", json=SCRIPT.model_dump(mode="json"))
    assert res.headers["content-type"] == "audio/wav"
    assert 'filename="podcast.wav"' in res.headers["content-disposition"]
    gen.synthesize.assert_called_once_with(SCRIPT, "wav")
    assert client.post("/api/audio?format=ogg", json=SCRIPT.model_dump(mode="json")).status_code == 422


def test_audio_rejects_unknown_speaker(client, gen):
    bad = {"title": "t", "summary": "s", "lines": [{"speaker": "Хтось", "text": "привіт"}]}
    assert client.post("/api/audio", json=bad).status_code == 422


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
    res = client.post("/api/audio", json=SCRIPT.model_dump(mode="json"))
    assert res.status_code == 503
    assert "GEMINI_API_KEY" in res.json()["detail"]


# ---------- логіка генератора ----------

def test_chunk_lines_respects_limit():
    lines = [Line(speaker="Олена", text="а" * 400) for _ in range(10)]
    chunks = chunk_lines(lines, limit=1000)
    assert [len(c) for c in chunks] == [2, 2, 2, 2, 2]
    assert sum(chunks, []) == lines


def test_tts_contents_mark_speaker_per_part():
    parts = to_tts_contents(SCRIPT.lines).parts
    assert [p.text for p in parts] == [line.text for line in SCRIPT.lines]
    assert [p.speech_metadata.speaker for p in parts] == ["Олена", "Віталій"]
    assert all(p.speech_metadata.style for p in parts)


def test_pcm_to_wav():
    wav = pcm_to_wav(b"\x00\x00" * 24000, rate=24000)
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        assert w.getnframes() == 24000  # рівно 1 секунда


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


def test_synthesize_uses_two_voices_and_joins_chunks(monkeypatch):
    monkeypatch.setattr("app.podcast.CHUNK_CHARS", 10)  # кожна репліка — окремий запит
    gen = _generator_with_fake_client()
    audio = MagicMock(data=b"\x01\x00" * 100, mime_type="audio/L16;codec=pcm;rate=24000")
    response = MagicMock()
    response.candidates[0].content.parts[0].inline_data = audio
    gen.client.models.generate_content.return_value = response

    wav = gen.synthesize(SCRIPT, "wav")

    calls = gen.client.models.generate_content.call_args_list
    assert len(calls) == 2
    speech = calls[0].kwargs["config"].speech_config.multi_speaker_voice_config
    assert {c.speaker for c in speech.speaker_voice_configs} == {"Олена", "Віталій"}
    pause = int(PAUSE_SECONDS * 24000)
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getnframes() == 200 + pause  # дві частини + пауза між ними
    assert gen.last_audio_seconds == pytest.approx((200 + pause) / 24000)


def test_pcm_to_mp3_is_smaller_than_wav():
    pcm = bytes(range(256)) * 375  # 2 с «шуму», 16 біт 24 кГц
    mp3 = pcm_to_mp3(pcm)
    assert mp3[:3] == b"ID3" or mp3[0] == 0xFF  # заголовок MP3
    assert len(mp3) < len(pcm_to_wav(pcm)) / 4


@pytest.mark.parametrize("focus, expected", [(None, False), ("  ", False), ("розділ 3", True)])
def test_write_script_adds_focus_only_when_given(focus, expected):
    gen = _generator_with_fake_client()
    gen.client.models.generate_content.return_value = MagicMock(
        parsed=SCRIPT, text="", usage_metadata=None
    )

    gen.write_script(b"%PDF", focus=focus)

    prompt = gen.client.models.generate_content.call_args.kwargs["contents"][1]
    assert ("«розділ 3»" in prompt) is expected
    assert "не пиши «з нами" in prompt  # документ не «гість» випуску


def _fake_response(**kwargs):
    return MagicMock(parsed=SCRIPT, text="", usage_metadata=None, **kwargs)


def test_falls_back_when_model_overloaded():
    gen = _generator_with_fake_client()
    overloaded = genai_errors.ServerError(
        503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}
    )
    gen.client.models.generate_content.side_effect = [overloaded, _fake_response()]

    gen.write_script(b"%PDF")

    models = [c.kwargs["model"] for c in gen.client.models.generate_content.call_args_list]
    assert models == [gen.text_model, "gemini-3.5-flash"]
    assert gen.last_model == "gemini-3.5-flash"


def test_no_fallback_on_client_error():
    gen = _generator_with_fake_client()
    gen.client.models.generate_content.side_effect = genai_errors.ClientError(
        400, {"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}}
    )
    with pytest.raises(genai_errors.ClientError):
        gen.write_script(b"%PDF")
    assert gen.client.models.generate_content.call_count == 1


def test_all_models_overloaded_raises_last_error():
    gen = _generator_with_fake_client()
    gen.client.models.generate_content.side_effect = genai_errors.ServerError(
        503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}
    )
    with pytest.raises(genai_errors.ServerError):
        gen.write_script(b"%PDF")
    assert gen.client.models.generate_content.call_count == 3  # основна + 2 запасні
