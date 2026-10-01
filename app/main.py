"""REST-сервіс «PDF → подкаст» на основі Google Gemini API.

Запуск:  uvicorn app.main:app --reload
Документація Swagger: http://127.0.0.1:8000/docs
"""

from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from google.genai import errors as genai_errors

from app.podcast import ConfigError, PodcastGenerator
from app.schemas import ScriptResponse

load_dotenv()

MAX_PDF_BYTES = 15 * 1024 * 1024  # inline-запит до Gemini обмежений ~20 МБ

app = FastAPI(
    title="PDF → Podcast",
    description="Перетворює PDF-документ на аудіоподкаст двох ведучих через Google Gemini API",
    version="1.0.0",
)


@lru_cache
def get_generator() -> PodcastGenerator:
    try:
        return PodcastGenerator()
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _call(fn, *args):
    """Перетворює помилки Gemini API на коректні HTTP-відповіді."""
    try:
        return fn(*args)
    except genai_errors.APIError as exc:
        status = exc.code if isinstance(exc.code, int) and 400 <= exc.code < 600 else 502
        raise HTTPException(status_code=status, detail=f"Gemini API: {exc.message}") from exc


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/script", response_model=ScriptResponse)
async def create_script(
    pdf: UploadFile = File(..., description="PDF-документ"),
    minutes: int = Form(3, ge=1, le=10, description="Бажана тривалість, хв"),
    gen: PodcastGenerator = Depends(get_generator),
):
    """Крок 1: PDF → сценарій діалогу (JSON)."""
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Потрібен PDF-файл")
    data = await pdf.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF більший за 15 МБ")
    script = _call(gen.write_script, data, minutes)
    return ScriptResponse(model=gen.text_model, script=script, tokens=gen.last_tokens)
