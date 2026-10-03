"""REST-сервіс «PDF → подкаст» на основі Google Gemini API.

Запуск:  uvicorn app.main:app --reload
Документація Swagger: http://127.0.0.1:8000/docs
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.genai import errors as genai_errors

from app.podcast import ConfigError, PodcastGenerator
from app.schemas import Script, ScriptResponse

load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"
MAX_PDF_BYTES = 15 * 1024 * 1024  # inline-запит до Gemini обмежений ~20 МБ
AUDIO_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav"}

app = FastAPI(
    title="PDF → Podcast",
    description="Перетворює PDF-документ на аудіоподкаст двох ведучих через Google Gemini API",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


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


@app.post(
    "/api/audio",
    response_class=Response,
    responses={200: {"content": {"audio/mpeg": {}, "audio/wav": {}}}},
)
def create_audio(
    script: Script,
    format: Literal["mp3", "wav"] = "mp3",
    gen: PodcastGenerator = Depends(get_generator),
):
    """Крок 2: сценарій → аудіо (сценарій можна відредагувати перед озвученням)."""
    audio = _call(gen.synthesize, script, format)
    return Response(
        content=audio,
        media_type=AUDIO_TYPES[format],
        headers={"Content-Disposition": f'attachment; filename="podcast.{format}"'},
    )
