"""REST-сервіс «PDF → подкаст» на основі Google Gemini API.

Запуск:  uvicorn app.main:app --reload
Документація Swagger: http://127.0.0.1:8000/docs
"""

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

app = FastAPI(
    title="PDF → Podcast",
    description="Перетворює PDF-документ на аудіоподкаст двох ведучих через Google Gemini API",
    version="1.0.0",
)


@app.get("/api/health")
def health():
    return {"status": "ok"}
