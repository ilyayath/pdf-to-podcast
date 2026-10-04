# PDF → Подкаст

PoC-проєкт інтеграції з **Google Gemini API**: перетворює PDF-документ (статтю, главу,
конспект) на аудіоподкаст — живу розмову двох ведучих українською мовою.

- **Олена** — ведуча: ставить питання, які виникли б у слухача;
- **Віталій** — експерт: пояснює зміст документа простими словами.

## Як це працює

```
 PDF ──► [1] Gemini 3.8 Flash ──► сценарій (JSON) ──► [2] Gemini TTS ──► podcast.mp3
         читає PDF напряму        можна відредагувати    два голоси
         structured output                                в одному запиті
```

1. **Сценарій.** PDF передається моделі як файл — без власного парсингу тексту. Модель
   повертає JSON за схемою `Script` (назва, опис, список реплік) завдяки `response_schema`.
2. **Озвучення.** Сценарій відправляється у TTS-модель Gemini з `MultiSpeakerVoiceConfig`:
   кожному ведучому призначено свій голос, а кожна репліка передається окремою частиною
   з `speech_metadata` (спікер і стиль мовлення). Довгий сценарій озвучується частинами, сирий PCM
   склеюється (з короткими паузами на стиках) і кодується в MP3 64 кбіт/с.

Між кроками сценарій можна переглянути та виправити у веб-інтерфейсі.

## Запуск

1. Отримайте безкоштовний API-ключ: <https://aistudio.google.com/apikey>
2. Встановіть залежності:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
3. Скопіюйте `.env.example` у `.env` і вставте ключ:
   ```
   GEMINI_API_KEY=ваш-ключ
   ```
4. Запустіть сервер:
   ```bash
   uvicorn app.main:app --reload
   ```
   - веб-інтерфейс: <http://127.0.0.1:8000>
   - Swagger UI: <http://127.0.0.1:8000/docs>

## CLI

```bash
python cli.py article.pdf                    # → article.json (сценарій) + article.mp3
python cli.py article.json                   # озвучити відредагований сценарій
python cli.py article.pdf -m 5 -o show.wav   # ≈5 хвилин, у форматі WAV
python cli.py article.pdf --script-only      # лише сценарій
python cli.py book.pdf -f "розділ 3, ПІД-регулятор"   # великий документ: лише про потрібне
```

## REST API

| Метод | Опис |
|---|---|
| `POST /api/script` | `multipart/form-data`: `pdf` (файл), `minutes` (1–10), `focus` (необов'язково: розділ чи тема) → JSON зі сценарієм |
| `POST /api/audio?format=mp3` | JSON сценарію → `audio/mpeg` (або `format=wav` → `audio/wav`) |

```bash
curl -F "pdf=@article.pdf" -F "minutes=3" http://127.0.0.1:8000/api/script > script.json
jq .script script.json | curl -X POST -H "Content-Type: application/json" \
     -d @- http://127.0.0.1:8000/api/audio -o podcast.mp3
```

## Тести

Gemini підмінюється заглушкою, тому ключ і мережа не потрібні:

```bash
pytest
```

## Структура

```
app/
  podcast.py         генерація сценарію та озвучення (Gemini SDK)
  schemas.py         Pydantic-схема сценарію (вона ж — response_schema для Gemini)
  main.py            REST API (FastAPI)
  static/index.html  веб-інтерфейс
cli.py               консольна версія
tests/test_api.py    тести
presentation/        презентація (LaTeX Beamer)
```

## Запасні моделі

Якщо основна модель відповідає `503` (перевантажена) або `429` (вичерпано ліміт), запит
автоматично повторюється на наступній моделі зі списку:

- сценарій: `gemini-3.8-flash` → `gemini-3.5-flash` → `gemini-2.5-flash`;
- озвучення: `gemini-3.8-flash-tts` → `gemini-3.8-flash-lite-tts`.

## Обмеження

- TTS підтримує не більше двох голосів в одному запиті.
- PDF до 15 МБ (обмеження inline-запиту до API).
- Модель може спростити або неточно передати зміст — сценарій варто переглянути перед озвученням.

## Приклад

У папці [`examples/`](examples) — справжній результат для статті
[«Attention Is All You Need»](https://arxiv.org/abs/1706.03762) (15 сторінок, англ.):

- [`attention.json`](examples/attention.json) — сценарій: 16 реплік, 398 слів;
- [`attention.mp3`](examples/attention.mp3) — подкаст: 3 хв 05 с, 1,4 МБ.

Генерація: сценарій — 20 с і 12 480 токенів (`gemini-3.8-flash`), озвучення — 28 с
(`gemini-3.8-flash-tts`).
