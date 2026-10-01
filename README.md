# PDF → Подкаст

PoC-проєкт інтеграції з **Google Gemini API**: перетворює PDF-документ на аудіоподкаст —
розмову двох ведучих українською мовою.

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # і вставте свій GEMINI_API_KEY
uvicorn app.main:app --reload
```
