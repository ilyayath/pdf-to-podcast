"""Генерація подкасту з PDF за допомогою Google Gemini API.

PDF -> сценарій діалогу (Gemini читає PDF напряму, відповідь — JSON за схемою Script).
"""

import os

from google import genai
from google.genai import types

from app.schemas import Script, Speaker

DEFAULT_TEXT_MODEL = "gemini-3.8-flash"

SCRIPT_PROMPT = """\
Ти — сценарист науково-популярного подкасту українською мовою.
Перетвори прикріплений документ на живий діалог двох ведучих:
- {host} — ведуча, задає питання, які виникли б у слухача, уточнює, підсумовує;
- {expert} — експерт, пояснює ідеї документа простими словами, з прикладами й аналогіями.

Вимоги:
- тривалість озвучення приблизно {minutes} хв (~{words} слів загалом);
- почни з короткого привітання та теми випуску, заверш підсумком з 2-3 головних думок;
- репліки короткі (1-4 речення), розмовна мова, без markdown, списків і формул;
- числа, абревіатури й терміни пиши так, як їх треба вимовляти;
- не вигадуй фактів, яких немає в документі.
"""


class ConfigError(RuntimeError):
    """API-ключ не задано."""


class PodcastGenerator:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ConfigError(
                "GEMINI_API_KEY не задано. Отримайте ключ на "
                "https://aistudio.google.com/apikey і додайте його у файл .env"
            )
        self.text_model = os.getenv("GEMINI_MODEL", DEFAULT_TEXT_MODEL)
        self.client = genai.Client(api_key=api_key)
        self.last_tokens: int | None = None

    def write_script(self, pdf: bytes, minutes: int = 3) -> Script:
        """Крок 1: PDF -> сценарій. Модель отримує PDF як файл, без попереднього парсингу."""
        prompt = SCRIPT_PROMPT.format(
            host=Speaker.host.value,
            expert=Speaker.expert.value,
            minutes=minutes,
            words=minutes * 140,
        )
        response = self.client.models.generate_content(
            model=self.text_model,
            contents=[types.Part.from_bytes(data=pdf, mime_type="application/pdf"), prompt],
            config=types.GenerateContentConfig(
                temperature=0.8,
                response_mime_type="application/json",
                response_schema=Script,
            ),
        )
        meta = response.usage_metadata
        self.last_tokens = meta.total_token_count if meta else None
        if isinstance(response.parsed, Script):
            return response.parsed
        return Script.model_validate_json(response.text)
