"""Генерація подкасту з PDF за допомогою Google Gemini API.

Конвеєр складається з двох викликів API:
1. PDF -> сценарій діалогу (Gemini читає PDF напряму, відповідь — JSON за схемою Script);
2. сценарій -> аудіо (Gemini TTS, два голоси в одному запиті).
"""

import io
import logging
import os
import re
import wave

import lameenc
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.schemas import Line, Script, Speaker

log = logging.getLogger(__name__)

DEFAULT_TEXT_MODEL = "gemini-3.8-flash"
DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts"

# Запасні моделі: якщо основна перевантажена (503) чи вичерпано її ліміт (429)
TEXT_FALLBACKS = ["gemini-3.5-flash", "gemini-2.5-flash"]
TTS_FALLBACKS = ["gemini-3.8-flash-lite-tts"]  # старіші TTS не підтримують speech_metadata
RETRYABLE_CODES = {429, 500, 503}

MP3_BITRATE = 64
PAUSE_SECONDS = 0.4

VOICES = {
    Speaker.host: "Kore",
    Speaker.expert: "Puck",
}

STYLES = {
    Speaker.host: "українською, дружньо й зацікавлено, жвавий темп",
    Speaker.expert: "українською, спокійно й упевнено, як лектор, що пояснює просто",
}

# Довгий сценарій озвучуємо частинами: кожен TTS-запит має обмеження на довжину,
# а коротші запити надійніші. Аудіо частин потім склеюється.
CHUNK_CHARS = 2500

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
- не вигадуй фактів, яких немає в документі;
- документ — це матеріал для обговорення, а не учасник розмови: не пиши «з нами
  книга / посібник / стаття», натомість «сьогодні розбираємо…», «у посібнику автор пише…»;
- якщо документ великий, не намагайся переказати все: обери 3-4 найважливіші ідеї
  і розкрий їх глибше.
"""

FOCUS_PROMPT = """
Побажання користувача: «{focus}».
- Якщо це частина чи тема документа — говори насамперед про неї, а решту згадуй лише
  для контексту. Якщо в документі про це нічого немає, чесно скажи про це на початку.
- Якщо це спосіб подачі (для кого, наскільки детально, на чому наголосити) — витримай
  його в усьому випуску.
"""


class ConfigError(RuntimeError):
    """API-ключ не задано."""


def _with_fallbacks(primary: str, fallbacks: list[str]) -> list[str]:
    return [primary] + [m for m in fallbacks if m != primary]


def chunk_lines(lines: list[Line], limit: int | None = None) -> list[list[Line]]:
    """Ділить репліки на групи, щоб текст кожної групи не перевищував limit символів."""
    limit = limit or CHUNK_CHARS
    chunks: list[list[Line]] = []
    current: list[Line] = []
    size = 0
    for line in lines:
        if current and size + len(line.text) > limit:
            chunks.append(current)
            current, size = [], 0
        current.append(line)
        size += len(line.text)
    if current:
        chunks.append(current)
    return chunks


def to_tts_contents(lines: list[Line]) -> types.Content:
    """Кожна репліка — окрема частина запиту з явно вказаним спікером і стилем мовлення."""
    return types.Content(
        role="user",
        parts=[
            types.Part(
                text=line.text,
                speech_metadata=types.SpeechMetadata(
                    speaker=line.speaker.value, style=STYLES[line.speaker]
                ),
            )
            for line in lines
        ],
    )


def _sample_rate(mime_type: str | None) -> int:
    # Gemini повертає сирий PCM, напр. "audio/L16;codec=pcm;rate=24000"
    match = re.search(r"rate=(\d+)", mime_type or "")
    return int(match.group(1)) if match else 24000


def voice(name: str) -> types.VoiceConfig:
    """Один із готових голосів Gemini TTS (Kore, Puck, Charon, Aoede, ...)."""
    return types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=name))


def silence(seconds: float, rate: int = 24000) -> bytes:
    return b"\x00\x00" * int(seconds * rate)


def pcm_to_mp3(pcm: bytes, rate: int = 24000) -> bytes:
    """MP3 64 кбіт/с — для мовлення якості досить, а файл ~6 разів менший за WAV."""
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(MP3_BITRATE)
    encoder.set_in_sample_rate(rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    return bytes(encoder.encode(pcm) + encoder.flush())


def pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16 біт
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buf.getvalue()


class PodcastGenerator:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ConfigError(
                "GEMINI_API_KEY не задано. Отримайте ключ на "
                "https://aistudio.google.com/apikey і додайте його у файл .env"
            )
        self.text_model = os.getenv("GEMINI_MODEL", DEFAULT_TEXT_MODEL)
        self.tts_model = os.getenv("GEMINI_TTS_MODEL", DEFAULT_TTS_MODEL)
        self.client = genai.Client(api_key=api_key)
        self.last_tokens: int | None = None
        self.last_audio_seconds: float | None = None
        self.last_model = self.text_model
        self.last_tts_model = self.tts_model

    def _generate(self, models: list[str], **kwargs) -> tuple[str, types.GenerateContentResponse]:
        """Викликає першу модель зі списку; якщо вона перевантажена чи вичерпано ліміт — наступну."""
        for model in models:
            try:
                return model, self.client.models.generate_content(model=model, **kwargs)
            except genai_errors.APIError as exc:
                if exc.code not in RETRYABLE_CODES or model == models[-1]:
                    raise
                log.warning("Модель %s недоступна (%s), пробую наступну", model, exc.code)
        raise ValueError("Порожній список моделей")

    def write_script(self, pdf: bytes, minutes: int = 3, focus: str | None = None) -> Script:
        """Крок 1: PDF -> сценарій. Модель отримує PDF як файл, без попереднього парсингу.

        focus — необов'язковий фокус: розділ великого документа або кут подачі
        («лише висновки», «поясни для першокурсника»).
        """
        prompt = SCRIPT_PROMPT.format(
            host=Speaker.host.value,
            expert=Speaker.expert.value,
            minutes=minutes,
            words=minutes * 140,
        )
        if focus and focus.strip():
            prompt += FOCUS_PROMPT.format(focus=focus.strip())
        self.last_model, response = self._generate(
            _with_fallbacks(self.text_model, TEXT_FALLBACKS),
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

    def synthesize(self, script: Script, fmt: str = "mp3") -> bytes:
        """Крок 2: сценарій -> MP3 або WAV. Кожна частина озвучується двома голосами за один запит."""
        speech = types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(speaker=speaker.value, voice_config=voice(name))
                    for speaker, name in VOICES.items()
                ]
            )
        )
        config = types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=speech)

        pcm = bytearray()
        rate = 24000
        models = _with_fallbacks(self.tts_model, TTS_FALLBACKS)
        for i, chunk in enumerate(chunk_lines(script.lines)):
            self.last_tts_model, response = self._generate(
                models, contents=to_tts_contents(chunk), config=config
            )
            # решту частин озвучуємо тією ж моделлю, щоб голоси звучали однаково
            models = _with_fallbacks(self.last_tts_model, models)
            audio = response.candidates[0].content.parts[0].inline_data
            rate = _sample_rate(audio.mime_type)
            if i:
                pcm += silence(PAUSE_SECONDS, rate)  # природна пауза на стику частин
            pcm += audio.data
        self.last_audio_seconds = len(pcm) / 2 / rate
        if fmt == "wav":
            return pcm_to_wav(bytes(pcm), rate)
        return pcm_to_mp3(bytes(pcm), rate)
