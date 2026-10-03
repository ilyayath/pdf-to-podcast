"""Генерація подкасту з PDF за допомогою Google Gemini API.

Конвеєр складається з двох викликів API:
1. PDF -> сценарій діалогу (Gemini читає PDF напряму, відповідь — JSON за схемою Script);
2. сценарій -> аудіо (Gemini TTS, два голоси в одному запиті).
"""

import io
import os
import re
import wave

import lameenc
from google import genai
from google.genai import types

from app.schemas import Line, Script, Speaker

DEFAULT_TEXT_MODEL = "gemini-3.8-flash"
DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts"

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
- не вигадуй фактів, яких немає в документі.
"""


class ConfigError(RuntimeError):
    """API-ключ не задано."""


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
        for i, chunk in enumerate(chunk_lines(script.lines)):
            response = self.client.models.generate_content(
                model=self.tts_model, contents=to_tts_contents(chunk), config=config
            )
            audio = response.candidates[0].content.parts[0].inline_data
            rate = _sample_rate(audio.mime_type)
            if i:
                pcm += silence(PAUSE_SECONDS, rate)  # природна пауза на стику частин
            pcm += audio.data
        self.last_audio_seconds = len(pcm) / 2 / rate
        if fmt == "wav":
            return pcm_to_wav(bytes(pcm), rate)
        return pcm_to_mp3(bytes(pcm), rate)
