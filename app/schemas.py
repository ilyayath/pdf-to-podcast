"""Pydantic-моделі сценарію подкасту."""

from enum import Enum

from pydantic import BaseModel, Field


class Speaker(str, Enum):
    """Двоє ведучих. Імена збігаються з іменами спікерів у TTS-запиті."""

    host = "Олена"    # ведуча: ставить питання, веде розмову
    expert = "Віталій"  # експерт: пояснює зміст статті


class Line(BaseModel):
    speaker: Speaker
    text: str = Field(..., min_length=1, max_length=2000)


class Script(BaseModel):
    """Схема, за якою Gemini повертає сценарій (structured output)."""

    title: str = Field(..., description="Назва випуску подкасту")
    summary: str = Field(..., description="Про що випуск, 1-2 речення")
    lines: list[Line] = Field(..., min_length=1, description="Репліки ведучих по черзі")


class ScriptResponse(BaseModel):
    model: str
    script: Script
    tokens: int | None = None
