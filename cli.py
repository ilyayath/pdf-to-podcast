"""Консольна версія: PDF → сценарій → подкаст.

Приклади:
    python cli.py article.pdf                    # створить article.mp3 і article.json
    python cli.py article.pdf -m 5 -o show.wav   # 5 хвилин, WAV
    python cli.py article.pdf --script-only      # лише сценарій, без озвучення
    python cli.py book.pdf -f "розділ 3, ПІД-регулятор"   # лише про потрібну частину
    python cli.py article.json                   # озвучити готовий (відредагований) сценарій
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google.genai import errors as genai_errors

from app.podcast import ConfigError, PodcastGenerator
from app.schemas import Script


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="PDF → Podcast (Gemini API)")
    parser.add_argument("input", type=Path, help="PDF-документ або JSON зі сценарієм")
    parser.add_argument("-m", "--minutes", type=int, default=3, help="тривалість, хв")
    parser.add_argument("-f", "--focus", help="про що говорити: розділ, тема чи питання")
    parser.add_argument("-o", "--output", type=Path, help="файл .mp3 або .wav")
    parser.add_argument("--script-only", action="store_true", help="не озвучувати")
    args = parser.parse_args()

    try:
        gen = PodcastGenerator()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.input.suffix.lower() == ".json":
        script = Script.model_validate_json(args.input.read_text(encoding="utf-8"))
    else:
        print("1/2 Пишу сценарій…", file=sys.stderr)
        t0 = time.perf_counter()
        script = gen.write_script(args.input.read_bytes(), args.minutes, args.focus)
        script_path = args.input.with_suffix(".json")
        script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n«{script.title}»\n{script.summary}\n")
        for line in script.lines:
            print(f"{line.speaker.value}: {line.text}")
        words = sum(len(line.text.split()) for line in script.lines)
        print(
            f"\nСценарій: {script_path} · {gen.last_model} · {time.perf_counter() - t0:.1f} с · "
            f"{gen.last_tokens} токенів · {len(script.lines)} реплік · {words} слів",
            file=sys.stderr,
        )

    if args.script_only:
        return 0

    out = args.output or args.input.with_suffix(".mp3")
    fmt = "wav" if out.suffix.lower() == ".wav" else "mp3"
    print("2/2 Озвучую…", file=sys.stderr)
    t0 = time.perf_counter()
    out.write_bytes(gen.synthesize(script, fmt))
    print(
        f"Готово: {out} · {gen.last_tts_model} · {time.perf_counter() - t0:.1f} с · "
        f"тривалість {gen.last_audio_seconds:.0f} с · {out.stat().st_size / 1024 / 1024:.1f} МБ",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except genai_errors.APIError as exc:
        print(f"Помилка Gemini API ({exc.code}): {exc.message}", file=sys.stderr)
        sys.exit(2)
