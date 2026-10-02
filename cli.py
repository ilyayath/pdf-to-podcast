"""Консольна версія: PDF → сценарій → подкаст.

Приклади:
    python cli.py article.pdf                    # створить article.wav і article.json
    python cli.py article.pdf -m 5 -o show.wav   # 5 хвилин, свій файл
    python cli.py article.pdf --script-only      # лише сценарій, без озвучення
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.podcast import ConfigError, PodcastGenerator


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="PDF → Podcast (Gemini API)")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("-m", "--minutes", type=int, default=3, help="тривалість, хв")
    parser.add_argument("-o", "--output", type=Path, help="файл .wav")
    parser.add_argument("--script-only", action="store_true", help="не озвучувати")
    args = parser.parse_args()

    try:
        gen = PodcastGenerator()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("1/2 Пишу сценарій…", file=sys.stderr)
    script = gen.write_script(args.pdf.read_bytes(), args.minutes)
    script_path = args.pdf.with_suffix(".json")
    script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n«{script.title}»\n{script.summary}\n")
    for line in script.lines:
        print(f"{line.speaker.value}: {line.text}")
    print(f"\nСценарій: {script_path} ({gen.last_tokens} токенів)", file=sys.stderr)

    if args.script_only:
        return 0

    print("2/2 Озвучую…", file=sys.stderr)
    out = args.output or args.pdf.with_suffix(".wav")
    out.write_bytes(gen.synthesize(script))
    print(f"Готово: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
