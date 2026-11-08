"""
Крок 0 (допоміжний) — завантаження датасету без Kaggle-акаунта.

Умова завдання пропонує:

    kaggle datasets download -d Cornell-University/arxiv
    unzip arxiv.zip          # -> arxiv-metadata-oai-snapshot.json

Для цього потрібен `~/.kaggle/kaggle.json` (API-токен Kaggle). Цей скрипт бере
**той самий файл** із дзеркала на HuggingFace — так само, як у попередній
домашній роботі датасет Spotify тягнувся з HF замість Kaggle:

    https://huggingface.co/datasets/jackkuo/arXiv-metadata-oai-snapshot

Це байт-у-байт оригінальний Kaggle-снапшот (перший запис у ньому — рівно той
"0704.0001 / Calculation of prompt diphoton production cross sections",
що наведений як приклад в умові завдання).

ПРОБЛЕМА РОЗМІРУ. Повний файл — 4.3 ГБ, а нам потрібні лише 10 000 записів.
Качати 4.3 ГБ заради 0.4% даних не має сенсу, тому скрипт користується
HTTP Range-запитами: він бере невеликі шматки файлу з N рівномірно
розподілених зсувів і збирає з них підмножину. Завантажується ~34 МіБ
замість 4.3 ГБ.

ЧОМУ САМЕ РІВНОМІРНО РОЗПОДІЛЕНІ ЗСУВИ, А НЕ ПЕРШІ 10 000 РЯДКІВ.
Снапшот відсортований за arXiv ID, а ID кодує дату подачі. Тому перші 10 000
рядків — це виключно статті за квітень 2007 року, переважно з hep-ph/math.
На такому корпусі фільтр "reinforcement learning за останні 5 років
і категорія cs.LG" з Частини 3 повернув би порожню видачу, а чанкінг
і гібридний пошук не мали б на чому працювати. Зсуви по всьому файлу
дають рівномірне покриття 2007-2025 і всіх категорій arXiv:

    зсув   0% -> 0704.xxxxx (2007)
    зсув  25% -> 1603.xxxxx (2016)
    зсув  50% -> 2010.xxxxx (2020)
    зсув  75% -> 2401.xxxxx (2024)

Структура рядка не змінюється — це той самий JSONL, що й з Kaggle,
тому `01_prepare_data.py` працює без жодних правок.
"""

import json
import os
import random
import sys
import time

import requests

DATASET_URL = (
    "https://huggingface.co/datasets/jackkuo/arXiv-metadata-oai-snapshot"
    "/resolve/main/arxiv-metadata-oai-snapshot.json"
)
OUTPUT_FILE = "arxiv-metadata-oai-snapshot.json"

TARGET_RECORDS = 10_400      # трохи з запасом: 01_prepare_data.py візьме 10 000
N_OFFSETS = 40               # з кількох десятків точок по всьому файлу
CHUNK_BYTES = 900_000        # скільки байтів читати з кожного зсуву
MAX_RETRIES = 4

PER_OFFSET = TARGET_RECORDS // N_OFFSETS + 1


def get_total_size() -> int:
    """Розмір файлу на HF (content-length після редіректу на CDN)."""
    resp = requests.head(DATASET_URL, allow_redirects=True, timeout=60)
    resp.raise_for_status()
    size = int(resp.headers["content-length"])
    print(f"Розмір повного снапшота: {size / 1024**3:.2f} ГБ")
    return size


def fetch_range(start: int, length: int) -> bytes:
    """Читає [start, start+length) через HTTP Range."""
    headers = {"Range": f"bytes={start}-{start + length - 1}"}
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(DATASET_URL, headers=headers, timeout=180)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = 2 ** attempt          # 2, 4, 8, 16 с — експоненційна пауза
            print(f"    ! шматок не завантажився ({exc}), "
                  f"спроба {attempt}, повтор через {wait} с", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"не вдалося прочитати діапазон {start}: {last_err}")


def records_from_chunk(blob: bytes, limit: int) -> list[dict]:
    """
    Витягує цілі JSON-рядки зі шматка байтів.
    Перший і останній рядки майже напевно обрізані посередині — відкидаємо їх.
    """
    lines = blob.split(b"\n")[1:-1]
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue   # биті межі шматка — просто пропускаємо
        if len(out) >= limit:
            break
    return out


def main() -> None:
    print("Завантажуємо arXiv-снапшот із дзеркала HuggingFace")
    print(f"  {DATASET_URL}\n")

    total = get_total_size()
    # Не чіпаємо найостанніші байти, щоб останній зсув не впав за кінець файлу
    usable = total - CHUNK_BYTES - 1
    offsets = [int(usable * i / N_OFFSETS) for i in range(N_OFFSETS)]

    print(f"Беремо {N_OFFSETS} шматків по {CHUNK_BYTES / 1000:.0f} КБ "
          f"(~{N_OFFSETS * CHUNK_BYTES / 1024**2:.0f} МБ замість "
          f"{total / 1024**3:.2f} ГБ)\n")

    seen: set[str] = set()
    records: list[dict] = []
    for i, off in enumerate(offsets, 1):
        blob = fetch_range(off, CHUNK_BYTES)
        got = 0
        for rec in records_from_chunk(blob, PER_OFFSET):
            rid = rec.get("id")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            records.append(rec)
            got += 1
        pct = 100 * off / total
        sample_id = records[-1]["id"] if records else "—"
        print(f"  [{i:>2}/{N_OFFSETS}] зсув {pct:5.1f}%  +{got:<4} записів  "
              f"(всього {len(records)}, напр. {sample_id})")

    # Перемішуємо, щоб у перших 10 000 рядках (саме стільки читає
    # 01_prepare_data.py) були статті з усіх епох, а не лише з перших зсувів.
    random.Random(42).shuffle(records)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"\nЗібрано унікальних статей: {len(records)}")
    print(f"Збережено в {OUTPUT_FILE} ({size_mb:.1f} МБ, формат JSONL)")

    years = sorted({r["versions"][0]["created"].split()[3] for r in records
                    if r.get("versions")})
    print(f"Роки в вибірці: {years[0]}-{years[-1]} ({len(years)} різних)")
    print("\nПоля першого запису:")
    print(" ", ", ".join(records[0].keys()))


if __name__ == "__main__":
    main()
