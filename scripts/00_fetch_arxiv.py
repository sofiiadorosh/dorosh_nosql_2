"""
Крок 0 (допоміжний) — отримання датасету без Kaggle.

Оригінальне завдання пропонує завантажити arXiv Dataset з Kaggle:

    kaggle datasets download -d Cornell-University/arxiv
    unzip arxiv.zip   # -> arxiv-metadata-oai-snapshot.json

Цей скрипт — заміна цього кроку: він тягне ті самі метадані напряму
з офіційного arXiv API (https://export.arxiv.org/api/query, без ключів
і без реєстрації) і записує файл `arxiv-metadata-oai-snapshot.json`
у ТОЧНО ТОМУ Ж форматі JSONL, що й Kaggle-снапшот:

    {"id", "authors", "title", "categories", "abstract",
     "versions": [{"version", "created"}], "update_date", "authors_parsed"}

Тому `01_prepare_data.py` працює без жодних змін — як із Kaggle-файлом,
так і з файлом, отриманим цим скриптом.

Чому не Kaggle: повний снапшот — це ~1.2 ГБ архіву / ~4.5 ГБ JSON, з якого
для завдання потрібно лише 10 000 записів, і він вимагає ~/.kaggle/kaggle.json.
Чому не OAI-PMH (https://oaipmh.arxiv.org/oai): запити з параметром `set`
стабільно падають у 503 first byte timeout на боці arXiv.

Корпус набирається зрізами (slices) так, щоб у ньому були і свіжі, і старі
статті, і різні категорії — інакше демонстрації з фільтрами за роком
і категорією в 04_search.py не мали б на чому працювати.
"""

import json
import os
import random
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

import requests  # requests тягне сертифікати через certifi — на macOS це важливо,
                 # бо python.org-збірка не має доступу до системного keychain

OUTPUT_FILE = "arxiv-metadata-oai-snapshot.json"
API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 200          # arXiv рекомендує не більше ~2000, стабільно працює 100-500
REQUEST_DELAY = 3.0      # arXiv просить паузу ~3 с між запитами
MAX_RETRIES = 5

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# Набираємо корпус ПОКАТЕГОРІЙНО І ПОРІЧНО: (категорія, перший рік, останній рік,
# скільки статей на рік). Якщо брати широкий діапазон дат одним запитом,
# arXiv віддає його відсортованим за датою, і в вибірку потрапляють лише
# найсвіжіші статті цього діапазону — корпус вироджується у 4-5 різних років.
# Окремий запит на кожен рік гарантує рівномірний розподіл за роками, без чого
# фільтри за роком у 04_search.py не мали б на чому працювати.
CATEGORY_PLAN = [
    # (категорія, рік_від, рік_до, статей на рік)
    ("cs.LG",    2009, 2026, 130),   # основна ML-категорія
    ("cs.CV",    2009, 2026,  95),   # computer vision -> запит про розпізнавання об'єктів
    ("cs.CL",    2009, 2026,  95),   # NLP -> BERT, аналіз тональності
    ("cs.AI",    2007, 2026,  45),
    ("stat.ML",  2012, 2026,  50),
    ("cs.NE",    2009, 2026,  35),   # нейромережі/еволюційні алгоритми
    ("hep-th",   2010, 2026,  50),   # фізика — для різноманіття категорій
    ("math.PR",  2010, 2026,  40),
    ("q-bio.NC", 2010, 2026,  30),
]

SLICES = [
    (f"{cat} {year}",
     f"cat:{cat} AND submittedDate:[{year}01010000 TO {year}12312359]",
     per_year)
    for cat, y_from, y_to, per_year in CATEGORY_PLAN
    for year in range(y_from, y_to + 1)
]

# Невеликий цільовий зріз: завдання вимагає демо-запиту за іменем автора
# ("Yann LeCun convolutional networks") у 06_hybrid_search.py. У випадковій
# вибірці з 10k статей робіт конкретного автора може просто не бути.
SLICES.append(("LeCun (author demo)", 'au:"LeCun_Y"', 60))


def http_get(url: str) -> bytes:
    """GET із ретраями: arXiv API інколи віддає порожню відповідь або 503."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "arxiv-semantic-search-hw/1.0"},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = REQUEST_DELAY * attempt
            print(f"    ! запит не вдався ({exc}); повтор через {wait:.0f} с", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"arXiv API недоступний після {MAX_RETRIES} спроб: {last_err}")


def parse_author(name: str) -> list[str]:
    """
    'Yann LeCun' -> ['LeCun', 'Yann', '']
    Формат authors_parsed у Kaggle-снапшоті: [прізвище, імена/ініціали, суфікс].
    """
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        return ["", "", ""]
    parts = name.split(" ")
    if len(parts) == 1:
        return [parts[0], "", ""]
    return [parts[-1], " ".join(parts[:-1]), ""]


def fmt_created(dt: datetime) -> str:
    """datetime -> 'Mon, 2 Apr 2007 19:18:42 GMT' (формат поля versions[].created)."""
    return f"{dt:%a}, {dt.day} {dt:%b} {dt:%Y} {dt:%H:%M:%S} GMT"


def entry_to_record(entry: ET.Element) -> dict | None:
    """Перетворює один <entry> з Atom-відповіді у рядок Kaggle-снапшота."""
    raw_id = (entry.findtext("atom:id", default="", namespaces=NS) or "").strip()
    if not raw_id:
        return None
    # http://arxiv.org/abs/2611.08487v1 -> ('2611.08487', 1)
    tail = raw_id.rsplit("/abs/", 1)[-1]
    m = re.match(r"^(.*?)v(\d+)$", tail)
    arxiv_id, version_no = (m.group(1), int(m.group(2))) if m else (tail, 1)

    title = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip()
    abstract = (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip()
    if not title or not abstract:
        return None

    published = entry.findtext("atom:published", default="", namespaces=NS) or ""
    updated = entry.findtext("atom:updated", default="", namespaces=NS) or published
    try:
        pub_dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
        upd_dt = datetime.strptime(updated, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None

    # versions[0] — завжди v1, саме з нього 01_prepare_data.py бере рік публікації
    versions = [{"version": "v1", "created": fmt_created(pub_dt)}]
    if version_no > 1:
        versions.append({"version": f"v{version_no}", "created": fmt_created(upd_dt)})

    names = [(e.findtext("atom:name", default="", namespaces=NS) or "").strip()
             for e in entry.findall("atom:author", NS)]
    names = [n for n in names if n]

    primary_el = entry.find("arxiv:primary_category", NS)
    primary = primary_el.get("term") if primary_el is not None else None
    cats = [c.get("term") for c in entry.findall("atom:category", NS) if c.get("term")]
    if primary:
        cats = [primary] + [c for c in cats if c != primary]
    if not cats:
        cats = ["unknown"]

    return {
        "id": arxiv_id,
        "authors": ", ".join(names),
        "title": title,
        "categories": " ".join(cats),
        "abstract": abstract,
        "versions": versions,
        "update_date": upd_dt.strftime("%Y-%m-%d"),
        "authors_parsed": [parse_author(n) for n in names],
    }


def fetch_slice(name: str, query: str, target: int, seen: set[str]) -> list[dict]:
    """Посторінково тягне один зріз, доки не набере `target` нових записів."""
    out: list[dict] = []
    start = 0
    empty_pages = 0
    page_size = min(max(target, 1), PAGE_SIZE)
    while len(out) < target and empty_pages < 3:
        params = urllib.parse.urlencode({
            "search_query": query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        body = http_get(f"{API_URL}?{params}")
        root = ET.fromstring(body)
        entries = root.findall("atom:entry", NS)
        if not entries:
            empty_pages += 1
            start += page_size
            time.sleep(REQUEST_DELAY)
            continue
        empty_pages = 0
        for entry in entries:
            rec = entry_to_record(entry)
            if rec is None or rec["id"] in seen:
                continue  # дедуплікація: cs.LG і stat.ML сильно перетинаються
            seen.add(rec["id"])
            out.append(rec)
            if len(out) >= target:
                break
        start += page_size
        time.sleep(REQUEST_DELAY)
    return out


def main() -> None:
    print("Збираємо метадані arXiv через офіційний API (заміна Kaggle-снапшота)")
    print(f"Зрізів (категорія × рік): {len(SLICES)}\n")
    seen: set[str] = set()
    records: list[dict] = []
    current_group = None
    group_count = 0
    for i, (name, query, target) in enumerate(SLICES, 1):
        group = name.rsplit(" ", 1)[0]
        if group != current_group:
            if current_group is not None:
                print(f"  {current_group:<22} зібрано {group_count}")
            current_group, group_count = group, 0
        got = fetch_slice(name, query, target, seen)
        records.extend(got)
        group_count += len(got)
        print(f"    [{i}/{len(SLICES)}] {name:<18} +{len(got):<4} "
              f"(всього {len(records)})", end="\r", flush=True)
    print(f"  {current_group:<22} зібрано {group_count}                    ")

    # Перемішуємо, щоб у перших 10 000 рядках (саме стільки читає 01_prepare_data.py)
    # були статті з усіх зрізів, а не лише з перших.
    random.Random(42).shuffle(records)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    print(f"\nВсього унікальних статей: {len(records)}")
    print(f"Збережено в {OUTPUT_FILE} ({size_mb:.1f} МБ, формат JSONL)")
    print("\nПриклад рядка:")
    print(json.dumps(records[0], ensure_ascii=False)[:600] + " ...")


if __name__ == "__main__":
    main()
