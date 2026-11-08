"""
Крок 5 — chunking: розбиваємо довгі анотації на частини, порівнюємо дві стратегії.

Стратегії:
  * fixed-size  — рівно CHUNK_WORDS слів із перекриттям OVERLAP слів, межі речень ігноруються;
  * semantic    — речення додаються до чанка, доки сумарна довжина не перевищить CHUNK_WORDS,
                  тому кожен чанк складається з цілих речень.

Для кожної стратегії створюється окремий індекс у Pinecone
(`arxiv-chunks-fixed` і `arxiv-chunks-semantic`), після чого виконується
пошук по чанках кількома тестовими запитами.
"""

import os
import re
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
N_PAPERS = 30          # беремо 30 статей із найдовшими анотаціями
CHUNK_WORDS = 80       # цільовий розмір чанка у словах
OVERLAP = 20           # перекриття для fixed-size
BATCH_SIZE = 100
TOP_K = 5

INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")


# ============================================================================
# 1. Вибираємо 30 статей із найдовшими анотаціями
# ============================================================================
df = df.copy()
df["n_words"] = df["abstract"].str.split().str.len()
longest = df.nlargest(N_PAPERS, "n_words").reset_index(drop=True)

print("=" * 78)
print(f"1. ВИБІРКА: {N_PAPERS} статей із найдовшими анотаціями")
print("=" * 78)
print(f"Довжина анотацій (слів): min={longest['n_words'].min()}, "
      f"max={longest['n_words'].max()}, median={int(longest['n_words'].median())}")
print(f"Для порівняння, по всьому корпусу: median={int(df['n_words'].median())}, "
      f"max={df['n_words'].max()}")
print("\nТоп-5 найдовших:")
for _, r in longest.head(5).iterrows():
    print(f"  {r['n_words']:>4} слів · arXiv:{r['id']} · {r['title'][:60]}")


# ============================================================================
# 2. Дві стратегії розбиття
# ============================================================================
# Розбиття на речення: крапка/!/? + пробіл + велика літера,
# але не після поширених скорочень наукового тексту.
_ABBREV = (r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bet al\.)(?<!\bFig\.)(?<!\bEq\.)"
           r"(?<!\bRef\.)(?<!\bcf\.)(?<!\bvs\.)(?<!\bapprox\.)"
           r"(?<!\s[A-Z]\.)(?<!^[A-Z]\.)")  # ініціали авторів: "A. Smith"
_SENT_RE = re.compile(rf"{_ABBREV}(?<=[.!?])\s+(?=[A-Z(\[])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


def chunk_fixed(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP) -> list[str]:
    """Фіксований розмір у словах із перекриттям. Межі речень не враховуються."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def chunk_semantic(text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    """Додаємо цілі речення, доки чанк не перевищить max_words."""
    chunks, current, current_len = [], [], 0
    for sent in split_sentences(text):
        n = len(sent.split())
        if current and current_len + n > max_words:
            chunks.append(" ".join(current))
            current, current_len = [sent], n
        else:
            current.append(sent)
            current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def starts_mid_sentence(chunk: str) -> bool:
    """
    Груба евристика: чанк вважається розрізаним, якщо починається не з
    великої літери. Це ОЦІНКА ЗНИЗУ — чанк, що почався з власної назви
    посеред речення ("ImageNet results show..."), сюди не потрапить.
    Точніший підрахунок (звірка з реальними межами речень) дає ~80%
    проти 76% у цієї евристики; висновок від цього не змінюється.
    """
    first = chunk.lstrip()[:1]
    return bool(first) and not (first.isupper() or first in "([")


def ends_mid_sentence(chunk: str) -> bool:
    return not chunk.rstrip().endswith((".", "!", "?"))


print("\n" + "=" * 78)
print(f"2. РОЗБИТТЯ НА ЧАНКИ (size={CHUNK_WORDS} слів, overlap={OVERLAP} для fixed)")
print("=" * 78)

records = {"fixed": [], "semantic": []}
for _, row in longest.iterrows():
    for strategy, fn in (("fixed", chunk_fixed), ("semantic", chunk_semantic)):
        for i, chunk in enumerate(fn(row["abstract"])):
            records[strategy].append({
                "chunk_id": f"{row['id']}_{strategy}_{i}",
                "arxiv_id": row["id"],
                "title": row["title"],
                "text": chunk,
                "chunk_index": i,
                "year": int(row["year"]),
                "category": row["category"],
            })

for strategy in ("fixed", "semantic"):
    recs = records[strategy]
    lens = [len(r["text"].split()) for r in recs]
    cut_start = sum(starts_mid_sentence(r["text"]) for r in recs)
    cut_end = sum(ends_mid_sentence(r["text"]) for r in recs)
    print(f"\n--- {strategy.upper()} ---")
    print(f"  чанків усього:          {len(recs)}")
    print(f"  слів у чанку:           min={min(lens)}, max={max(lens)}, "
          f"avg={sum(lens) / len(lens):.1f}")
    print(f"  починається з середини речення: {cut_start}/{len(recs)} "
          f"({100 * cut_start / len(recs):.0f}%)")
    print(f"  обривається на середині речення: {cut_end}/{len(recs)} "
          f"({100 * cut_end / len(recs):.0f}%)")

print("\n--- Приклад: та сама стаття, різні стратегії ---")
sample_id = longest.iloc[0]["id"]
for strategy in ("fixed", "semantic"):
    sample = [r for r in records[strategy] if r["arxiv_id"] == sample_id][:2]
    print(f"\n  [{strategy}] перші 2 чанки статті arXiv:{sample_id}:")
    for r in sample:
        print(f"    #{r['chunk_index']} ({len(r['text'].split())} слів): ...{r['text'][:150]}...")

# --- Вплив overlap на кількість чанків і покриття --------------------------
print("\n--- Вплив розміру overlap (стратегія fixed, size=80 слів) ---")
print(f"  {'overlap':>8} | {'чанків':>7} | {'слів усього':>12} | {'надлишок':>9}")
print(f"  {'-' * 8}-+-{'-' * 7}-+-{'-' * 12}-+-{'-' * 9}")
base_words = int(longest["n_words"].sum())
for ov in (0, 10, 20, 40, 60):
    n_chunks = sum(len(chunk_fixed(a, CHUNK_WORDS, ov)) for a in longest["abstract"])
    n_words = sum(len(c.split()) for a in longest["abstract"] for c in chunk_fixed(a, CHUNK_WORDS, ov))
    print(f"  {ov:>8} | {n_chunks:>7} | {n_words:>12} | {n_words / base_words:>8.2f}x")


# ============================================================================
# 3-5. Створюємо індекси і заливаємо чанки
# ============================================================================
def ensure_index(name: str) -> None:
    existing = [i["name"] for i in pc.list_indexes()]
    if name in existing:
        print(f"  індекс '{name}' вже існує")
        return
    print(f"  створюємо індекс '{name}' ...")
    pc.create_index(
        name=name,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(name).status["ready"]:
        time.sleep(1)


def upload(index_name: str, recs: list[dict]) -> None:
    index = pc.Index(index_name)
    texts = [r["text"] for r in recs]
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)
    for start in tqdm(range(0, len(recs), BATCH_SIZE), desc=f"Upsert -> {index_name}"):
        batch = recs[start:start + BATCH_SIZE]
        index.upsert(vectors=[{
            "id": r["chunk_id"],
            "values": vecs[start + j].tolist(),
            "metadata": {
                "arxiv_id": r["arxiv_id"],
                "title": r["title"][:500],
                "text": r["text"][:1000],
                "chunk_index": r["chunk_index"],
                "year": r["year"],
                "category": r["category"],
            },
        } for j, r in enumerate(batch)])
    stats = dict(index.describe_index_stats())
    print(f"  в індексі '{index_name}': {stats.get('total_vector_count')} векторів")


print("\n" + "=" * 78)
print("3-5. ІНДЕКСИ PINECONE ДЛЯ ЧАНКІВ")
print("=" * 78)
for name in (INDEX_FIXED, INDEX_SEMANTIC):
    ensure_index(name)

upload(INDEX_FIXED, records["fixed"])
upload(INDEX_SEMANTIC, records["semantic"])


# ============================================================================
# 6. Пошук по чанках
# ============================================================================
def search_chunks(index_name: str, query: str, top_k: int = TOP_K):
    index = pc.Index(index_name)
    qv = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    return index.query(vector=qv.astype(np.float32).tolist(),
                       top_k=top_k, include_metadata=True)


TEST_QUERIES = [
    "neural network training on large datasets",
    "statistical guarantees and theoretical bounds",
    "experimental evaluation on benchmark datasets",
]

print("\n" + "=" * 78)
print("6. ПОШУК ПО ЧАНКАХ")
print("=" * 78)

for query in TEST_QUERIES:
    print(f"\n{'─' * 78}\nЗапит: \"{query}\"\n{'─' * 78}")
    for label, index_name in (("FIXED", INDEX_FIXED), ("SEMANTIC", INDEX_SEMANTIC)):
        print(f"\n  --- {label} ---")
        res = search_chunks(index_name, query)
        for rank, m in enumerate(res["matches"], 1):
            md = m["metadata"]
            print(f"  {rank}. [{m['score']:.4f}] {md['title'][:62]}")
            print(f"     чанк #{int(md['chunk_index'])} · arXiv:{md['arxiv_id']} · "
                  f"{md['category']} · {md['year']}")
            print(f"     \"{md['text'][:170].strip()}...\"")

print("\nГотово.")
