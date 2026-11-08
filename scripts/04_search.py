"""
Крок 4 — пошукові запити.

1) Чистий семантичний пошук у Pinecone.
2) Пошук із фільтрами за метаданими (рік, категорія).
3) Порівняння метрик схожості (cosine / dot product / L2) на локальних ембеддингах.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
EMBEDDINGS_FILE = "embeddings/embeddings.npy"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract

CURRENT_YEAR = datetime.now().year


# --- допоміжні функції ------------------------------------------------------
def encode_query(text: str) -> list[float]:
    """Кодує текстовий запит у нормалізований 768-вимірний вектор."""
    vec = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32).tolist()


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show_pinecone_results(res) -> None:
    if not res["matches"]:
        print("  (нічого не знайдено)")
        return
    for rank, match in enumerate(res["matches"], 1):
        md = match["metadata"]
        # Повний abstract беремо з parquet за arxiv_id (в Pinecone лежить обрізаний)
        full = df.loc[df["id"] == md["arxiv_id"], "abstract"]
        abstract = full.iloc[0] if len(full) else md.get("abstract", "")
        print(f"\n  {rank}. [{match['score']:.4f}] {md['title']}")
        print(f"     {md['category']} · {md['year']} · arXiv:{md['arxiv_id']}")
        print(f"     {md['authors'][:90]}")
        print(f"     {abstract[:220].strip()}...")


def show_local_results(order, scores, score_name: str) -> None:
    for rank, i in enumerate(order, 1):
        row = df.iloc[i]
        print(f"  {rank}. [{score_name}={scores[i]:+.4f}] {row['title'][:72]}")
        print(f"     {row['category']} · {row['year']} · arXiv:{row['id']}")


# ============================================================================
# 1. ЧИСТИЙ СЕМАНТИЧНИЙ ПОШУК
# ============================================================================
QUERY = "teaching machines to recognize objects in pictures"

header(f"1. СЕМАНТИЧНИЙ ПОШУК\nЗапит: \"{QUERY}\"")
print("\nУ запиті немає жодного з термінів 'computer vision', 'object detection',")
print("'image classification' — збіг має бути за змістом, а не за словами.")

res = index.query(vector=encode_query(QUERY), top_k=TOP_K, include_metadata=True)
show_pinecone_results(res)


# ============================================================================
# 2. ПОШУК ІЗ ФІЛЬТРАМИ
# ============================================================================
RL_QUERY = "reinforcement learning agents that learn from rewards"

header(f"2. ПОШУК ІЗ ФІЛЬТРАМИ\nЗапит: \"{RL_QUERY}\"")

rl_vector = encode_query(RL_QUERY)

# --- Приклад A: останні 5 років + категорія cs.LG ---------------------------
year_from = CURRENT_YEAR - 5
filter_a = {"year": {"$gte": year_from}, "category": {"$eq": "cs.LG"}}
print(f"\n--- Приклад A: year >= {year_from} AND category == 'cs.LG' ---")
print(f"    filter = {filter_a}")
res_a = index.query(vector=rl_vector, top_k=TOP_K, include_metadata=True, filter=filter_a)
show_pinecone_results(res_a)

# --- Приклад B: статті до 2015 року, будь-яка категорія ---------------------
filter_b = {"year": {"$lt": 2015}}
print(f"\n--- Приклад B: year < 2015, будь-яка категорія ---")
print(f"    filter = {filter_b}")
res_b = index.query(vector=rl_vector, top_k=TOP_K, include_metadata=True, filter=filter_b)
show_pinecone_results(res_b)

# --- Без фільтра, для порівняння -------------------------------------------
print("\n--- Для порівняння: той самий запит без фільтрів ---")
res_none = index.query(vector=rl_vector, top_k=TOP_K, include_metadata=True)
show_pinecone_results(res_none)

# --- Коротке порівняння видач ----------------------------------------------
def summarize(res, label):
    if not res["matches"]:
        print(f"  {label:<22} (порожня видача)")
        return set()
    years = [m["metadata"]["year"] for m in res["matches"]]
    cats = {m["metadata"]["category"] for m in res["matches"]}
    scores = [m["score"] for m in res["matches"]]
    ids = {m["id"] for m in res["matches"]}
    print(f"  {label:<22} роки: {min(years)}-{max(years)}  "
          f"категорії: {','.join(sorted(cats)):<22} "
          f"score: {min(scores):.3f}-{max(scores):.3f}")
    return ids


print("\n--- Порівняння видач ---")
ids_none = summarize(res_none, "без фільтра")
ids_a = summarize(res_a, "A (>=%d, cs.LG)" % year_from)
ids_b = summarize(res_b, "B (<2015)")
print(f"\n  Перетин A і B:            {len(ids_a & ids_b)} документів")
print(f"  Перетин 'без фільтра' і A: {len(ids_none & ids_a)} документів")
print(f"  Перетин 'без фільтра' і B: {len(ids_none & ids_b)} документів")


# ============================================================================
# 3. ПОРІВНЯННЯ МЕТРИК СХОЖОСТІ НА ЛОКАЛЬНИХ ЕМБЕДДИНГАХ
# ============================================================================
header(f"3. ПОРІВНЯННЯ МЕТРИК\nЗапит: \"{QUERY}\"")

emb = np.load(EMBEDDINGS_FILE)            # (N, 768), вже L2-нормалізовані
q = np.array(encode_query(QUERY), dtype=np.float32)  # теж нормалізований

print(f"\nЕмбеддинги: {emb.shape}, норма запиту = {np.linalg.norm(q):.6f}")
print(f"Норми документів: min={np.linalg.norm(emb, axis=1).min():.6f}, "
      f"max={np.linalg.norm(emb, axis=1).max():.6f}")

# dot product
dot = emb @ q

# cosine similarity (ділимо на норми явно — щоб не спиратися на нормалізацію)
cosine = dot / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q))

# L2-відстань (менше = ближче), для єдиного напрямку сортування беремо -L2
l2 = np.linalg.norm(emb - q, axis=1)

top_cos = np.argsort(-cosine)[:TOP_K]
top_dot = np.argsort(-dot)[:TOP_K]
top_l2 = np.argsort(l2)[:TOP_K]

print("\n--- COSINE SIMILARITY (більше = краще) ---")
show_local_results(top_cos, cosine, "cos")

print("\n--- DOT PRODUCT (більше = краще) ---")
show_local_results(top_dot, dot, "dot")

print("\n--- L2 DISTANCE (менше = краще) ---")
show_local_results(top_l2, l2, "L2")

print("\n--- Порівняння ---")
print(f"  top-5 cosine == top-5 dot?  {list(top_cos) == list(top_dot)}")
print(f"  top-5 cosine == top-5 L2?   {list(top_cos) == list(top_l2)}")
print(f"  max |cosine - dot| по всьому корпусу: {np.abs(cosine - dot).max():.3e}")
# Для одиничних векторів:  ||a-b||^2 = 2 - 2*cos  =>  L2 = sqrt(2 - 2*cos)
predicted_l2 = np.sqrt(np.maximum(2 - 2 * cosine, 0))
print(f"  max |L2 - sqrt(2-2*cos)| :            {np.abs(l2 - predicted_l2).max():.3e}")
print("  => для нормалізованих векторів L2 — монотонно спадна функція від cosine,")
print("     тому порядок top-k збігається (з точністю до похибки float32).")
