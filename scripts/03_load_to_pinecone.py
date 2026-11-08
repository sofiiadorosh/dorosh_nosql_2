"""
Крок 3 — створюємо індекс у Pinecone і завантажуємо туди вектори з метаданими.

Вхід:  data/arxiv_subset.parquet + embeddings/embeddings.npy
Вихід: індекс `arxiv-papers` у Pinecone (10 000 векторів по 768 вимірів)

Метрика індексу — cosine. Ембеддинги вже L2-нормалізовані в 02_embed.py,
тому cosine і dotproduct дали б ідентичне ранжування; cosine обрано як
метрику, стійку до того, що хтось покладе в індекс ненормалізований вектор.
"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
METRIC = "cosine"
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# --- Створюємо індекс (якщо не існує) ---------------------------------------
existing = [idx["name"] for idx in pc.list_indexes()]
print(f"Наявні індекси: {existing or '—'}")

if INDEX_NAME not in existing:
    print(f"Створюємо індекс '{INDEX_NAME}' (dim={VECTOR_DIM}, metric={METRIC}) ...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Чекаємо, доки індекс стане готовим до запитів
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        pass
    print("Індекс створено.")
else:
    print(f"Індекс '{INDEX_NAME}' вже існує — використовуємо його.")

index = pc.Index(INDEX_NAME)

# --- Завантажуємо дані ------------------------------------------------------
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)
assert len(df) == len(embeddings), "Кількість рядків і векторів не збігається"
assert embeddings.shape[1] == VECTOR_DIM, f"Очікується розмірність {VECTOR_DIM}"
print(f"Даних до завантаження: {len(df)} записів, вектори {embeddings.shape}")

# --- Готуємо і заливаємо батчами -------------------------------------------
# abstract обрізаємо до 500 символів: Pinecone обмежує сумарний розмір метаданих
# одного вектора до 40 KB, а повний текст у нас і так лежить у parquet
# і підтягується за arxiv_id після пошуку.
total = 0
for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Upsert у Pinecone"):
    chunk = df.iloc[start:start + BATCH_SIZE]
    vectors = []
    for pos, (_, row) in enumerate(chunk.iterrows()):
        i = start + pos
        vectors.append({
            "id": f"paper_{i}",
            "values": embeddings[i].tolist(),
            "metadata": {
                "arxiv_id": str(row["id"]),
                "title": str(row["title"])[:500],
                "abstract": str(row["abstract"])[:500],
                "authors": str(row["authors"])[:200],
                "year": int(row["year"]),
                "category": str(row["category"]),
            },
        })
    index.upsert(vectors=vectors)
    total += len(vectors)

print(f"\nВідправлено векторів: {total}")

# --- Підсумкова статистика індексу -----------------------------------------
stats = dict(index.describe_index_stats())
print(f"Всього векторів в індексі '{INDEX_NAME}': {stats.get('total_vector_count')}")
print(f"Розмірність: {stats.get('dimension')}, "
      f"заповненість індексу: {stats.get('index_fullness')}")
