"""
Крок 6 (додатковий) — гібридний пошук: BM25 + вектори, об'єднані через
Reciprocal Rank Fusion (RRF).

RRF не порівнює скори різних систем (вони в різних шкалах), а працює лише
з позиціями в ранжованих списках:

    RRF(d) = Σ_i  1 / (k + rank_i(d))

де rank_i(d) — позиція документа d у видачі i-го методу (1-based),
а k — згладжувальна константа (класичне значення з праці Cormack et al. 2009 — 60).
"""

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше, щоб RRF міг переранжувати
SHOW_K = 5
RRF_K = 60

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


# ============================================================================
# 1. Локальний BM25-індекс за заголовками й анотаціями
# ============================================================================
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


print("Будуємо BM25-індекс ...")
corpus_texts = (df["title"] + " " + df["abstract"]).tolist()
tokenized_corpus = [tokenize(t) for t in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)
print(f"  документів: {len(tokenized_corpus)}, "
      f"середня довжина: {np.mean([len(t) for t in tokenized_corpus]):.0f} токенів")
print("  УВАГА: за умовою індексуються лише title + abstract — поле authors")
print("  у BM25-документ НЕ входить (це важливо для запиту за іменем автора).\n")

# Позиція в df -> id вектора в Pinecone (03_load_to_pinecone.py клав "paper_<i>")
pos_to_vid = {i: f"paper_{i}" for i in range(len(df))}
vid_to_pos = {v: k for k, v in pos_to_vid.items()}


# ============================================================================
# 2-4. Три методи пошуку
# ============================================================================
def search_bm25(query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
    """-> [(позиція в df, bm25 score)], відсортовано за спаданням."""
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in order]


def search_vector(query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
    """-> [(позиція в df, cosine score)] з Pinecone."""
    qv = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    res = index.query(vector=qv.astype(np.float32).tolist(),
                      top_k=top_k, include_metadata=False)
    return [(vid_to_pos[m["id"]], float(m["score"])) for m in res["matches"]
            if m["id"] in vid_to_pos]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """
    rankings — список ранжованих списків (кожен: позиції документів за спаданням релевантності).
    Повертає [(позиція, rrf_score)], відсортовано за спаданням RRF.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def search_hybrid(query: str, top_k: int = TOP_K, k: int = RRF_K):
    bm = search_bm25(query, top_k)
    vec = search_vector(query, top_k)
    fused = reciprocal_rank_fusion([[d for d, _ in bm], [d for d, _ in vec]], k=k)
    bm_rank = {d: r for r, (d, _) in enumerate(bm, 1)}
    vec_rank = {d: r for r, (d, _) in enumerate(vec, 1)}
    return [(d, s, bm_rank.get(d), vec_rank.get(d)) for d, s in fused], bm, vec


# ============================================================================
# Виведення
# ============================================================================
def show_simple(results, score_name: str) -> None:
    for rank, (pos, score) in enumerate(results[:SHOW_K], 1):
        row = df.iloc[pos]
        print(f"  {rank}. [{score_name}={score:.4f}] {row['title'][:66]}")
        print(f"     {row['category']} · {row['year']} · arXiv:{row['id']}")


def show_hybrid(results) -> None:
    for rank, (pos, score, br, vr) in enumerate(results[:SHOW_K], 1):
        row = df.iloc[pos]
        src = f"BM25 #{br}" if br else "—"
        src += f" / vec #{vr}" if vr else " / —"
        print(f"  {rank}. [RRF={score:.5f}] {row['title'][:60]}")
        print(f"     {row['category']} · {row['year']} · arXiv:{row['id']}   ({src})")


QUERIES = [
    ("точний термін",        "BERT fine-tuning"),
    ("ім'я автора",          "Yann LeCun convolutional networks"),
    ("перефразування",       "making computers understand human emotions from text"),
]

summary_rows = []

for label, query in QUERIES:
    print("\n" + "=" * 78)
    print(f"ЗАПИТ ({label}): \"{query}\"")
    print("=" * 78)

    hybrid, bm, vec = search_hybrid(query)

    print(f"\n--- BM25 (топ-{SHOW_K}) ---")
    show_simple(bm, "bm25")
    print(f"\n--- ВЕКТОРНИЙ / Pinecone (топ-{SHOW_K}) ---")
    show_simple(vec, "cos")
    print(f"\n--- ГІБРИДНИЙ RRF, k={RRF_K} (топ-{SHOW_K}) ---")
    show_hybrid(hybrid)

    bm_top = {d for d, _ in bm[:SHOW_K]}
    vec_top = {d for d, _ in vec[:SHOW_K]}
    hyb_top = {d for d, _, _, _ in hybrid[:SHOW_K]}
    new_docs = hyb_top - bm_top - vec_top
    print(f"\n  Перетин BM25 ∩ вектор (топ-{SHOW_K}): {len(bm_top & vec_top)}")
    print(f"  Документів у гібридному топ-{SHOW_K}, яких немає в жодному з топ-{SHOW_K}: "
          f"{len(new_docs)}")
    for pos in new_docs:
        row = df.iloc[pos]
        br = next((r for r, (d, _) in enumerate(bm, 1) if d == pos), None)
        vr = next((r for r, (d, _) in enumerate(vec, 1) if d == pos), None)
        print(f"    + {row['title'][:58]} (BM25 #{br}, vec #{vr})")

    summary_rows.append({
        "запит": label,
        "BM25 #1": df.iloc[bm[0][0]]["title"][:42] if bm else "—",
        "Вектор #1": df.iloc[vec[0][0]]["title"][:42] if vec else "—",
        "RRF #1": df.iloc[hybrid[0][0]]["title"][:42] if hybrid else "—",
    })

    # --- Вплив параметра k -------------------------------------------------
    print(f"\n  --- Вплив k у RRF (топ-3) ---")
    for k in (1, 10, 60, 1000):
        fused, _, _ = search_hybrid(query, k=k)
        titles = " | ".join(df.iloc[d]["title"][:26] for d, _, _, _ in fused[:3])
        print(f"    k={k:>4}: {titles}")


# ============================================================================
# Підсумкова таблиця
# ============================================================================
print("\n" + "=" * 78)
print("ПІДСУМКОВА ТАБЛИЦЯ: 3 запити × 3 методи (документ на першому місці)")
print("=" * 78)
summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
