"""
Крок 2 — кодуємо тексти статей моделлю allenai/specter2_base у вектори.

Вхід:  data/arxiv_subset.parquet
Вихід: embeddings/embeddings.npy  (shape = [N, 768], L2-нормалізовані)

Формат входу моделі: "title [SEP] abstract" — SPECTER/SPECTER2 навчалися
саме на конкатенації заголовка й анотації через роздільник [SEP],
тому цей токен обов'язковий.
"""

import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64

# 1. Завантажуємо датасет
df = pd.read_parquet(INPUT_FILE)
print(f"Прочитано записів: {len(df)}")

# 2. Готуємо тексти: title [SEP] abstract
texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()
print(f"Приклад тексту для кодування:\n{texts[0][:300]}...\n")

# 3. Завантажуємо модель
print(f"Завантажуємо модель {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
print(f"max_seq_length = {model.max_seq_length} токенів")

# 4. Кодуємо батчами з нормалізацією
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,   # ембеддинги одиничної довжини -> cosine == dot
    convert_to_numpy=True,
)
embeddings = embeddings.astype(np.float32)

# 5. Контрольний вивід
print(f"\nОброблено текстів:      {embeddings.shape[0]}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
print(f"Норма першого вектора:   {np.linalg.norm(embeddings[0]):.6f}")
print(f"Мін/макс норма по всіх:  {np.linalg.norm(embeddings, axis=1).min():.6f} / "
      f"{np.linalg.norm(embeddings, axis=1).max():.6f}")

# 6-7. Зберігаємо (директорію створюємо за потреби)
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(OUTPUT_FILE, embeddings)
size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
print(f"\nЗбережено в {OUTPUT_FILE} ({size_mb:.1f} МБ, dtype={embeddings.dtype})")
