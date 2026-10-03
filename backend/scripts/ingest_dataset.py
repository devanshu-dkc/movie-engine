import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from qdrant_client.models import VectorParams, Distance

# 1. Resolve relative directory anchors:
# SCRIPT_DIR  -> .../backend/scripts
# BACKEND_DIR -> .../backend
# PROJECT_ROOT -> .../ (root where data/ and .env live)
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Add both PROJECT_ROOT and BACKEND_DIR to sys.path for runtime execution
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 2. Load environment variables
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 3. Import SemanticSearch safely with type ignore to suppress LSP/linter diagnostics
try:
    from backend.core.semantic_search import SemanticSearch  # type: ignore # noqa: F401
except (ImportError, ModuleNotFoundError):
    from core.semantic_search import SemanticSearch  # type: ignore # noqa: F401

# Explicit path to the JSON dataset in root data/
JSON_DATA_PATH = PROJECT_ROOT / "data" / "movies_temp.json"
COLLECTION_NAME = "movies"
EMBEDDING_DIM = 384


def load_dataset(limit: Optional[int] = None) -> List[Dict]:
    logging.info(f"Loading JSON dataset from: {JSON_DATA_PATH}")

    if not JSON_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {JSON_DATA_PATH}")

    with open(JSON_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_movies = data if isinstance(data, list) else data.get("movies", [])
    movies = []

    for idx, item in enumerate(raw_movies):
        if limit and len(movies) >= limit:
            break

        plot = str(item.get("plot") or item.get("overview", "")).strip()
        title = str(item.get("title", "")).strip()

        if not plot or not title:
            continue

        year = str(item.get("year") or item.get("release_year") or "N/A").strip()
        director = str(item.get("director", "")).strip()
        genre = str(item.get("genre", "")).strip()

        movies.append({
            "id": idx + 1,
            "title": title,
            "plot": plot,
            "year": year,
            "release_year": year,
            "director": director,
            "genre": genre,
        })

    logging.info(f"Loaded {len(movies)} valid movie records.")
    return movies


def recreate_and_ingest(movies: List[Dict]):
    ss = SemanticSearch()

    logging.info(f"Checking existing collection '{COLLECTION_NAME}'...")
    if ss.client.collection_exists(collection_name=COLLECTION_NAME):
        ss.client.delete_collection(collection_name=COLLECTION_NAME)

    logging.info(f"Creating fresh Qdrant collection '{COLLECTION_NAME}' (dim={EMBEDDING_DIM})...")
    ss.client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    logging.info(f"Ingesting {len(movies)} movies into Qdrant...")
    ss.index_movies(movies, batch_size=100)
    logging.info("Ingestion completed successfully.")


if __name__ == "__main__":
    records = load_dataset()
    recreate_and_ingest(records)