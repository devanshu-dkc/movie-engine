import logging
import os
from typing import Dict, List, Optional

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)


class SemanticSearch:

  def __init__(
      self,
      collection_name: str = "movies",
      model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
      qdrant_url: Optional[str] = None,
      qdrant_api_key: Optional[str] = None,
  ):
    self.collection_name = collection_name

    # Initialize fastembed (uses ONNX runtime on CPU, ultra-low memory)
    self.model = TextEmbedding(model_name=model_name)

    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

    # Set explicit 60-second timeout to prevent WriteTimeout on cloud calls
    if qdrant_api_key:
      self.client = QdrantClient(
          url=qdrant_url, api_key=qdrant_api_key, timeout=60.0
      )
    else:
      self.client = QdrantClient(url=qdrant_url, timeout=60.0)

    self._ensure_collection()

  def _ensure_collection(self):
    try:
      collections = [
          c.name for c in self.client.get_collections().collections
      ]
      if self.collection_name not in collections:
        # MiniLM-L6-v2 produces 384-dimensional vectors
        embedding_size = 384
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=embedding_size, distance=Distance.COSINE
            ),
        )
        logger.info(f"Created Qdrant collection: {self.collection_name}")
      else:
        logger.info(
            f"Using existing Qdrant collection: {self.collection_name}"
        )
    except Exception as e:
      logger.error(f"Error ensuring collection exists: {e}")

  def index_movies(self, movies: List[Dict], batch_size: int = 100):
    valid_movies = [
        m for m in movies if m.get("plot") and str(m["plot"]).strip()
    ]
    logger.info(
        f"Indexing {len(valid_movies)} movies into Qdrant in batches of"
        f" {batch_size}..."
    )

    total_batches = (len(valid_movies) + batch_size - 1) // batch_size
    for i in range(0, len(valid_movies), batch_size):
      batch = valid_movies[i : i + batch_size]

      rich_texts = [
          f"Title: {m.get('title', '')}. Plot: {m.get('plot', '')}"
          for m in batch
      ]

      # fastembed generates a list of numpy arrays
      embeddings = [vec.tolist() for vec in self.model.embed(rich_texts)]

      points = []
      for j, (movie, vector) in enumerate(zip(batch, embeddings)):
        point_id = movie.get("id", i + j + 1)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "title": movie.get("title", ""),
                    "plot": movie.get("plot", ""),
                    "year": str(
                        movie.get("year")
                        or movie.get("release_year")
                        or "N/A"
                    ),
                    "release_year": str(
                        movie.get("year")
                        or movie.get("release_year")
                        or "N/A"
                    ),
                },
            )
        )

      # wait=False avoids holding open HTTP write streams
      self.client.upsert(
          collection_name=self.collection_name, points=points, wait=False
      )
      logger.info(f"Indexed batch {i // batch_size + 1}/{total_batches}")

    logger.info("Dataset indexing completed successfully.")

  def search(self, query: str, top_k: int = 10) -> List[Dict]:
    # Extract query vector from fastembed generator
    query_vector = list(self.model.embed([query]))[0].tolist()

    response = self.client.query_points(
        collection_name=self.collection_name,
        query=query_vector,
        limit=top_k,
    )

    results = []
    for r in response.points:
      payload = r.payload or {}
      results.append({
          "id": r.id,
          "title": payload.get("title", "Unknown Title"),
          "plot": payload.get("plot", ""),
          "year": payload.get("year") or payload.get("release_year") or "N/A",
          "score": round(float(r.score), 4),
      })

    results.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return results