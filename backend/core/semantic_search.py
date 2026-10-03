import os
import logging
from typing import List, Dict, Optional
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

try:
    from config import EMBEDDING_MODEL
except ImportError:
    from backend.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class SemanticSearch:
    def __init__(
        self,
        collection_name: str = "movies",
        model_name: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.model_name = model_name or EMBEDDING_MODEL
        
        logger.info(f"Initializing FastEmbed model: {self.model_name}...")
        self.model = TextEmbedding(model_name=self.model_name)

        qdrant_url = (
            qdrant_url
            or os.getenv("QDRANT_URL")
            or os.getenv("QDRANT_HOST", "http://localhost:6333")
        )
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        client_kwargs = {
            "url": qdrant_url,
            "timeout": 10.0,
            "check_compatibility": False,
        }
        if qdrant_api_key:
            client_kwargs["api_key"] = qdrant_api_key

        self.client = QdrantClient(**client_kwargs)

    def index_movies(self, movies: List[Dict], batch_size: int = 100):
        """
        Embeds movie plots as raw passages and upserts them to Qdrant.
        """
        total = len(movies)
        for i in range(0, total, batch_size):
            batch = movies[i : i + batch_size]
            # Embed raw plot summaries without query instructions
            texts = [m.get("plot", "").strip() for m in batch]
            embeddings = list(self.model.embed(texts))

            points = [
                models.PointStruct(
                    id=m["id"],
                    vector=emb.tolist(),
                    payload={
                        "title": m.get("title", ""),
                        "plot": m.get("plot", ""),
                        "year": m.get("year", "N/A"),
                        "release_year": m.get("release_year", "N/A"),
                    },
                )
                for m, emb in zip(batch, embeddings)
            ]

            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Upserted {min(i + batch_size, total)}/{total} movies.")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Applies BGE query instruction prefix and executes similarity search.
        """
        # BGE models require asymmetric retrieval prefix for queries
        bge_query = f"Represent this sentence for searching relevant passages: {query.strip()}"
        query_vector = list(self.model.embed([bge_query]))[0].tolist()

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