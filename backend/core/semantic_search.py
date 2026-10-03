import os
import logging
from typing import List, Dict, Optional
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class SemanticSearch:
    def __init__(
        self,
        collection_name: str = "movies",
        model_name: str = "BAAI/bge-small-en-v1.5",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        
        logger.info("Initializing FastEmbed model...")
        self.model = TextEmbedding(model_name=model_name)

        qdrant_url = qdrant_url or os.getenv("QDRANT_URL") or os.getenv("QDRANT_HOST", "http://localhost:6333")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        client_kwargs = {
            "url": qdrant_url,
            "timeout": 10.0,
            "check_compatibility": False
        }
        if qdrant_api_key:
            client_kwargs["api_key"] = qdrant_api_key

        self.client = QdrantClient(**client_kwargs)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
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