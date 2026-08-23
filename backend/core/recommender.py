import logging
from typing import List, Dict, Set

from .semantic_search import SemanticSearch
from .tmdb_service import TMDBService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Recommender:
    def __init__(self):
        """
        Streamlined Vector + TMDB Recommender Engine.
        - Semantic Vector Search: Qdrant via SemanticSearch
        - Metadata & Poster Enrichment: TMDB API
        """
        self.semantic_search = SemanticSearch()
        self.tmdb = TMDBService()

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Helper to normalize titles for deduplication."""
        return title.strip().lower() if title else ""

    def _search_qdrant(self, query: str, limit: int = 10) -> List[Dict]:
        """Perform vector search on Qdrant and format the initial hits."""
        search_results = self.semantic_search.search(query, top_k=limit)
        return [
            {
                "title": hit.get("title"),
                "genre": hit.get("genre"),
                "director": hit.get("director"),
                "year": hit.get("year"),
                "score": round(float(hit.get("score", 0.0)), 4),
                "source": "vector",
            }
            for hit in search_results
        ]

    def recommend(self, query: str, limit: int = 8) -> List[Dict]:
        """
        Retrieves vector matches for a query and enriches candidates via TMDB.
        """
        logger.info(f"Processing recommendation query: '{query}'")
        
        # Step 1: Retrieve vector search matches
        vector_results = self._search_qdrant(query, limit=limit)
        if not vector_results:
            logger.info("No vector search matches found.")
            return []

        # Step 2: Deduplicate results safely
        seen_titles: Set[str] = set()
        deduped_candidates: List[Dict] = []

        for movie in vector_results:
            raw_title = movie.get("title")
            norm_title = self._normalize_title(raw_title)

            if norm_title and norm_title not in seen_titles:
                seen_titles.add(norm_title)
                deduped_candidates.append(movie)

        # Step 3: Concurrently enrich candidates via TMDB
        logger.info(f"Enriching {len(deduped_candidates)} unique candidates via TMDB...")
        return self.tmdb.enrich_movies(deduped_candidates)

    def get_trending(self, limit: int = 6) -> List[Dict]:
        """Fetch trending movies directly from TMDB as a fallback or landing page feature."""
        return self.tmdb.get_trending_movies(limit=limit)