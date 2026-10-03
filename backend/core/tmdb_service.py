import os
import re
import requests
from functools import lru_cache
from typing import Dict, List, Optional


class TMDBService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TMDB_API_KEY", "")
        self.base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()

    def _clean_title(self, title: str) -> str:
        # Strip embedded year strings like "(1999)" from titles
        return re.sub(r"\s*\(\d{4}\)", "", title).strip()

    @lru_cache(maxsize=2048)
    def get_movie_details(self, title: str, year: Optional[str] = None) -> Dict:
        if not self.api_key:
            return {"poster_url": None, "tmdb_id": None, "year": year or "N/A"}

        clean_name = self._clean_title(title)
        params = {"api_key": self.api_key, "query": clean_name}

        if year and str(year).strip() and str(year).strip().isdigit():
            params["primary_release_year"] = int(str(year).strip())

        try:
            # 1. Primary lookup: Clean Title + Year
            res = self.session.get(f"{self.base_url}/search/movie", params=params, timeout=4)
            results = res.json().get("results", [])

            # 2. Fallback lookup: Clean Title only if initial search missed
            if not results and "primary_release_year" in params:
                del params["primary_release_year"]
                res = self.session.get(f"{self.base_url}/search/movie", params=params, timeout=4)
                results = res.json().get("results", [])

            if results:
                top = results[0]
                poster_path = top.get("poster_path")
                release_date = top.get("release_date", "")
                extracted_year = release_date[:4] if (release_date and len(release_date) >= 4) else year

                return {
                    "poster_url": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
                    "tmdb_id": top.get("id"),
                    "year": extracted_year or year or "N/A",
                }
        except Exception:
            pass

        return {"poster_url": None, "tmdb_id": None, "year": year or "N/A"}

    def get_trending_movies(self, limit: int = 20) -> List[Dict]:
        if not self.api_key:
            return []

        url = f"{self.base_url}/trending/movie/day"
        params = {"api_key": self.api_key}

        try:
            res = self.session.get(url, params=params, timeout=5)
            data = res.json()
            movies = []
            for item in data.get("results", [])[:limit]:
                release_date = item.get("release_date", "")
                movies.append({
                    "id": item.get("id"),
                    "tmdb_id": item.get("id"),
                    "title": item.get("title", "Unknown"),
                    "plot": item.get("overview", ""),
                    "year": release_date[:4] if release_date else "N/A",
                    "poster_url": f"https://image.tmdb.org/t/p/w500{item.get('poster_path')}" if item.get("poster_path") else None,
                    "vote_average": item.get("vote_average"),
                    "release_date": release_date,
                })
            return movies
        except Exception:
            return []

    def enrich_movies(self, movies: List[Dict]) -> List[Dict]:
        for m in movies:
            existing_year = str(m.get("year") or m.get("release_year") or "")
            details = self.get_movie_details(m.get("title", ""), existing_year)

            m["poster_url"] = details.get("poster_url")
            m["tmdb_id"] = details.get("tmdb_id")

            # Preserve existing year or update from TMDB
            if not existing_year or existing_year == "N/A":
                m["year"] = details.get("year", "N/A")
            else:
                m["year"] = existing_year

        # Sort strictly descending from highest to lowest similarity score
        movies.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return movies