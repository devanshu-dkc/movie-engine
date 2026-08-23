from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from core.semantic_search import SemanticSearch
from core.tmdb_service import TMDBService

load_dotenv()

# Global service instances (loaded once during startup)
semantic_search_service = None
tmdb_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Lifecycle manager to instantiate heavy models once on app startup."""
  global semantic_search_service, tmdb_service
  print("Initializing services...")
  semantic_search_service = SemanticSearch()
  tmdb_service = TMDBService()
  print("Services initialized successfully.")
  yield
  print("Shutting down application...")


app = FastAPI(
    title="Movie Recommendation API",
    description="Backend API for the Movie Engine project",
    version="1.0.0",
    lifespan=lifespan,
)

# Allowed CORS Origins - updated with your new Vercel production URL
origins = [
    "https://movie-engine-dusky.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
  return {"status": "online", "message": "Movie Engine backend is running!"}


@app.get("/trending")
def get_trending(limit: int = Query(6, ge=1, le=20)):
  """Fetch daily trending movies via TMDB."""
  if not tmdb_service:
    raise HTTPException(
        status_code=500, detail="TMDB service not initialized."
    )

  trending_movies = tmdb_service.get_trending_movies(limit=limit)
  return {"results": trending_movies}


@app.get("/recommend")
def recommend(
    query: str = Query(..., min_length=1), top_k: int = Query(6, ge=1, le=20)
):
  """Perform semantic vector search and enrich results with TMDB posters/links."""
  if not semantic_search_service or not tmdb_service:
    raise HTTPException(
        status_code=500, detail="Search services not initialized."
    )

  # 1. Vector similarity search via Qdrant
  raw_results = semantic_search_service.search(query=query, top_k=top_k)

  # 2. Enrich results with posters and IMDb links
  enriched_results = tmdb_service.enrich_movies(raw_results)

  return {"query": query, "results": enriched_results}


if __name__ == "__main__":
  import uvicorn

  port = int(os.getenv("PORT", 8000))
  uvicorn.run("main:app", host="0.0.0.0", port=port)