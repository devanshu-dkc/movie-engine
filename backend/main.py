import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from core.semantic_search import SemanticSearch
from core.tmdb_service import TMDBService

load_dotenv()

# Global service instances (loaded once during startup)
semantic_search_service = None
tmdb_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to safely instantiate heavy models on app startup."""
    global semantic_search_service, tmdb_service
    print("Initializing services...")

    # Initialize TMDB Service safely
    try:
        tmdb_service = TMDBService()
        print("TMDB Service initialized successfully.")
    except Exception as e:
        print(f"Error initializing TMDB Service: {e}")

    # Initialize Qdrant / Semantic Search safely
    try:
        semantic_search_service = SemanticSearch()
        print("Semantic Search Service initialized successfully.")
    except Exception as e:
        print(f"WARNING: Semantic Search Service failed to initialize: {e}")
        semantic_search_service = None

    yield
    print("Shutting down application...")


app = FastAPI(
    title="Movie Recommendation API",
    description="Backend API for the Movie Engine project",
    version="1.0.0",
    lifespan=lifespan
)


# Allowed CORS Origins
origins = [
    "https://movie-engine-dusky.vercel.app",
    "https://movie-engine-em7prrl40-dev-346f.vercel.app",
    "https://movie-engine-bg9836t23-dev-346f.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://movie-engine.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Movie Engine backend is running!",
        "services": {
            "tmdb": tmdb_service is not None,
            "semantic_search": semantic_search_service is not None
        }
    }


@app.get("/trending")
def get_trending(limit: int = Query(6, ge=1, le=20)):
    """Fetch daily trending movies via TMDB."""
    if not tmdb_service:
        raise HTTPException(
            status_code=503, 
            detail="TMDB service is currently unavailable."
        )
    
    try:
        trending_movies = tmdb_service.get_trending_movies(limit=limit)
        return {"results": trending_movies}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch trending movies: {str(e)}"
        )


@app.get("/recommend")
def recommend(query: str = Query(..., min_length=1), top_k: int = Query(6, ge=1, le=20)):
    """Perform semantic vector search and enrich results with TMDB posters/links."""
    if not semantic_search_service:
        raise HTTPException(
            status_code=503, 
            detail="Search engine is offline (Qdrant connection active issue). Check QDRANT_HOST and QDRANT_API_KEY environment variables."
        )
    if not tmdb_service:
        raise HTTPException(
            status_code=503, 
            detail="TMDB service is unavailable."
        )

    try:
        # 1. Vector similarity search via Qdrant
        raw_results = semantic_search_service.search(query=query, top_k=top_k)

        # 2. Enrich results with posters and IMDb links
        enriched_results = tmdb_service.enrich_movies(raw_results)

        return {
            "query": query,
            "results": enriched_results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error executing vector search: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)