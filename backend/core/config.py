import os
from pathlib import Path
from dotenv import load_dotenv

# Path references
# BASE_DIR points to backend/
BASE_DIR = Path(__file__).resolve().parent

# Load local backend/.env first, then fallback to root .env
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv(dotenv_path=BASE_DIR.parent / ".env")

# === Embedding Model ===
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# === Qdrant Cloud ===
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "movies")

# === Neo4j Aura ===
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# === Optional / Data Directory ===
DATA_DIR = BASE_DIR.parent / "data"

# === External Services ===
TMDB_API_KEY = os.getenv("TMDB_API_KEY")