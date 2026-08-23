import sys
from pathlib import Path
from dotenv import load_dotenv

# Set project root (parent directory of frontend/ or current directory if at root)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "frontend" else SCRIPT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from backend or root
load_dotenv(dotenv_path=PROJECT_ROOT / "backend" / ".env")
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from backend.core.recommender import Recommender


def test_recommender(query: str):
    """
    Tests the vector recommendation + TMDB enrichment pipeline.
    """
    print(f"\n{'*'*60}")
    print(f"RECOMMENDER QUERY: '{query}'")
    print(f"{'*'*60}")

    recommender = Recommender()
    
    # Use standard limit parameter matching recommender.py
    results = recommender.recommend(query=query, limit=6)

    if not results:
        print("No recommendations found.")
        return

    print(f"--- Found {len(results)} recommendations ---")
    for i, rec in enumerate(results, 1):
        score_str = f" | Score: {rec['score']:.3f}" if rec.get('score') is not None else ""
        print(
            f"{i}. {rec.get('title')} "
            f"| {rec.get('genre', 'N/A')} "
            f"| Year: {rec.get('year', 'N/A')}"
            f"{score_str} [Source: {rec.get('source', 'vector')}]"
        )


if __name__ == "__main__":
    queries_to_test = [
        "an action flick set in bombay",
        "period film set in world war 1",
        "a fun sci-fi movie about aliens visiting earth"
    ]

    for q in queries_to_test:
        test_recommender(q)