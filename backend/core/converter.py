import csv
import json

CSV_PATH = "data/TMDB 10000 Movies Dataset.csv"          # Replace with your source CSV file path
JSON_OUTPUT_PATH = "data/movies_temp.json"

movies = []

with open(CSV_PATH, mode="r", encoding="utf-8-sig") as csv_file:
    reader = csv.DictReader(csv_file)
    for idx, row in enumerate(reader):
        # Map your CSV headers to standard movie fields
        title = row.get("title") or row.get("original_title") or ""
        plot = row.get("overview") or row.get("plot") or row.get("description") or ""
        genres = row.get("genres") or row.get("genre") or "Unknown"
        year = row.get("release_date", "")[:4] if row.get("release_date") else row.get("year", "N/A")
        
        if not title.strip() or not plot.strip():
            continue

        movies.append({
            "id": idx + 1,
            "title": title.strip(),
            "plot": plot.strip(),
            "genre": genres.strip(),
            "year": year,
            "release_year": year
        })

with open(JSON_OUTPUT_PATH, mode="w", encoding="utf-8") as json_file:
    json.dump(movies, json_file, indent=2, ensure_ascii=False)

print(f"Successfully converted {len(movies)} movies to {JSON_OUTPUT_PATH}")