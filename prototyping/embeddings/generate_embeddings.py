"""
-generates embeddings for processed chunks

-script loads chunks from chunks.json, creates embeddings using all-MiniLM-L6-v2 model,
    and saves them to embeddings.json
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

INPUT_FILE = PROJECT_ROOT / "prototyping" / "document_preprocessing" / "chunks.json"
OUTPUT_FILE = BASE_DIR / "embeddings.json"

# limit to 500 chunks for testing
MAX_CHUNKS = 500


print("Loading chunks...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# use only a subset of chunks for testing to keep the generation fast
chunks = chunks[:MAX_CHUNKS]

print(f"Loaded {len(chunks)} chunks")

# small efficient embedding model that creates 384-dimensional vectors
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# extract text from chunks (metadata such as title is kept separately)
texts = [chunk["content"] for chunk in chunks]

# convert all chunk texts into semantic vectors
print("Creating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

results = []

#combine chunk metadata with embeddings
for chunk, embedding in zip(chunks, embeddings):
    results.append({
        "id": chunk["id"],
        "title": chunk["title"],
        "section": chunk["section"],
        "content": chunk["content"],
        "links": chunk["links"],
        "embedding": embedding.tolist()
    })

# save embeddings to a file
print("Saving embeddings...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Done. Saved {len(results)} embeddings to {OUTPUT_FILE}")