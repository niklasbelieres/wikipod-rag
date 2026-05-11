"""
-semantic retrieval over embeddings

-loads the generated chunk embeddings, converts a user query into
    an embedding using the same SentenceTransformer model, and compares the query
    embedding with all stored chunk embeddings using cosine similarity

- most semantically relevant chunks are then returned and displayed together
    with their similarity score and a short content preview
"""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

EMBEDDINGS_FILE = PROJECT_ROOT / "prototyping" / "embeddings" / "embeddings.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def load_chunks_with_embeddings(file_path):
    #Load chunks including their embedding vectors from a JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(vector_a, vector_b):
    #Calculate cosine similarity between two vectors.
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


def search(query, chunks, model, top_k=TOP_K):
    #Search for the most similar chunks for a user query.
    query_embedding = model.encode(query)

    results = []

    # compare the query embedding with every stored chunk embedding
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        results.append({
            "score": score,
            "id": chunk["id"],
            "title": chunk["title"],
            "section": chunk["section"],
            "content": chunk["content"]
        })

    #sort by similarity score so the most relevant chunks appear first
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def main():
    if len(sys.argv) < 2:
        print("Usage: python semantic_search.py \"your search query\"")
        return

    query = " ".join(sys.argv[1:])

    if not EMBEDDINGS_FILE.exists():
        print(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        return

    print("Loading embeddings...")
    chunks = load_chunks_with_embeddings(EMBEDDINGS_FILE)

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Searching for: {query}\n")
    results = search(query, chunks, model)

    for index, result in enumerate(results, start=1):
        preview = result["content"][:500].replace("\n", " ")

        print(f"{index}. {result['title']} — {result['section']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Content: {preview}...")
        print("-" * 80)


if __name__ == "__main__":
    main()