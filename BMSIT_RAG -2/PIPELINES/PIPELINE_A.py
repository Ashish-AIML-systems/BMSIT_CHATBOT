"""
PIPELINE_A.PY — BMSIT RAG
--------------------------
Pure dense retrieval using FAISS with cosine similarity.
Best for: focused single-concept queries with clear semantic meaning.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .pipeline_utils import compute_dynamic_top_k

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "BMSIT INDEX")

model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_confidence(scores_array, k):
    top_score = float(scores_array[0][0])
    avg_score = float(np.mean(scores_array[0]))
    score_gap = float(scores_array[0][0] - scores_array[0][1]) if k > 1 else 0

    if top_score >= 0.50:
        level = "HIGH"
    elif top_score >= 0.40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level":            level,
        "confidence_score": round(
            (top_score * 0.6) + (avg_score * 0.3) + (score_gap * 0.1), 4
        )
    }


def run_pipeline_a(query):
    ACTIVE_DOC  = os.environ.get("ACTIVE_DOC")
    if not ACTIVE_DOC:
        raise ValueError("ACTIVE_DOC not set by router")

    DOC_DIR     = os.path.join(INDEX_DIR, ACTIVE_DOC)
    INDEX_PATH  = os.path.join(DOC_DIR, "faiss.index")
    CHUNKS_PATH = os.path.join(DOC_DIR, "chunks.json")

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    k = compute_dynamic_top_k(query)
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    scores, indices = index.search(query_embedding, k)
    confidence = compute_confidence(scores, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        c = chunks[idx]
        results.append({
            "pipeline": "A",
            "pdf_name": ACTIVE_DOC,
            "chunk_id": c["chunk_id"],
            "page":     c.get("page", "N/A"),
            "text":     c["text"],
            "score":    float(scores[0][rank])
        })

    return results, confidence
