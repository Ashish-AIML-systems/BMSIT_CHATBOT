"""
PIPELINE_D.PY — BMSIT RAG
--------------------------
Two-stage: dense FAISS + CrossEncoder reranking.
Best for: long, complex, multi-aspect queries.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from .pipeline_utils import compute_dynamic_top_k

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "BMSIT INDEX")

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
reranker   = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

RERANK_HIGH = 0.75
RERANK_MED  = 0.50


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_confidence(norm_scores):
    if not norm_scores:
        return {"level": "LOW", "confidence_score": 0.0}

    top = float(norm_scores[0])
    avg = float(np.mean(norm_scores))

    if top >= RERANK_HIGH:
        level = "HIGH"
    elif top >= RERANK_MED:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {"level": level, "confidence_score": round(avg, 4)}


def run_pipeline_d(query, top_k=None):
    if top_k is None:
        top_k = compute_dynamic_top_k(query)

    ACTIVE_DOC = os.environ.get("ACTIVE_DOC")
    DOC_DIR    = os.path.join(INDEX_DIR, ACTIVE_DOC)
    FAISS_PATH = os.path.join(DOC_DIR, "faiss.index")
    CHUNK_PATH = os.path.join(DOC_DIR, "chunks.json")

    index = faiss.read_index(FAISS_PATH)
    with open(CHUNK_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Stage 1 — wide candidate pool
    query_embedding         = bi_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices         = index.search(query_embedding, 30)

    candidates = []
    for r, idx in enumerate(indices[0]):
        c = chunks[idx]
        candidates.append({
            "pipeline": "D",
            "pdf_name": ACTIVE_DOC,
            "chunk_id": c["chunk_id"],
            "page":     c.get("page", "N/A"),
            "text":     c["text"],
            "score":    float(scores[0][r])
        })

    # Stage 2 — CrossEncoder reranking
    pairs       = [[query, c["text"]] for c in candidates]
    raw_scores  = reranker.predict(pairs)
    norm_scores = np.array([sigmoid(float(s)) for s in raw_scores])

    ranked = sorted(
        zip(norm_scores, candidates), key=lambda x: x[0], reverse=True
    )[:top_k]

    results         = []
    norm_top_scores = []
    for norm_s, c in ranked:
        c["score"] = round(float(norm_s), 4)
        results.append(c)
        norm_top_scores.append(norm_s)

    return results, compute_confidence(norm_top_scores)
