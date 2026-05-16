"""
PIPELINE_C.PY — BMSIT RAG
--------------------------
Hybrid retrieval: FAISS dense + BM25 sparse, merged by reciprocal rank fusion.
Best for: mixed signals, medium-length queries, general fallback.
"""

import os
import json
import numpy as np
import string
import faiss
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from .pipeline_utils import compute_dynamic_top_k

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "BMSIT INDEX")

model = SentenceTransformer("all-MiniLM-L6-v2")

RRF_K = 60  # standard constant for reciprocal rank fusion


class SparsePreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        text   = text.lower()
        text   = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        return [self.lemmatizer.lemmatize(w) for w in tokens]


prep = SparsePreprocessor()


def reciprocal_rank_fusion(dense_ranked, sparse_ranked, chunks, top_k):
    """
    Merges two ranked lists using RRF.
    Returns top_k chunk dicts sorted by fused score.
    """
    rrf_scores = {}

    for rank, idx in enumerate(dense_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

    for rank, idx in enumerate(sparse_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    results = []
    for idx in sorted_ids:
        c = chunks[idx]
        results.append({
            "pipeline": "C",
            "pdf_name": os.environ.get("ACTIVE_DOC"),
            "chunk_id": c["chunk_id"],
            "page":     c.get("page", "N/A"),
            "text":     c["text"],
            "score":    round(rrf_scores[idx], 6)
        })

    return results


def run_pipeline_c(query, top_k=None):
    if top_k is None:
        top_k = compute_dynamic_top_k(query)

    ACTIVE_DOC  = os.environ.get("ACTIVE_DOC")
    DOC_DIR     = os.path.join(INDEX_DIR, ACTIVE_DOC)
    FAISS_PATH  = os.path.join(DOC_DIR, "faiss.index")
    CHUNKS_PATH = os.path.join(DOC_DIR, "chunks.json")

    index = faiss.read_index(FAISS_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Dense pass
    query_embedding       = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    dense_scores, d_indices = index.search(query_embedding, min(top_k * 3, len(chunks)))
    dense_ranked          = list(d_indices[0])

    # Sparse pass
    corpus        = [prep.clean(c["text"]) for c in chunks]
    bm25          = BM25Okapi(corpus)
    q             = prep.clean(query)
    bm25_scores   = bm25.get_scores(q)
    sparse_ranked = list(np.argsort(bm25_scores)[::-1][: top_k * 3])

    # RRF merge
    results = reciprocal_rank_fusion(dense_ranked, sparse_ranked, chunks, top_k)

    top_score = results[0]["score"] if results else 0
    if top_score >= 0.015:
        level = "HIGH"
    elif top_score >= 0.010:
        level = "MEDIUM"
    else:
        level = "LOW"

    confidence = {
        "level":            level,
        "confidence_score": round(top_score, 6)
    }

    return results, confidence
