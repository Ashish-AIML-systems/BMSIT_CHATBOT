"""
PIPELINE_B.PY — BMSIT RAG
--------------------------
Sparse BM25 retrieval with lemmatization.
Best for: rare terms, acronyms, subject codes, exam keywords.
"""

import os
import json
import numpy as np
import string
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
from .pipeline_utils import compute_dynamic_top_k

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "BMSIT INDEX")


class SparsePreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        text   = text.lower()
        text   = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        return [self.lemmatizer.lemmatize(w) for w in tokens]


prep = SparsePreprocessor()


def compute_confidence(scores):
    if not scores:
        return {"level": "LOW", "confidence_score": 0.0}

    top = float(scores[0])
    avg = float(np.mean(scores))

    if top > 5:
        level = "HIGH"
    elif top > 2:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {"level": level, "confidence_score": avg}


def run_pipeline_b(query, top_k=None):
    if top_k is None:
        top_k = compute_dynamic_top_k(query)

    ACTIVE_DOC  = os.environ.get("ACTIVE_DOC")
    DOC_DIR     = os.path.join(INDEX_DIR, ACTIVE_DOC)
    CHUNKS_PATH = os.path.join(DOC_DIR, "chunks.json")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    corpus = [prep.clean(c["text"]) for c in chunks]
    bm25   = BM25Okapi(corpus)

    q      = prep.clean(query)
    scores = bm25.get_scores(q)
    ranked = np.argsort(scores)[::-1][:top_k]

    results    = []
    score_list = []

    for idx in ranked:
        if scores[idx] <= 0:
            continue
        c = chunks[idx]
        results.append({
            "pipeline": "B",
            "pdf_name": ACTIVE_DOC,
            "chunk_id": c["chunk_id"],
            "page":     c.get("page", "N/A"),
            "text":     c["text"],
            "score":    float(scores[idx])
        })
        score_list.append(scores[idx])

    return results, compute_confidence(score_list)
