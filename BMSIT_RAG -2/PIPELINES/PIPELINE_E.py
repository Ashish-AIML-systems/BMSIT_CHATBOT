"""
PIPELINE_E.PY — BMSIT RAG
--------------------------
Page-level retrieval with cross-reference following.
Best for: "see page N" queries, last-resort fallback, structural references.

SPECIAL MODES:
  FORCE_PAGE  — returns exact requested page directly
  LAST_RESORT — fetches top-3 pages + follows cross-references from all of them
"""

import os
import re
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "BMSIT INDEX")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Cross-reference regex patterns (generic — works on any academic/subject doc)
_DIRECT_PAGE_RE = re.compile(
    r'\b(?:see\s+)?(?:page|p\.?)\s*(\d{1,3})\b', re.IGNORECASE
)
_NAMED_REF_RE = re.compile(
    r'\b(?:Table|Figure|Fig\.?|Section|Appendix|Equation|Eq\.?)\s+(\d+(?:\.\d+)?)\b',
    re.IGNORECASE
)


def _extract_referenced_pages(text, all_pages):
    referenced = set()

    for match in _DIRECT_PAGE_RE.finditer(text):
        try:
            referenced.add(int(match.group(1)))
        except ValueError:
            continue

    for match in _NAMED_REF_RE.finditer(text):
        label = match.group(0).lower()
        for page_data in all_pages:
            if label in page_data["text"].lower():
                referenced.add(page_data["page"])
                break

    return referenced


def _load_index_and_pages(active_doc):
    data_path  = os.path.join(INDEX_DIR, active_doc)
    index_path = os.path.join(data_path, "page_index.faiss")
    meta_path  = os.path.join(data_path, "page_metadata.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"[Pipeline E] Page index not found for '{active_doc}'. "
            f"Re-run index_builder.py to generate it."
        )

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    return index, pages


def _pages_to_results(page_list, score=1.0):
    return [
        {
            "pipeline": "E",
            "chunk_id": f"page_{p['page']}",
            "page":     p["page"],
            "text":     p["text"],
            "score":    score,
        }
        for p in page_list
    ]


def run_pipeline_e(query, top_k=2):
    ACTIVE_DOC   = os.environ.get("ACTIVE_DOC")
    index, pages = _load_index_and_pages(ACTIVE_DOC)

    # ── MODE 1: FORCE_PAGE ─────────────────────────────────────────────────────
    forced_page = os.environ.get("FORCE_PAGE")
    if forced_page:
        page_num      = int(forced_page)
        forced_chunks = [p for p in pages if p.get("page") == page_num]

        if forced_chunks:
            print(f"[Pipeline E] FORCE_PAGE {page_num} → returned directly")
            return _pages_to_results(forced_chunks, score=1.0), {
                "level": "HIGH", "confidence_score": 1.0,
                "reason": "literal_page_match"
            }
        print(f"[Pipeline E] FORCE_PAGE {page_num} not found — falling through")

    # ── MODE 2: LAST_RESORT ────────────────────────────────────────────────────
    last_resort = os.environ.get("LAST_RESORT")
    if last_resort:
        print("[Pipeline E] LAST_RESORT — fetching top-3 pages + cross-references")

        query_embedding = np.array(model.encode([query])).astype("float32")
        n_top           = min(3, len(pages))
        scores, indices = index.search(query_embedding, n_top)

        top_pages       = [pages[indices[0][i]] for i in range(n_top)]
        best_score      = float(scores[0][0])

        print(f"[Pipeline E] Best pages: {[p['page'] for p in top_pages]}")

        # Follow cross-references from ALL top pages
        ref_page_nums = set()
        for page_data in top_pages:
            refs = _extract_referenced_pages(page_data["text"], pages)
            ref_page_nums.update(refs)

        top_page_nums = {p["page"] for p in top_pages}
        ref_page_nums = [n for n in ref_page_nums if n not in top_page_nums]

        if ref_page_nums:
            print(f"[Pipeline E] Cross-references → fetching pages: {sorted(ref_page_nums)}")

        ref_pages = [p for p in pages if p["page"] in ref_page_nums]

        results = (
            _pages_to_results(top_pages, score=best_score) +
            _pages_to_results(ref_pages, score=best_score * 0.9)
        )

        confidence = {
            "level":            "HIGH" if best_score < 0.5 else "MEDIUM",
            "confidence_score": round(best_score, 4),
            "reason":           (
                f"last_resort: pages {sorted(top_page_nums)} + "
                f"{len(ref_pages)} cross-referenced"
            )
        }

        os.environ.pop("LAST_RESORT", None)
        return results, confidence

    # ── MODE 3: NORMAL SEMANTIC SCORING ───────────────────────────────────────
    query_embedding = np.array(model.encode([query])).astype("float32")
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        page_data = pages[idx]
        results.append({
            "pipeline": "E",
            "chunk_id": f"page_{page_data['page']}",
            "page":     page_data["page"],
            "text":     page_data["text"],
            "score":    float(scores[0][rank])
        })

    confidence = {
        "level":            "HIGH" if scores[0][0] < 0.5 else "MEDIUM",
        "confidence_score": float(np.mean(scores[0]))
    }

    return results, confidence
