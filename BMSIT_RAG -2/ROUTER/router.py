# ==============================================================================
# ROUTER.PY — BMSIT CHATBOT
# ==============================================================================
#
# Two-layer routing:
#   Layer 1 → selects best PDF from BMSIT INDEX using embedding + keyword scoring
#   Layer 2 → selects best pipeline (A-E) based on query features
#
# Features:
#   - Router Memory: avoids pipelines that failed on similar past queries
#   - Session Anchor: sticks to a document across follow-up queries
#   - Low-confidence fallback: auto-falls back to Pipeline C then E
#   - Literal page detection: forces Pipeline E for "see page N" queries
# ==============================================================================

import os
import re
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# PATHS
# ==============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, BASE_DIR)

INDEX_DIR        = os.path.join(BASE_DIR, "BMSIT INDEX")
FAILURE_LOG_PATH = os.path.join(BASE_DIR, "router_failure_log.json")

# ==============================================================================
# PIPELINE IMPORTS
# ==============================================================================

from PIPELINES.PIPELINE_A import run_pipeline_a
from PIPELINES.PIPELINE_B import run_pipeline_b
from PIPELINES.PIPELINE_C import run_pipeline_c
from PIPELINES.PIPELINE_D import run_pipeline_d
from PIPELINES.PIPELINE_E import run_pipeline_e

# ==============================================================================
# MODEL
# ==============================================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================================================================
# THRESHOLDS
# ==============================================================================

GENERAL_FALLBACK_THRESHOLD = 0.30
MEMORY_SIM_THRESHOLD       = 0.85
ANCHOR_MIN_SCORE           = 0.35
ANCHOR_SWITCH_RELATIVE     = 1.05
ANCHOR_SWITCH_FLOOR        = 0.60
ANCHOR_TTL                 = 8

LENGTH_VERY_SHORT   = 3
LENGTH_LONG_QUERY   = 15
RARITY_HIGH         = 0.55
DIVERSITY_LOW       = 0.60
DISPERSION_FOCUSED  = 0.10
DISPERSION_MIXED    = 0.11

# ==============================================================================
# WORD SETS
# ==============================================================================

COMMON_WORDS = {
    "what", "is", "the", "a", "an", "of", "in", "to", "and", "or",
    "how", "why", "when", "where", "which", "who", "does", "do",
    "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "will", "would", "could", "should", "may", "might",
    "can", "this", "that", "these", "those", "it", "its", "for",
    "with", "on", "at", "by", "from", "as", "about", "between",
    "than", "more", "less", "better", "used", "use", "using",
    "explain", "describe", "tell", "me", "give", "list", "show",
    "define", "compare", "difference", "vs", "versus",
}

STRUCTURE_TERMS = {"page", "figure", "fig", "section", "chapter", "appendix"}

TABLE_IMAGE_TERMS = {
    "table", "chart", "graph", "diagram", "image", "figure",
    "plot", "row", "column", "cell", "formula", "equation",
    "calculate", "calculation", "percentage", "percent",
}

DOC_HINT_TERMS = {
    "placement": ["placement", "company", "companies", "ctc", "recruit", "recruited", "campus"],
    "faculty data - sheet1": ["faculty", "professor", "specialization", "specializations", "hod", "staff"],
    "criteria 7.docx": ["lab", "labs", "laboratory", "laboratories", "facility", "facilities", "technical staff"],
    "student_details": ["admission", "admit", "intake", "students", "academic year", "sanctioned"],
}

SUPPLEMENT_PAGE_TERMS = {
    "how many", "current academic year", "sanctioned", "intake", "companies",
    "faculty", "specialization", "specializations", "labs", "laboratories",
    "placement", "technical staff"
}

FOLLOWUP_TERMS = {
    "it", "its", "they", "them", "their", "those", "these",
    "this", "that", "he", "she", "his", "her", "more",
    "further", "also", "again", "else", "too", "then",
}

FOLLOWUP_PHRASES = (
    "what about", "how about", "tell me more",
    "explain more", "go deeper", "and this", "and that",
    "what else", "why is that", "how does that",
)

# ==============================================================================
# SESSION ANCHOR
# ==============================================================================

_SESSION = {
    "anchor_doc":   None,
    "anchor_score": 0.0,
    "query_count":  0,
}

# ==============================================================================
# LITERAL PAGE PATTERN
# ==============================================================================

_PAGE_RE = re.compile(
    r'\bp\.?\s*(\d{1,3})\b|\bpage\s+(\d{1,3})\b',
    re.IGNORECASE,
)

# ==============================================================================
# CROSS-REFERENCE PATTERN
# ==============================================================================

_CHUNK_REF_RE = re.compile(
    r'\b(?:see|refer\s+to|shown\s+in)\s+'
    r'(?:Table|Figure|Fig\.?|Section|page|p\.?)\s*\d+'
    r'|\((?:Table|Figure|Fig\.?|Section)\s*\d+\)'
    r'|\bTable\s+\d+\b|\bFigure\s+\d+\b|\bFig\.\s*\d+\b',
    re.IGNORECASE,
)

# ==============================================================================
# ROUTER MEMORY
# ==============================================================================

def _load_failure_log() -> list:
    if not os.path.exists(FAILURE_LOG_PATH):
        return []
    try:
        with open(FAILURE_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _get_pipelines_to_avoid(query_embedding: np.ndarray) -> set:
    avoid = set()
    for entry in _load_failure_log():
        try:
            past_emb = np.array(entry["embedding"]).reshape(1, -1)
            sim = float(cosine_similarity(query_embedding, past_emb)[0][0])
            if sim >= MEMORY_SIM_THRESHOLD:
                avoid.add(entry["pipeline"])
        except Exception:
            continue
    return avoid


def _log_pipeline_failure(query: str, query_embedding: np.ndarray, pipeline: str):
    log = _load_failure_log()
    for entry in log:
        if entry.get("query") == query and entry.get("pipeline") == pipeline:
            return
    log.append({
        "query":     query,
        "pipeline":  pipeline,
        "embedding": query_embedding.flatten().tolist(),
    })
    try:
        with open(FAILURE_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
    except Exception:
        pass

# ==============================================================================
# LAYER 1 — DOCUMENT SELECTION
# ==============================================================================

def _forced_doc_from_query(query: str) -> str | None:
    q = query.lower()
    if any(term in q for term in ["placement", "company", "companies", "recruit", "recruited", "ctc", "campus"]):
        return "placement data"
    if any(term in q for term in ["faculty", "professor", "specialization", "specializations", "hod"]):
        return "faculty data - Sheet1"
    if any(term in q for term in ["lab", "labs", "laboratory", "laboratories", "facility", "facilities", "technical staff"]):
        return "criteria 7.docx"
    if any(term in q for term in ["admission", "admit", "intake", "current academic year", "sanctioned"]):
        return "student_details"
    return None


def _doc_name_bonus(query: str, doc: str) -> float:
    q = query.lower()
    for doc_key, hints in DOC_HINT_TERMS.items():
        if doc_key in doc.lower() and any(h in q for h in hints):
            return 0.15
    return 0.0


def _should_add_page_context(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in SUPPLEMENT_PAGE_TERMS)


def _keyword_score(query: str, chunks: list) -> float:
    query_words = set(query.lower().split())
    score = 0
    for chunk in chunks[:5]:
        words = set(chunk["text"].lower().split())
        score += len(query_words & words)
    return min(score / 10, 1.0)


def _score_document(query: str, query_embedding: np.ndarray, doc: str) -> float:
    doc_path = os.path.join(INDEX_DIR, doc)
    if not os.path.isdir(doc_path):
        return -1.0
    try:
        doc_emb    = np.load(os.path.join(doc_path, "doc_embedding.npy"))
        chunk_embs = np.load(os.path.join(doc_path, "embeddings.npy"))
        with open(os.path.join(doc_path, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        doc_score   = float(cosine_similarity(query_embedding, doc_emb)[0][0])
        chunk_score = float(np.max(cosine_similarity(query_embedding, chunk_embs)))
        kw_score    = _keyword_score(query, chunks)

        name_bonus = _doc_name_bonus(query, doc)
        return round(0.45 * chunk_score + 0.25 * doc_score + 0.15 * kw_score + name_bonus, 4)
    except Exception:
        return -1.0


def _is_followup_query(query: str) -> bool:
    q      = query.strip().lower()
    tokens = q.split()
    if len(tokens) <= 6:
        return True
    if any(phrase in q for phrase in FOLLOWUP_PHRASES):
        return True
    if tokens and tokens[0] in FOLLOWUP_TERMS:
        return True
    if sum(1 for t in tokens if t in FOLLOWUP_TERMS) >= 2 and len(tokens) <= 12:
        return True
    return False


def _select_document(query: str, query_embedding: np.ndarray) -> tuple:
    """
    Scores all docs in BMSIT INDEX. Manages session anchor for follow-up queries.
    Returns (doc_name, score).
    """

    if os.environ.get("LAST_RESORT"):
        current = os.environ.get("ACTIVE_DOC")
        if current:
            return current, _score_document(query, query_embedding, current)

    all_docs = [
        d for d in os.listdir(INDEX_DIR)
        if os.path.isdir(os.path.join(INDEX_DIR, d))
    ]

    if not all_docs:
        raise ValueError("[Router] No indexed documents found in BMSIT INDEX. Run index_builder.py first.")

    score_map = {doc: _score_document(query, query_embedding, doc) for doc in all_docs}

    forced_doc = _forced_doc_from_query(query)
    if forced_doc in score_map:
        best_doc = forced_doc
        best_score = score_map[best_doc]
    else:
        best_doc = max(score_map, key=score_map.get)
        best_score = score_map[best_doc]

    anchor  = _SESSION["anchor_doc"]
    q_count = _SESSION["query_count"]

    # No anchor yet
    if anchor is None:
        if best_score >= ANCHOR_MIN_SCORE:
            _SESSION["anchor_doc"]   = best_doc
            _SESSION["anchor_score"] = best_score
            _SESSION["query_count"]  = 1
        return best_doc, best_score

    # Anchor expired
    if q_count >= ANCHOR_TTL:
        _SESSION["anchor_doc"]   = None
        _SESSION["anchor_score"] = 0.0
        _SESSION["query_count"]  = 0
        return _select_document(query, query_embedding)

    _SESSION["query_count"]  += 1
    anchor_score = score_map.get(anchor, 0.0)
    _SESSION["anchor_score"] = anchor_score

    # Same doc as anchor — confirm
    if best_doc == anchor:
        return anchor, anchor_score

    # Standalone query — switch if score is meaningful
    if not _is_followup_query(query):
        if best_score >= ANCHOR_MIN_SCORE:
            _SESSION["anchor_doc"]   = best_doc
            _SESSION["anchor_score"] = best_score
        return best_doc, best_score

    # Follow-up query — only switch anchor if clearly better
    relative_win = best_score > anchor_score * ANCHOR_SWITCH_RELATIVE
    floor_win    = best_score >= ANCHOR_SWITCH_FLOOR and best_score > anchor_score

    if relative_win or floor_win:
        _SESSION["anchor_doc"]   = best_doc
        _SESSION["anchor_score"] = best_score
        return best_doc, best_score

    return anchor, anchor_score


def reset_anchor():
    _SESSION["anchor_doc"]   = None
    _SESSION["anchor_score"] = 0.0
    _SESSION["query_count"]  = 0

# ==============================================================================
# LAYER 2 — QUERY FEATURE EXTRACTION
# ==============================================================================

def _analyze_query(query: str) -> dict:
    tokens     = query.lower().split()
    length     = len(tokens)
    rarity     = sum(1 for t in tokens if t not in COMMON_WORDS) / max(length, 1)
    diversity  = len(set(tokens)) / max(length, 1)
    emb        = model.encode([query], normalize_embeddings=True)
    dispersion = float(np.std(emb))
    structure  = any(t in STRUCTURE_TERMS   for t in tokens)
    table_img  = any(t in TABLE_IMAGE_TERMS for t in tokens)

    return {
        "length":     length,
        "rarity":     round(rarity,    4),
        "diversity":  round(diversity, 4),
        "dispersion": round(dispersion,4),
        "structure":  structure,
        "table_img":  table_img,
    }

# ==============================================================================
# LAYER 2 — PIPELINE SELECTION
# ==============================================================================

def _choose_pipeline(features: dict, avoid: set) -> str:
    length     = features["length"]
    rarity     = features["rarity"]
    diversity  = features["diversity"]
    dispersion = features["dispersion"]
    structure  = features["structure"]
    table_img  = features["table_img"]

    if structure:
        preference = ["E", "C", "D", "A", "B"]
    elif table_img:
        preference = ["E", "C", "D", "A", "B"]
    elif length <= LENGTH_VERY_SHORT:
        preference = ["C", "A", "B", "D", "E"]
    elif length <= 6 and rarity >= 0.5:
        preference = ["B", "C", "D", "A", "E"]
    elif rarity >= RARITY_HIGH and length <= 12:
        preference = ["B", "C", "D", "A", "E"]
    elif diversity < DIVERSITY_LOW and length <= 10:
        preference = ["B", "C", "A", "D", "E"]
    elif length > LENGTH_LONG_QUERY:
        preference = ["D", "C", "E", "A", "B"]
    elif dispersion <= DISPERSION_FOCUSED and diversity >= 0.75:
        preference = ["A", "C", "D", "B", "E"]
    elif dispersion > DISPERSION_MIXED or (0.60 <= diversity < 0.75):
        preference = ["C", "D", "A", "B", "E"]
    else:
        preference = ["C", "D", "A", "B", "E"]

    for p in preference:
        if p not in avoid:
            return p

    return "C"  # absolute fallback

# ==============================================================================
# DISPATCHER
# ==============================================================================

_DISPATCH = {
    "A": run_pipeline_a,
    "B": run_pipeline_b,
    "C": run_pipeline_c,
    "D": run_pipeline_d,
    "E": run_pipeline_e,
}


def _dispatch(pipeline_id: str, query: str) -> tuple:
    return _DISPATCH[pipeline_id](query)

# ==============================================================================
# MAIN ROUTER
# ==============================================================================

def run_router(query: str, force_pipeline: str = None) -> dict:
    """
    Entry point.

    Returns:
        {
            "document":      str,
            "pipeline":      str,
            "chunks":        list,
            "confidence":    dict,
            "use_general_llm": bool,
        }
    """

    if not query or not query.strip():
        raise ValueError("[Router] Query cannot be empty.")

    query_embedding = model.encode([query])

    # ── Layer 1: document selection ───────────────────────────────────────────

    doc, doc_score = _select_document(query, query_embedding)
    os.environ["ACTIVE_DOC"] = doc

    if doc_score < GENERAL_FALLBACK_THRESHOLD and not force_pipeline:
        return {
            "document":        doc,
            "pipeline":        None,
            "chunks":          [],
            "confidence":      {"level": "LOW", "confidence_score": 0.0},
            "use_general_llm": True,
        }

    # ── Literal page shortcut ─────────────────────────────────────────────────

    page_match = _PAGE_RE.search(query)
    if page_match:
        num = page_match.group(1) or page_match.group(2)
        os.environ["FORCE_PAGE"] = str(num)
    else:
        os.environ.pop("FORCE_PAGE", None)

    # ── Layer 2: pipeline selection ───────────────────────────────────────────

    avoid_pipelines = set()
    if not force_pipeline:
        avoid_pipelines = _get_pipelines_to_avoid(query_embedding)

    if force_pipeline:
        pipeline_id = force_pipeline
    else:
        features    = _analyze_query(query)
        pipeline_id = _choose_pipeline(features, avoid_pipelines)

    chunks, confidence = _dispatch(pipeline_id, query)

    # ── Low-confidence fallback ───────────────────────────────────────────────

    if not force_pipeline and confidence.get("level") == "LOW":
        _log_pipeline_failure(query, query_embedding, pipeline_id)

        if pipeline_id != "C":
            chunks, confidence = _dispatch("C", query)
            pipeline_id = "C"

            if confidence.get("level") == "LOW":
                _log_pipeline_failure(query, query_embedding, "C")
                os.environ["LAST_RESORT"] = "1"
                chunks, confidence = _dispatch("E", query)
                pipeline_id = "E"
                os.environ.pop("LAST_RESORT", None)

    # ── Supplementary Pipeline E for cross-page refs ──────────────────────────

    if chunks:
        should_supplement = pipeline_id not in ("E",) and (
            any(_CHUNK_REF_RE.search(c.get("text", "")) for c in chunks) or
            _should_add_page_context(query)
        )
        if should_supplement:
            try:
                e_chunks, _ = _dispatch("E", query)
                existing = {(c.get("chunk_id"), c.get("page")) for c in chunks}
                new_chunks = [
                    c for c in e_chunks
                    if (c.get("chunk_id"), c.get("page")) not in existing
                ]
                if new_chunks:
                    chunks = chunks + new_chunks
            except Exception:
                pass

    return {
        "document":        doc,
        "pipeline":        pipeline_id,
        "chunks":          chunks,
        "confidence":      confidence,
        "use_general_llm": False,
    }
