"""
PIPELINE_UTILS.PY — BMSIT RAG
------------------------------
Shared utilities imported by all pipeline files.
Single source of truth for dynamic top_k logic.
"""

import os


# ==============================================================================
# DYNAMIC TOP-K
#
# Base k scales with query length.
# QUERY_COMPLEXITY_HINT env var (set by answer_generator for "including X,Y,Z"
# queries) adds extra chunks on top of the base.
#
# Length brackets:
#   ≤ 5 tokens   → k = 3   (very short / single concept)
#   6-10 tokens  → k = 5
#   11-15 tokens → k = 7
#   > 15 tokens  → k = 10  (long / multi-aspect)
# ==============================================================================

def compute_dynamic_top_k(query: str) -> int:
    tokens = query.strip().split()
    length = len(tokens)

    if length <= 5:
        base_k = 3
    elif length <= 10:
        base_k = 5
    elif length <= 15:
        base_k = 7
    else:
        base_k = 10

    # Bonus chunks for enumerated queries ("including X, Y, Z")
    hint = int(os.environ.get("QUERY_COMPLEXITY_HINT", "0"))
    return base_k + hint
