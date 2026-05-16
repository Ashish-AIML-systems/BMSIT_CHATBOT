# ==============================================================================
# ANS_EVALUATOR.PY — BMSIT CHATBOT
# ==============================================================================

import os
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ==============================================================================
# PIPELINE FALLBACK ORDER
# ==============================================================================

FALLBACK_MAP = {
    "A": ["B", "E", "C", "D"],
    "B": ["A", "E", "C", "D"],
    "C": ["D", "E", "A", "B"],
    "D": ["E", "C", "A", "B"],
    "E": ["D", "C", "A", "B"],
}

# ==============================================================================
# HELPERS
# ==============================================================================

def _extract_sub_questions(question: str) -> list:
    parts = re.split(r"\band\b|\n", question, flags=re.IGNORECASE)
    return [p.strip() for p in parts if len(p.strip()) > 10] or [question]


def _is_complete(answer: str) -> bool:
    bad_signals = [
        "insufficient context",
        "i don't know",
        "cannot find",
        "not available",
        "i'm sorry, i could not",
    ]
    return not any(sig in answer.lower() for sig in bad_signals)

# ==============================================================================
# EVALUATOR
# ==============================================================================

class AnswerEvaluator:

    def __init__(self, groq_api_key: str = None, max_retries: int = 3):
        self.api_key     = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.max_retries = max_retries
        self.client      = Groq(api_key=self.api_key)
        self.model       = "openai/gpt-oss-120b"

    # ── Chatbot tone rewriter ─────────────────────────────────────────────────

    def _rewrite_as_chatbot(self, answer: str) -> str:
        prompt = f"""You are a college admission assistant chatbot for BMS Institute of Technology & Management.

Rewrite the following answer so that it is:
- Warm, welcoming, and polite
- Suitable for parents and students visiting for admissions
- Clear, professional, and reassuring
- Ends with an offer for further help

Do NOT change factual meaning. Do NOT add new information.

Answer:
{answer}"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    # ── Main evaluation + retry ───────────────────────────────────────────────

    async def evaluate_and_retry(
        self,
        question: str,
        initial_answer: str,
        initial_pipeline: str,
        pipeline_runner=None,
    ) -> dict:

        if _is_complete(initial_answer):
            return {
                "final_answer":    self._rewrite_as_chatbot(initial_answer),
                "pipeline_used":   initial_pipeline,
                "result":          "good",
                "all_retries_failed": False,
            }

        best_answer  = initial_answer
        best_pipeline = initial_pipeline
        fallbacks    = FALLBACK_MAP.get(initial_pipeline, [])

        for fb in fallbacks[:self.max_retries]:
            if not pipeline_runner:
                break

            try:
                retry_answer = await pipeline_runner(question, fb)
            except Exception:
                continue

            if _is_complete(retry_answer):
                best_answer   = retry_answer
                best_pipeline = fb
                break

            best_answer   = retry_answer
            best_pipeline = fb

        final_answer = self._rewrite_as_chatbot(best_answer)
        success      = _is_complete(best_answer)

        return {
            "final_answer":       final_answer,
            "pipeline_used":      best_pipeline,
            "result":             "good" if success else "bad",
            "all_retries_failed": not success,
        }