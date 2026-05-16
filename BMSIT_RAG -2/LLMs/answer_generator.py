# ==============================================================================
# ANSWER_GENERATOR.PY — BMSIT CHATBOT
# ==============================================================================

import os
import sys
import asyncio
from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ==============================================================================
# PATH FIX
# ==============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(CURRENT_DIR)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from ROUTER.router import run_router, reset_anchor
from LLMs.ANS_EVALUATOR import AnswerEvaluator

# ==============================================================================
# CLIENT SETUP
# ==============================================================================

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set in environment variables.")

client    = Groq(api_key=api_key)
evaluator = AnswerEvaluator(groq_api_key=api_key, max_retries=3)

# ==============================================================================
# PROMPT BUILDER
# ==============================================================================

def _build_prompt(query: str, context: str) -> str:
    return f"""You are a helpful college admission assistant chatbot for BMS Institute of Technology & Management.

STRICT RULES:
- Use ONLY the given context.
- Do NOT hallucinate or add outside information.
- If the information is not present in the context, say it is not available.
- When the context contains tables, read headers and values carefully before answering.
- If the question asks about the current academic year, prefer values labeled CAY or 2025-26 when present.
- If the question asks for companies, faculty, laboratories, or other lists, mention the items explicitly from the context instead of giving a generic reply.
- Keep the answer factual and concise before adding any polite closing.

Question:
{query}

Context:
{context}

Answer:"""

# ==============================================================================
# SINGLE GENERATION CALL
# ==============================================================================

def _generate(prompt: str) -> str:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

# ==============================================================================
# PIPELINE RUNNER — passed to evaluator for retries
# ==============================================================================

async def _pipeline_runner(question: str, fallback_pipeline: str) -> str:
    """
    Called by ANS_EVALUATOR when retrying with a different pipeline.
    Runs the router with a forced pipeline and returns a fresh answer string.
    """
    router_result = run_router(question, force_pipeline=fallback_pipeline)

    chunks = router_result.get("chunks", [])
    if not chunks:
        return "I'm sorry, the information is not available in our documents."

    context = "\n\n".join(c["text"] for c in chunks)
    prompt  = _build_prompt(question, context)
    return _generate(prompt)

# ==============================================================================
# MAIN LOOP
# ==============================================================================

async def main():
    print("\nBMSIT Admission Assistant — type 'exit' to quit\n")

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() == "exit":
                break

            if not query:
                continue

            reset_anchor()

            router_result = run_router(query)

            # ── General LLM fallback ──────────────────────────────────────────
            if router_result.get("use_general_llm"):
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": query}],
                )
                print("\nAssistant:", response.choices[0].message.content.strip(), "\n")
                continue

            # ── RAG path ─────────────────────────────────────────────────────
            chunks = router_result.get("chunks", [])

            if not chunks:
                print("\nAssistant: I'm sorry, I could not find relevant information for your query.\n")
                continue

            context = "\n\n".join(c["text"] for c in chunks)
            prompt  = _build_prompt(query, context)

            initial_answer = _generate(prompt)

            # ── Evaluate + retry if incomplete ────────────────────────────────
            eval_result = await evaluator.evaluate_and_retry(
                question=query,
                initial_answer=initial_answer,
                initial_pipeline=router_result.get("pipeline"),
                pipeline_runner=_pipeline_runner,
            )

            print("\nAssistant:", eval_result["final_answer"], "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

        except EOFError:
            break

        except Exception as e:
            print(f"\nError: {e}\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    asyncio.run(main())