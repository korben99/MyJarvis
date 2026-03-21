"""
Test suite for web_search.py — deep search pipeline.

Run from jarvis-core/:
    python3 test_web_search.py

Tests:
  1. Weather          — Open-Meteo routing
  2. News             — DDG news routing
  3. LLM judge        — relevance assessment with good vs. bad results
  4. Query refiner    — generates a better query given thin results
  5. Deep pipeline    — stage progression for a question that needs page fetching
  6. Full search_web  — routing + deep pipeline end-to-end
"""

import asyncio
import logging
import time

# ── Logging: show pipeline stages clearly ─────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
# Quieten httpx noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from web_search import (
    _llm_judge_relevance,
    _refine_web_query,
    _ddg_text_deep,
    search_weather,
    search_news,
    search_web,
)

SEP = "─" * 70


def header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def show_results(results: list[dict], max_body: int = 200) -> None:
    if not results:
        print("  (no results)")
        return
    for i, r in enumerate(results, 1):
        body = r.get("body", "")[:max_body]
        ellipsis = "…" if len(r.get("body", "")) > max_body else ""
        print(f"  [{i}] {r.get('title', '(no title)')}")
        print(f"      {body}{ellipsis}")
        print(f"      → {r.get('url', '')}")


# ══════════════════════════════════════════════════════════════════════════
#  TEST 1 — Weather
# ══════════════════════════════════════════════════════════════════════════

async def test_weather():
    header("TEST 1 — Weather (Open-Meteo routing)")
    t0 = time.perf_counter()
    results = await search_web("météo à Paris", original_message="météo à Paris")
    elapsed = time.perf_counter() - t0
    print(f"  ✓ {len(results)} result(s) in {elapsed:.2f}s")
    show_results(results)


# ══════════════════════════════════════════════════════════════════════════
#  TEST 2 — News
# ══════════════════════════════════════════════════════════════════════════

async def test_news():
    header("TEST 2 — News (DDG news routing)")
    t0 = time.perf_counter()
    results = await search_web("actualités intelligence artificielle", original_message="quelles sont les dernières actualités sur l'IA ?")
    elapsed = time.perf_counter() - t0
    print(f"  ✓ {len(results)} result(s) in {elapsed:.2f}s")
    show_results(results, max_body=150)


# ══════════════════════════════════════════════════════════════════════════
#  TEST 3 — LLM Judge (relevance assessment)
# ══════════════════════════════════════════════════════════════════════════

async def test_judge():
    header("TEST 3 — LLM Judge (relevance assessment)")

    question = "What is the current population of Tokyo?"

    good_results = [
        {
            "title": "Tokyo population 2024",
            "body": "Tokyo's population is approximately 13.96 million in the city proper and 37.4 million in the greater metropolitan area, making it the most populous metropolitan area in the world.",
            "url": "https://example.com/tokyo-population",
        }
    ]
    bad_results = [
        {
            "title": "Japan travel tips",
            "body": "Tokyo is a great city to visit. There are many restaurants and temples.",
            "url": "https://example.com/japan-travel",
        },
        {
            "title": "Japanese culture overview",
            "body": "Japan has a rich cultural heritage including tea ceremonies and martial arts.",
            "url": "https://example.com/japanese-culture",
        },
    ]

    t0 = time.perf_counter()
    verdict_good = await _llm_judge_relevance(question, good_results)
    verdict_bad  = await _llm_judge_relevance(question, bad_results)
    elapsed = time.perf_counter() - t0

    print(f"  Question : {question}")
    print(f"  Good results → sufficient={verdict_good}  (expected: True)")
    print(f"  Bad results  → sufficient={verdict_bad}   (expected: False)")
    print(f"  ✓ Both judge calls in {elapsed:.2f}s")

    ok = (verdict_good is True) and (verdict_bad is False)
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")


# ══════════════════════════════════════════════════════════════════════════
#  TEST 4 — Query refiner
# ══════════════════════════════════════════════════════════════════════════

async def test_refiner():
    header("TEST 4 — Query refiner")

    question      = "Quel est le meilleur framework Python pour du machine learning en 2024 ?"
    current_query = "python framework machine learning"
    thin_results  = [
        {"title": "Python programming", "body": "Python is a programming language.", "url": "https://python.org"},
        {"title": "Machine learning basics", "body": "Machine learning is a subset of AI.", "url": "https://example.com"},
    ]

    t0 = time.perf_counter()
    refined = await _refine_web_query(question, current_query, thin_results)
    elapsed = time.perf_counter() - t0

    print(f"  Original query : {current_query}")
    print(f"  Refined query  : {refined}")
    print(f"  ✓ Refiner call in {elapsed:.2f}s")
    print(f"  {'✓ PASS' if refined and refined != current_query else '✗ FAIL (same query returned)'}")


# ══════════════════════════════════════════════════════════════════════════
#  TEST 5 — Deep pipeline stage progression
# ══════════════════════════════════════════════════════════════════════════

async def test_deep_pipeline():
    header("TEST 5 — Deep pipeline (3-stage)")

    # Use a specific factual question that DDG snippets may answer partially
    query   = "PyTorch vs TensorFlow 2024 comparison"
    message = "Quelle est la différence entre PyTorch et TensorFlow en 2024, lequel choisir ?"

    print(f"  Query   : {query}")
    print(f"  Message : {message}")
    print()

    t0 = time.perf_counter()
    results = await _ddg_text_deep(query, message, max_results=5)
    elapsed = time.perf_counter() - t0

    print(f"\n  ✓ {len(results)} result(s) returned in {elapsed:.2f}s")
    show_results(results, max_body=300)


# ══════════════════════════════════════════════════════════════════════════
#  TEST 6 — Full search_web end-to-end
# ══════════════════════════════════════════════════════════════════════════

async def test_search_web_general():
    header("TEST 6 — search_web end-to-end (general question)")

    message = "Quel est le prix actuel du Bitcoin ?"
    query   = "prix Bitcoin aujourd'hui"

    print(f"  Message : {message}")
    print(f"  Query   : {query}")
    print()

    t0 = time.perf_counter()
    results = await search_web(query, original_message=message)
    elapsed = time.perf_counter() - t0

    print(f"\n  ✓ {len(results)} result(s) in {elapsed:.2f}s")
    show_results(results, max_body=300)


# ══════════════════════════════════════════════════════════════════════════
#  RUNNER
# ══════════════════════════════════════════════════════════════════════════

async def main():
    print(f"\n{'═' * 70}")
    print(f"  Jarvis web_search.py — deep pipeline test")
    print(f"{'═' * 70}")

    await test_weather()
    await test_news()
    await test_judge()
    await test_refiner()
    await test_deep_pipeline()
    await test_search_web_general()

    print(f"\n{SEP}")
    print("  All tests complete.")
    print(SEP)


if __name__ == "__main__":
    asyncio.run(main())
