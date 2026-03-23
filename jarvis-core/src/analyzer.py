"""
PROJECT JARVIS v8
Jarvis Conversation Analyzer
=============================
After each exchange, extracts:
- Topics discussed
- User facts to remember
- Mood/sentiment
- Projects mentioned

Uses the LLM to analyze — costs ~$0.001 per analysis.

Episodic Salience Score (ESS)
Le score d'importance devient la somme de plusieurs signaux :

Personal relevance
informations sur l'utilisateur (facts, projets)

Emotional intensity
émotions positives ou négatives fortes

Novelty
nouveau sujet ou nouvelle information

Goal relevance
lié à un projet ou une action concrète

Memory summary signal
le LLM a identifié quelque chose à retenir

Message depth
message long / détaillé
"""

import json

from config import (
    IMPORTANCE_THRESHOLD,
    PRIMARY_API_KEY,
    PRIMARY_API_URL,
    PRIMARY_MODEL,
)
from helpers import call_llm_async, extract_llm_json, get_logger
from prompts import get_prompt

logger = get_logger("jarvis-analyzer")


async def analyze_exchange(user_msg: str, assistant_msg: str, existing_projects: list = None, existing_profile_keys: list = None) -> dict:
    """Analyze a conversation exchange using the LLM."""
    try:
        projects_context = (
            ", ".join(p["name"] for p in existing_projects if isinstance(p, dict) and p.get("name") and p.get("status") != "done")
            if existing_projects else "aucun"
        )
        profile_keys_str = ", ".join(existing_profile_keys) if existing_profile_keys else "aucune"
        prompt = get_prompt("ANALYSIS_PROMPT").format(
            user_message=user_msg[:1000],
            assistant_message=assistant_msg[:1000],
            existing_projects=projects_context,
            existing_profile_keys=profile_keys_str,
        )

        content = await call_llm_async(
            [{"role": "user", "content": prompt}],
            model=PRIMARY_MODEL,
            api_url=PRIMARY_API_URL,
            api_key=PRIMARY_API_KEY,
            temperature=0.1,
            max_tokens=500,
            json_response=True,
            no_think=True,
            timeout=30.0,
        )

        try:
            result = extract_llm_json(content)
        except json.JSONDecodeError as exc:
            logger.error("Analyzer JSON parse error: %s", exc.doc[:200])
            raise

        # Episodic Salience Score (ESS) — signals combined into [0, 1]
        # IMPORTANCE_THRESHOLD = 0.35 → stored as episodic vector
        # AUTOBIO_IMPORTANCE_THRESHOLD = 0.60 → stored as autobiographical
        importance = 0.0

        # LLM's own judgment is the primary signal: 0.4 alone clears
        # IMPORTANCE_THRESHOLD so any exchange the LLM deems worth
        # remembering is captured, even with no other signals.
        # Only count the bonus when the LLM provided an actual summary
        # sentence — a bare boolean true gives no usable memory_summary.
        remember_text = result.get("should_remember")

        if isinstance(remember_text, str) and remember_text.strip():
            importance += 0.40

        # Personal facts revealed by the user
        importance += min(len(result.get("user_facts", [])), 3) * 0.20

        # Projects / goal context
        importance += min(len(result.get("projects", [])), 2) * 0.15

        # Emotional intensity (mild boost — avoid over-storing rants)
        mood = result.get("mood", "neutral")
        if mood in ["happy", "curious", "focused"]:
            importance += 0.10
        elif mood in ["stressed", "frustrated"]:
            importance += 0.15

        # Message depth (minor signal — long messages often carry more info)
        if len(user_msg) > 80:
            importance += 0.05

        # Clamp score
        importance = min(importance, 1.0)
        result["importance"] = round(importance, 3)

        # Convert "should_remember" sentence into boolean
        remember_text = result.get("should_remember")

        if remember_text and isinstance(remember_text, str):
            result["memory_summary"] = remember_text
            result["should_remember"] = result["importance"] > IMPORTANCE_THRESHOLD
        else:
            result["memory_summary"] = None
            result["should_remember"] = False

        return result

    except Exception as e:
        logger.error("Analysis error: %s", e)
        return {
            "topics": [],
            "mood": "neutral",
            "user_facts": [],
            "projects": [],
            "importance": 0.0,
            "memory_summary": None,
            "should_remember": False,
        }
