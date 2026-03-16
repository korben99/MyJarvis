"""
PROJECT JARVIS v7
Jarvis Conversation Analyzer
=============================
After each exchange, extracts:
- Topics discussed
- User facts to remember
- Mood/sentiment
- Projects mentioned

Uses the LLM to analyze — costs ~$0.001 per analysis.

Episodic Salience Score (ESS)
Le score d’importance devient la somme de plusieurs signaux :

Personal relevance
informations sur l’utilisateur (facts, projets)

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
import logging

import httpx

from config import (
    ANALYSIS_MODEL,
    IMPORTANCE_THRESHOLD,
    OPENAI_API_KEY,
    OPENAI_API_URL,
)

logger = logging.getLogger("jarvis-analyzer")

ANALYSIS_PROMPT = """Analyze this conversation exchange between a user and their AI assistant Jarvis.
Return a JSON object with these fields:
- "topics": list of 1-3 topic keywords discussed (lowercase)
- "mood": the user's apparent mood (one of: happy, neutral, focused, stressed, frustrated, curious, tired)
- "user_facts": list of new facts learned about the user (empty if none). Each fact as {{"key": "...", "value": "..."}}
  Examples: {{"key": "current_project", "value": "CRA compliance"}}, {{"key": "expertise", "value": "cybersecurity"}}
- "projects": list of project names mentioned (empty if none)
- "should_remember": a single sentence summarizing what's worth remembering from this exchange (or null if routine)

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.

User said: {user_message}
Assistant said: {assistant_message}"""


async def analyze_exchange(user_msg: str, assistant_msg: str) -> dict:
    """Analyze a conversation exchange using the LLM."""
    try:
        prompt = ANALYSIS_PROMPT.format(
            user_message=user_msg[:1000],
            assistant_message=assistant_msg[:1000],
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OPENAI_API_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ANALYSIS_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500,
                },
            )
            data = resp.json()

            if "choices" not in data:
                logger.error(f"Analyzer invalid response: {data}")
                raise ValueError("Invalid analyzer response")
            content = data["choices"][0]["message"]["content"]

            # Parse JSON (strip markdown code fences if present)
            content = content.strip()
            if "```" in content:
                # Extract content between first and last ``` markers
                inner = content.split("```")[1]
                # Strip optional language identifier on first line (e.g. "json\n")
                first_newline = inner.find("\n")
                if first_newline != -1 and not inner[:first_newline].strip().startswith("{"):
                    inner = inner[first_newline:].strip()
                content = inner.strip()

            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Analyzer JSON parse error: {content[:200]}")
                raise

            # Episodic Salience Score (ESS) — signals combined into [0, 1]
            # IMPORTANCE_THRESHOLD = 0.35 → stored as episodic vector
            # AUTOBIO_IMPORTANCE_THRESHOLD = 0.60 → stored as autobiographical
            importance = 0.0

            # LLM's own judgment is the primary signal: 0.4 alone clears
            # IMPORTANCE_THRESHOLD so any exchange the LLM deems worth
            # remembering is captured, even with no other signals.
            if result.get("should_remember"):
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
        logger.error(f"Analysis error: {e}")
        return {
            "topics": [],
            "mood": "neutral",
            "user_facts": [],
            "projects": [],
            "importance": 0.0,
            "memory_summary": None,
            "should_remember": False,
        }
