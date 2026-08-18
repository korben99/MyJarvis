"""
Jarvis LLM Router — two-tier edition
=====================================
Tier 1 (this file): fast intent classifier + complexity detector.
Tier 2 (main.py):   reasoning model selected when use_reasoning=True.

Router backend is fully OpenAI-compatible (/v1/chat/completions).
Swap from GPT-4.1-nano to Qwen2.5-7B via mlx-lm by changing three env vars:
    ROUTER_API_URL=http://<mac-ip>:8080/v1
    ROUTER_API_KEY=mlx        (mlx-lm ignores auth but httpx must send something)
    ROUTER_MODEL=Qwen/Qwen2.5-7B-Instruct-8bit

mlx-lm note: response_format is supported from mlx-lm ≥ 0.21. For older
versions the JSON is extracted from the raw text as a fallback.

If ROUTER_MODEL is empty or the call fails for any reason, returns None
and main.py falls back to the embedding router automatically.

RouterResult fields:
    use_memory, use_rag, use_web, use_gmail, use_calendar,
    use_briefing, use_self   — data-source flags
    use_reasoning            — True → route to Tier-2 reasoning model
    gmail_query              — Gmail search string (or "")
    calendar_days            — days ahead to fetch (default 7)
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from config import (
    MAX_TOKENS_SHORT,
    ROUTER_API_KEY,
    ROUTER_API_URL,
    ROUTER_DATA_DIR,
    ROUTER_MODEL,
    ROUTER_TIMEOUT,
)
from helpers import call_llm_async, extract_llm_json, fmt_now_fr, get_logger
from prompts import get_prompt

logger = get_logger("jarvis-llm-router")

# Fuseau de la date injectée dans <date>. Le routeur n'a pas de user_code sous la main
# (llm_route ne reçoit que le message), et tous les utilisateurs sont sur le même
# fuseau ; à défaut, une heure de décalage ne change pas un calcul en jours.
_ROUTER_TZ = "Europe/Paris"


# ── Result dataclass ──────────────────────────────────────────────────────
@dataclass
class RouterResult:
    use_memory: bool
    use_rag: bool
    use_web: bool
    use_weather: bool
    use_gmail: bool
    use_calendar: bool
    use_briefing: bool
    use_self: bool
    use_portfolio: bool
    gmail_query: str
    calendar_days: int
    weather_location: str = field(default="")
    rag_query: str = field(default="")
    use_small_talk: bool = field(
        default=False
    )  # skip profile/memory injection entirely
    use_reasoning: bool = field(default=False)
    project_name: str = field(default="")


_ALLOWED_INTENTS = {
    "memory",
    "rag",
    "web",
    "weather",
    "gmail",
    "calendar",
    "briefing",
    "portfolio",
    "self",
}

# ── Training data collector ───────────────────────────────────────────────


def _log_routing_sample(
    message: str,
    result: "RouterResult",
    model: str,
    source: str = "llm",
    last_jarvis: str | None = None,
) -> None:
    """Append one JSONL entry to the router training file.

    File: {ROUTER_DATA_DIR}/routing_samples.jsonl
    Each line is a self-contained JSON object with:
      - id / ts   : deduplication + timeline
      - message   : raw user input
      - routing   : the full RouterResult as a dict
      - model     : which router model produced this result
      - source    : "embed" (fast-path) ou "llm" (routeur LLM)
      - last_jarvis : dernier tour assistant, ou null. Ajouté le 18/08/2026 : sans
                    lui, un message elliptique (« confirme », « la couronne ») est
                    inétiquetable — le ré-étiquetage du 17/08 a rabattu une vingtaine
                    de ces messages sur `web` faute d'antécédent, et le LoRA v2 a
                    appris l'erreur. C'est LA donnée manquante du corpus.
      - ok        : null = uncurated, true = validated, false = wrong

    `source` existe parce que jusqu'au 17/08/2026 seul le routeur LLM
    journalisait : le fast-path embedding tranchait la moitié du trafic sans
    laisser de trace. Le fichier ne contenait donc pas le trafic réel mais le
    RÉSIDU que l'embedding avait refusé de trancher — les échantillons les plus
    ambigus du corpus. Toute statistique calculée avant cette date est biaisée
    dans ce sens ; filtrer sur `source` pour comparer ce qui est comparable.
    """
    os.makedirs(ROUTER_DATA_DIR, exist_ok=True)
    # Règle stricte de ROUTER_SYSTEM : un champ n'est renseigné que si son intent
    # est présent. RouterResult ne l'applique pas (calendar_days vaut 7 par défaut
    # même sur une requête météo), et sans cette normalisation l'échantillon
    # apprendrait au modèle à émettre 7 partout. curate_router_dataset.py applique
    # déjà la règle en aval ; on la pose ici pour que la ligne journalisée soit
    # utilisable telle quelle comme cible d'entraînement.
    _in = lambda name: getattr(result, f"use_{name}")  # noqa: E731
    sample = {
        "id": str(uuid.uuid4()),
        "ts": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "message": message,
        "routing": {
            "intents": [
                intent
                for intent, flag in [
                    ("memory", result.use_memory),
                    ("rag", result.use_rag),
                    ("web", result.use_web),
                    ("weather", result.use_weather),
                    ("gmail", result.use_gmail),
                    ("calendar", result.use_calendar),
                    ("briefing", result.use_briefing),
                    ("self", result.use_self),
                    ("portfolio", result.use_portfolio),
                ]
                if flag
            ],
            "gmail_query": result.gmail_query or None if _in("gmail") else None,
            "calendar_days": result.calendar_days if _in("calendar") else None,
            "weather_location": result.weather_location or None if _in("weather") else None,
            "rag_query": result.rag_query or None if _in("rag") else None,
            # Absent des échantillons jusqu'au 17/08/2026, alors que ROUTER_SYSTEM
            # impose la clé et que RouterResult la porte. router_lora_adapterv1.py
            # la lisait donc via .get() → None sur 100 % des exemples : le LoRA v1 a
            # appris à toujours émettre project_name:null, sans jamais pouvoir
            # apprendre à l'extraire.
            "project_name": result.project_name or None,
            "use_reasoning": result.use_reasoning,
        },
        "model": model,
        "source": source,
        "last_jarvis": last_jarvis,
        # Décision à part entière du fast-path (aucun intent levé), invisible dans
        # `intents` qui ne liste que les use_* — sans ce champ le petit talk serait
        # indistinguable d'un routage vide.
        #
        # HORS de `routing` volontairement : curate_router_dataset.py sérialise ce
        # sous-dict EN BLOC comme cible d'entraînement du LoRA
        # (`json.dumps(routing)`). Toute clé ajoutée dans `routing` apprendrait au
        # modèle à émettre un champ de plus, alors que ROUTER_SYSTEM impose
        # « 7 clés, ni plus ni moins ». `routing` doit rester le miroir exact du
        # schéma de sortie ; les métadonnées vont au niveau du dessus.
        "use_small_talk": result.use_small_talk,
        "ok": None,  # null = not yet reviewed by human
    }
    path = os.path.join(ROUTER_DATA_DIR, "routing_samples.jsonl")
    try:
        if os.path.exists(path) and os.path.getsize(path) > 10 * 1024 * 1024:
            import shutil

            shutil.move(path, path + ".bak")  # rotation simple
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Could not write routing sample: %s", exc)


# ── Core call ─────────────────────────────────────────────────────────────


async def llm_route(message: str, google_available: bool = True, last_jarvis: str | None = None) -> RouterResult | None:
    """
    Call the router model (OpenAI-compatible /v1/chat/completions).
    Returns RouterResult on success, None on any failure → caller falls back
    to the embedding router automatically.
    last_jarvis: last assistant response (truncated) — injected as <last_jarvis> for context-aware routing.
    """
    # Plafond à 1000 caractères. À 400 (valeur précédente), 6 % des messages réels
    # étaient tronqués — souvent au milieu de la phrase qui portait l'intention, le
    # sujet arrivant après une mise en contexte. Médiane mesurée : 101 caractères,
    # p90 315, donc 1000 couvre 98,4 % des messages en entier et ne coûte rien au cas
    # courant. Le plafond reste nécessaire : le corpus contient un message de 81 513
    # caractères (document collé) qui saturerait le contexte du routeur.
    routing_message = message[:1000]
    # 600 et non 300 : c'est <last_jarvis> qui porte l'antécédent des messages
    # elliptiques (« confirme », « la couronne », « oui je pense aussi »), et 300
    # caractères tronquaient souvent la réponse avant le sujet dont elle parlait.
    last_jarvis_block = (
        f"<last_jarvis>{last_jarvis[:600]}</last_jarvis>\n" if last_jarvis else ""
    )
    # La date est indispensable à calendar_days : sans elle, « vendredi » ou « la
    # semaine prochaine » ne sont pas calculables — ce n'était pas une faiblesse du
    # modèle mais une donnée absente. Jour de la semaine inclus, c'est lui qui sert
    # au calcul relatif.
    routing_date = fmt_now_fr(_ROUTER_TZ).split(",")[0]
    prompt = get_prompt("ROUTER_USER").format(
        date=routing_date, message=routing_message, last_jarvis_block=last_jarvis_block
    )

    try:
        # response_format is supported by OpenAI and mlx-lm ≥ 0.21.
        # Older mlx-lm versions ignore it — extract_llm_json() handles that.
        raw = await call_llm_async(
            [
                {"role": "system", "content": get_prompt("ROUTER_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=ROUTER_MODEL,
            api_url=ROUTER_API_URL,
            api_key=ROUTER_API_KEY,
            temperature=0.0,  # Hermes is designed for deterministic structured output
            max_tokens=MAX_TOKENS_SHORT,
            json_response=True,
            no_think=True,
            timeout=ROUTER_TIMEOUT,
        )
        parsed = extract_llm_json(raw)

    except httpx.TimeoutException:
        logger.warning(
            "LLM router timeout (%.1fs) — no routing info (all intents off)",
            ROUTER_TIMEOUT,
        )
        return None
    except Exception as exc:
        logger.warning(
            "LLM router error (%s) — no routing info (all intents off): %s",
            type(exc).__name__,
            exc,
        )
        return None

    # ── Guard: parsed must be a non-empty dict ──────────────────────────────
    if not isinstance(parsed, dict):
        logger.warning(
            "LLM router returned non-dict (%s): %r — falling back",
            type(parsed).__name__,
            str(parsed)[:200],
        )
        return None

    logger.debug("LLM router raw output: %r", raw[:300])

    # ── Extract and validate fields ──
    try:
        intents: list[str] = parsed.get("intents", [])
        if not isinstance(intents, list):
            intents = []
        intents = [i for i in intents if i in _ALLOWED_INTENTS]
        if not intents:
            intents = ["memory"]

        gmail_query: str = parsed.get("gmail_query") or ""
        _cal_raw = parsed.get("calendar_days")
        if _cal_raw is None:
            calendar_days = 7
        else:
            try:
                calendar_days = int(_cal_raw)
            except (ValueError, TypeError):
                calendar_days = 7

        weather_location: str = (parsed.get("weather_location") or "") if "weather" in intents else ""
        rag_query: str = (parsed.get("rag_query") or "") if "rag" in intents else ""
        project_name: str = parsed.get("project_name") or ""
        use_reasoning: bool = bool(parsed.get("use_reasoning", False))
    except Exception as exc:
        logger.warning(
            "LLM router field extraction failed (%s): %s — parsed=%r",
            type(exc).__name__,
            exc,
            str(parsed)[:300],
        )
        return None

    # ── Guardrail: prevent over-triggering reasoning on simple queries ──
    # SUPPRESSION on fait confiance aux deux etages de routage embed + 3B
    # _simple_query = (
    #    len(message) < 80
    #    and "?" not in message
    #    and not any(
    #        k in message.lower() for k in ["analyse", "explique", "compare", "pourquoi"]
    #    )
    # )

    # if _simple_query:
    #    use_reasoning = False

    calendar_days = max(1, min(calendar_days, 90))

    result = RouterResult(
        use_memory="memory" in intents,
        use_rag="rag" in intents,
        use_web="web" in intents,
        use_weather="weather" in intents,
        use_gmail="gmail" in intents and google_available,
        use_calendar="calendar" in intents and google_available,
        use_briefing="briefing" in intents,
        use_self="self" in intents,
        use_portfolio="portfolio" in intents,
        use_reasoning=use_reasoning,
        gmail_query=gmail_query,
        calendar_days=calendar_days,
        weather_location=weather_location,
        rag_query=rag_query,
        project_name=project_name,
    )

    logger.info(
        "LLM router [%s]: intents=%s | reasoning=%s → final=%s",
        ROUTER_MODEL,
        intents,
        parsed.get("use_reasoning"),
        result.use_reasoning,
    )
    _log_routing_sample(message, result, ROUTER_MODEL, last_jarvis=last_jarvis)
    return result
