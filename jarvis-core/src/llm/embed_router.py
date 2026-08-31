"""
embed_router.py — Routeur d'intent par similarité d'embeddings
==============================================================
Fast-path avant le routeur LLM (3B, ~1.3s).

Principe :
  1. Règles déterministes (URL, messages très courts, briefing exact, reasoning)
  2. Similarité cosinus entre le message et des phrases-exemples par intent
  3. Si score ≥ THRESHOLD et pas ambigu → RouterResult direct, sans appel LLM
  4. Sinon → None → le routeur LLM 3B prend le relais

Désactiver : EMBED_ROUTER_ENABLED = False ci-dessous, ou env var EMBED_ROUTER=no.

Modifier les exemples : éditer INTENT_EXAMPLES — une phrase par ligne.
Modifier les seuils   : EMBED_ROUTE_THRESHOLD, AMBIGUITY_MARGIN.
Ajouter un intent     : ajouter la clé dans INTENT_EXAMPLES + mettre à jour
                        _make_result() et RouterResult dans llm_router.py.

Retourne use_reasoning=True pour les ordres explicites de l'utilisateur
("mode expert", "analyse approfondie", etc.) — ces phrases sont non-ambiguës
et n'ont pas besoin du routeur LLM pour être classifiées.
"""

import os
import re
import threading

import numpy as np
from helpers import get_logger
from .router import RouterResult, _log_routing_sample
from .lexique import (
    BRIEFING_EXACT,
    CITY_AFTER_PREP_RE,
    INTENT_EXAMPLES,
    RAG_CMD_RE,
    RAG_LEAD_NOISE_RE,
    REASON_EXACT,
    REASON_REGEX,
    SMALL_TALK_EXACT,
    TEMPORAL_WORDS,
)

logger = get_logger("jarvis-embed-router")


EMBED_ROUTER_ENABLED: bool = os.getenv("EMBED_ROUTER", "yes").lower() != "no"

# Seuil minimum de similarité cosinus pour accepter un intent
EMBED_ROUTE_THRESHOLD: float = 0.74

# Si les deux meilleurs intents sont à moins de cette marge, c'est ambigu → LLM
AMBIGUITY_MARGIN: float = 0.06

# Longueur au-delà de laquelle on n'essaie même pas l'embedding — voir _embed_route().
EMBED_MAX_CHARS: int = int(os.getenv("EMBED_MAX_CHARS", "130"))


# ── Cache des embeddings d'exemples (initialisé une seule fois) ───────────────
_cache_lock = threading.Lock()
_examples_vectors: dict[str, np.ndarray] | None = None  # intent → (N, D) matrix


def preload_embed_router() -> None:
    """Précharge les vecteurs d'exemples au démarrage (évite la latence sur la première requête)."""
    if EMBED_ROUTER_ENABLED:
        _load_example_vectors()


def _load_example_vectors() -> dict[str, np.ndarray]:
    """Encode tous les exemples et met en cache les matrices de vecteurs."""
    global _examples_vectors
    with _cache_lock:
        if _examples_vectors is not None:
            return _examples_vectors
        from memory import get_embed_model

        model = get_embed_model()
        result: dict[str, np.ndarray] = {}
        for intent, phrases in INTENT_EXAMPLES.items():
            vecs = model.encode(phrases, normalize_embeddings=True)  # (N, D)
            result[intent] = vecs.astype(np.float32)
        _examples_vectors = result
        logger.info(
            "Embed router: %d intents, %d phrases total chargées",
            len(result),
            sum(v.shape[0] for v in result.values()),
        )
        return _examples_vectors


# ── Extracteurs de paramètres simples (sans LLM) ─────────────────────────────


def _extract_weather_location(message: str) -> str:
    """Extrait une ville depuis un message météo.

    Stratégie :
      1. Après préposition (à|au|aux|pour|sur|vers|en), capture la séquence
         capitalisée qui suit — gère les noms composés français.
         Couverts : 'Paris', 'La Rochelle', 'Aix-en-Provence',
                    'Saint-Germain-en-Laye', 'Boulogne-sur-Mer'.
      2. Fallback : nettoyage + premier token significatif — pour les
         messages en minuscules ou sans préposition ('météo paris').

    Limitations acceptées (cas rares → Hermes rattrape) :
      - 'météo Le Havre' (pas de préposition, minuscule liant) → 'Havre'
      - 'météo à Lyon et Paris' (multi-villes) → 'Lyon' seulement
    """
    # ── 1. Séquence capitalisée après préposition ──
    m = CITY_AFTER_PREP_RE.search(message)
    if m:
        city = m.group(1).strip(" ,;:!?.")
        first_word = city.split()[0].lower().rstrip("-")
        if first_word not in TEMPORAL_WORDS:
            return city

    # ── 2. Fallback (minuscules, villes simples) ──
    msg = message.lower()
    clean = re.sub(
        r"\b(météo|meteo|temps|température|temperature|prévisions?|pluie|"
        r"soleil|chaud|froid|vent|neige|orage|nuages?|brouillard|"
        r"aujourd'hui|demain|hier|ce soir|ce matin|ce week-?end|cette semaine|"
        r"lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|"
        r"dehors|extérieur|il fait|quel temps|parapluie|"
        r"à|au|aux|pour|sur|vers|en|de|du|des)\b",
        " ",
        msg,
        flags=re.IGNORECASE,
    ).strip()
    words = [w.strip("?,!.") for w in clean.split() if len(w.strip("?,!.")) > 2]
    return words[0].capitalize() if words else ""


def _extract_calendar_days(message: str) -> int:
    """Déduit le nombre de jours d'agenda à récupérer."""
    m = message.lower()
    if any(w in m for w in ["aujourd'hui", "ce soir", "ce matin", "maintenant"]):
        return 1
    if "demain" in m:
        return 2
    if any(w in m for w in ["cette semaine", "la semaine", "7 jours", "semaine"]):
        return 7
    if any(w in m for w in ["ce mois", "du mois", "30 jours", "mois"]):
        return 30
    return 7


def _extract_rag_query(message: str) -> str:
    """Strip RAG routing phrases, return semantic topic keywords."""
    cleaned = RAG_CMD_RE.sub(" ", message)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?,!.")
    cleaned = RAG_LEAD_NOISE_RE.sub("", cleaned).strip()
    return cleaned or message


def _extract_gmail_query(message: str) -> str:
    """Construit une requête Gmail depuis le message."""
    m = message.lower()
    if any(w in m for w in ["non lu", "non lus", "pas lu", "unread"]):
        return "is:unread"
    if any(w in m for w in ["important", "prioritaire"]):
        return "is:important"
    if "facture" in m:
        return "subject:facture"
    if "aujourd'hui" in m or "du jour" in m:
        return "newer_than:1d"
    # Numeric duration: "il y a 3 semaines", "depuis 2 jours", "les 4 derniers mois"
    _dur = re.search(r"(\d+)\s*(jour|semaine|mois)", m)
    if _dur:
        n, unit = int(_dur.group(1)), _dur.group(2)
        days = n if unit == "jour" else (n * 7 if unit == "semaine" else n * 30)
        return f"newer_than:{days}d"
    if any(
        w in m for w in ["récent", "récents", "derniers", "semaine", "cette semaine"]
    ):
        return "newer_than:7d"
    return "newer_than:7d"  # défaut : mails récents


# ── Fonction principale ───────────────────────────────────────────────────────


def embed_route(message: str, google_available: bool = True,
                last_jarvis: str | None = None) -> RouterResult | None:
    """Fast-path de routage par embeddings, avec journalisation de la décision.

    `last_jarvis` n'entre PAS dans la décision — le fast-path tranche sur la seule
    similarité du message. Il n'est là que pour être journalisé : c'est l'antécédent
    des messages elliptiques, et sans lui dans les échantillons le professeur du
    prochain ré-étiquetage travaillera en aveugle comme le mien l'a fait.

    Enveloppe `_embed_route`. Quand le fast-path tranche, la décision est écrite
    dans routing_samples.jsonl avec source="embed" ; quand il défère (None), rien
    n'est écrit ici car le routeur LLM journalisera lui-même en aval.

    Sans cette journalisation, le jeu d'échantillons ne contenait que le résidu
    déféré au LLM — soit les messages les plus ambigus — et pas le trafic réel.
    """
    result = _embed_route(message, google_available)
    if result is not None:
        try:
            _log_routing_sample(message, result, "embed-router", source="embed",
                                last_jarvis=last_jarvis)
        except Exception as exc:  # jamais bloquer le routage pour une trace
            logger.debug("Embed router: journalisation impossible (%s)", type(exc).__name__)
    return result


def _embed_route(message: str, google_available: bool = True) -> RouterResult | None:
    """
    Tente de router le message par similarité d'embeddings.

    Retourne un RouterResult si l'intent est clair et le score suffisant,
    None sinon → le routeur LLM 3B prend le relais.

    Toujours synchrone — appelé avant asyncio.gather dans chat.py.
    Coût : ~2-5 ms (dot product numpy, modèle embed déjà chargé).
    """
    if not EMBED_ROUTER_ENABLED:
        return None

    msg = message.strip()
    if not msg:
        return None

    msg_lower = msg.lower()

    # Message long → routeur LLM directement, sans calculer d'embedding.
    #
    # Les phrases-exemples font 4 à 21 jetons (médiane 7). Un message long est encodé en un
    # vecteur moyenné sur toutes ses propositions : sa similarité maximale avec un exemple
    # court s'effondre mécaniquement, quelle que soit la clarté de l'intention. Mesuré sur
    # 959 messages réels — part de messages franchissant le seuil, par longueur :
    #
    #     0-10 jetons  33 %      20-30 jetons   1 %
    #     10-15        10 %      30-45          0 %   (185 messages)
    #     15-20         4 %      45 et plus     0 %   (270 messages)
    #
    # Au-delà de 30 jetons, le fast-path n'a jamais tranché : le calcul est perdu d'avance.
    # La barrière est donc gratuite en décisions et elle borne un risque à venir — enrichir
    # les pools d'exemples finit par produire des accrochages fortuits sur des messages longs,
    # là où le score n'a plus de sens.
    #
    # Seuil exprimé en CARACTÈRES pour rester sans coût : tokeniser juste pour décider de ne
    # pas tokeniser n'aurait aucun intérêt. 130 caractères est calibré sur le même corpus —
    # aucun message de moins de 30 jetons n'y est écarté.
    if len(msg) >= EMBED_MAX_CHARS:
        logger.debug(
            "Embed router: message long (%d car.) → LLM router sans embedding", len(msg)
        )
        return None

    # ── 1. Règles déterministes ───────────────────────────────────────────────

    # Commandes explicites de raisonnement ("mode expert", "analyse approfondie"…) :
    # retour immédiat memory+reasoning, sans embedding ni LLM router.
    if any(t in msg_lower for t in REASON_EXACT):
        logger.debug("Embed router: REASON_EXACT → memory + use_reasoning=True")
        result = _build_result("memory", msg, google_available)
        result.use_reasoning = True
        return result

    # Détection reasoning via regex (flag, pas intent) — doit précéder les early-returns
    force_reasoning = bool(REASON_REGEX.search(msg_lower))

    # URL → memory (la page est déjà fetchée automatiquement en amont)
    if re.search(r"https?://", msg):
        logger.debug("Embed router: URL détectée → memory")
        result = _build_result("memory", msg, google_available)
        if force_reasoning:
            result.use_reasoning = True
        return result

    # Mention explicite du mot RAG → rag direct (le jargon métier noie le signal embedding)
    if re.search(r"\brag\b", msg_lower):
        logger.debug("Embed router: RAG keyword → rag direct")
        result = _build_result("rag", msg, google_available)
        if force_reasoning:
            result.use_reasoning = True
        return result

    # Commandes de gestion des propositions de prompt → self direct
    _proposal_explicit = (
        any(kw in msg_lower for kw in ("proposition", "proposals", "propositions"))
        and any(
            kw in msg_lower
            for kw in (
                "accepte",
                "rejette",
                "approuve",
                "refuse",
                "montre",
                "liste",
                "show",
                "list",
                "détail",
            )
        )
    ) or bool(
        re.search(r"\b[a-f0-9]{6,8}\b", msg_lower)
        and any(
            kw in msg_lower
            for kw in ("accepte", "rejette", "approuve", "refuse", "montre")
        )
    )
    if _proposal_explicit:
        logger.debug("Embed router: proposal trigger → self direct")
        result = _build_result("self", msg, google_available)
        if force_reasoning:
            result.use_reasoning = True
        return result

    if len(msg) <= 50 and "?" not in msg:
        _norm = msg_lower.rstrip(" !.,")
        if _norm in SMALL_TALK_EXACT:
            logger.debug("Embed router: small talk → no context injection")
            return _build_result("small_talk", msg, google_available)

    if msg_lower.rstrip("! ?,") in BRIEFING_EXACT:
        logger.debug("Embed router: briefing exact → briefing")
        return _build_result("briefing", msg, google_available)

    # ── 2. Similarité cosinus ─────────────────────────────────────────────────

    try:
        from memory import get_embed_model

        model = get_embed_model()
        query_vec = model.encode(msg, normalize_embeddings=True).astype(np.float32)

        example_vecs = _load_example_vectors()

        scores: dict[str, float] = {
            intent: float((vecs @ query_vec).max())
            for intent, vecs in example_vecs.items()
        }

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_score = sorted_scores[0]
        second_intent, second_score = sorted_scores[1]

        logger.debug(
            "Embed router: 1=%s %.3f  2=%s %.3f",
            best_intent,
            best_score,
            second_intent,
            second_score,
        )

        if best_score < EMBED_ROUTE_THRESHOLD:
            logger.debug("Embed router: score %.3f < seuil → LLM router", best_score)
            return None

        if best_score - second_score < AMBIGUITY_MARGIN:
            logger.debug(
                "Embed router: ambiguïté %.3f entre %s et %s → LLM router",
                best_score - second_score,
                best_intent,
                second_intent,
            )
            return None

        # Project intent → force LLM router so it can extract project_name
        if best_intent == "project":
            logger.debug(
                "Embed router: project intent (%.3f) → LLM router pour extraction du nom",
                best_score,
            )
            return None

        logger.info(
            "Embed router: %s (%.3f) — LLM router évité",
            best_intent,
            best_score,
        )
        result = _build_result(best_intent, msg, google_available)
        if force_reasoning:
            result.use_reasoning = True
        return result

    except Exception as exc:
        logger.warning("Embed router: erreur (%s) → LLM router", exc)
        return None


def _build_result(intent: str, message: str, google_available: bool) -> RouterResult:
    """Construit un RouterResult pour l'intent classifié."""
    return RouterResult(
        use_memory=intent == "memory",
        use_rag=intent == "rag",
        use_web=intent == "web",
        use_weather=intent == "weather",
        use_gmail=intent == "gmail" and google_available,
        use_calendar=intent == "calendar" and google_available,
        use_briefing=intent == "briefing",
        use_self=intent == "self",
        use_portfolio=intent == "portfolio",
        use_small_talk=intent == "small_talk",
        gmail_query=_extract_gmail_query(message) if intent == "gmail" else "",
        calendar_days=_extract_calendar_days(message) if intent == "calendar" else 7,
        weather_location=_extract_weather_location(message)
        if intent == "weather"
        else "",
        rag_query=_extract_rag_query(message) if intent == "rag" else "",
    )
