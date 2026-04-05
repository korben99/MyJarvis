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

Ne retourne jamais use_reasoning=True — les requêtes "mode expert" tombent
toujours sur le routeur LLM qui a le contexte pour juger.
"""

import os
import re
import threading

import numpy as np
from helpers import get_logger
from llm_router import RouterResult

logger = get_logger("jarvis-embed-router")

EMBED_ROUTER_ENABLED: bool = os.getenv("EMBED_ROUTER", "yes").lower() != "no"

# Seuil minimum de similarité cosinus pour accepter un intent
EMBED_ROUTE_THRESHOLD: float = 0.82

# Si les deux meilleurs intents sont à moins de cette marge, c'est ambigu → LLM
AMBIGUITY_MARGIN: float = 0.06

# ── Phrases-exemples par intent (toutes en français) ─────────────────────────
#
# Règles d'édition :
#   • 8 à 15 phrases par intent — assez pour couvrir les formulations courantes
#   • Phrases courtes, naturelles, telles qu'un utilisateur les taperait
#   • Pas de doublons inter-intents (sinon l'ambiguïté monte)
#   • Pas de noms propres sauf si vraiment caractéristiques (ex: "PEA" pour portfolio)

INTENT_EXAMPLES: dict[str, list[str]] = {
    # ── Conversation / mémoire générale ──────────────────────────────────────
    "memory": [
        "salut, ça va ?",
        "comment vas-tu Jarvis ?",
        "merci beaucoup",
        "c'est quoi ton avis là-dessus ?",
        "aide-moi à rédiger un email",
        "traduis ce texte en anglais",
        "explique-moi comment ça marche",
        "tu te souviens de notre conversation sur",
        "qu'est-ce que tu penses de ça ?",
        "c'est bon, parfait",
        "aide-moi avec ce code",
        "qu'est-ce que tu sais sur moi ?",
        "résume ce texte",
        "donne-moi des idées pour",
        "c'est quoi la différence entre",
    ],
    # ── Météo ────────────────────────────────────────────────────────────────
    "weather": [
        "quelle est la météo à",
        "quel temps fait-il ?",
        "météo à Paris demain",
        "température ce week-end",
        "météo de la semaine prochaine",
        "prévisions météo pour demain matin",
        "quel temps demain ?",
    ],
    # ── Emails / Gmail ───────────────────────────────────────────────────────
    "gmail": [
        "mes mails",
        "vérifie mes emails",
        "nouveaux mails ?",
        "résume mes emails d'aujourd'hui",
        "montre-moi mes messages récents",
        "emails avec le mot facture",
        "j'ai reçu quelque chose d'important ?",
        "check mes mails",
        "regarde dans mes mails",
    ],
    # ── Agenda / Calendrier ──────────────────────────────────────────────────
    "calendar": [
        "mon agenda aujourd'hui",
        "qu'est-ce que j'ai de prévu ?",
        "planning cette semaine",
        "mes rendez-vous du jour",
        "agenda de la semaine",
        "qu'est-ce que j'ai prévu ce soir ?",
        "montre-moi mon calendrier",
        "regarde dans mon agenda",
    ],
    # ── Briefing matinal ─────────────────────────────────────────────────────
    "briefing": [
        "briefing du matin",
        "briefing matinal",
        "lance le briefing",
        "donne-moi le briefing",
        "le point du matin",
        "briefing s'il te plaît",
    ],
    # ── Recherche web / actualités ───────────────────────────────────────────
    "web": [
        "cherche sur internet",
        "dernières actualités sur",
        "qu'est-ce qui se passe dans le monde ?",
        "recherche des infos sur",
        "recherche sur le net",
        "trouve-moi des informations sur",
        "quelles sont les news",
        "recherche en ligne",
    ],
    # ── Documents personnels / RAG ───────────────────────────────────────────
    "rag": [
        "cherche dans mes documents",
        "dans mes notes",
        "j'ai un fichier sur ce sujet",
        "retrouve le document sur",
        "mes notes sur",
        "cherche dans ma base documentaire",
        "regarde dans mes fichiers",
        "dans mes fichiers",
        "dans mes documents",
        "RAG",
        "rag",
    ],
    # ── Portefeuille boursier ────────────────────────────────────────────────
    "portfolio": [
        "mon portefeuille",
        "mon PEA",
        "mes actions",
        "performance de mes actions",
        "analyse mon portefeuille",
        "analyse mes actionsmes positions boursières",
        "comment va mon portefeuille ?",
    ],
    # ── État interne de Jarvis ───────────────────────────────────────────────
    "self": [
        "comment tu vas Jarvis ?",
        "Salut Jarvis, en forme ?quel est ton état actuel ?",
        "qu'est-ce que tu fais en ce moment ?",
        "ton état interne",
        "tes dernières réflexions",
        "qu'est-ce que tu as appris récemment ?",
        "comment tu te sens ?",
        "donne-moi ton introspection",
        "tu as réfléchi à quoi récemment ?",
        "tes auto-réflexions",
    ],
}

# ── Cache des embeddings d'exemples (initialisé une seule fois) ───────────────
_cache_lock = threading.Lock()
_examples_vectors: dict[str, np.ndarray] | None = None  # intent → (N, D) matrix


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
    """Tente d'extraire une ville depuis un message météo."""
    msg = message.lower()
    # Retire les mots déclencheurs météo et temporels
    clean = re.sub(
        r"\b(météo|meteo|temps|température|temperature|prévisions?|pluie|"
        r"soleil|chaud|froid|vent|neige|orage|nuages?|brouillard|"
        r"aujourd'hui|demain|ce soir|ce matin|ce week-?end|cette semaine|"
        r"lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|"
        r"là|la|dehors|extérieur|il fait|quel temps|parapluie)\b",
        " ",
        msg,
        flags=re.IGNORECASE,
    ).strip()
    # Les mots restants de > 2 caractères sont candidats ville
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
    if any(
        w in m for w in ["récent", "récents", "derniers", "semaine", "cette semaine"]
    ):
        return "newer_than:7d"
    return "newer_than:7d"  # défaut : mails récents


# ── Fonction principale ───────────────────────────────────────────────────────


def embed_route(message: str, google_available: bool = True) -> RouterResult | None:
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

    # ── 1. Règles déterministes ───────────────────────────────────────────────

    # URL → memory (la page est déjà fetchée automatiquement en amont)
    if re.search(r"https?://", msg):
        logger.debug("Embed router: URL détectée → memory")
        return _build_result("memory", msg, google_available)

    # Requête "mode expert" / "analyse approfondie" → LLM router (use_reasoning possible)
    reasoning_triggers = [
        "mode expert",
        "analyse approfondie",
        "réfléchis en profondeur",
        "raisonne",
        "debug complet",
        "analyse complète",
    ]
    if any(t in msg_lower for t in reasoning_triggers):
        logger.debug("Embed router: reasoning trigger → LLM router")
        return None

    # Messages très courts (≤ 12 chars) → memory (salutations, acquiescences)
    if len(msg) <= 12:
        logger.debug("Embed router: message court → memory")
        return _build_result("memory", msg, google_available)

    # Briefing exact (avant l'embedding, ces formulations sont sans ambiguïté)
    briefing_exact = {
        "briefing",
        "mon briefing",
        "briefing matinal",
        "briefing du matin",
        "lance le briefing",
        "le briefing",
        "fais le briefing",
    }
    if msg_lower.rstrip("! ?,") in briefing_exact:
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

        logger.info(
            "Embed router: %s (%.3f) — LLM router évité",
            best_intent,
            best_score,
        )
        return _build_result(best_intent, msg, google_available)

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
        use_reasoning=False,  # jamais — laissé au routeur LLM
        gmail_query=_extract_gmail_query(message) if intent == "gmail" else "",
        calendar_days=_extract_calendar_days(message) if intent == "calendar" else 7,
        weather_location=_extract_weather_location(message)
        if intent == "weather"
        else "",
    )
