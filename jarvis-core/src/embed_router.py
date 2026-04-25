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
from llm_router import RouterResult

logger = get_logger("jarvis-embed-router")

EMBED_ROUTER_ENABLED: bool = os.getenv("EMBED_ROUTER", "yes").lower() != "no"

# Seuil minimum de similarité cosinus pour accepter un intent
EMBED_ROUTE_THRESHOLD: float = 0.74

# Si les deux meilleurs intents sont à moins de cette marge, c'est ambigu → LLM
AMBIGUITY_MARGIN: float = 0.06

# ── Small talk whitelist ───────────────────────────────────────────────────────
# Acquiescements purs : aucun contenu informationnel, le LLM n'a besoin que de
# l'historique de conversation. Détectés dans chat.py AVANT le keyword dispatch
# (step 1) pour bypasser les lectures Redis inutiles.
# Critères bloquants appliqués par is_small_talk() : "?" présent, longueur > 50.
# Garde calendrier : is_small_talk() doit être appelé uniquement s'il n'y a PAS
# d'action calendrier en attente (oui/non peuvent être des confirmations).
SMALL_TALK_EXACT: frozenset[str] = frozenset({
    "merci", "merci !", "merci beaucoup", "super merci", "merci bien",
    "parfait", "c'est parfait", "top", "génial", "excellent", "nickel",
    "super", "très bien", "bien", "c'est bon", "c'est bien",
    "ok", "okay", "oki", "d'accord", "ok ok",
    "oui oui", "non non",
    "vas-y", "go", "continue", "allez", "fais-le", "fais",
    "bonne idée", "oui bonne idée", "oui c'est ça",
    "ah ok", "ah je vois", "ah d'accord", "ah oui",
    "je vois", "j'ai compris", "compris", "reçu",
    "haha", "lol", "😄", "👍",
})


def is_small_talk(message: str) -> bool:
    """
    Retourne True si le message est un acquiescement pur (aucun contenu).
    NE PAS appeler quand une action calendrier est en attente : oui/non/ok
    peuvent être des confirmations d'événement.
    """
    if len(message) > 50 or "?" in message:
        return False
    return message.lower().rstrip(" !.,") in SMALL_TALK_EXACT

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
        # questions / demandes d'aide — nécessitent le profil pour personnaliser
        "c'est quoi ton avis là-dessus ?",
        "aide-moi à rédiger un email",
        "explique-moi comment ça marche",
        "qu'est-ce que tu penses de ça ?",
        "aide-moi avec ce code",
        "qu'est-ce que tu sais sur moi ?",
        "résume ce texte",
        "donne-moi des idées pour",
        "c'est quoi la différence entre",
        # partage d'informations personnelles — profil à mettre à jour
        "je viens de décider de",
        "j'ai changé d'avis sur",
        "depuis quelque temps je",
        "j'adore ce sport",
        "je déteste travailler le soir",
        "j'ai toujours préféré",
        "je ne supporte pas le",
        # référence au passé / continuité de conversation
        "tu te souviens de",
        "comme je t'avais expliqué",
        "rappelle-toi ce projet",
        "on avait parlé de",
        "tu sais ce que j'ai fait hier",
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
        "lit mon dernier mail",
        "lit mes mails non lus",
        "lit mes mails",
        "vérifie mes emails",
        "résume mes emails non lus",
        "recherche dans mes mails",
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
        "cherche sur le net",
        "dernières actualités",
        "recherche des infos",
        "recherche sur le net",
        "trouve-moi des informations",
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
    # ── Raisonnement / mode expert ───────────────────────────────────────────
    "reasoning": [
        "mode expert",
        "analyse approfondie",
        "réfléchis en profondeur",
        "réfléchis bien",
        "raisonne sur",
        "raisonne étape par étape",
        "raisonne par étape",
        "pense par étape",
        "debug",
        "analyse complète",
        "réflexion approfondie",
        "donne-moi ton analyse complète",
        "prends le temps de réfléchir",
    ],
    # ── Portefeuille boursier ────────────────────────────────────────────────
    "portfolio": [
        "mon portefeuille",
        "mon PEA",
        "mes actions",
        "performance de mes actions",
        "analyse mon portefeuille",
        "mes positions boursières",
        "comment va mon portefeuille ?",
    ],
    # ── État interne de Jarvis ───────────────────────────────────────────────
    "self": [
        "comment vas-tu Jarvis ?",
        "Salut Jarvis, en forme ?",
        "qu'est-ce que tu fais en ce moment ?",
        "ton état interne",
        "tes dernières réflexions",
        "qu'est-ce que tu as appris récemment ?",
        "comment tu te sens ?",
        "donne-moi ton introspection",
        "tu as réfléchi à quoi récemment ?",
        "tes auto-réflexions",
        "tes réflexions",
        "montre les propositions de prompt",
        "montre les prompts en attente",
        "montre les prompts",
        "liste les propositions en attente",
        "accepte la proposition",
        "rejette la proposition",
        "montre la proposition",
        "approuve la proposition de prompt",
    ],
}

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


def _extract_rag_query(message: str) -> str:
    """Retire les phrases de commande du message pour n'en garder que les mots-clés sémantiques."""
    cleaned = re.sub(
        r"\b(cherche(?:r)?|retrouve(?:r)?|trouve(?:r)?|recherche(?:r)?|regarde(?:r)?|"
        r"dans mes (documents?|notes?|fichiers?|base documentaire)|"
        r"mes (documents?|notes?|fichiers?)|"
        r"mode expert\s*:?|j'ai un fichier sur|RAG|rag)\b",
        " ",
        message,
        flags=re.IGNORECASE,
    ).strip()
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?,!.")
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

    # Requête "mode expert" / "analyse approfondie" → reasoning direct (no_think=False)
    # Retourne use_reasoning=True sans passer par le LLM router.
    reasoning_triggers = [
        "mode expert",
        "analyse approfondie",
        "réfléchis en profondeur",
        "raisonne",
        "debug complet",
        "analyse complète",
    ]
    if any(t in msg_lower for t in reasoning_triggers):
        logger.debug("Embed router: reasoning trigger → reasoning direct")
        return _build_result("reasoning", msg, google_available)

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
        return _build_result("self", msg, google_available)

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
        use_memory=intent
        in ("memory", "reasoning"),  # reasoning inclut le contexte mémoire
        use_rag=intent == "rag",
        use_web=intent == "web",
        use_weather=intent == "weather",
        use_gmail=intent == "gmail" and google_available,
        use_calendar=intent == "calendar" and google_available,
        use_briefing=intent == "briefing",
        use_self=intent == "self",
        use_portfolio=intent == "portfolio",
        use_reasoning=intent == "reasoning",

        gmail_query=_extract_gmail_query(message) if intent == "gmail" else "",
        calendar_days=_extract_calendar_days(message) if intent == "calendar" else 7,
        weather_location=_extract_weather_location(message)
        if intent == "weather"
        else "",
        rag_query=_extract_rag_query(message) if intent == "rag" else "",
    )
