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
        "les news du jour",
        "cours actuel du pétrole",
        "qui a gagné le match hier",
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
        "j'ai un fichier sur ce sujet",
        "retrouve le document sur",
        "cherche dans ma base documentaire",
        "regarde dans mes fichiers",
        "dans mes fichiers",
        "dans mes documents",
        "RAG",
        "lis ma fiche",
        "lis un extrait de mon fichier",
        "base-toi sur mon fichier",
        "base-toi sur ma fiche",
        "ma fiche sur",
        "extrait de mon document",
        "depuis mon RAG",
        "qui est dans le rag",
        "montre-moi ma fiche",
        "consulte mon fichier",
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
    # ── Questions sur l'état d'un projet ────────────────────────────────────
    # Couvre les requêtes de STATUS ("où en est", "comment avance").
    # Les mises à jour conversationnelles ("j'ai avancé sur X") routent vers
    # "memory" — la mémoire épisodique fournit le contexte, l'analyzer capture
    # l'update dans la timeline. L'injection de détail n'est pas nécessaire
    # quand l'utilisateur donne une information (il connaît son propre projet).
    "project": [
        "comment avance le projet",
        "où en est le projet",
        "état d'avancement du projet",
        "donne-moi l'avancement",
        "mets à jour le projet",
        "j'ai avancé sur le projet",
        "j'ai terminé la partie",
        "j'ai fini le projet",
        "on avance sur le projet",
        "prochaine étape du projet",
        "il reste encore à faire",
        "j'ai commencé un nouveau projet",
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
        "parle-moi de toi",
        "ton identité",
    ],
}

# ── Cache des embeddings d'exemples (initialisé une seule fois) ───────────────
_cache_lock = threading.Lock()
_examples_vectors: dict[str, np.ndarray] | None = None  # intent → (N, D) matrix


_REASON_EXACT = {
    "mode expert",
    "analyse approfondie",
    "analyse complète",
    "analyse détaillée",
    "réflexion approfondie",
    "réfléchis en profondeur",
    "réfléchis bien",
    "prends le temps de réfléchir",
    "prends le temps d'analyser",
    "debug complet",
}

_REASON_REGEX = re.compile(
    r"\braisonne\b|\bréfléchis\b|\bétape par étape\b|\bpas à pas\b|\ben profondeur\b",
    re.IGNORECASE,
)

# ── Small talk — acquiescements purs (≤ 50 chars, pas de ?, pas de contenu) ──
# Bypasse profil, mémoire et opinions : le LLM n'a besoin que de l'historique.
# WHITELIST conservative : uniquement mots qui n'apportent aucun fait nouveau.
# Critères bloquants : présence de "?" OU longueur > 50 chars → jamais small talk.
_SMALL_TALK_EXACT = {
    "merci",
    "merci !",
    "merci beaucoup",
    "super merci",
    "merci bien",
    "parfait",
    "c'est parfait",
    "top",
    "génial",
    "excellent",
    "nickel",
    "super",
    "très bien",
    "bien",
    "c'est bon",
    "c'est bien",
    "ok",
    "okay",
    "oki",
    "d'accord",
    "ok ok",
    "oui oui",
    "non non",
    "vas-y",
    "go",
    "continue",
    "allez",
    "fais-le",
    "fais",
    "bonne idée",
    "oui bonne idée",
    "oui c'est ça",
    "ah ok",
    "ah je vois",
    "ah d'accord",
    "ah oui",
    "je vois",
    "j'ai compris",
    "compris",
    "reçu",
    "haha",
    "lol",
    "😄",
    "👍",
    # Salutations pures (ajoutées)
    "bonjour",
    "salut",
    "salut jarvis",
    "hello",
    "hey",
    "yo",
    "coucou",
    "bonsoir",
    "hi",
    "hola",
    "re",
    "rebonjour",
}

# Briefing exact (avant l'embedding, ces formulations sont sans ambiguïté)
_BRIEFING_EXACT = {
    "briefing",
    "mon briefing",
    "briefing matinal",
    "briefing du matin",
    "lance le briefing",
    "le briefing",
    "fais le briefing",
}

# Liants minuscules dans les noms de villes françaises
_FR_CITY_LIANTS = r"(?:de|du|des|le|la|les|aux|en|sur|sous|sainte?|saint)"

# Capture une ville après préposition, en préservant les noms composés :
# La Rochelle · Aix-en-Provence · Saint-Germain-en-Laye · Boulogne-sur-Mer
_CITY_AFTER_PREP_RE = re.compile(
    r"\b(?:à|au|aux|pour|sur|vers|en)\s+"
    r"("
    r"[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ][\wÀ-ÿ'\-]*"
    r"(?:[-\s]+(?:" + _FR_CITY_LIANTS + r"[-\s]+)?"
    r"[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]?[\wÀ-ÿ'\-]+){0,3}"
    r")"
)

# Faux positifs fréquents après préposition (jours, moments)
_TEMPORAL_WORDS = {
    "aujourd'hui",
    "demain",
    "hier",
    "lundi",
    "mardi",
    "mercredi",
    "jeudi",
    "vendredi",
    "samedi",
    "dimanche",
    "matin",
    "soir",
    "midi",
}


# ── RAG query extraction ─────────────────────────────────────────────────────
# Phase 1: strip command/routing phrases (compound patterns first, then single verbs)
_RAG_CMD_RE = re.compile(
    r"(?:"
    r"base-toi sur (?:mon|ma)\s*|"
    r"lis (?:un extrait de )?(?:mon|ma)\s*|"
    r"montre(?:-moi)? (?:mon|ma)\s*|"
    r"extrait de (?:mon|ma)\s*|"
    r"consulte(?:r)? (?:mon|ma|mes)\s*|"
    r"depuis (?:le |la |mon |ma |mes )?RAG\b\s*|"
    r"dans (?:le |la |mon |ma |mes )?RAG\b\s*|"
    r"sur (?:le |la |mon |ma |mes )?RAG\b\s*|"
    r"qui est dans le RAG\b\s*|"
    r"dans mes (?:documents?|notes?|fichiers?|base documentaire)\s*|"
    r"dans (?:mon|ma) (?:document|fichier|fiche|note)\s*|"
    r"(?:mes|mon|ma) (?:documents?|notes?|fichiers?)\s*|"
    r"j'ai (?:un|une) (?:fichier|fiche|document|note) sur\s*|"
    r"retrouve(?:r)? (?:le |la |un |une )?(?:document|fichier|fiche) sur\s*|"
    r"\b(?:"
    r"cherche(?:r)?|retrouve(?:r)?|trouve(?:r)?|recherche(?:r)?|"
    r"regarde(?:r)?|consulte(?:r)?|montre(?:-moi)?|lis|lit|"
    r"extrais?|extrait"
    r")\b\s*|"
    r"\bRAG\b\s*"
    r")",
    re.IGNORECASE,
)

# Phase 2: strip leading articles/possessives left after command removal
_RAG_LEAD_NOISE_RE = re.compile(
    r"^(?:(?:mon|ma|mes|le|la|les|l'|un|une|du|de|des|d')\s+)+",
    re.IGNORECASE,
)


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
    m = _CITY_AFTER_PREP_RE.search(message)
    if m:
        city = m.group(1).strip(" ,;:!?.")
        first_word = city.split()[0].lower().rstrip("-")
        if first_word not in _TEMPORAL_WORDS:
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
    cleaned = _RAG_CMD_RE.sub(" ", message)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?,!.")
    cleaned = _RAG_LEAD_NOISE_RE.sub("", cleaned).strip()
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

    # Détection reasoning (flag, pas intent) — doit précéder les early-returns
    force_reasoning = bool(
        any(t in msg_lower for t in _REASON_EXACT) or _REASON_REGEX.search(msg_lower)
    )

    # URL → memory (la page est déjà fetchée automatiquement en amont)
    if re.search(r"https?://", msg):
        logger.debug("Embed router: URL détectée → memory")
        result = _build_result("memory", msg, google_available)
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
        if _norm in _SMALL_TALK_EXACT:
            logger.debug("Embed router: small talk → no context injection")
            return _build_result("small_talk", msg, google_available)

    if msg_lower.rstrip("! ?,") in _BRIEFING_EXACT:
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
