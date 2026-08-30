"""
lexique.py — Sélection du lexique de reconnaissance, selon `JARVIS_LANG`.
===============================================================================
Pendant de `prompts.py` pour ce qui RECONNAÎT la langue en entrée : phrases-
exemples du routeur sémantique, small talk, déclencheurs de raisonnement, motifs
d'extraction.

Les imports ci-dessous sont **explicites, jamais `import *`**. C'est délibéré :
cette liste EST le contrat entre les langues. Un nom oublié dans `lexique_en.py`
fait échouer l'import au démarrage — bruyamment, et avant la première requête —
là où un `import *` l'aurait laissé passer jusqu'à un `NameError` en pleine
conversation.
"""

import importlib
import os

_JEUX = {"fr": "llm.lexique_fr", "en": "llm.lexique_en"}
_LANG = os.getenv("JARVIS_LANG", "fr").strip().lower()

if _LANG not in _JEUX:  # pragma: no cover - garde de démarrage
    raise RuntimeError(
        f"JARVIS_LANG={_LANG!r} inconnue — valeurs acceptées : {', '.join(sorted(_JEUX))}"
    )

_jeu = importlib.import_module(_JEUX[_LANG])

# ── Le contrat ────────────────────────────────────────────────────────────
# Tout nom ajouté ici doit exister dans CHAQUE lexique de langue.
INTENT_EXAMPLES = _jeu.INTENT_EXAMPLES

REASON_EXACT = _jeu._REASON_EXACT
REASON_REGEX = _jeu._REASON_REGEX

SMALL_TALK_EXACT = _jeu._SMALL_TALK_EXACT
BRIEFING_EXACT = _jeu._BRIEFING_EXACT
TEMPORAL_WORDS = _jeu._TEMPORAL_WORDS

CITY_AFTER_PREP_RE = _jeu._CITY_AFTER_PREP_RE
RAG_CMD_RE = _jeu._RAG_CMD_RE
RAG_LEAD_NOISE_RE = _jeu._RAG_LEAD_NOISE_RE

LANG = _LANG
