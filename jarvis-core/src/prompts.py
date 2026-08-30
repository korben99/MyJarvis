"""
prompts.py — Résolution des prompts, indépendante de la langue.
===============================================================================
Ce module ne contient AUCUN texte de prompt. Il charge le jeu correspondant à
`JARVIS_LANG` (`prompts_fr` ou `prompts_en`) et expose la mécanique commune :
`get_prompt()`, les surcharges à chaud de l'autocoding, et la liste blanche des
prompts modifiables.

**Une langue par instance, pas par utilisateur.** Le choix est fait au démarrage
et ne change plus. C'est ce qui permet au shim de servir les 48 sites d'appel
sans qu'aucun n'ait à connaître la langue, et ce qui met l'autocoding à l'abri
du mélange : une instance, une langue, une surcharge sans ambiguïté.

Pour ajouter une langue : écrire `prompts_<code>.py` exposant exactement les
mêmes noms que `prompts_fr.py`, et l'ajouter à `_JEUX`. Un test vérifie la
parité des noms — un prompt manquant deviendrait un `KeyError` au moment de
servir une requête, c'est-à-dire au pire moment.
"""

import importlib
import json
import os

# ── Sélection du jeu de langue ────────────────────────────────────────────
# Lue ici et non via `config` : `config` importe `helpers`, qui importe
# `llm/local`, qui importe ce module. Passer par config créerait un cycle.
_JEUX = {"fr": "prompts_fr", "en": "prompts_en"}
_LANG_DEMANDEE = os.getenv("JARVIS_LANG", "fr").strip().lower()

if _LANG_DEMANDEE not in _JEUX:  # pragma: no cover - garde de démarrage
    raise RuntimeError(
        f"JARVIS_LANG={_LANG_DEMANDEE!r} inconnue — valeurs acceptées : "
        f"{', '.join(sorted(_JEUX))}"
    )

LANG = _LANG_DEMANDEE
_jeu = importlib.import_module(_JEUX[LANG])

# Import en masse des constantes du jeu choisi. On filtre sur les MAJUSCULES :
# tout ce qui commence par `_` est privé au jeu, et rien d'autre n'a à traverser.
globals().update({
    nom: valeur for nom, valeur in vars(_jeu).items() if nom.isupper()
})


# ══════════════════════════════════════════════════════════════════════════
#  PROMPTS MODIFIABLES PAR L'AUTOCODING
# ══════════════════════════════════════════════════════════════════════════
# Source unique de vérité pour `refine_prompt`. La liste n'existait qu'en PROSE, dans
# REFLECTION_PROMPT ; côté code, le seul garde était « la constante existe », si bien que
# les 41 prompts du module étaient acceptés — 24 hors de la liste annoncée au modèle,
# dont AGENT_SYSTEM et REFINE_PROMPT_SYSTEM lui-même.
#
# Ne sont refinables que les prompts dont un échec est OBSERVABLE dans une conversation ou
# dans un cycle : identité, routage, analyse, briefing, réflexion, revue nocturne. En sont
# exclus les prompts de la boucle agentique (jugés sur un livrable, pas sur un échange) et
# les prompts de service courts, où une réécriture coûte plus qu'elle ne rapporte.
REFINABLE_PROMPTS: frozenset[str] = frozenset({
    "SYSTEM_BASE", "IDENTITY",
    "ROUTER_SYSTEM", "ROUTER_USER",
    "ANALYSIS_PROMPT", "BRIEFING_USER", "WEB_RELEVANCE_JUDGE",
    "NIGHTLY_FACTS_PROMPT", "NIGHTLY_FACTS_SYSTEM",
    "NIGHTLY_SELF_PROMPT", "NIGHTLY_SELF_SYSTEM",
    "NIGHTLY_CLEANING_PROMPT", "NIGHTLY_CLEANING_SYSTEM",
    "REFLECTION_PROMPT", "REFLECTION_SYSTEM",
    "REFLECTION_USER_PROMPT", "REFLECTION_USER_SYSTEM",
})

# Garde de démarrage, sur le modèle de celui de `self/engine.py` pour le catalogue
# d'actions : un nom annoncé au modèle mais sans constante derrière produirait une
# proposition sur un prompt fantôme, et l'échec ne serait visible qu'au moment d'appliquer
# la surcharge. On préfère refuser de démarrer.
#
# Le garde vaut désormais aussi contrôle de complétude du jeu de langue : si `prompts_en`
# oublie un prompt refinable, l'instance anglaise ne démarre pas.
_REFINABLES_FANTOMES = sorted(
    n for n in REFINABLE_PROMPTS if not isinstance(globals().get(n), str)
)
if _REFINABLES_FANTOMES:  # pragma: no cover - garde de démarrage
    raise RuntimeError(
        f"prompts[{LANG}]: REFINABLE_PROMPTS cite des constantes inexistantes : "
        f"{_REFINABLES_FANTOMES}"
    )


# ══════════════════════════════════════════════════════════════════════════
#  LIVE OVERRIDE LOADER
# ══════════════════════════════════════════════════════════════════════════
# get_prompt(name) is the canonical way to retrieve any prompt at runtime.
# It checks prompt_overrides.json first (mtime-cached, no restart needed).
# Falls back to the module constant if no override is active.
# All callers should use get_prompt("NAME") instead of the bare constant.

_overrides_path: str | None = None  # resolved lazily to avoid circular import
_override_cache: dict = {}
_override_mtime: float = -1.0


def _resolve_overrides_path() -> str:
    """Lazily resolve the overrides file path via config (avoids circular import)."""
    global _overrides_path
    if _overrides_path is None:
        try:
            from config import PROMPT_DATA_DIR

            _overrides_path = os.path.join(PROMPT_DATA_DIR, "prompt_overrides.json")
        except Exception:
            _overrides_path = ""  # mark as failed so we don't retry forever
    return _overrides_path


def get_prompt(name: str) -> str:
    """
    Return the current text for prompt constant `name`, in the instance language.

    Priority:
      1. Active override in prompt_overrides.json  (live, mtime-cached)
      2. Constant from the loaded language set      (compile-time default)

    The overrides file is only re-read when its mtime changes, so the overhead
    on hot paths (router, analyzer) is a single os.stat() call.
    """
    global _override_cache, _override_mtime
    path = _resolve_overrides_path()
    if path and os.path.exists(path):
        try:
            mtime = os.path.getmtime(path)
            if mtime != _override_mtime:
                with open(path, encoding="utf-8") as f:
                    _override_cache = json.load(f)
                _override_mtime = mtime
        except Exception:
            pass
    elif _override_cache:
        # Overrides file deleted (manual rollback) — drop the stale cache so the
        # module constants take effect again without a restart.
        _override_cache = {}
        _override_mtime = -1.0
    if name in _override_cache:
        return _override_cache[name]
    valeur = globals().get(name)
    if not isinstance(valeur, str):
        # Rendait `""` jusqu'ici. Un prompt vide part au modèle sans lever la moindre
        # erreur : le seul symptôme est une dégradation des réponses, constatée des jours
        # plus tard et impossible à rattacher à sa cause. Même famille que le champ
        # pydantic supprimé en silence par `extra=ignore`.
        raise KeyError(f"prompt inconnu en {LANG} : {name!r}")
    return valeur
