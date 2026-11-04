"""
test_lru_prefix_snapshot.py — banc du snapshot de préfixe LRU (`_LRU_PREFIX_SNAPSHOT`)

**Ce n'est pas un test unitaire, c'est un BANC.** Il charge le modèle réel et exige que
Jarvis soit arrêté pour ne pas se disputer le GPU. Ignoré par défaut :
`pytest -m benchmark jarvis-core/tests/test_lru_prefix_snapshot.py -s`.

Il rejoue une conversation d'agent — des tours assistants porteurs de `tool_calls`, seuls
à produire des prompts où celui d'un tour est préfixe exact du suivant — et compare deux
passes sur la même conversation, drapeau éteint puis allumé.

Deux questions, deux critères :

  • GAIN      le nombre de jetons réutilisés doit suivre le prompt du tour précédent au
              lieu de retomber sur le bloc système. Mesuré, pas asserté : un gain nul
              n'est pas une régression, c'est une absence de bénéfice.

  • INTÉGRITÉ chaque tour doit produire une vraie génération. Un cache qui déclare couvrir
              plus qu'il ne couvre réellement fait émettre un EOS immédiat, donc une
              réponse vide ou de quelques jetons. C'est le critère d'échec, et il porte
              sur la passe allumée comme sur la passe éteinte.

L'effondrement observé en production survenait à la QUATRIÈME réutilisation : le banc
tourne donc sur six tours au minimum, un nombre plus faible ne prouverait rien.
"""

import asyncio
import os
import subprocess
import sys
import time

import pytest

# Env vars AVANT import du module (lus au niveau module).
os.environ.setdefault("LLM_LOCAL", "yes")
os.environ.setdefault("LLM_DEBUG_PROMPTS", "no")
os.environ.setdefault("RAW_DEBUG_PROMPTS", "no")
os.environ.setdefault("QUANT_KV", "yes")
os.environ.setdefault("QUANT_KV_BITS", "4")
os.environ.setdefault("LRU_KV_SIZE", "16")
os.environ.setdefault("LRU_KV_GB", "4.0")

# Chemin d'import : conftest.py le pose sous pytest, rien ne le pose en exécution directe.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _jarvis_running() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-f", "uvicorn main:app"], text=True)
        return [p for p in out.strip().splitlines() if p != str(os.getpid())] != []
    except subprocess.CalledProcessError:
        return False


# `integration` en plus de `benchmark` : le `skipif` ne protège la suite unitaire que
# pendant que Jarvis tourne, et c'est précisément arrêté que ce banc se lance. Sans ce
# marqueur, `pytest -m "not integration"` chargerait le modèle.
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.integration,
    pytest.mark.skipif(
        _jarvis_running(),
        reason="Jarvis tourne — le banc a besoin du GPU seul (jarvis-stop)",
    ),
]

import llm.local as local_mod  # noqa: E402
from config import PRIMARY_MODEL  # noqa: E402
from tool_calls import normalise_messages_for_template, parse_tool_calls  # noqa: E402

TOURS = 6
THINKING_BUDGET = 3000
MAX_TOKENS = 16000

# En deçà, la génération n'a pas vraiment eu lieu : c'est la signature de l'EOS immédiat.
MIN_JETONS_SORTIE = 20

OUTILS = [
    {
        "type": "function",
        "function": {
            "name": "doc_search",
            "description": "Recherche dans la base documentaire d'un produit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "description": "Nom du produit."},
                    "query": {"type": "string", "description": "Termes recherchés."},
                },
                "required": ["product", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ticket_search",
            "description": "Recherche des tickets ouverts correspondant à des termes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termes recherchés."},
                    "limit": {"type": "integer", "description": "Nombre de résultats."},
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEME = (
    "Tu es un assistant technique. Tu disposes d'outils de recherche documentaire. "
    "Tu appelles un outil quand la réponse exige une information que tu n'as pas, "
    "et tu réponds directement sinon. Tu réponds en français."
)

# UNE seule question, puis la boucle d'outils — c'est la forme d'une conversation
# d'agent, et elle n'est pas interchangeable avec une suite de questions. Le template
# n'écrit le bloc de réflexion vide que pour les tours assistants qui SUIVENT le dernier
# message utilisateur ; intercaler une question à chaque tour rejette les tours
# précédents avant ce repère, les fait rendre sans ce bloc, et rompt la chaîne de
# préfixes que le mécanisme exploite.
QUESTION = (
    "Quel est le comportement d'un équipement réseau quand sa licence expire, "
    "et que se passe-t-il si elle est renouvelée après la date d'expiration ?"
)


def _resultat_outil(tour: int) -> str:
    """Résultat d'outil volumineux — le prompt doit grossir comme en production.

    Le volume compte autant que le contenu : c'est lui qui fait diverger le prompt du
    bloc système, donc qui rend visible l'écart entre réutiliser le préfixe du tour
    précédent et retomber sur l'entrée système.
    """
    entree = (
        "Article {n} — procédure de vérification de l'état de licence. "
        "L'équipement conserve ses règles actives et cesse de recevoir les mises à jour "
        "de signatures. L'interface d'administration affiche un bandeau d'avertissement. "
        "Les sessions établies ne sont pas interrompues par la bascule d'état. "
    )
    return "\n".join(entree.format(n=tour * 10 + i) for i in range(1, 21))


async def _un_tour(messages: list[dict]) -> tuple[str, dict]:
    """Génère un tour et retourne (sortie brute, stats LRU)."""
    stats: dict = {"pose": None}
    original = local_mod._lru_get_cache
    original_pose = local_mod._lru_insert_prompt_prefix

    def _capture(model_path, model, tokenizer, sys_content, prompt_token_ids, tools=None):
        cache, remaining = original(
            model_path, model, tokenizer, sys_content, prompt_token_ids, tools
        )
        stats["prompt"] = len(prompt_token_ids)
        stats["restants"] = len(remaining)
        stats["reutilises"] = len(prompt_token_ids) - len(remaining)
        # Les jetons servent à vérifier la chaîne de préfixes, l'invariant dont dépend
        # tout le mécanisme : sans elle, l'entrée posée ne peut jamais être servie.
        stats["jetons"] = list(prompt_token_ids)
        return cache, remaining

    def _capture_pose(model_path, prompt_token_ids, cache, model_short):
        stats["pose"] = len(prompt_token_ids) - 1
        return original_pose(model_path, prompt_token_ids, cache, model_short)

    local_mod._lru_get_cache = _capture
    local_mod._lru_insert_prompt_prefix = _capture_pose
    try:
        t0 = time.time()
        morceaux: list[str] = []
        async for morceau in local_mod.stream_local(
            messages,
            PRIMARY_MODEL,
            no_think=False,
            thinking_budget=THINKING_BUDGET,
            max_tokens=MAX_TOKENS,
            tools=OUTILS,
            skip_debug_log=True,
        ):
            morceaux.append(morceau)
        stats["secondes"] = time.time() - t0
    finally:
        local_mod._lru_get_cache = original
        local_mod._lru_insert_prompt_prefix = original_pose

    sortie = "".join(morceaux)
    stats["sortie_car"] = len(sortie.strip())
    return sortie, stats


def _passe(actif: bool) -> list[dict]:
    """Rejoue la conversation complète avec le drapeau dans l'état demandé."""
    local_mod._LRU_PREFIX_SNAPSHOT = actif
    local_mod._lru_caches.clear()  # chaque passe part d'un LRU vierge

    messages = [
        {"role": "system", "content": SYSTEME},
        {"role": "user", "content": QUESTION},
    ]
    mesures = []

    for tour in range(TOURS):
        sortie, stats = asyncio.run(_un_tour(normalise_messages_for_template(messages)))

        texte, appels = parse_tool_calls(sortie.strip(), OUTILS)
        if not appels:
            # Le modèle a répondu sans outil. On force la forme agentique : sans tour
            # assistant porteur de tool_calls, le template n'écrit pas le bloc de
            # réflexion vide et la chaîne de préfixes ne peut pas s'établir.
            appels = [{
                "id": f"call_banc{tour}",
                "type": "function",
                "function": {"name": "doc_search", "arguments": "{\"product\": \"x\"}"},
            }]
        messages.append({"role": "assistant", "content": texte, "tool_calls": appels})
        for appel in appels:
            messages.append({
                "role": "tool",
                "tool_call_id": appel["id"],
                "content": _resultat_outil(tour),
            })

        stats["tour"] = tour + 1
        stats["appels"] = len(appels)
        mesures.append(stats)

        jetons = stats["sortie_car"] // 4
        drapeau = "⚠️ " if jetons < MIN_JETONS_SORTIE else "   "
        pose = stats["pose"] if stats["pose"] is not None else "—"
        print(
            f"  {drapeau}tour {stats['tour']}  prompt={stats['prompt']:>6}  "
            f"réutilisés={stats['reutilises']:>6}  restants={stats['restants']:>6}  "
            f"posé={str(pose):>6}  sortie={stats['sortie_car']:>6}car  "
            f"{stats['secondes']:.1f}s"
        )

    # Chaîne de préfixes : la clé posée au tour N doit être préfixe des jetons du tour
    # N+1. Si elle ne l'est pas, l'entrée est inutilisable quoi que fasse le cache, et
    # c'est là qu'il faut chercher — pas dans le LRU.
    for a, b in zip(mesures, mesures[1:]):
        if a["pose"] is None:
            a["chaine"] = None
            continue
        cle = a["jetons"][: a["pose"]]
        a["chaine"] = b["jetons"][: len(cle)] == cle
    ruptures = [m["tour"] for m in mesures if m.get("chaine") is False]
    if any(m["pose"] is not None for m in mesures):
        print(
            f"  chaîne de préfixes : {'intacte' if not ruptures else f'rompue aux tours {ruptures}'}"
        )

    return mesures


def test_snapshot_prefixe():
    print(f"\n{'─' * 78}\nPASSE 1 — drapeau ÉTEINT (référence)\n{'─' * 78}")
    eteint = _passe(False)

    print(f"\n{'─' * 78}\nPASSE 2 — drapeau ALLUMÉ\n{'─' * 78}")
    allume = _passe(True)

    print(f"\n{'─' * 78}\nCOMPARAISON\n{'─' * 78}")
    print("  tour   réutilisés éteint   réutilisés allumé   restants éteint   restants allumé")
    for a, b in zip(eteint, allume):
        print(
            f"  {a['tour']:>4}   {a['reutilises']:>16}   {b['reutilises']:>17}   "
            f"{a['restants']:>15}   {b['restants']:>15}"
        )
    economie = sum(a["restants"] for a in eteint) - sum(b["restants"] for b in allume)
    print(f"\n  jetons de prefill économisés sur {TOURS} tours : {economie}")

    # INTÉGRITÉ — le critère d'échec. Une sortie quasi vide signe le cache incohérent.
    for mesures, nom in ((eteint, "éteint"), (allume, "allumé")):
        courts = [m["tour"] for m in mesures if m["sortie_car"] // 4 < MIN_JETONS_SORTIE]
        assert not courts, (
            f"drapeau {nom} : génération effondrée aux tours {courts} "
            f"(< {MIN_JETONS_SORTIE} jetons de sortie) — cache incohérent"
        )


# Exécution directe, hors pytest : `python jarvis-core/tests/test_lru_prefix_snapshot.py`.
# Sous pytest, le thread de travail de `stream_local` ne démarre pas et la boucle attend
# indéfiniment son sentinel — le même code tourne normalement en exécution directe.
if __name__ == "__main__":
    try:
        test_snapshot_prefixe()
    except AssertionError as echec:
        print(f"\n❌ ÉCHEC — {echec}")
        raise SystemExit(1)
    print("\n✅ intégrité préservée sur les deux passes")
