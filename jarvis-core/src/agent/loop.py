"""La boucle : objectif → outil → observation → … → finish.

Elle emprunte le SEUL chemin d'inférence qui rende `tools` disponible, stream_local(), en
priorité background (`priority="bg"`, cf. llm/local.py) : un pas d'agent ne prend le GPU
que si aucun appel de chat n'attend.

Trois budgets bornent la dérive, et ils sont indépendants :
    max_steps   nombre de tours          — borne le raisonnement en rond
    timeout     temps réel               — borne l'attente derrière le chat
    no-progress deux appels identiques   — borne la boucle serrée sur un outil qui échoue

Rien n'est perdu à un redémarrage : le contexte est écrit sur disque après chaque pas et
la tâche est remise en file au boot suivant (store.requeue_interrupted).
"""

import asyncio
import json
import os
import time

from config import (
    AGENT_MAX_STEPS,
    AGENT_QUIET_SECONDS,
    AGENT_STEP_MAX_TOKENS,
    AGENT_TASK_TIMEOUT_MINUTES,
    AGENT_THINKING_BUDGET,
    AGENT_WRITE_MAX_CHARS,
    AGENT_WRITE_MAX_TOKENS,
    PRIMARY_MODEL,
)
from helpers import get_logger
from prompts import get_prompt
from tool_calls import normalise_messages_for_template, parse_tool_calls

from . import store
from .store import append_transcript, save_messages, save_task
from .tools import FINISH, PLAN, TOOL_SCHEMAS, execute_tool, render_plan

logger = get_logger("jarvis-agent")

# Au-delà, on élide les plus vieux résultats d'outil. En caractères, pas en tokens : on ne
# tokenise pas pour ça, l'ordre de grandeur suffit (~4 car/token → ~25 k tokens).
_CONTEXT_SOFT_CAP = 100_000
_ELIDED = "[résultat élidé — trop ancien pour tenir dans le contexte]"

# Fenêtre de détection de boucle, en nombre d'appels. 6 couvre un aller-retour A→B→A→B
# sans pénaliser une reprise légitime du même outil à quelques pas d'intervalle.
_LOOP_WINDOW = 6


class _Cancelled(Exception):
    """Annulation demandée entre deux pas."""


# ── Contexte ──────────────────────────────────────────────────────────────


def _initial_messages(task: dict) -> list[dict]:
    system = get_prompt("AGENT_SYSTEM").format(
        workspace=task["workspace"],
        max_steps=AGENT_MAX_STEPS,
        write_max_chars=AGENT_WRITE_MAX_CHARS,
    )
    objective = get_prompt("AGENT_OBJECTIVE").format(
        objective=task["objective"], max_steps=AGENT_MAX_STEPS
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": objective},
    ]


def _compact(messages: list[dict]) -> list[dict]:
    """Élide les plus anciens résultats d'outil quand le contexte devient trop gros.

    On ne touche ni au system, ni à l'objectif, ni aux tours de l'assistant : ce sont eux
    qui portent le fil. Les résultats d'outil, eux, ont déjà été exploités au tour où ils
    sont arrivés — et ce qui devait en être retenu a normalement été écrit sur disque.
    """
    total = sum(len(m.get("content") or "") for m in messages)
    if total <= _CONTEXT_SOFT_CAP:
        return messages
    for message in messages:
        if total <= _CONTEXT_SOFT_CAP:
            break
        if message.get("role") == "tool" and message.get("content") != _ELIDED:
            total -= len(message["content"]) - len(_ELIDED)
            message["content"] = _ELIDED
    return messages


def _relance(step: int, rien_ecrit: bool) -> str:
    """La relance à joindre au pas `step`. Vide avant la mi-parcours.

    DEUX relances, plus trois. Celle des derniers pas (« il te reste peu de pas, écris
    maintenant ») protégeait la fin de partie : la phase de conclusion la garantit
    désormais mécaniquement, elle n'avait plus de rôle propre.

    Les deux qui restent disent des choses différentes, et c'est ce qui les justifie :
    l'une rappelle un rythme, l'autre porte un ÉTAT que le modèle ne peut pas observer —
    il ne sait pas ce qu'il y a sur le disque sans le lister. Faible mesure mais nette :
    sur quatre exécutions, les deux fois où la relance lui est parvenue il a écrit dans
    les deux pas suivants ; les deux fois où elle ne lui est pas parvenue (mécanisme
    absent, puis pied de page manquant sur les tours « plan seul »), rien n'a été écrit.
    """
    if AGENT_MAX_STEPS - step > AGENT_MAX_STEPS // 2:
        return ""
    return get_prompt("AGENT_HINT_NO_FILE" if rien_ecrit else "AGENT_HINT_HALF_BUDGET")


def _step_footer(step: int, task: dict, rien_ecrit: bool = False) -> str:
    """Pied ajouté à chaque résultat d'outil : compteur de pas, relance, plan.

    Le plan est réaffiché ICI, sous le résultat que le modèle vient d'obtenir : il voit
    d'un coup ce qu'il a appris et où il en est, sans avoir à le redéduire.

    `rien_ecrit` est CALCULÉ PAR L'APPELANT, une fois par pas. Il l'était auparavant ici,
    ce qui faisait un os.listdir du workspace à chaque résultat d'outil — de l'I/O disque
    caché dans une fonction de rendu, appelée plusieurs fois par tour.
    """
    footer = get_prompt("AGENT_STEP_FOOTER").format(
        step=step, max_steps=AGENT_MAX_STEPS, hint=_relance(step, rien_ecrit),
    )
    return footer + render_plan(task)


# Le raisonnement du dernier tour est réinjecté ; les précédents sont élagués. Le cap est
# large : ThinkingBudgetProcessor borne déjà la production à AGENT_THINKING_BUDGET.
_THINK_KEEP_CHARS = 4000


def _split_think(raw: str) -> tuple[str, str]:
    """Sépare (raisonnement, sortie visible).

    Le découpage doit précéder parse_tool_calls : le modèle ébauche souvent un
    <tool_call> À L'INTÉRIEUR de sa réflexion, et parser le brut ferait exécuter un
    appel qu'il avait justement écarté en réfléchissant.
    """
    if "</think>" not in raw:
        return "", raw.strip()
    think, visible = raw.split("</think>", 1)
    return think.replace("<think>", "").strip(), visible.strip()


def _context_for_model(messages: list[dict]) -> list[dict]:
    """Copies prêtes pour le template : retire `_think`, qui est un champ interne.

    LE RAISONNEMENT N'EST PAS RÉINJECTÉ, et c'est délibéré.

    Le problème est réel — un tour d'agent laissait `content=''`, tout le « pourquoi »
    vivant dans le <think> qu'on jette. Le contexte se réduisait à un objectif suivi d'un
    tas de résultats bruts, et le modèle rejouait six fois la même recherche.

    Mais le rapatrier par la tuyauterie a échoué DEUX fois le 19/08/2026, de deux façons
    distinctes, en figeant la boucle à chaque fois :
      1. fusionné dans le `content` du tour assistant → le modèle en déduit que « contenu
         assistant = mon raisonnement », émet sa réflexion, ferme </think>, puis réécrit le
         MÊME texte en sortie visible (think=3556 car, visible=3556 car) et se fige ;
      2. ajouté en message `user` en fin de contexte, donc APRÈS le résultat de l'outil →
         le plan périmé devient le signal le plus récent. Le modèle relit « ton plan était
         d'aller lire l'article RTL » juste après avoir reçu cet article, et le refetche.
         Trois fois de suite.

    La solution retenue ne coûte pas une ligne de mécanique : AGENT_SYSTEM demande au
    modèle d'écrire UNE phrase en clair avant chaque appel d'outil. Elle atterrit dans
    `content`, se persiste d'elle-même, occupe la bonne place chronologique, et ne peut
    donc reproduire ni l'une ni l'autre des deux pannes ci-dessus.

    `_think` reste capturé — il alimente le journal et le transcript, jamais le prompt.
    """
    return [{k: v for k, v in m.items() if k != "_think"} for m in messages]


# ── Inférence d'un pas ────────────────────────────────────────────────────


def _fichiers_produits(task: dict) -> list[str]:
    """Fichiers réellement écrits dans le workspace, hors fichiers de service.

    Sert le cas où la tâche s'arrête sans passer par finish (budget épuisé, annulation) :
    le livrable existe sur disque mais personne ne l'a déclaré. Au run du 19/08/2026,
    l'article LinkedIn était écrit et complet, et l'utilisateur recevait « livrables: [] ».
    """
    from .tools import _FICHIERS_INTERNES

    try:
        noms = sorted(
            n for n in os.listdir(task["workspace"])
            if n not in _FICHIERS_INTERNES and not n.startswith(".")
            and os.path.isfile(os.path.join(task["workspace"], n))
        )
    except OSError:
        return []
    return noms


def _has_sources(task: dict, deliverables: list) -> bool:
    """True si au moins un livrable renvoie à quelque chose que l'agent a réellement consulté.

    Pas « contient une URL » : une doc technique tirée du code source n'en a légitimement
    aucune, et exiger une URL pousserait le modèle à en INVENTER pour passer le garde-fou —
    on aurait fabriqué le mensonge qu'on prétend empêcher.

    On compare donc aux sources effectivement ouvertes pendant la tâche (`sources_seen`,
    alimenté par fetch_url et read_file) : une URL citée, un nom de fichier lu. Lecture
    volontairement grossière : le but est d'attraper le document écrit de mémoire, pas de
    noter la qualité de la bibliographie. Un livrable illisible passe — le doute profite à
    la tâche.
    """
    from .sandbox import SandboxError, resolve

    seen = task.get("sources_seen") or []
    needles = {"http"} | {s for s in seen} | {os.path.basename(s) for s in seen}

    for name in deliverables:
        try:
            path = resolve(task["id"], str(name), write=False)
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (SandboxError, OSError):
            return True
        if any(n and n in content for n in needles):
            return True
    return False


def _truncated_tool_call(raw: str) -> bool:
    """True si un bloc <tool_call> a été ouvert sans être refermé.

    Signature d'une génération coupée par max_tokens au milieu d'un paramètre. Sans cette
    détection, `parse_tool_calls` ne trouve rien (sa regex exige la fermeture), le pas est
    compté comme « aucun outil appelé » et le modèle recommence à l'identique jusqu'à
    épuiser son budget.
    """
    return raw.count("<tool_call>") > raw.count("</tool_call>")


async def _generate(
    messages: list[dict],
    with_tools: bool,
    *,
    max_tokens: int = 0,
    thinking_budget: int = -1,
    tools_override: list | None = None,
) -> str:
    from llm.local import _AGENT_PROMPTS_LOG_PATH, stream_local

    # OBLIGATOIRE dès le second tour. parse_tool_calls rend `arguments` en CHAÎNE JSON
    # (convention OpenAI) alors que le template itère `arguments|items` et exige un dict :
    # sans cette conversion le prompt est corrompu à partir du moment où l'historique
    # contient un appel d'outil, et le modèle n'émet plus rien du tout. Mesuré le
    # 19/08/2026 : 20 pas vides en 5 s. Idempotent — un dict déjà converti est laissé tel
    # quel — et non destructif : la fonction renvoie des copies, `messages` garde sa forme
    # OpenAI pour la persistance.
    messages = normalise_messages_for_template(_context_for_model(messages))

    budget = thinking_budget if thinking_budget >= 0 else AGENT_THINKING_BUDGET

    full = ""
    async for chunk in stream_local(
        messages,
        model=PRIMARY_MODEL,
        max_tokens=max_tokens or AGENT_STEP_MAX_TOKENS,
        no_think=budget == 0,
        thinking_budget=budget,
        tools=(tools_override or TOOL_SCHEMAS) if with_tools else None,
        priority="bg",
        # Journal dédié : un pas d'agent pèse les schémas de dix outils plus tout le
        # contexte accumulé, et noyait prompts.log, qui sert le chat.
        skip_debug_log=False,
        debug_log_path=_AGENT_PROMPTS_LOG_PATH,
    ):
        full += chunk
    # Rendu BRUT, balises comprises : c'est l'appelant qui sépare raisonnement et sortie
    # visible (_split_think), parce que lui seul sait qu'il faut conserver le premier pour
    # le tour suivant.
    return full.strip()


async def _jouer_tour(
    task: dict, messages: list[dict], outils: list | None = None
) -> tuple[str, str, list[dict]]:
    """Un tour de modèle : génère, rattrape une troncature, sépare, parse.

    Rend (raisonnement, texte visible, appels d'outil). Partagé par la boucle principale
    et la phase de conclusion, qui faisaient la même chose à quelques lignes près.
    """
    # `outils` reste None dans le cas courant : _generate retombe alors sur TOOL_SCHEMAS.
    # Passer un « override » égal au défaut brouillerait la lecture — et la trace.
    raw = await _generate(_compact(messages), with_tools=True, tools_override=outils)

    # Bloc d'appel coupé en plein vol : on rejoue le pas avec le budget d'écriture et SANS
    # raisonnement — la réflexion a déjà eu lieu, tout le budget doit aller au contenu. Le
    # cache LRU rend le re-prefill quasi gratuit ; seule la génération est repayée. Une
    # seule reprise : si ça retronque, le document dépasse le budget et il faut le découper.
    if _truncated_tool_call(raw):
        logger.info("agent: %s pas %d tronqué — reprise à %d tokens",
                    task["id"], task["steps"], AGENT_WRITE_MAX_TOKENS)
        append_transcript(task, {"event": "truncated_retry", "step": task["steps"]})
        raw = await _generate(
            _compact(messages), with_tools=True, tools_override=outils,
            max_tokens=AGENT_WRITE_MAX_TOKENS, thinking_budget=0,
        )

    think, visible = _split_think(raw)
    text, calls = parse_tool_calls(visible, outils or TOOL_SCHEMAS)

    # Trace en INFO, pas en debug : c'est la seule mesure qui dise si la réflexion est
    # coupée par ThinkingBudgetProcessor (dont le log, lui, est en debug). think= qui frôle
    # systématiquement AGENT_THINKING_BUDGET × 4 caractères ⇒ budget trop court.
    logger.info(
        "agent: %s pas %d — think=%d car, visible=%d car, outil=%s",
        task["id"], task["steps"], len(think), len(visible),
        calls[0]["function"]["name"] if calls else "aucun",
    )
    return think, text, calls


def _signature(call: dict) -> str:
    """Identité d'un appel d'outil : nom + arguments normalisés."""
    fn = call.get("function") or {}
    args = fn.get("arguments") or "{}"
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return f"{fn.get('name')}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"


def _repetition(recent: list[str], signature: str) -> int:
    """Nombre de fois que `signature` a DÉJÀ été vue dans la fenêtre. L'y enregistre.

    Mute `recent` sur place. Fenêtre glissante et non répétitions consécutives : un
    aller-retour A→B→A remettait un compteur consécutif à zéro à chaque alternance.
    Les appelants lisent : 1 = avertir, 2 ou plus = arrêter.
    """
    vue = recent.count(signature)
    recent.append(signature)
    del recent[:-_LOOP_WINDOW]
    return vue


async def _wait_for_quiet(task: dict) -> None:
    """Retarde le pas tant qu'une conversation est en cours.

    Le lock GPU background ne protège qu'entre deux générations : il ignore le tour de chat
    À VENIR. Sans cette fenêtre, l'agent prend le GPU pendant que l'utilisateur lit, et le
    message suivant attend la fin du pas (jusqu'à ~25 s).
    """
    if AGENT_QUIET_SECONDS <= 0:
        return
    from llm.local import seconds_since_chat

    while True:
        idle = seconds_since_chat()
        if idle >= AGENT_QUIET_SECONDS:
            return
        if store.is_cancelled(task["id"]):
            raise _Cancelled
        await asyncio.sleep(min(AGENT_QUIET_SECONDS - idle, 10.0))


# ── Boucle principale ─────────────────────────────────────────────────────


async def run_task(task: dict) -> dict:
    """Exécute une tâche jusqu'à finish, épuisement du budget, annulation ou échec.

    Retourne l'enregistrement mis à jour. Ne lève pas : tout échec est consigné dans le
    champ `error` et le statut passe à failed — le worker doit enchaîner sur la suivante.
    """
    deadline = time.time() + AGENT_TASK_TIMEOUT_MINUTES * 60
    messages = store.load_messages(task) or _initial_messages(task)
    resumed = len(messages) > 2

    task["status"] = store.STATUS_RUNNING
    task["started_at"] = task.get("started_at") or store.now_iso()
    save_task(task)
    append_transcript(task, {"event": "resumed" if resumed else "start",
                             "objective": task["objective"], "step": task["steps"]})
    logger.info("agent: %s %s — %s", task["id"],
                "reprise" if resumed else "démarrage", task["objective"][:80])

    # Fenêtre glissante des derniers appels, et non compteur de répétitions CONSÉCUTIVES.
    # Un aller-retour A→B→A remettait le compteur à zéro à chaque alternance : mesuré le
    # 19/08/2026, read_file / fetch_url / read_file en boucle sans jamais atteindre le
    # seuil fatal. Compter les occurrences dans la fenêtre attrape les deux formes.
    recent_signatures: list[str] = []

    try:
        while task["steps"] < AGENT_MAX_STEPS:
            if store.is_cancelled(task["id"]):
                raise _Cancelled
            if time.time() > deadline:
                return await _conclure(
                    task, messages,
                    motif=f"délai dépassé ({AGENT_TASK_TIMEOUT_MINUTES} min)",
                )

            await _wait_for_quiet(task)
            task["steps"] += 1
            step = task["steps"]
            # Un seul relevé du disque par pas, partagé par tous les pieds de page du tour.
            rien_ecrit = not _fichiers_produits(task)

            think, text, tool_calls = await _jouer_tour(task, messages)

            # ── Aucun outil appelé : on relance, sans consommer de tour supplémentaire
            #    en silence — le pas est compté, mais le contexte dit pourquoi.
            if not tool_calls:
                # Un tour sans outil qui se répète à l'identique : le modèle est figé.
                #
                # Un compteur séparé traitait à part le cas de la sortie VIDE (prompt
                # cassé). Il faisait doublon : une sortie vide donne la signature
                # « no_tool: », qui se répète tout autant. Deux mécanismes pour un seul
                # phénomène, avec deux seuils et deux messages — supprimé.
                if _repetition(recent_signatures, f"no_tool:{text}") >= 2:
                    motif = ("3 générations vides d'affilée — le modèle ne répond plus"
                             if not text else "figé : 3 tours identiques sans appel d'outil")
                    return await _conclure(task, messages, motif=motif)
                messages.append({"role": "assistant", "content": text or "(vide)",
                                 "_think": think})
                messages.append({
                    "role": "user",
                    "content": get_prompt("AGENT_NO_TOOL_NUDGE") + _step_footer(step, task, rien_ecrit),
                })
                append_transcript(task, {"event": "no_tool", "step": step, "text": text[:500]})
                save_messages(task, messages)
                save_task(task)
                continue

            # Un seul appel d'action par tour : les suivants porteraient sur des résultats
            # que le modèle n'a pas encore vus.
            #
            # UNE exception, `plan`, et elle est de principe : marquer une étape faite ne
            # dépend d'aucun résultat, c'est de l'état, pas une action. L'exiger dans un
            # tour séparé ferait payer un pas de budget par étape — sur un plan en 5
            # étapes, un quart des 20 pas partirait en comptabilité.
            plan_call = next(
                (c for c in tool_calls if c["function"]["name"] == PLAN), None
            )
            action_calls = [c for c in tool_calls if c["function"]["name"] != PLAN]

            def _args_of(c: dict) -> dict:
                try:
                    return json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    return {}

            emitted = [plan_call] if plan_call else []
            if action_calls:
                emitted.append(action_calls[0])
            ignored = len(action_calls) - 1 if action_calls else 0

            messages.append({"role": "assistant", "content": text or None,
                             "tool_calls": emitted, "_think": think})

            # Le plan est appliqué d'abord : c'est l'état sur lequel le tour s'appuie.
            if plan_call:
                plan_result = await execute_tool(task, PLAN, _args_of(plan_call))
                # Le pied porte le compteur de pas, l'alerte « workspace vide » et le
                # plan : l'omettre ici privait le modèle de tout signal précisément sur les
                # tours où il dérive en replanifiant — mesuré le 19/08/2026, 4 plan de
                # suite sans jamais voir qu'il n'avait rien écrit.
                messages.append({
                    "role": "tool", "tool_call_id": plan_call["id"], "name": PLAN,
                    "content": plan_result + _step_footer(step, task, rien_ecrit),
                })
                append_transcript(task, {"event": "plan", "step": step,
                                         "args": _args_of(plan_call)})
                if not action_calls:
                    # Un tour sans action réelle compte dans la fenêtre de boucle : sans
                    # ça, le modèle peut reposer son plan indéfiniment sans jamais
                    # déclencher la détection (mesuré : 4 pas de suite, pas 8 à 11).
                    if _repetition(recent_signatures, f"plan_only:{_signature(plan_call)}") >= 2:
                        return await _conclure(
                            task, messages, motif="figé : plan reposé 3 fois sans action",
                        )
                    save_messages(task, messages)
                    save_task(task)
                    continue

            call = action_calls[0]
            name = call["function"]["name"]
            args = _args_of(call)

            # ── finish : sortie normale
            if name == FINISH:
                return _finir(task, messages, args, text)

            # ── Boucle : même appel, mêmes arguments, dans la fenêtre récente.
            seen = _repetition(recent_signatures, _signature(call))
            if seen >= 2:
                return await _conclure(
                    task, messages,
                    motif=f"boucle détectée : {name} appelé 3 fois avec les mêmes arguments",
                )

            # Deuxième appel STRICTEMENT identique : on n'exécute pas, on le dit. Rejouer
            # rendrait le même résultat, que le modèle a déjà sous les yeux — et sans cet
            # avertissement il découvre le problème en se faisant tuer au troisième
            # (mesuré le 19/08/2026 : read_file rejoué à l'identique, tâche perdue au pas 4
            # alors que le résultat portait « reprends avec offset=318 »).
            if seen == 1:
                result = get_prompt("AGENT_REPEATED_CALL").format(name=name)
                logger.info("agent: %s — %s rejoué à l'identique, avertissement", task["id"], name)
                append_transcript(task, {"event": "repeated_call", "step": step, "name": name})
            else:
                result = await execute_tool(task, name, args)
            if ignored:
                result += f"\n\n[{ignored} autre(s) appel(s) ignoré(s) : un outil par tour.]"

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": result + _step_footer(step, task, rien_ecrit),
            })
            append_transcript(task, {"event": "tool", "step": step, "name": name,
                                     "args": args, "result": result[:1000]})
            save_messages(task, messages)
            save_task(task)

        # ── Budget épuisé : PHASE DE CONCLUSION, outillée.
        # L'ancienne version demandait une synthèse en prose, SANS outils — le modèle ne
        # pouvait donc rien sauvegarder. Deux tâches ont ainsi été perdues le 19/08/2026
        # (article LinkedIn, analyse de logs) : le travail était fait, il ne manquait
        # qu'un tour pour l'écrire. On lui rend ce tour, avec de quoi écrire et conclure.
        return await _conclure(task, messages)

    except _Cancelled:
        append_transcript(task, {"event": "cancelled", "step": task["steps"]})
        return _terminate(task, store.STATUS_CANCELLED, error="annulée",
                          deliverables=_fichiers_produits(task))
    except asyncio.CancelledError:
        # Arrêt du service : on laisse le statut à running pour que requeue_interrupted
        # la reprenne au démarrage suivant, contexte intact.
        save_messages(task, messages)
        save_task(task)
        append_transcript(task, {"event": "interrupted", "step": task["steps"]})
        raise
    except Exception as exc:
        logger.exception("agent: tâche %s en échec", task["id"])
        append_transcript(task, {"event": "error", "error": f"{type(exc).__name__}: {exc}"})
        return _terminate(task, store.STATUS_FAILED, error=f"{type(exc).__name__}: {exc}")


# Tours accordés à la conclusion, en plus du budget. Trois : écrire, compléter, conclure.
_TOURS_CONCLUSION = 3


async def _conclure(task: dict, messages: list[dict], motif: str = "") -> dict:
    """Dernière chance de sauvegarder le travail avant de rendre la main.

    Empruntée par TOUS les arrêts prématurés, pas seulement l'épuisement du budget : une
    boucle détectée, un blocage, un délai dépassé surviennent le plus souvent APRÈS que
    l'essentiel a été trouvé. Terminer sec revenait à jeter ce travail.

    Outils volontairement réduits à write_file et finish : à ce stade il ne s'agit plus
    de chercher mais de rendre. Un jeu d'outils complet relancerait l'exploration —
    c'est précisément ce qui a consommé le budget.
    """
    from .tools import TOOL_SCHEMAS as _ALL

    outils = [t for t in _ALL if t["function"]["name"] in ("write_file", FINISH)]
    messages.append({
        "role": "user",
        "content": get_prompt("AGENT_FINAL_TURN").format(objective=task["objective"]),
    })
    append_transcript(task, {"event": "conclusion", "step": task["steps"]})
    logger.info("agent: %s — budget épuisé, phase de conclusion", task["id"])

    for _ in range(_TOURS_CONCLUSION):
        think, text, calls = await _jouer_tour(task, messages, outils)

        if not calls:
            messages.append({"role": "assistant", "content": text or "(vide)", "_think": think})
            break

        call = calls[0]
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        messages.append({"role": "assistant", "content": text or None,
                         "tool_calls": [call], "_think": think})

        # `tools_override` ne restreint que ce qui est DÉCLARÉ au modèle : parse_tool_calls
        # ne filtre pas, et le modèle rappelle de mémoire les outils vus plus haut dans le
        # contexte. Sans ce refus, la conclusion repartait analyser — mesuré le 19/08/2026,
        # shell et plan exécutés pendant la phase censée être réduite à rendre.
        if name not in ("write_file", FINISH):
            messages.append({
                "role": "tool", "tool_call_id": call["id"], "name": name,
                "content": (
                    f"{name} n'est plus disponible : ton budget est épuisé, il ne reste "
                    "que write_file et finish. Écris ton livrable, puis conclus."
                ),
            })
            append_transcript(task, {"event": "conclusion_refus", "name": name})
            save_messages(task, messages)
            continue

        if name == FINISH:
            summary = (args.get("summary") or text or "").strip()
            deliverables = args.get("deliverables") or _fichiers_produits(task)
            if isinstance(deliverables, str):
                deliverables = [deliverables]
            append_transcript(task, {"event": "finish", "step": task["steps"], "summary": summary})
            save_messages(task, messages)
            return _terminate(task, store.STATUS_DONE, result=summary, error=motif,
                              deliverables=[str(d) for d in deliverables])

        result = await execute_tool(task, name, args)
        messages.append({"role": "tool", "tool_call_id": call["id"], "name": name,
                         "content": result})
        append_transcript(task, {"event": "tool", "step": task["steps"], "name": name,
                                 "args": args, "result": result[:1000]})
        save_messages(task, messages)

    # Ni finish ni rien de plus à écrire : on rend ce qui existe sur disque.
    fichiers = _fichiers_produits(task)
    resume = (
        f"Budget épuisé. Fichiers produits : {', '.join(fichiers)}."
        if fichiers else
        "Budget épuisé sans qu'aucun fichier n'ait pu être produit."
    )
    save_messages(task, messages)
    return _terminate(task, store.STATUS_DONE, result=resume, deliverables=fichiers,
                      error=motif or f"budget de {AGENT_MAX_STEPS} pas épuisé sans finish")


def _finir(task: dict, messages: list[dict], args: dict, text: str, motif: str = "") -> dict:
    """Traite un appel à `finish`. Partagé par la boucle principale et la conclusion.

    `deliverables` retombe sur ce qui existe réellement sur disque : le modèle oublie
    régulièrement de les déclarer, et un fichier produit mais non listé est un fichier
    perdu pour l'utilisateur.
    """
    summary = (args.get("summary") or text or "").strip()
    deliverables = args.get("deliverables") or _fichiers_produits(task)
    if isinstance(deliverables, str):
        deliverables = [deliverables]

    # Absence de sources : SIGNALÉE, jamais bloquante.
    #
    # La version précédente refusait le finish. Bilan sur une journée d'essais : zéro
    # fabrication rattrapée, deux rustines pour éviter les faux positifs (doc de code, base
    # documentaire), et un rapport de 11 ko détruit parce que l'objection a poussé le modèle
    # à réécrire son fichier. Beaucoup de livrables n'ont légitimement rien à citer : un
    # script, un fichier de configuration, une synthèse des données de l'utilisateur.
    #
    # Ce qui corrige réellement les inventions, ce sont les règles de sourçage
    # d'AGENT_SYSTEM. Ici on informe l'humain, il tranche.
    caveat = ""
    if deliverables and not _has_sources(task, deliverables):
        caveat = get_prompt("AGENT_CAVEAT_NO_SOURCE")
        append_transcript(task, {"event": "no_source", "step": task["steps"]})
        logger.info("agent: %s — livrable sans source citée (signalé)", task["id"])

    append_transcript(task, {"event": "finish", "step": task["steps"], "summary": summary})
    save_messages(task, messages)
    return _terminate(task, store.STATUS_DONE, result=summary + caveat, error=motif,
                      deliverables=[str(d) for d in deliverables])


def _terminate(task: dict, status: str, *, result: str = "", error: str = "",
               deliverables: list[str] | None = None) -> dict:
    # Filet de dernier recours : `result` part en notification et en historique iOS. Aucun
    # chemin de sortie ne doit y laisser de raisonnement, quel que soit l'appelant.
    if result and "</think>" in result:
        logger.warning("agent: %s — raisonnement retiré du résultat", task["id"])
        result = _split_think(result)[1]
    task["status"] = status
    task["finished_at"] = store.now_iso()
    task["result"] = result
    task["error"] = error
    if deliverables is not None:
        task["deliverables"] = deliverables
    save_task(task)
    logger.info("agent: %s → %s (%d pas)%s", task["id"], status, task["steps"],
                f" — {error}" if error else "")
    return task
