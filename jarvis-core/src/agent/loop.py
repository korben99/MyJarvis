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
from .tools import FINISH, TOOL_SCHEMAS, execute_tool

logger = get_logger("jarvis-agent")

# Au-delà, on élide les plus vieux résultats d'outil. En caractères, pas en tokens : on ne
# tokenise pas pour ça, l'ordre de grandeur suffit (~4 car/token → ~25 k tokens).
_CONTEXT_SOFT_CAP = 100_000
_ELIDED = "[résultat élidé — trop ancien pour tenir dans le contexte]"


class _Cancelled(Exception):
    """Annulation demandée entre deux pas."""


# ── Contexte ──────────────────────────────────────────────────────────────


def _initial_messages(task: dict) -> list[dict]:
    system = get_prompt("AGENT_SYSTEM").format(
        workspace=task["workspace"],
        max_steps=AGENT_MAX_STEPS,
        write_max_chars=AGENT_WRITE_MAX_CHARS,
    )
    objective = get_prompt("AGENT_OBJECTIVE").format(objective=task["objective"])
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


def _step_footer(step: int) -> str:
    remaining = AGENT_MAX_STEPS - step
    # Deux paliers. À mi-parcours on demande de consolider — attendre les 3 derniers pas
    # pour le signaler ne laisse pas le temps de rédiger un livrable de plusieurs morceaux.
    if remaining <= 3:
        hint = get_prompt("AGENT_HINT_LOW_BUDGET")
    elif remaining <= AGENT_MAX_STEPS // 2:
        hint = get_prompt("AGENT_HINT_HALF_BUDGET")
    else:
        hint = ""
    return get_prompt("AGENT_STEP_FOOTER").format(
        step=step, max_steps=AGENT_MAX_STEPS, hint=hint
    )


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
    """Copies prêtes pour le template, raisonnement du DERNIER tour conservé.

    Sans ça, un tour d'agent ne laisse aucune trace : sa sortie visible se réduit à
    l'appel d'outil, tout le « pourquoi » vit dans le <think> — et le jeter donne un
    contexte fait d'un objectif suivi d'un tas de résultats bruts, sans fil conducteur.
    Mesuré le 19/08/2026 : 32 messages, 17 000 tokens, TOUS les tours assistant à 0
    caractère de contenu, et le modèle qui rejoue six fois la même recherche.

    Un seul tour conservé, pas N : la croissance resterait sinon quadratique, et c'est le
    plan le plus récent qui porte la continuité.
    """
    last = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" and messages[i].get("_think"):
            last = i
            break

    out = []
    for i, message in enumerate(messages):
        clean = {k: v for k, v in message.items() if k != "_think"}
        if i == last:
            think = message["_think"][-_THINK_KEEP_CHARS:].strip()
            body = (clean.get("content") or "").strip()
            clean["content"] = f"[Mon raisonnement au tour précédent]\n{think}\n\n{body}".strip()
        out.append(clean)
    return out


# ── Inférence d'un pas ────────────────────────────────────────────────────


def _has_sources(task: dict, deliverables: list) -> bool:
    """True si au moins un livrable cite une URL.

    Lecture volontairement grossière — on cherche « http », pas une bibliographie bien
    formée. Le but est d'attraper la note entièrement rédigée de mémoire, pas de noter la
    qualité du sourçage. Un livrable illisible passe : le doute profite à la tâche.
    """
    from .sandbox import SandboxError, resolve

    for name in deliverables:
        try:
            path = resolve(task["id"], str(name), write=False)
            with open(path, encoding="utf-8", errors="replace") as f:
                if "http" in f.read():
                    return True
        except (SandboxError, OSError):
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
) -> str:
    from llm.local import stream_local

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
        tools=TOOL_SCHEMAS if with_tools else None,
        priority="bg",
        # Journalisé comme le reste, sous le gate global LLM_DEBUG_PROMPTS : sans ça, le
        # prompt réellement envoyé à chaque pas d'agent est indiagnosticable.
        skip_debug_log=False,
    ):
        full += chunk
    # Rendu BRUT, balises comprises : c'est l'appelant qui sépare raisonnement et sortie
    # visible (_split_think), parce que lui seul sait qu'il faut conserver le premier pour
    # le tour suivant.
    return full.strip()


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

    last_call_signature = ""
    repeat_count = 0
    empty_count = 0
    sources_challenged = False

    try:
        while task["steps"] < AGENT_MAX_STEPS:
            if store.is_cancelled(task["id"]):
                raise _Cancelled
            if time.time() > deadline:
                return _terminate(
                    task, store.STATUS_FAILED,
                    error=f"délai dépassé ({AGENT_TASK_TIMEOUT_MINUTES} min)",
                )

            await _wait_for_quiet(task)
            task["steps"] += 1
            step = task["steps"]

            raw = await _generate(_compact(messages), with_tools=True)

            # Bloc d'appel coupé en plein vol : on rejoue le pas avec le budget d'écriture
            # et SANS raisonnement — la réflexion a déjà eu lieu, tout le budget doit aller
            # au contenu. Le cache LRU rend le re-prefill quasi gratuit ; seule la
            # génération est repayée. Une seule reprise : si ça retronque, c'est que le
            # document dépasse le budget et il faut le découper (write_file append).
            if _truncated_tool_call(raw):
                logger.info("agent: %s pas %d tronqué — reprise à %d tokens",
                            task["id"], task["steps"] + 1, AGENT_WRITE_MAX_TOKENS)
                append_transcript(task, {"event": "truncated_retry", "step": task["steps"] + 1})
                raw = await _generate(
                    _compact(messages), with_tools=True,
                    max_tokens=AGENT_WRITE_MAX_TOKENS, thinking_budget=0,
                )

            think, visible = _split_think(raw)
            text, tool_calls = parse_tool_calls(visible, TOOL_SCHEMAS)

            # Trace en INFO, pas en debug : c'est la seule mesure qui dise si la réflexion
            # est coupée par ThinkingBudgetProcessor (dont le log, lui, est en debug).
            # think= qui frôle systématiquement AGENT_THINKING_BUDGET × 4 caractères ⇒
            # budget trop court pour la tâche.
            logger.info(
                "agent: %s pas %d — think=%d car, visible=%d car, outil=%s",
                task["id"], step, len(think), len(visible),
                tool_calls[0]["function"]["name"] if tool_calls else "aucun",
            )

            # ── Aucun outil appelé : on relance, sans consommer de tour supplémentaire
            #    en silence — le pas est compté, mais le contexte dit pourquoi.
            if not tool_calls:
                # Sortie VIDE et sans outil : le modèle ne produit plus rien. Relancer
                # n'y changera rien — c'est un prompt cassé, pas une hésitation. Sans ce
                # compteur, le budget entier se consume en quelques secondes (mesuré le
                # 19/08/2026 : 20 pas en 5 s) et l'échec réel reste invisible.
                empty_count = empty_count + 1 if not text else 0
                if empty_count >= 3:
                    return _terminate(
                        task, store.STATUS_FAILED,
                        error="3 générations vides d'affilée — le modèle ne répond plus",
                    )
                messages.append({"role": "assistant", "content": text or "(vide)",
                                 "_think": think})
                messages.append({
                    "role": "user",
                    "content": get_prompt("AGENT_NO_TOOL_NUDGE") + _step_footer(step),
                })
                append_transcript(task, {"event": "no_tool", "step": step, "text": text[:500]})
                save_messages(task, messages)
                save_task(task)
                continue

            # Un seul appel par tour : les suivants porteraient sur des résultats que le
            # modèle n'a pas encore vus. On garde le premier et on le dit.
            call = tool_calls[0]
            ignored = len(tool_calls) - 1
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            messages.append({"role": "assistant", "content": text or None,
                             "tool_calls": [call], "_think": think})

            # ── finish : sortie normale
            if name == FINISH:
                summary = (args.get("summary") or text or "").strip()
                deliverables = args.get("deliverables") or []
                if isinstance(deliverables, str):
                    deliverables = [deliverables]

                # Garde-fou mécanique sur les sources. La consigne du prompt ne suffit
                # pas : au run du 19/08/2026 le modèle a cité « Mallory.ai » et
                # « Cyber Daily » sans une seule URL, dans une note de renseignement.
                # UNE seule objection, jamais deux — au-delà, on rendrait la tâche
                # impossible à terminer pour un livrable qui n'a légitimement pas de
                # source (un script, une liste de fichiers).
                if deliverables and not sources_challenged and not _has_sources(task, deliverables):
                    sources_challenged = True
                    messages.append({
                        "role": "tool", "tool_call_id": call["id"], "name": name,
                        "content": get_prompt("AGENT_MISSING_SOURCES") + _step_footer(step),
                    })
                    append_transcript(task, {"event": "missing_sources", "step": step})
                    logger.info("agent: %s — finish refusé, aucune URL dans les livrables",
                                task["id"])
                    save_messages(task, messages)
                    save_task(task)
                    continue

                append_transcript(task, {"event": "finish", "step": step, "summary": summary})
                save_messages(task, messages)
                return _terminate(task, store.STATUS_DONE, result=summary,
                                  deliverables=[str(d) for d in deliverables])

            # ── Boucle serrée : même appel, mêmes arguments, deux fois de suite.
            signature = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            repeat_count = repeat_count + 1 if signature == last_call_signature else 0
            last_call_signature = signature
            if repeat_count >= 2:
                return _terminate(
                    task, store.STATUS_FAILED,
                    error=f"boucle détectée : {name} appelé 3 fois avec les mêmes arguments",
                )

            result = await execute_tool(task, name, args)
            if ignored:
                result += f"\n\n[{ignored} autre(s) appel(s) ignoré(s) : un outil par tour.]"

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": result + _step_footer(step),
            })
            append_transcript(task, {"event": "tool", "step": step, "name": name,
                                     "args": args, "result": result[:1000]})
            save_messages(task, messages)
            save_task(task)

        # ── Budget épuisé sans finish : on récupère une synthèse plutôt qu'un échec sec.
        messages.append({
            "role": "user",
            "content": get_prompt("AGENT_FORCED_SUMMARY").format(objective=task["objective"]),
        })
        summary = await _generate(_compact(messages), with_tools=False)
        append_transcript(task, {"event": "budget_exhausted", "summary": summary[:1000]})
        save_messages(task, messages)
        return _terminate(task, store.STATUS_DONE, result=summary,
                          error=f"budget de {AGENT_MAX_STEPS} pas épuisé sans finish")

    except _Cancelled:
        append_transcript(task, {"event": "cancelled", "step": task["steps"]})
        return _terminate(task, store.STATUS_CANCELLED, error="annulée")
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


def _terminate(task: dict, status: str, *, result: str = "", error: str = "",
               deliverables: list[str] | None = None) -> dict:
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
