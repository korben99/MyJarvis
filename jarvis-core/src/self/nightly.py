"""Revue nocturne (APScheduler 23:00) — LA NUIT APPREND, elle n'agit jamais vers l'extérieur.

Deux temps, depuis le découpage du 21/08/2026 :
  • par utilisateur ayant conversé : faits durables, curation autobio, dédup et narratif
    de profil — 4 appels chacun ;
  • une fois, sur la journée entière : révision des axes d'introspection et des opinions,
    depuis les conversations ET l'état opérationnel. Global, donc hors de la boucle.
Puis l'entretien déterministe (purge des opinions, consolidation mensuelle le 1er).

Module indépendant : ni la réflexion ni les actions ne l'appellent (planifié à part).
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

from config import (
    DEFAULT_TEMP,
    GROWTH_LOG_MAX_ENTRIES,
    INTROSPECTION_AXES,
    INTROSPECTION_LOG_MAX_ENTRIES,
    MAX_TOKENS_COMPACT,
    MAX_TOKENS_THINK_MEDIUM,
    REASONING_API_KEY,
    REASONING_API_URL,
    REASONING_MODEL,
    THINKING_BUDGET_MEDIUM,
    USER_CODES,
    USERS,
    llm_timeout,
)
from helpers import call_llm_async_bg, extract_llm_json, get_logger, get_redis
from llm.local import _NIGHTLY_PROMPTS_LOG_PATH, journal_de_cycle
from memory import (
    archive_autobiographical_event,
    consolidate_memories,
    curative_profile_cleanup,
    get_autobiographical_facts,
    get_self_memory,
    retract_autobiographical_event,
    save_self_memory,
    self_memory_lock,
    store_autobiographical_event,
    update_profile_narrative,
)
from prompts import get_prompt
from vitals import recent_incidents

from .actions import _execute_action
from .context import (
    _check_memory_health,
    _check_service_health,
    _fmt_memory_health,
)
from .state import _DEFAULT_RELATION, _upsert_opinion_inplace, get_user_relation

logger = get_logger("jarvis-self")

_VALID_STYLES = {"direct", "gentle", "formal", "playful"}
_VALID_MOODS = {"warm", "enthusiastic", "measured", "playful", "professional"}


# ══════════════════════════════════════════════════
#  NIGHTLY REVIEW  (replaces nightly-reflection.py)
# ══════════════════════════════════════════════════

# prompts accessed via get_prompt() for live-override support


def _build_conv_text(conversations: list[dict]) -> str:
    """Sort conversations by importance desc and build the conv_text string."""
    sorted_convs = sorted(
        conversations, key=lambda c: c.get("importance", 0), reverse=True
    )
    conv_text = ""
    for c in sorted_convs:
        imp = c.get("importance", 0.0)
        summary = (c.get("memory_summary") or "").strip()
        topics = c.get("topics") or []
        mood = c.get("mood", "?")
        header = f"[importance:{imp:.2f}] [mood:{mood}]"
        if topics:
            header += f" [topics: {', '.join(topics)}]"
        if summary:
            # Analyzer already distilled this exchange — use summary only
            conv_text += f"{header}\n{summary}\n\n"
        else:
            # No summary available — fall back to raw exchange
            conv_text += (
                f"{header}\n"
                f"User: {c.get('user', '')[:350]}\n"
                f"Jarvis: {c.get('assistant', '')[:350]}\n\n"
            )
    return conv_text[:6000] or "(no conversation content)"


async def _nightly_facts_user(
    user_code: str, user_name: str, conversations: list[dict], review_date: str
) -> dict | None:
    """Call 1 — extract durable user facts, relation update, tomorrow suggestions."""
    current_relation = get_user_relation(user_code)
    existing_autobio = await asyncio.to_thread(
        get_autobiographical_facts, user_code, 8, True  # newest first
    )
    existing_autobio_str = (
        "\n".join(f"- {f}" for f in existing_autobio) if existing_autobio else "aucun"
    )
    prompt = get_prompt("NIGHTLY_FACTS_PROMPT").format(
        user_name=user_name,
        user_code=user_code,
        review_date=review_date,
        count=len(conversations),
        conv_text=_build_conv_text(conversations),
        current_relation=json.dumps(current_relation, ensure_ascii=False),
        existing_autobio=existing_autobio_str,
    )
    try:
        content = await call_llm_async_bg(
            [
                {"role": "system", "content": get_prompt("NIGHTLY_FACTS_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_THINK_MEDIUM,
            thinking_budget=THINKING_BUDGET_MEDIUM,
            json_response=True,
            no_think=False,
            timeout=llm_timeout(MAX_TOKENS_THINK_MEDIUM),
        )
        return extract_llm_json(content)
    except Exception as exc:
        logger.error(
            "Nightly facts LLM call failed for %s: %s",
            user_code,
            type(exc).__name__,
            exc_info=True,
        )
        return None


def _normalise_introspection(brut, actuel: dict | None = None) -> dict[str, str]:
    """Les révisions d'axes retenues, filtrées sur les noms d'axes connus.

    Un axe inventé est jeté sans bruit plutôt que créé : la liste est fermée par
    construction (config.INTROSPECTION_AXES), c'est ce qui borne le coût de réinjection et
    empêche la liste de redevenir un tas. Un objet vide — le cas attendu la plupart des
    nuits — ne produit aucune écriture.

    Une réécriture À L'IDENTIQUE est écartée : mesuré le 20/08/2026, le modèle réémet
    parfois un axe au mot près. Sans effet sur le contenu, mais elle gonflerait
    introspection_log et ferait croire à une révision là où il n'y en a pas.
    """
    if not isinstance(brut, dict):
        return {}
    actuel = actuel or {}
    revisions = {}
    for axe, texte in brut.items():
        cle = str(axe).strip().lower()
        if cle not in INTROSPECTION_AXES:
            logger.info("Nightly self: axe inconnu ignoré (%s)", cle[:40])
            continue
        if isinstance(texte, str) and texte.strip():
            propre = texte.strip()
            if propre == (actuel.get(cle) or "").strip():
                continue
            revisions[cle] = propre
    return revisions


def _render_introspection(data: dict) -> str:
    """L'état courant des axes, tel qu'il est montré à la revue nocturne.

    Les axes vides sont montrés VIDES et non masqués : c'est ce qui permet à la nuit de
    les remplir, et de voir qu'ils le sont restés.
    """
    slots = data.get("self_introspection") or {}
    return "\n".join(
        f"{axe} ({question})\n    {slots.get(axe) or '— vide —'}"
        for axe, question in INTROSPECTION_AXES.items()
    )


def _etat_operationnel() -> str:
    """L'état de Jarvis lui-même sur la journée : services, incidents, santé mémoire.

    Sans ce bloc, l'introspection ne se nourrit que des conversations et l'axe
    `meta_personne` ne peut rien apprendre de son propre fonctionnement — un angle mort
    relevé le 21/08/2026, quand `self_notes` (la seule à voir ces signaux) a été retirée.

    Déterministe et tolérant : si une sonde est indisponible, on n'invente rien, la ligne
    disparaît.
    """
    lignes = []
    try:
        services = _check_service_health()
        ko = [s for s, v in services.items() if v != "ok"]
        lignes.append(f"services : {'KO — ' + ', '.join(ko) if ko else 'tous nominaux'}")
        lignes.append("santé mémoire :\n" + _fmt_memory_health(_check_memory_health()))
    except Exception as exc:
        logger.debug("état opérationnel : santé indisponible (%s)", type(exc).__name__)
    try:
        recents = recent_incidents(1)
        if recents:
            lignes.append(
                "incidents des dernières 24 h :\n"
                + "\n".join(
                    f"  - {i.get('kind', '?')} ({i.get('severity', '?')}) : "
                    f"{i.get('detail', '')}"
                    for i in recents[-5:]
                )
            )
    except Exception as exc:
        logger.debug("état opérationnel : vitals indisponible (%s)", type(exc).__name__)
    return "\n".join(lignes) or "aucun signal opérationnel"


async def _nightly_introspection(
    conversations: list[dict], review_date: str
) -> dict | None:
    """Apprendre sur soi — révision des axes d'introspection et des opinions.

    UN SEUL appel par nuit, toutes conversations confondues. Était dans la boucle
    utilisateur jusqu'au 21/08/2026, donc exécuté N fois pour un modèle de soi qui est
    GLOBAL : chaque passage ne voyait qu'un utilisateur et réécrivait par-dessus le
    précédent. Les axes oscillaient entre les points de vue au lieu de se consolider.

    EN MODE RAISONNANT, contrairement aux trois autres appels de la nuit. Ceux-là
    extraient ou trient — le modèle classe et reformate, il n'a rien à intégrer. Celui-ci
    doit rapprocher une journée de conversations et un état opérationnel d'une
    connaissance de soi déjà écrite, puis décider si quelque chose a bougé. C'est
    exactement la tâche où l'A/B du 19/08/2026 (DOCS/AGENTIC.md) a mesuré que `no_think`
    ne fait pas gagner de temps et rend le modèle « confiant et faux » : il rédige depuis
    ses a priori au lieu de ce qu'il vient de lire.

    Ce que ça corrigeait concrètement, le 22/08/2026 : en `no_think`, la révision du
    20/08 avait réécrit `meta_personne` avec une paraphrase à 76 % de l'exemple qui
    figurait alors dans le prompt, sur une journée qui ne parlait ni de santé ni de
    médecin — et avait ainsi effacé une observation juste. Signature d'une génération qui
    puise dans son contexte de consignes faute d'avoir intégré sa matière. Les exemples du
    prompt ont été portés à un par axe dans la même passe : trois seulement étaient
    illustrés, et ce sont les seuls axes qui aient jamais été remplis.
    """
    data = get_self_memory()
    recent_opinions = [
        f"{o['topic']}: {o['opinion']}" for o in data.get("opinions", [])[-10:]
    ]
    prompt = get_prompt("NIGHTLY_SELF_PROMPT").format(
        review_date=review_date,
        count=len(conversations),
        conv_text=_build_conv_text(conversations),
        etat_operationnel=_etat_operationnel(),
        self_introspection=_render_introspection(data),
        recent_opinions=json.dumps(recent_opinions, ensure_ascii=False)
        if recent_opinions
        else "aucune",
    )
    try:
        content = await call_llm_async_bg(
            [
                {"role": "system", "content": get_prompt("NIGHTLY_SELF_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_THINK_MEDIUM,
            thinking_budget=THINKING_BUDGET_MEDIUM,
            json_response=True,
            no_think=False,
            timeout=llm_timeout(MAX_TOKENS_THINK_MEDIUM),
        )
        return extract_llm_json(content)
    except Exception as exc:
        logger.error(
            "Nightly introspection LLM call failed: %s",
            type(exc).__name__,
            exc_info=True,
        )
        return None


async def _nightly_cleaning_user(
    user_code: str, user_name: str, user_insights: list[str], review_date: str
) -> dict | None:
    """Call 3 — memory curator: archive outdated facts, delete strict duplicates."""
    autobio_facts = await asyncio.to_thread(get_autobiographical_facts, user_code, 40)
    if not autobio_facts:
        logger.info("Nightly cleaning skipped for %s — no autobio facts yet", user_code)
        return None

    facts_numbered = "\n".join(
        f"{i + 1}. {text}" for i, text in enumerate(autobio_facts)
    )
    new_insights_str = (
        json.dumps(user_insights, ensure_ascii=False) if user_insights else "aucun"
    )
    prompt = get_prompt("NIGHTLY_CLEANING_PROMPT").format(
        user_name=user_name,
        review_date=review_date,
        facts_count=len(autobio_facts),
        autobio_facts=facts_numbered,
        new_user_insights=new_insights_str,
    )
    try:
        content = await call_llm_async_bg(
            [
                {"role": "system", "content": get_prompt("NIGHTLY_CLEANING_SYSTEM")},
                {"role": "user", "content": prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_COMPACT,
            json_response=True,
            no_think=True,
            timeout=llm_timeout(MAX_TOKENS_COMPACT),
        )
        return extract_llm_json(content)
    except Exception as exc:
        logger.error(
            "Nightly cleaning LLM call failed for %s: %s",
            user_code,
            type(exc).__name__,
            exc_info=True,
        )
        return None


# En deçà de ce TTL restant, le narratif de profil est régénéré. Il en faut un parce que
# l'entretien tourne maintenant pour tout le monde chaque nuit : sans seuil, on repaierait
# un appel LLM par utilisateur et par nuit pour un profil qui n'a pas bougé.
_NARRATIF_SEUIL_REGEN = 2 * 86400


async def _entretien_nocturne(now: datetime, review_date: str) -> None:
    """Entretien de la mémoire — pour TOUS les utilisateurs, qu'ils aient parlé ou non.

    Vivait dans la boucle par utilisateur, après deux `continue` : le verrou d'idempotence,
    puis « aucune conversation hier ». Un utilisateur silencieux le 31 ne recevait donc
    jamais `consolidate_memories` — ni compression épisodique, ni décroissance
    autobiographique. Même sort pour la dédup de profil et le narratif, dont le TTL est de
    7 jours : une semaine de silence faisait disparaître le portrait, aucun chemin ne
    pouvait le régénérer, et `build_memory_context` retombait sans un mot sur le rendu
    clé/valeur brut.

    C'est l'inverse exact de ce qu'il faut. Un utilisateur silencieux est précisément celui
    dont la mémoire a le plus besoin d'être entretenue — c'est chez lui que les faits
    vieillissent et que les projets s'endorment.

    Verrou d'idempotence propre : celui de la boucle d'apprentissage est consommé par
    l'apprentissage, et deux passages dans la même nuit ne doivent pas doubler la purge.
    """
    r = get_redis()
    for user_code in USER_CODES:
        if not r.set(f"jarvis:{user_code}:nightly_maint:{review_date}", "1", nx=True, ex=90000):
            logger.info("Entretien déjà fait pour %s le %s", user_code, review_date)
            continue

        stable_profile = USERS.get(user_code, {}).get("profile", {})

        try:
            await asyncio.to_thread(curative_profile_cleanup, user_code, stable_profile)
        except Exception as exc:
            logger.warning(
                "Entretien : dédup de profil en échec pour %s (%s)",
                user_code, type(exc).__name__,
            )

        try:
            ttl = r.ttl(f"user:{user_code}:profile_narrative")
            # -2 (absent) et -1 (sans TTL) passent tous deux le seuil : dans les deux cas
            # il n'y a pas de narratif utilisable à conserver.
            if ttl < _NARRATIF_SEUIL_REGEN:
                await asyncio.to_thread(
                    update_profile_narrative, user_code, stable_profile
                )
        except Exception as exc:
            logger.warning(
                "Entretien : narratif de profil en échec pour %s (%s)",
                user_code, type(exc).__name__,
            )

        if now.day == 1:
            try:
                await asyncio.to_thread(consolidate_memories, user_code)
                logger.info("Monthly memory consolidation done for %s", user_code)
            except Exception as exc:
                logger.warning(
                    "Monthly consolidation failed for %s: %s",
                    user_code,
                    type(exc).__name__,
                )


async def run_nightly_interaction_review() -> None:
    """
    Nightly per-user conversation review. Called by APScheduler at 23:00.

    LA NUIT APPREND — elle n'agit jamais vers l'extérieur. Découpage arrêté le 21/08/2026 :
    ce qui SORT (push, mail, alerte) appartient au cycle de réflexion, ce qui S'ÉCRIT en
    mémoire appartient ici.

    APPRENDRE SUR L'UTILISATEUR — 4 appels par utilisateur ayant conversé la veille :
      NIGHTLY_FACTS      : faits durables → Qdrant autobio, relation, suggestions du lendemain
      NIGHTLY_CLEANING   : curation autobio (archive le périmé, supprime les erreurs)
      (l'entretien du profil a quitté cette boucle — voir ENTRETIEN plus bas)

    APPRENDRE SUR SOI — 1 appel, APRÈS la boucle, sur la journée entière :
      NIGHTLY_SELF       : révision des 9 axes d'introspection, opinions, et lacunes de
                           connaissance — depuis les conversations ET l'état opérationnel
                           (services, incidents, santé mémoire). Hors boucle parce que ces
                           mémoires sont GLOBALES : les réviser par utilisateur revenait à
                           les réécrire N fois, chaque passage écrasant le précédent.
                           Les lacunes sont ici et pas dans le cycle d'action parce
                           qu'elles exigent un échec CONCRET : seule la nuit voit les
                           conversations.

    ENTRETIEN — pour TOUS les utilisateurs, y compris ceux qui n'ont pas parlé :
      curative_profile_cleanup() : dédup du hash profil Redis (sans LLM si < 5 clés)
      update_profile_narrative() : portrait ~300 tokens, régénéré si son TTL de 7 j approche
      consolidate_memories() le 1er du mois : compression épisodique + décroissance autobio
      prune_self_memory  : opinions obsolètes, global (cooldown 24 h intégré)

    Les trois premiers vivaient dans la boucle ci-dessus, donc derrière « a conversé
    hier » — un silencieux n'était jamais entretenu, alors que c'est chez lui que les
    faits vieillissent. Voir _entretien_nocturne.

    Chaque écriture dans jarvis-self.json se fait sous self_memory_lock immédiatement après
    l'appel — aucune donnée n'est retenue au travers d'un await.
    Idempotent : verrou Redis par utilisateur et par date (TTL 25 h).
    """
    with journal_de_cycle(_NIGHTLY_PROMPTS_LOG_PATH):
        await _revue_nocturne()


async def _revue_nocturne() -> None:
    """Corps de la revue. Séparé pour que TOUT ce qu'elle appelle — y compris la curation
    et le narratif de profil, qui vivent dans memory/ — écrive dans nightly-prompts.log."""
    logger.info("=== Nightly interaction review starting ===")
    r = get_redis()
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    review_date = yesterday.strftime("%Y-%m-%d")
    start_ts = yesterday.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    end_ts = yesterday.replace(
        hour=23, minute=59, second=59, microsecond=999999
    ).timestamp()

    toutes_conversations: list[dict] = []

    for user_code, user_name in USER_CODES.items():
        lock_key = f"jarvis:{user_code}:nightly_review:{review_date}"
        if not r.set(lock_key, "1", nx=True, ex=90000):  # 25h TTL
            logger.info(
                "Nightly review already done for %s on %s — skipping",
                user_code,
                review_date,
            )
            continue

        entries_raw = r.zrangebyscore(f"convlog:{user_code}", start_ts, end_ts)
        if not entries_raw:
            logger.info(
                "No conversations for %s on %s — skipping", user_code, review_date
            )
            continue

        conversations = []
        for raw in entries_raw:
            try:
                conversations.append(json.loads(raw))
            except Exception:
                pass

        logger.info(
            "Nightly review for %s: %d conversations", user_code, len(conversations)
        )

        # ── Call 1: extract user facts ────────────────────────────────────
        facts = await _nightly_facts_user(
            user_code, user_name, conversations, review_date
        )
        user_insights: list[str] = []

        if facts:
            durables = [i for i in facts.get("insights_durables", []) if i]
            evenements = [i for i in facts.get("insights_evenements", []) if i]
            user_insights = durables + evenements  # full context for cleaning

            # Only durable states go to autobio — nightly is the sole autobio writer
            for item in durables:
                if isinstance(item, dict):
                    insight = (item.get("text") or "").strip()
                    importance = round(max(0.5, min(0.9, float(item.get("importance", 0.7)))), 2)
                else:
                    insight = str(item).strip()
                    importance = 0.7
                if insight:
                    store_autobiographical_event(user_code, insight, importance=importance)

            # Store tomorrow's suggestions in Redis (24h)
            suggestions = facts.get("tomorrow_suggestions", [])
            if suggestions:
                r.setex(
                    f"jarvis:{user_code}:tomorrow_suggestions",
                    86400,
                    json.dumps(suggestions),
                )

        # L'introspection et les opinions sont GLOBALES : elles sont révisées une seule
        # fois, après la boucle, sur la journée entière. Voir _nightly_introspection.
        toutes_conversations.extend(conversations)

        # ── Persist facts → jarvis-self.json ─────────────────────────────
        summary = facts.get("daily_summary", "") if facts else ""
        rel_update = facts.get("user_relation_update", {}) if facts else {}

        if summary or rel_update:
            with self_memory_lock:
                data = get_self_memory()
                if summary:
                    data.setdefault("growth_log", []).append(
                        {
                            "date": review_date,
                            "user_code": user_code,
                            "user_name": user_name,
                            "summary": summary,
                            "mood": (facts or {}).get("mood_summary", ""),
                            "conversations": len(conversations),
                        }
                    )
                if rel_update:
                    current = {
                        **_DEFAULT_RELATION,
                        **data.get("user_relations", {}).get(user_code, {}),
                    }
                    new_affinity = rel_update.get("affinity", current["affinity"])
                    try:
                        new_affinity = round(max(0.0, min(1.0, float(new_affinity))), 2)
                    except (TypeError, ValueError):
                        new_affinity = current["affinity"]
                    new_style = rel_update.get(
                        "interaction_style", current["interaction_style"]
                    )
                    if new_style not in _VALID_STYLES:
                        new_style = current["interaction_style"]
                    new_mood = rel_update.get(
                        "average_interaction_mood", current["average_interaction_mood"]
                    )
                    if new_mood not in _VALID_MOODS:
                        new_mood = current["average_interaction_mood"]
                    data.setdefault("user_relations", {})[user_code] = {
                        "affinity": new_affinity,
                        "interaction_style": new_style,
                        "average_interaction_mood": new_mood,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    logger.info(
                        "User relation updated for %s: affinity=%.2f style=%s mood=%s",
                        user_code,
                        new_affinity,
                        new_style,
                        new_mood,
                    )
                data["growth_log"] = data.get("growth_log", [])[
                    -GROWTH_LOG_MAX_ENTRIES:
                ]
                data["last_nightly"] = review_date
                save_self_memory(data)

        # ── Call 3: memory cleaning (Qdrant autobio) ─────────────────────
        cleaning = await _nightly_cleaning_user(
            user_code, user_name, user_insights, review_date
        )
        if cleaning:
            # Hard cap: trust the prompt constraint but enforce it in code too.
            # A runaway LLM should not wipe out more than 3 memories in one night.
            to_archive = cleaning.get("to_archive", [])[:3]
            to_delete = cleaning.get("to_delete", [])[:2]
            for text in to_archive:
                if isinstance(text, str) and text.strip():
                    await asyncio.to_thread(
                        archive_autobiographical_event, user_code, text
                    )
            for text in to_delete:
                if isinstance(text, str) and text.strip():
                    await asyncio.to_thread(
                        retract_autobiographical_event, user_code, text
                    )
            rationale = cleaning.get("rationale", "")
            logger.info(
                "Nightly cleaning for %s — archive:%d delete:%d — %s",
                user_code,
                len(to_archive),
                len(to_delete),
                rationale[:80],
            )

        logger.info("Nightly review done for %s — %s", user_code, summary[:80])

    # L'entretien de la mémoire (dédup de profil, narratif, consolidation mensuelle) ne
    # vit plus dans cette boucle — voir _entretien_nocturne pour le pourquoi.
    await _entretien_nocturne(now, review_date)

    # ── APPRENDRE SUR SOI — un seul appel, sur la journée entière ─────────
    # Hors de la boucle : introspection et opinions sont GLOBALES. Les réviser par
    # utilisateur revenait à les réécrire N fois, chaque passage ne voyant qu'un
    # interlocuteur et écrasant le précédent.
    if toutes_conversations:
        resultat = await _nightly_introspection(toutes_conversations, review_date)
        revisions = _normalise_introspection(
            (resultat or {}).get("self_introspection"),
            get_self_memory().get("self_introspection"),
        )
        opinions = [
            o
            for o in (resultat or {}).get("jarvis_opinions", [])
            if isinstance(o, dict) and o.get("topic") and o.get("opinion")
        ]

        # Lacunes de connaissance — déplacées ici depuis le cycle de réflexion le
        # 21/08/2026. Elles exigent de décrire un échec CONCRET, et seule la nuit a les
        # conversations sous les yeux : la réflexion ne voyait que des compteurs, et
        # produisait donc des lacunes sur son propre comportement de boucle faute d'autre
        # matière observable. On réutilise l'action telle quelle — elle porte déjà les
        # garde-fous qui comptent (contexte substantiel, cooldown de 7 j par sujet, refus
        # si une proposition est en attente sur le même sujet).
        for gap in (resultat or {}).get("knowledge_gaps") or []:
            if not isinstance(gap, dict) or not gap.get("topic"):
                continue
            issue = await asyncio.to_thread(
                _execute_action,
                "flag_knowledge_gap",
                {"topic": gap["topic"], "context": gap.get("context", "")},
            )
            logger.info("Nightly lacune : %s", issue)

        if revisions or opinions:
            with self_memory_lock:
                data = get_self_memory()
                # Un axe est RÉÉCRIT, jamais empilé : une seule ligne par axe, la dernière.
                # C'est ce qui borne le coût de réinjection et évite que la connaissance de
                # soi redevienne la liste sans fin qu'elle était avant le 20/08/2026.
                axes = data.setdefault("self_introspection", {})
                for axe, texte in revisions.items():
                    axes[axe] = texte
                    data.setdefault("introspection_log", []).append(
                        {"axe": axe, "text": texte, "date": review_date}
                    )
                    logger.info("Nightly introspection: %s → %s", axe, texte[:70])
                for op in opinions:
                    _upsert_opinion_inplace(
                        data, op["topic"], op["opinion"].strip(), review_date
                    )
                    logger.info(
                        "Nightly opinion: %s → %s", op["topic"], op["opinion"][:60]
                    )
                # Les axes sont bornés par construction (9) ; seule leur trace historique
                # est tronquée — elle sert à relire comment un axe a évolué.
                data["introspection_log"] = data.get("introspection_log", [])[
                    -INTROSPECTION_LOG_MAX_ENTRIES:
                ]
                save_self_memory(data)
        else:
            logger.info("Nightly introspection : aucune révision (réponse attendue)")

    # ── Entretien de la mémoire de soi ────────────────────────────────────
    # Hors boucle utilisateur : les opinions sont globales, pas per-user. Déplacé ici
    # depuis le catalogue d'actions le 21/08/2026 — purger n'est pas agir, et le cycle de
    # réflexion ne fait plus qu'agir. Le cooldown de 24 h vit dans l'action elle-même,
    # donc un appel par nuit ne peut pas emballer la purge.
    try:
        resultat = await asyncio.to_thread(_execute_action, "prune_self_memory", {})
        logger.info("Nightly: entretien des opinions — %s", resultat)
    except Exception as exc:
        logger.warning("Nightly: purge des opinions échouée (%s)", type(exc).__name__)

    logger.info("=== Nightly interaction review complete ===")
