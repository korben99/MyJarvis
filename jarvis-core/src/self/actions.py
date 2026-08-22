"""Catalogue d'actions et handlers de la réflexion + livraison push + génération de
push proactif.

Chaque _action_* est appelée via _execute_action(action, params) depuis l'engine.
Cœur volontairement groupé : ces handlers, la livraison push et le push proactif sont
mutuellement récursifs (queue_push ↔ ask_user ↔ flag_project_stall ↔ generate_proactive_push).
"""

import asyncio
import json
import time
from datetime import datetime, timezone

import pytz
from config import (
    DEFAULT_TEMP,
    MAX_TOKENS_COMPACT,
    REASONING_API_KEY,
    REASONING_API_URL,
    REASONING_MODEL,
    USER_ADMINS,
    USER_CODES,
    USER_EMAILS,
    USERS,
    llm_timeout,
)
from apns import is_real_apns_token, send_apns_push
from google_services import is_google_available, send_gmail_message
from helpers import (
    call_llm_async_bg,
    call_llm_bg,
    extract_llm_json,
    get_logger,
    get_redis,
    rel_time_fr,
)
import emotional_state
from memory import (
    append_conversation_message,
    get_self_memory,
    get_user_projects,
    save_self_memory,
    self_memory_lock,
)
from prompts import get_prompt
from trade_keys import idx_key, pos_key

from .context import _check_memory_health, _check_service_health, _fmt_memory_health
from .proposals import _action_refine_prompt, _load_proposals
from .state import (
    _DEVICE_TOKEN_PREFIX,
    _KNOWLEDGE_GAPS_KEY,
    _PUSH_COOLDOWN_PREFIX,
    _PUSH_COOLDOWN_TTL,
    slug_de_sujet,
)

logger = get_logger("jarvis-self")

# _NS_GUARDS supprimés le 21/08/2026 avec _action_correct_profile, seule à les utiliser.
# À noter : ils protégeaient les espaces `financial` et `travel` du profil (une valeur
# écrite sous `placement:*` devait contenir un montant, un nom de fonds…). Cette
# protection n'existait QUE sur ce chemin — ni l'analyseur ni la revue nocturne ne l'ont
# jamais eue. Sa disparition ne dégrade donc rien, mais si elle a de la valeur, sa place
# est dans update_user_profile, là où le profil est réellement écrit.

class EchecAction(str):
    """Sortie d'action qui signale un échec. C'est une chaîne, et rien d'autre.

    Remplace l'heuristique de préfixe de l'engine, qui déduisait l'échec de la FORME du
    message : `outcome.startswith(f"{action}:")`. Toutes les sorties de
    `_action_flag_project_stall` commencent par ce préfixe, réussite comprise — donc chaque
    rappel effectivement envoyé était compté comme un échec système, interdisait l'action
    pour le reste du cycle et faisait baisser la confiance.

    Sous-classe de `str` à dessein : tout ce qui journalise, sérialise en JSON, concatène
    ou tronque une sortie d'action continue de fonctionner sans un seul changement. Seul
    l'engine lit le type, par `isinstance`.
    """


# ── Actions-local Redis keys / cooldowns ──────────────────────────────────
_NOTIF_KEY_PREFIX = "jarvis:self:notif"
_NOTIF_TTL = 86400  # 24h — one notification per user per day
_PUSH_PENDING_PREFIX = "jarvis:push:pending"  # list of pending push messages per user
_IOS_SESSION_ID = "iphone-main"  # session used by the iOS app (hardcoded in Swift)


# ══════════════════════════════════════════════════
#  ACTION CATALOG
# ══════════════════════════════════════════════════


def _action_nothing(params: dict) -> str:
    reason = params.get("reason", "no reason given")
    logger.info("Self action: nothing (%s)", reason)
    return f"no-op: {reason}"


# _action_store_insight() supprimée le 21/08/2026 : la revue nocturne est propriétaire
# de la mémoire autobiographique (Call 1, `insights_durables`), et elle décide sur les
# conversations complètes de la journée là où cette action ne voyait qu'un résumé
# d'activité. Il y avait trois écrivains pour la même destination.

_GAP_GENERIC_PHRASES = {
    "lacune de connaissance identifiée dans les capacités d'assistance",
    "lacune identifiée dans les capacités",
    "knowledge gap identified",
}
_GAP_COOLDOWN_TTL = 7 * 86400  # 7 days per topic


def _action_flag_knowledge_gap(params: dict) -> str:
    topic = params.get("topic", "").strip()
    context = params.get("context", "").strip()
    if not topic:
        return EchecAction("flag_knowledge_gap: missing topic")

    # Guard 1 — context must be substantive (not generic filler)
    if len(context) < 30 or context.lower().rstrip(".") in _GAP_GENERIC_PHRASES:
        return (
            f"flag_knowledge_gap: context too generic for '{topic}' — "
            "describe a specific observed failure, not a general statement"
        )

    slug = slug_de_sujet(topic)
    r = get_redis()
    cooldown_key = f"jarvis:self:gap_cooldown:{slug}"

    # Guard 2 — per-topic cooldown (7 days)
    if r.exists(cooldown_key):
        ttl = r.ttl(cooldown_key)
        return EchecAction(f"flag_knowledge_gap: '{topic}' already flagged recently — cooldown active ({ttl // 3600}h remaining)")

    # Guard 3 — block if a proposal already exists for this topic (pending or approved < 30 days)
    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff = now_ts - 30 * 86400
    for p in _load_proposals():
        p_slug = slug_de_sujet(p.get("topic", ""))
        if p_slug != slug:
            continue
        if p.get("status") == "pending":
            return EchecAction(f"flag_knowledge_gap: proposal already pending for '{topic}' — no need to re-flag")
        if p.get("status") == "approved":
            approved_ts = datetime.fromisoformat(
                p.get("approved_at", "2000-01-01T00:00:00+00:00")
            ).timestamp()
            if approved_ts > cutoff:
                return EchecAction(f"flag_knowledge_gap: proposal for '{topic}' approved recently — cooldown active (30 days)")

    r.setex(cooldown_key, _GAP_COOLDOWN_TTL, "1")

    entry = json.dumps(
        {
            "topic": topic,
            "context": context,
            "date": datetime.now(timezone.utc).isoformat(),
        }
    )
    # Plus de compteur d'occurrences. Le seuil « ×3 » qu'il alimentait n'a jamais été lu
    # en code, et il ne pouvait de toute façon pas mesurer la récurrence : le slug est
    # tronqué à 40 caractères sans rapprochement sémantique, si bien que deux formulations
    # du même problème comptaient séparément (mesuré le 21/08/2026 —
    # « gestion_des_notifications_techniques_ave » = 2 et
    # « gestion_des_défaillances_de_notification » = 1, même sujet de fond, jamais 3).
    # refine_prompt est désormais borné en DÉBIT, pas en récurrence : voir
    # _action_refine_prompt.
    pipe = r.pipeline()
    pipe.zadd(_KNOWLEDGE_GAPS_KEY, {entry: time.time()})
    pipe.zremrangebyrank(_KNOWLEDGE_GAPS_KEY, 0, -51)  # keep last 50
    pipe.execute()

    emotional_state.update({"confiance": -0.15})
    logger.info("Self action: knowledge gap flagged — %s", topic)
    return f"flagged knowledge gap: {topic}"


def _action_send_notification(params: dict) -> str:
    user_code = params.get("user_code", "")
    subject = params.get("subject", "").strip()
    message = params.get("message", "").strip()

    if not user_code or not subject or not message or user_code not in USER_CODES:
        return EchecAction("send_notification: invalid params")

    to = USER_EMAILS.get(user_code, "")
    if not to:
        return EchecAction(f"send_notification: no email configured for {user_code}")

    if not is_google_available(user_code):
        return EchecAction("send_notification: Google not configured")

    # One notification per user per day guard (uses user's local timezone)
    r = get_redis()
    user_tz_str = USERS.get(user_code, {}).get("timezone", "Europe/Paris")
    user_tz = pytz.timezone(user_tz_str)
    today = datetime.now(user_tz).strftime("%Y-%m-%d")
    notif_key = f"{_NOTIF_KEY_PREFIX}:{user_code}:{today}"
    if r.exists(notif_key):
        logger.info(
            "Self action: notification suppressed for %s (already sent today)",
            user_code,
        )
        return EchecAction(f"send_notification: suppressed (already sent to {user_code} today)")

    user_name = USER_CODES[user_code]
    html = f"<p>Bonjour {user_name},</p><p>{message}</p><p><em>— Jarvis</em></p>"
    success = send_gmail_message(
        to=to,
        subject=f"Jarvis — {subject}",
        html_body=html,
        text_body=message,
        user_code=user_code,
    )

    if success:
        r.setex(notif_key, _NOTIF_TTL, "1")
        logger.info("Self action: notification sent to %s (%s)", user_code, to)
        return f"notification sent to {user_code}"
    # Guard: mark as attempted today even on failure to avoid retry loops in the chain
    r.setex(notif_key, _NOTIF_TTL, "failed")
    return EchecAction("send_notification: delivery failed")


def _deliver_push(user_code: str, message: str, cooldown_key: str,
                  cooldown_ttl: int) -> str | None:
    """Livre un push iOS : file Redis (fallback polling) + APNs immédiat si token réel +
    injection dans la conversation iOS. Pose le cooldown fourni. Retourne None en cas de
    succès, ou une chaîne d'erreur (device absent, cooldown actif).

    Cœur partagé entre queue_push (conversationnel, par utilisateur) et alert_admin
    (maintenance/sécurité, cooldown distinct). Un seul endroit pour la logique APNs — dont
    le correctif boucle-vs-worker-thread ci-dessous.
    """
    r = get_redis()
    device_token = r.get(f"{_DEVICE_TOKEN_PREFIX}:{user_code}") or ""
    if not device_token:
        return f"no device registered for {user_code}"
    # Cooldown appliqué en code — le LLM de réflexion peut ignorer la contrainte du prompt.
    if r.exists(cooldown_key):
        return f"cooldown active for {user_code}"

    # Toujours filer dans Redis — repli par polling si APNs échoue ou si l'app est au premier plan.
    pending_key = f"{_PUSH_PENDING_PREFIX}:{user_code}"
    r.rpush(pending_key, json.dumps({
        "message": message,
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }))
    r.expire(pending_key, 86400)  # auto-expire si non relevé sous 24h
    r.setex(cooldown_key, cooldown_ttl, "1")

    # Injecte aussi dans la conversation iOS persistante : visible à l'ouverture même si la
    # notification a été manquée.
    append_conversation_message(user_code, _IOS_SESSION_ID, "assistant", message)

    # APNs immédiat si token réel (livraison instantanée, app même tuée).
    if is_real_apns_token(device_token):
        # Cette fonction tourne dans deux contextes : boucle principale (requête) et worker
        # thread (self-reflection via asyncio.to_thread). Dans le second, aucune boucle n'est
        # associée au thread, et asyncio.ensure_future levait RuntimeError — ce qui faisait
        # avorter TOUT le cycle de réflexion.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(send_apns_push(device_token, body=message))
        except RuntimeError:
            asyncio.run(send_apns_push(device_token, body=message))
        logger.info("APNs push scheduled for %s — %s", user_code, message[:80])
    return None


def _action_queue_push(params: dict) -> str:
    """
    Queue an iOS push notification for a user.
    - Always queues to Redis (polled by the app as fallback).
    - If device has a real APNs token, also fires an immediate APNs push.
    """
    user_code = params.get("user_code", "")
    message = params.get("message", "").strip()

    if not user_code or user_code not in USER_CODES:
        return EchecAction("queue_push: invalid user_code")
    if not message:
        return EchecAction("queue_push: empty message")

    err = _deliver_push(user_code, message,
                        f"{_PUSH_COOLDOWN_PREFIX}:{user_code}", _PUSH_COOLDOWN_TTL)
    if err:
        return EchecAction(f"queue_push: {err}")
    logger.info("Self action: push queued for %s — %s", user_code, message[:80])
    return f"push queued for {user_code}: {message[:80]}"


# Canal d'alerte administrateur : cooldown propre, distinct du push conversationnel, pour
# qu'une reco de maintenance/sécurité ne soit ni bloquée par le push du jour ni ne le bloque.
# Vocation à porter plus tard une action LLM automatique (le message deviendra une commande).
_ADMIN_ALERT_COOLDOWN_PREFIX = "jarvis:admin_alert:cooldown"
_ADMIN_ALERT_COOLDOWN_TTL = 86400  # 24 h — au plus une alerte admin par jour


def _action_alert_admin(params: dict) -> str:
    """Pousse une alerte à l'administrateur (maintenance, sécurité, dérive).

    Action GLOBALE (phase 1) : c'est le canal par lequel Jarvis, ayant vu son état
    (<etat_disparition>, <vulnerabilites>), peut enfin recommander une action concrète —
    p.ex. « monter openssl vers 3.5.6 sur qdrant ». Cible l'admin, cooldown 24h dédié.
    """
    message = params.get("message", "").strip()
    if not message:
        return EchecAction("alert_admin: empty message")
    admin = next(iter(USER_ADMINS), "")
    if not admin:
        return EchecAction("alert_admin: no admin configured")
    corps = message if message.startswith(("[", "⚠")) else f"⚠️ Maintenance — {message}"
    # Clé SUFFIXÉE par l'administrateur. Le préfixe nu était passé tel quel, donc la clé
    # posée était littéralement « jarvis:admin_alert:cooldown » : sans effet avec un seul
    # administrateur, mais avec deux la première alerte faisait taire le second.
    err = _deliver_push(admin, corps,
                        f"{_ADMIN_ALERT_COOLDOWN_PREFIX}:{admin}",
                        _ADMIN_ALERT_COOLDOWN_TTL)
    if err:
        return EchecAction(f"alert_admin: {err}")
    logger.info("Self action: admin alert → %s — %s", admin, message[:80])
    return f"admin alerted ({admin}): {message[:80]}"


# _action_correct_profile() supprimée le 21/08/2026 : `curative_profile_cleanup`
# (revue nocturne, Call 4) applique des `updates` par hset autant que des suppressions,
# donc elle corrige aussi les valeurs — avec le profil entier sous les yeux, ce que
# cette action n'avait pas.

def _action_ask_user(params: dict) -> str:
    """
    Queue a short clarification question as an iOS push notification.
    The user answers naturally in the next chat message; the analyzer captures the reply.
    Uses the same push cooldown as queue_push (max 1 per 48h per user).
    """
    user_code = params.get("user_code", "")
    question = params.get("question", "").strip()

    if not user_code or user_code not in USER_CODES:
        return EchecAction("ask_user: invalid user_code")
    if not question:
        return EchecAction("ask_user: empty question")

    return _action_queue_push({"user_code": user_code, "message": question})


# _action_update_self_note() et la liste `self_notes` supprimées le 21/08/2026.
#
# Le bloc <notes_jarvis> qui devait porter ces notes en conversation lisait la clé `text`
# alors que l'écriture posait `note` : masqué par un `if n.get("text")`, il n'a JAMAIS rien
# injecté depuis sa création le 25/04/2026. Les notes n'étaient donc lues que par le cycle
# qui les écrivait — un circuit fermé, dont le contenu observé portait la marque : six
# notes en trois jours sur sa propre inertie, dont une constatant que les précédentes
# étaient « redondantes et stériles ».
#
# La connaissance de soi a désormais un seul foyer, `self_introspection` (neuf axes, revus
# la nuit, injectés en permanence). Si un besoin d'observation opérationnelle se confirme,
# il passera par l'axe `meta_personne` plutôt que par une seconde liste.


# _action_consolidate_memory() supprimée le 21/08/2026 : la consolidation reste appelée
# par la revue nocturne le 1er du mois (`consolidate_memories`). On perd le déclenchement
# à la demande — c'est assumé : comprimer la mémoire n'est pas agir, et cette action
# portait un cooldown de 48 h qui la rendait de toute façon rare.

def alerter_si_anomalie_critique(
    health: dict | None = None, mem_health: dict | None = None
) -> str:
    """Alerte l'admin sur les anomalies critiques. DÉTERMINISTE, sans LLM.

    `health` et `mem_health` sont RÉUTILISÉS s'ils sont fournis. `gather_global_context`
    calcule déjà les deux, et cette fonction les recalculait juste avant : deux sondes de
    santé mémoire par cycle, chacune faisant un scroll Qdrant par utilisateur. Les sondes
    par défaut restent en place pour les appelants qui n'ont pas de contexte sous la main.

    Était l'action `check_health` jusqu'au 21/08/2026. Ce n'en était pas une : la moitié
    du travail (sonder les services et la santé mémoire) est déjà faite par
    `gather_global_context`, et l'autre moitié — cette alerte — ne demandait aucun
    jugement. La laisser dans le catalogue rendait un signal critique dépendant du choix
    du modèle, alors qu'un service injoignable ou des vecteurs non normalisés sont
    anormaux sans discussion.

    Appelée en tête de `run_self_reflection`, à côté de `consolidate_incidents`, pour que
    l'alerte parte même si la chaîne LLM échoue ensuite. Le LLM garde la même information
    dans son contexte et peut toujours en tirer un `alert_admin` argumenté ; les deux
    canaux ne font pas doublon — celui-ci constate, l'autre recommande.

    Cooldown de 4 h propre, pour qu'un service instable ne génère pas un flot de mails.
    """
    health = health if health is not None else _check_service_health()
    issues = [svc for svc, status in health.items() if status != "ok"]
    if issues:
        logger.warning("Self health check: services KO — %s", issues)

    mem_health = mem_health if mem_health is not None else _check_memory_health()
    mem_lines = _fmt_memory_health(mem_health)
    logger.info("Self memory health:\n%s", mem_lines)

    # Alertes critiques → email admin (cooldown 4h pour éviter le spam)
    norm_issues = [
        f"{uc}: {s['norm_anomalies']} vecteurs non-normalisés"
        for uc, s in mem_health.items()
        if s.get("norm_anomalies", 0) > 0
    ]
    critical = [f"service KO: {svc}" for svc in issues] + norm_issues
    if critical:
        r = get_redis()
        if not r.exists(_HEALTH_ALERT_KEY):
            r.setex(_HEALTH_ALERT_KEY, _HEALTH_ALERT_TTL, "1")
            alert_body = "Anomalies détectées :\n" + "\n".join(f"• {c}" for c in critical)
            for admin_code in USER_ADMINS:
                _action_send_notification({
                    "user_code": admin_code,
                    "subject": "Alerte santé système",
                    "message": alert_body,
                })
        else:
            logger.info("Self health alert suppressed (cooldown actif)")

    return f"{len(critical)} anomalie(s) critique(s)" if critical else "aucune anomalie"


def _action_update_trade_threshold(params: dict) -> str:
    user_code = params.get("user_code", "")
    isin = params.get("isin", "").strip().upper()
    th = params.get("threshold_high")
    tl = params.get("threshold_low")

    if not user_code or user_code not in USER_CODES:
        return EchecAction("update_trade_threshold: invalid user_code")
    if not isin:
        return EchecAction("update_trade_threshold: missing isin")
    if th is None and tl is None:
        return EchecAction("update_trade_threshold: at least one of threshold_high / threshold_low is required")

    r = get_redis()
    if not r.sismember(idx_key(user_code), isin):
        return EchecAction(f"update_trade_threshold: ISIN {isin} not in portfolio for {user_code}")

    key = pos_key(user_code, isin)
    mapping = {}
    parts = []

    if th is not None:
        try:
            th = round(float(th), 2)
        except (TypeError, ValueError):
            return EchecAction("update_trade_threshold: threshold_high must be a number")
        mapping["threshold_high"] = str(th)
        parts.append(f"high={th}€")

    if tl is not None:
        try:
            tl = round(float(tl), 2)
        except (TypeError, ValueError):
            return EchecAction("update_trade_threshold: threshold_low must be a number")
        mapping["threshold_low"] = str(tl)
        parts.append(f"low={tl}€")

    r.hset(key, mapping=mapping)
    pos_name = r.hget(key, "name") or isin

    result = f"threshold updated for {pos_name} ({isin}): {', '.join(parts)}"
    logger.info("Self action: %s", result)
    return result


_PRUNE_COOLDOWN_KEY = "jarvis:self:last_prune"
_PRUNE_COOLDOWN_TTL = 86400  # 24h
_CONSOLIDATE_COOLDOWN_PREFIX = "jarvis:self:last_consolidate"
_CONSOLIDATE_COOLDOWN_TTL = 48 * 3600  # 48h
_STALL_COOLDOWN_PREFIX = "jarvis:self:stall"
_STALL_COOLDOWN_TTL = 14 * 86400  # 14j par projet — évite l'effet "relance en boucle"
_HEALTH_ALERT_KEY = "jarvis:self:health_alert"
_HEALTH_ALERT_TTL = 4 * 3600  # 4h — évite le spam en cas de service instable


def _action_prune_self_memory(params: dict) -> str:
    """
    Call the Primary LLM to identify obsolete/redundant entries in `opinions`, then delete
    them from jarvis-self.json.
    Runs synchronously (called via asyncio.to_thread from run_self_reflection).

    Ne porte plus que sur les opinions depuis le 21/08/2026 : `self_notes` a été supprimée,
    et `self_introspection` n'est pas purgeable — neuf axes fixes, révisés par la revue
    nocturne, jamais supprimés. Une purge par index n'aurait aucun sens sur un dict borné
    par construction, et un axe qu'on efface reviendrait vide le lendemain.
    """
    r = get_redis()
    if r.exists(_PRUNE_COOLDOWN_KEY):
        return EchecAction("prune_self_memory: cooldown active (24h)")

    with self_memory_lock:
        data = get_self_memory()

    opinions = data.get("opinions", [])

    if len(opinions) < 2:
        return "prune_self_memory: nothing to prune (fewer than 2 opinions)"

    def _fmt(items: list, text_key: str = "text") -> str:
        """Format a memory list for the LLM prompt.

        Uses the explicit text_key (e.g. 'note', 'opinion', 'text') so the model
        sees clean prose rather than Python dict repr.  Falls back to the first
        non-empty string value it can find before resorting to str(item).
        """
        if not items:
            return "  (vide)"
        lines = []
        for i, item in enumerate(items):
            if isinstance(item, dict):
                text = (
                    item.get(text_key)
                    or item.get("text")
                    or item.get("note")
                    or item.get("opinion")
                    or str(item)
                )
                date = item.get("date") or item.get("created") or ""
                date_str = f" ({date[:10]})" if date else ""  # YYYY-MM-DD only
            else:
                text = str(item)
                date_str = ""
            lines.append(f"  [{i}] {text}{date_str}")
        return "\n".join(lines)

    user_prompt = get_prompt("PRUNE_SELF_MEMORY_USER").format(
        opinions=_fmt(opinions, "opinion"),
    )

    try:
        content = call_llm_bg(
            [
                {"role": "system", "content": get_prompt("PRUNE_SELF_MEMORY_SYSTEM")},
                {"role": "user", "content": user_prompt},
            ],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_COMPACT,
            json_response=True,
            no_think=True,  # classification task — thinking loops indefinitely, never emits JSON
            timeout=llm_timeout(MAX_TOKENS_COMPACT),
        )
    except Exception as exc:
        logger.error(
            "prune_self_memory LLM call failed: %s", type(exc).__name__, exc_info=True
        )
        return EchecAction(f"prune_self_memory: LLM call failed ({type(exc).__name__})")

    try:
        result = extract_llm_json(content)
    except (ValueError, Exception) as exc:
        logger.warning("prune_self_memory: extract_llm_json failed (%s) — raw=%r…", exc, content[:80])
        return EchecAction("prune_self_memory: invalid LLM response")
    if not result or "to_delete" not in result:
        return EchecAction("prune_self_memory: invalid LLM response")

    to_delete = result["to_delete"]
    total_deleted = 0

    with self_memory_lock:
        data = get_self_memory()
        for field in ("opinions",):
            raw_indices = to_delete.get(field, [])
            if not raw_indices:
                continue
            lst = data.get(field, [])
            cap = max(0, int(len(lst) * 0.30))  # never delete more than 30 %
            indices = sorted(
                set(int(i) for i in raw_indices if 0 <= int(i) < len(lst))
            )[:cap]
            for i in reversed(indices):
                lst.pop(i)
            data[field] = lst
            if indices:
                total_deleted += len(indices)
                logger.info(
                    "prune_self_memory: deleted %d from %s: %s",
                    len(indices),
                    field,
                    indices,
                )
        save_self_memory(data)

    r.setex(_PRUNE_COOLDOWN_KEY, _PRUNE_COOLDOWN_TTL, "1")
    return f"prune_self_memory: deleted {total_deleted} entries total"


def _get_active_projects(user_code: str) -> list[dict]:
    """Return in_progress / active projects for a user from Redis."""
    try:
        return [
            p
            for p in get_user_projects(user_code)
            if p.get("status") in ("in_progress", "active")
        ]
    except Exception:
        return []


def _action_flag_project_stall(params: dict) -> str:
    """
    Détecte les projets actifs sans mise à jour depuis > 21j et envoie un
    push de prise de nouvelles. Cooldown 14j par projet pour éviter le harcèlement.

    21j (et non 14) car un projet de fond peut légitimement rester silencieux
    plusieurs semaines sans être "à la traîne" — voir generate_proactive_push
    pour le raisonnement au cas par cas basé sur l'âge réel du projet.
    """
    user_code = params.get("user_code", "")
    if not user_code or user_code not in USER_CODES:
        return EchecAction("flag_project_stall: invalid user_code")

    projects = _get_active_projects(user_code)
    if not projects:
        return "flag_project_stall: aucun projet actif"

    now = time.time()
    r = get_redis()

    # Sans appareil enregistré, aucun rappel ne peut partir : on le dit tout de suite au
    # lieu de scanner les projets pour buter dessus projet par projet. Contrôle d'ÉTAT et
    # non cooldown temporel — il se lève de lui-même à la seconde où un appareil est
    # enregistré, là où un cooldown de 24 h ferait attendre un jour.
    if not r.exists(f"{_DEVICE_TOKEN_PREFIX}:{user_code}"):
        return f"flag_project_stall: aucun appareil enregistré pour {user_code}"

    sent, skipped, injoignables = [], [], []

    for p in projects:
        name = p.get("name", "")
        lu = p.get("last_update", "")
        if not lu:
            continue
        try:
            ts = (
                datetime.strptime(lu[:19], "%Y-%m-%dT%H:%M:%S")
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
        except (ValueError, TypeError):
            continue

        # A task with a passed due date is chased on its date, not on staleness:
        # "revenir vers James le 12" is an appointment, not a stalled project. Naive dates
        # are read as UTC — day granularity, so the drift is irrelevant.
        due = p.get("due_at", "")
        overdue = False
        if due:
            try:
                d = datetime.fromisoformat(due)
                overdue = (
                    d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d
                ).timestamp() <= now
            except (ValueError, TypeError):
                logger.warning(
                    "flag_project_stall: unreadable due_at on '%s' (%r) — ignored", name, due
                )

        days = int((now - ts) / 86400)
        if not overdue and days <= 21:
            continue

        cooldown_key = f"{_STALL_COOLDOWN_PREFIX}:{user_code}:{name.lower()[:30]}"
        if r.exists(cooldown_key):
            skipped.append(name)
            continue

        # Prise de nouvelles générale, pas une demande de statut chiffrée —
        # évite le ton "surveillance" d'un décompte de jours explicite.
        if overdue:
            msg = f"Petit rappel : « {name} », c'était prévu pour le {due[:10]}. Où ça en est ?"
        else:
            msg = f"Ça fait un moment qu'on n'a pas reparlé de « {name} » — il y a du nouveau ?"
        result = _action_queue_push({"user_code": user_code, "message": msg})
        if "push queued" in result:
            r.setex(cooldown_key, _STALL_COOLDOWN_TTL, "1")
            sent.append(f"{name} (échéance {due[:10]})" if overdue else f"{name} ({days}j)")
        else:
            # On NOTE et on continue, au lieu de sortir. Sortir au premier push non
            # livrable annulait toute la passe : les projets suivants n'étaient jamais
            # examinés, aucun cooldown n'était posé, et la passe repartait à l'identique au
            # cycle suivant. Mesuré les 21/08/2026 sur trois cycles d'affilée, à la lettre
            # près : « ZSXEDC — push indisponible : no device registered ». Il n'y a pas
            # d'appareil enregistré et il n'y en aura pas tant que l'utilisateur n'en
            # enregistre pas un ; la tentative se rejouait toutes les cinq heures.
            injoignables.append(name)

    parts = []
    if sent:
        parts.append(f"rappel envoyé pour {', '.join(sent)}")
    if skipped:
        parts.append(f"{len(skipped)} en cooldown")
    if injoignables:
        parts.append(f"{len(injoignables)} non livrable(s) — canal indisponible")
    if not parts:
        return "flag_project_stall: aucun projet en retard (> 21j)"

    message = "flag_project_stall: " + ", ".join(parts)
    # Échec seulement si RIEN n'est parti alors qu'il y avait quelque chose à envoyer.
    return message if sent else EchecAction(message)


# Doit correspondre exactement à _SELF_ACTIONS | _USER_ACTIONS (self/engine.py) : une
# entrée en trop est un handler que plus personne ne peut atteindre, une entrée manquante
# fait retomber la chaîne sur "nothing".
#
# Deux entrées ne sont plus des actions du LLM mais restent ici, appelées directement par
# la revue nocturne : `prune_self_memory` (l'entretien de mémoire a quitté le cycle
# d'action) et `flag_knowledge_gap` (elle exige un échec concret, que seule la nuit voit).
_ACTION_CATALOG = {
    "nothing": _action_nothing,
    "flag_knowledge_gap": _action_flag_knowledge_gap,
    "send_notification": _action_send_notification,
    "queue_push": _action_queue_push,
    "ask_user": _action_ask_user,
    "update_trade_threshold": _action_update_trade_threshold,
    "refine_prompt": _action_refine_prompt,
    "flag_project_stall": _action_flag_project_stall,
    "alert_admin": _action_alert_admin,
    "prune_self_memory": _action_prune_self_memory,  # appelée par la nuit, pas par le LLM
}


def _execute_action(action: str, params: dict) -> str:
    fn = _ACTION_CATALOG.get(action)
    if fn is None:
        logger.warning(
            "Self: unknown action requested — %r (defaulting to nothing)", action
        )
        return EchecAction(f"unknown action: {action}")
    return fn(params or {})


# ══════════════════════════════════════════════════
#  PROACTIVE PUSH GENERATION
# ══════════════════════════════════════════════════


def _iso_to_ts(iso_str: str) -> float | None:
    """Parse a project's first_mentioned/last_update ISO string to a Unix timestamp."""
    try:
        return datetime.fromisoformat(iso_str).timestamp()
    except (ValueError, TypeError):
        return None


def _last_conversation_ts(user_code: str) -> float:
    """Return Unix timestamp of the most recent episodic conversation, or 0."""
    r = get_redis()
    entries = r.zrevrangebyscore(
        f"convlog:{user_code}",
        "+inf",
        "-inf",
        start=0,
        num=1,
        withscores=True,
    )
    return entries[0][1] if entries else 0.0


async def generate_proactive_push(user_code: str) -> str:
    """
    Per-user LLM call: read recent conversations + active projects + mood,
    decide if there is something worth checking on proactively.

    Two trigger paths:
      A) Recent conversation (last 24h) — reactive follow-up on what was discussed.
         Each exchange is timestamped ("il y a 1h", "il y a 2 jours", …) so the
         LLM can judge whether *enough time has actually passed* for the topic
         at hand, instead of following up on e.g. a minor health complaint an
         hour after it was mentioned.
      B) Active project + silence > 96h — proactive check-in on ongoing work
         even when the user hasn't talked to Jarvis recently. Each project is
         annotated with its own age (first_mentioned) and last real update, so
         the LLM can gauge whether the elapsed time is even plausible given
         the project's apparent scope (a few days silence on a months-long
         initiative is not "stalled").

    Guards:
      - Device must be registered (jarvis:device:token:{user_code})
      - Cooldown 48h between pushes per user (jarvis:push:cooldown:{user_code})
      - At least one of: recent conversation OR active project with silence > 96h
    """
    r = get_redis()

    # Guard: device registered?
    if not r.exists(f"{_DEVICE_TOKEN_PREFIX}:{user_code}"):
        return "no device registered"

    # Guard: cooldown active?
    if r.exists(f"{_PUSH_COOLDOWN_PREFIX}:{user_code}"):
        return "cooldown active"

    now = time.time()

    # ── Path A: recent conversations (last 24h), timestamped ────────────
    cutoff = now - 24 * 3600
    entries_raw = r.zrangebyscore(
        f"convlog:{user_code}", cutoff, "+inf", withscores=True
    )

    conv_lines: list[str] = []
    for raw, score in entries_raw[-10:]:
        try:
            e = json.loads(raw)
            user_msg = e.get("user", "")[:150]
            asst_msg = e.get("assistant", "")[:150]
            topics = ", ".join(e.get("topics", []))
            elapsed = rel_time_fr(score)
            if user_msg:
                conv_lines.append(f"[{elapsed}] User: {user_msg}")
            if asst_msg:
                conv_lines.append(f"[{elapsed}] Jarvis: {asst_msg}")
            if topics:
                conv_lines.append(f"Topics: {topics}")
            conv_lines.append("")
        except Exception:
            pass

    # ── Path B: active projects + silence > 96h ──────────────────────────
    active_projects = _get_active_projects(user_code)
    last_ts = _last_conversation_ts(user_code)
    silence_hours = (now - last_ts) / 3600 if last_ts else 999

    project_lines: list[str] = []
    if active_projects and silence_hours > 96:
        for p in active_projects[:5]:
            age_bits = []
            first_ts = _iso_to_ts(p.get("first_mentioned", ""))
            update_ts = _iso_to_ts(p.get("last_update", ""))
            if first_ts:
                age_bits.append(f"mentionné pour la 1ère fois {rel_time_fr(first_ts)}")
            if update_ts:
                age_bits.append(f"dernière mise à jour {rel_time_fr(update_ts)}")
            age_str = f" ({', '.join(age_bits)})" if age_bits else ""
            project_lines.append(f"- {p['name']}{age_str}: {p.get('description', '')[:120]}")

    # Neither path has anything to work with → skip
    if not conv_lines and not project_lines:
        return "no recent conversations and no active projects"

    mood = emotional_state.describe()

    user_name = USER_CODES.get(user_code, user_code)
    conv_text = (
        "\n".join(conv_lines)[:2000] if conv_lines else "(aucune conversation récente)"
    )

    projects_section = ""
    if project_lines:
        projects_section = (
            f"\nProjets actifs de {user_name} (aucun échange avec Jarvis depuis {silence_hours:.0f}h) :\n"
            + "\n".join(project_lines)
            + "\n"
        )

    prompt = get_prompt("PROACTIVE_PUSH_PROMPT").format(
        user_name=user_name,
        conv_text=conv_text,
        projects_section=projects_section,
        mood=mood,
    )

    try:
        content = await call_llm_async_bg(
            [{"role": "user", "content": prompt}],
            model=REASONING_MODEL,
            api_url=REASONING_API_URL,
            api_key=REASONING_API_KEY,
            temperature=DEFAULT_TEMP,
            max_tokens=MAX_TOKENS_COMPACT,
            json_response=True,
            no_think=True,
            timeout=llm_timeout(MAX_TOKENS_COMPACT),
        )
        message = extract_llm_json(content).get("message")
    except Exception as exc:
        logger.warning(
            "generate_proactive_push: LLM call failed for %s: %s",
            user_code,
            type(exc).__name__,
            exc_info=True,
        )
        return "LLM call failed"

    if not message or str(message).strip().lower() == "null":
        return "no proactive message generated"

    outcome = _action_queue_push({"user_code": user_code, "message": message})
    logger.info("generate_proactive_push for %s: %s", user_code, outcome)
    return outcome
