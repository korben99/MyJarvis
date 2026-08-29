"""Renseignement sur les groupes d'attaquants — agrégateurs de sites de fuite.

Le web de surface ne dit presque rien d'utile sur un groupe de ransomware : les sources
primaires sont les sites de fuite en .onion que ces groupes opèrent eux-mêmes. Des
agrégateurs les moissonnent en continu et republient en clearnet. On s'appuie sur eux
plutôt que d'aller sur Tor : lecture seule, pas d'infrastructure, pas d'exposition.

Sources retenues (vérifiées) :

  RansomLook    www.ransomlook.io/api    liste des groupes, posts récents. Libre, sans clé.
  ransomwatch   raw.githubusercontent    métadonnées des groupes (.onion, alias, notes) et
                                         historique des victimes publiées. Libre, sans clé.

Écartée : ransomware.live, passée sous clé API — toutes ses routes rendent désormais du
HTML. À reprendre si une clé est obtenue, sa couverture est la meilleure des trois.

DEUX PIÈGES DE VOLUMÉTRIE, tous deux rencontrés à la mise au point :
  · RansomLook /api/group/<nom> rend jusqu'à 161 Mo pour un groupe actif — jamais appelé.
  · ransomwatch posts.json fait 2,3 Mo : téléchargé une fois, gardé en cache mémoire.
Tout ce qui sort d'ici est de toute façon plafonné par l'appelant.
"""

import time

from deps import HTTP_CLIENT
from helpers import get_logger

logger = get_logger("jarvis-agent")

_RANSOMLOOK = "https://www.ransomlook.io/api"
_RANSOMWATCH = "https://raw.githubusercontent.com/joshhighet/ransomwatch/main"

# Jeux de données entiers, en cache pour tout le process. Ils bougent au rythme des publications
# des groupes (quelques fois par jour) : 6 h est large, et évite de retélécharger 2,3 Mo à
# chaque pas d'agent.
_CACHE_TTL = 6 * 3600
_cache: dict[str, tuple[float, object]] = {}


async def _fetch_json(url: str, cache_key: str = "", timeout: float = 25.0):
    """GET JSON, avec cache mémoire optionnel. Rend None en cas d'échec — jamais d'exception."""
    if cache_key and cache_key in _cache:
        ts, data = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data
    try:
        resp = await HTTP_CLIENT.get(
            url, timeout=timeout, headers={"User-Agent": "jarvis-cti/1.0", "accept": "application/json"}
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("cti: %s indisponible (%s)", url, type(exc).__name__)
        return None
    if cache_key:
        _cache[cache_key] = (time.time(), data)
    return data


def _matches(name: str, needle: str) -> bool:
    """Rapprochement tolérant : « zero bytes », « zerobyte », « ZeroBytes » désignent le même
    groupe selon la source. On compare sans espaces, tirets ni casse."""
    squash = lambda s: "".join(c for c in s.lower() if c.isalnum())
    a, b = squash(name), squash(needle)
    return bool(b) and (b in a or a in b)


async def list_groups(needle: str = "") -> str:
    groups = await _fetch_json(f"{_RANSOMLOOK}/groups", cache_key="rl_groups")
    if not groups:
        return "Source RansomLook indisponible."
    if needle:
        hits = [g for g in groups if _matches(str(g), needle)]
        if not hits:
            return (
                f"Aucun groupe correspondant à « {needle} » dans RansomLook "
                f"({len(groups)} groupes suivis). Le nom est peut-être un alias — "
                f"cherche-le sur le web pour trouver sa dénomination courante."
            )
        return f"Groupes correspondant à « {needle} » : " + ", ".join(str(g) for g in hits)
    return f"{len(groups)} groupes suivis. Extrait : " + ", ".join(str(g) for g in groups[:60])


async def group_profile(needle: str) -> str:
    """Fiche d'un groupe : alias, sites .onion connus, notes de l'agrégateur."""
    data = await _fetch_json(f"{_RANSOMWATCH}/groups.json", cache_key="rw_groups")
    if not data:
        return "Source ransomwatch indisponible."

    hits = [g for g in data if _matches(g.get("name", ""), needle)]
    if not hits:
        return (
            f"Aucune fiche pour « {needle} » chez ransomwatch ({len(data)} entités suivies). "
            f"Vérifie l'orthographe ou l'alias du groupe."
        )

    out = []
    for g in hits[:3]:
        lines = [f"## {g.get('name')}"]
        if g.get("meta"):
            lines.append(f"Note de la source : {g['meta']}")
        locations = g.get("locations") or []
        if locations:
            lines.append(f"Sites connus ({len(locations)}) :")
            for loc in locations[:8]:
                title = loc.get("title") or ""
                available = loc.get("available")
                state = "en ligne" if available else ("hors ligne" if available is not None else "état inconnu")
                lines.append(f"  · {loc.get('fqdn')} — {state}{' — ' + title if title else ''}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


async def group_victims(needle: str, limit: int = 25) -> str:
    """Victimes publiées par un groupe, les plus récentes d'abord."""
    posts = await _fetch_json(f"{_RANSOMWATCH}/posts.json", cache_key="rw_posts", timeout=40.0)
    if not posts:
        return "Source ransomwatch indisponible."

    hits = [p for p in posts if _matches(p.get("group_name", ""), needle)]
    if not hits:
        return f"Aucune victime publiée au nom de « {needle} » dans ransomwatch."

    hits.sort(key=lambda p: p.get("discovered", ""), reverse=True)
    lines = [
        f"· {p.get('discovered', '?')[:10]} — {p.get('post_title', '?')}"
        for p in hits[:limit]
    ]
    total = len(hits)
    header = f"{total} victime(s) publiée(s) au nom de « {needle} »"
    if total > limit:
        header += f" — les {limit} plus récentes"
    return header + " :\n" + "\n".join(lines)


async def cert_fr(needle: str = "", limit: int = 20) -> str:
    """Avis, alertes et actualités du CERT-FR (ANSSI).

    Comble l'angle mort des deux agrégateurs ci-dessus : ils ne voient QUE les sites de
    fuite de ransomware. Un acteur d'extorsion qui publie sur un forum de breach ou sur
    Telegram — ZeroBytes, DGFiP et Éducation nationale en 2026 — leur est complètement
    invisible. Le CERT-FR, lui, est la source de référence sur les cibles françaises, et
    il est public.
    """
    import re
    from email.utils import parsedate_to_datetime
    from xml.etree import ElementTree

    out, seen = [], set()
    for feed in ("alerte", "avis", "actualite"):
        try:
            resp = await HTTP_CLIENT.get(
                f"https://www.cert.ssi.gouv.fr/{feed}/feed/",
                timeout=20.0, headers={"User-Agent": "jarvis-cti/1.0"},
            )
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.content)
        except Exception as exc:
            logger.warning("cti: flux CERT-FR %s indisponible (%s)", feed, type(exc).__name__)
            continue

        for item in root.iterfind(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            raw_date = (item.findtext("pubDate") or "").strip()
            desc = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
            desc = re.sub(r"\s+", " ", desc).strip()
            if needle and not (_matches(title, needle) or needle.lower() in desc.lower()):
                continue
            if link in seen:
                continue
            seen.add(link)
            # Les flux sortent dans l'ordre de publication, plus ANCIEN d'abord, et le flux
            # « alerte » garde des alertes de 2023 toujours ouvertes en tête. Sans tri, une
            # veille remonterait systématiquement de l'obsolète.
            try:
                sort_key = parsedate_to_datetime(raw_date).timestamp()
            except (TypeError, ValueError):
                sort_key = 0.0
            out.append((sort_key, f"· [{feed}] {raw_date[:16]} — {title}\n  {link}\n  {desc[:220]}"))

    out.sort(key=lambda x: x[0], reverse=True)
    out = [line for _, line in out]

    if not out:
        return (
            f"Rien au CERT-FR sur « {needle} » dans les flux courants."
            if needle else "Flux CERT-FR indisponibles."
        )
    header = f"CERT-FR{' — filtré sur « ' + needle + ' »' if needle else ''} :"
    return header + "\n" + "\n".join(out[:limit])


async def recent_activity(limit: int = 30) -> str:
    """Dernières publications, tous groupes confondus — sert à situer l'actualité."""
    posts = await _fetch_json(f"{_RANSOMLOOK}/recent", cache_key="rl_recent", timeout=25.0)
    if not posts:
        return "Source RansomLook indisponible."
    if isinstance(posts, dict):
        posts = list(posts.values())
    lines = [
        f"· {(p.get('discovered') or '?')[:10]} — {p.get('group_name', '?')} — {p.get('post_title', '?')}"
        for p in posts[:limit]
        if isinstance(p, dict)
    ]
    return "Publications récentes (tous groupes) :\n" + "\n".join(lines)
