"""Helpers de date/heure en fuseau utilisateur + formatage français.

Tout datetime visible par l'utilisateur passe par le fuseau lu dans users_list.json ;
les horodatages internes (Redis, logs) restent en UTC.
"""

import time
from datetime import date, datetime

import pytz
from config import USERS

from .logging_setup import get_logger

logger = get_logger("jarvis-helpers")

_UTC = pytz.UTC


def get_user_tz(user_code: str) -> pytz.BaseTzInfo:
    """Return the pytz timezone for a user. Defaults to UTC on unknown code or bad name."""
    tz_name = USERS.get(user_code, {}).get("timezone", "UTC")
    try:
        return pytz.timezone(tz_name)
    except pytz.UnknownTimeZoneError:
        logger.warning(
            "Unknown timezone %r for user %s — falling back to UTC", tz_name, user_code
        )
        return _UTC


def now_user(user_code: str) -> datetime:
    """Current datetime in the user's timezone."""
    return datetime.now(get_user_tz(user_code))


def today_user(user_code: str) -> date:
    """Current date in the user's timezone."""
    return now_user(user_code).date()


def build_iso_dt(date_str: str, time_str: str, tz_name: str) -> str:
    """
    Build an ISO 8601 datetime string with timezone offset.
    date_str: "YYYY-MM-DD", time_str: "HH:MM", tz_name: e.g. "Europe/Paris"
    Returns e.g. "2026-03-25T14:00:00+01:00"
    """
    tz = pytz.timezone(tz_name)
    naive = datetime(
        int(date_str[:4]),
        int(date_str[5:7]),
        int(date_str[8:10]),
        int(time_str[:2]),
        int(time_str[3:5]),
    )
    return tz.localize(naive).isoformat()


def fmt_event_time(iso: str, user_code: str, fmt: str = "%d/%m %H:%M") -> str:
    """
    Convert an ISO 8601 datetime string (with or without UTC offset) to the
    user's local timezone and format it.

    All-day events (date-only strings like "2026-03-21") are returned as-is.
    Returns the raw string on any parse error.
    """
    if not iso or len(iso) <= 10:
        return iso  # all-day event — no time component

    try:
        # Python < 3.11 does not accept "Z" as UTC in fromisoformat — normalize first.
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_UTC)
        return dt.astimezone(get_user_tz(user_code)).strftime(fmt)
    except (ValueError, OverflowError):
        return iso


_JOURS_FR = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MOIS_FR = (
    "",
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)


_SEASONS_FR = {
    12: "hiver",
    1: "hiver",
    2: "hiver",
    3: "printemps",
    4: "printemps",
    5: "printemps",
    6: "été",
    7: "été",
    8: "été",
    9: "automne",
    10: "automne",
    11: "automne",
}


def fmt_now_fr(tz_name: str) -> str:
    """Return current datetime + season formatted in French for the given IANA timezone.

    Example: 'lundi 30 mars 2026, 14:32 (printemps)'
    """
    now = datetime.now(pytz.timezone(tz_name))
    jour = _JOURS_FR[now.weekday()]
    saison = _SEASONS_FR[now.month]
    return f"{jour} {now.day} {_MOIS_FR[now.month]} {now.year}, {now.strftime('%H:%M')} ({saison})"


def fmt_date_fr(d: date) -> str:
    """Return a short French date label: 'Dimanche 10 mai'."""
    return f"{_JOURS_FR[d.weekday()].capitalize()} {d.day} {_MOIS_FR[d.month]}"


def rel_time_fr(ts: float) -> str:
    """Return a French relative time string for a Unix timestamp.

    Examples: 'il y a 3 jours', 'il y a 2 semaines', 'il y a 1 mois'
    """
    delta = time.time() - ts
    if delta < 3600:
        m = max(1, int(delta / 60))
        return f"il y a {m} min"
    if delta < 86400:
        h = int(delta / 3600)
        return f"il y a {h}h"
    if delta < 7 * 86400:
        d = int(delta / 86400)
        return f"il y a {d} jour{'s' if d > 1 else ''}"
    if delta < 30 * 86400:
        w = int(delta / (7 * 86400))
        return f"il y a {w} semaine{'s' if w > 1 else ''}"
    if delta < 365 * 86400:
        mo = int(delta / (30 * 86400))
        return f"il y a {mo} mois"
    y = int(delta / (365 * 86400))
    return f"il y a {y} an{'s' if y > 1 else ''}"
