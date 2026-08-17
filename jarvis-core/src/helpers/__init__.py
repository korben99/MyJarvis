"""
Jarvis — Common Helpers
=======================
Utilitaires partagés dans tout le projet.

Ce paquet remplace l'ancien monofichier `helpers.py`. L'API publique est **identique** :
tous les `from helpers import …` existants continuent de fonctionner sans changement,
car ce `__init__` ré-exporte les noms depuis les sous-modules thématiques :

    logging_setup  configuration du logging + get_logger (racine du paquet)
    timefmt        fuseaux utilisateur + formatage date/heure français
    text           normalisation de clés + score de recouvrement de mots-clés
    store          singletons Redis/Qdrant + helpers JSON/session
    llm_json       extraction robuste de JSON LLM + filtrage <think>
    llm_http       clients HTTP LLM partagés + call_llm(_bg/_async/_async_bg)
    weather        codes météo WMO → français

Public API
----------
Timezone
  get_user_tz(user_code)            -> pytz.BaseTzInfo
  now_user(user_code)               -> datetime (tz-aware, user local)
  today_user(user_code)             -> date (user local)
  fmt_event_time(iso, user_code)    -> str  "DD/MM HH:MM" or "YYYY-MM-DD" for all-day

Connections
  get_redis()                       -> redis.Redis
  get_qdrant()                      -> QdrantClient
  redis_get_json(key, default)      -> any
  redis_set_json(key, data, ttl)    -> None

LLM calls
  call_llm / call_llm_bg / call_llm_async / call_llm_async_bg
  All share a persistent, connection-pooled httpx client. API keys are never logged.

LLM parsing
  extract_llm_json(raw)             -> dict  (raises ValueError on failure)
  filter_think_chunk(chunk, in_think) -> (visible_text, think_fragment, new_in_think)

Weather
  WEATHER_CODES                     -> dict[int, str]
"""

from .llm_http import (
    call_llm,
    call_llm_async,
    call_llm_async_bg,
    call_llm_bg,
)
from .llm_json import extract_llm_json, filter_think_chunk
from .logging_setup import get_logger, setup_logging
from .store import (
    get_qdrant,
    get_redis,
    get_session_summary_data,
    get_sticky_rag,
    redis_get_json,
    redis_set_json,
    set_session_summary_data,
    set_sticky_rag,
)
from .text import _FR_STOPWORDS, keyword_overlap_score, normalize_key
from .timefmt import (
    build_iso_dt,
    fmt_date_fr,
    fmt_event_time,
    fmt_now_fr,
    get_user_tz,
    now_user,
    rel_time_fr,
    today_user,
)
from .weather import WEATHER_CODES

__all__ = [
    # logging
    "setup_logging", "get_logger",
    # timefmt
    "get_user_tz", "now_user", "today_user", "build_iso_dt", "fmt_event_time",
    "fmt_now_fr", "fmt_date_fr", "rel_time_fr",
    # text
    "normalize_key", "keyword_overlap_score", "_FR_STOPWORDS",
    # store
    "get_redis", "get_qdrant", "get_session_summary_data", "set_session_summary_data",
    "get_sticky_rag", "set_sticky_rag", "redis_get_json", "redis_set_json",
    # llm_json
    "filter_think_chunk", "extract_llm_json",
    # llm_http
    "call_llm", "call_llm_bg", "call_llm_async", "call_llm_async_bg",
    # weather
    "WEATHER_CODES",
]
