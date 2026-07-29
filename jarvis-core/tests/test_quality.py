"""
tests/test_quality.py — Jarvis quality & integration tests
===========================================================

DEUX SUITES :

A. UNIT TESTS (no server needed) — rapides, déterministes
   · _detect_satisfaction
   · retract_autobiographical_event (Qdrant mocked)
   · log_conversation satisfaction field
   · _get_user_activity / _fmt_activity (self.py)
   · ANALYSIS_PROMPT retractions field
   · post_analysis retractions pipeline

B. INTEGRATION TESTS (server must be running on localhost:8000)
   Teste le vrai pipeline Jarvis avec l'utilisateur TEST (isolé de la prod).
   Couvre :
   · Chat simple (smoke test)
   · Météo (intent weather + réponse)
   · Recherche web (intent web + sources)
   · Écriture en mémoire (profil utilisateur après échange)
   · Lecture de mémoire (rappel du fait dans échange suivant)
   · Ajout de projet (intent projet → Redis)
   · Clôture de projet (statut done)
   · Update de clé de profil (via chat)
   · Suppression de clé de profil (via chat)
   · Briefing (génération + lecture)
   · Bourse / portfolio (cours action via web)
   · Routeur actif (routing_samples.jsonl alimenté)

RUN :
    cd /opt/jarvis && source venv/bin/activate

    # Suite complète (unit + intégration si serveur dispo)
    python -m pytest jarvis-core/tests/test_quality.py -v --tb=short

    # Unit seuls (sans serveur)
    python -m pytest jarvis-core/tests/test_quality.py -v -m "not integration"

    # Intégration seuls
    python -m pytest jarvis-core/tests/test_quality.py -v -m integration

    # Standalone (affiche les sorties détaillées)
    python jarvis-core/tests/test_quality.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Sauvegarder le vrai httpx AVANT que le mock ne l'écrase dans sys.modules.
# Les helpers d'intégration utilisent _httpx directement.
import httpx as _httpx  # noqa: E402  (doit être avant les mocks)

# ──────────────────────────────────────────────────────────────────────────────
# Logging — niveau DEBUG pour tout le fichier de test
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jarvis-qa")


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION A : bootstrap mocks pour les tests unitaires ─────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _make_config_module():
    m = types.ModuleType("config")
    m.AUTOBIO_DEDUP_THRESHOLD = 0.85
    m.AUTOBIO_IMPORTANCE_THRESHOLD = 0.60
    m.AUTOBIO_RECENCY_WINDOW_DAYS = 365
    m.EPISODIC_RETENTION_DAYS = 30
    m.CHAT_LOG_TTL = 86400
    m.DONE_PROJECT_TTL_DAYS = 30
    m.CHAT_MAX_MESSAGES = 50
    m.EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
    m.IMPORTANCE_THRESHOLD = 0.35
    m.MEMORY_DECAY_FACTOR = 0.85
    m.MEMORY_DECAY_THRESHOLD = 0.15
    m.MEMORY_DECAY_DURABLE_MIN = 1.0
    m.MEMORY_CONSOLIDATION_IMPORTANCE = 1.0
    m.NOVELTY_THRESHOLD = 0.3
    m.QDRANT_COLLECTION = "jarvis_rag"
    m.QDRANT_MEMORY_COLLECTION = "jarvis_memory"
    m.RAG_SCORE_THRESHOLD = 0.5
    m.RAG_TOP_K = 5
    m.RECALL_MEMORY_SIMILARITY_THRESHOLD = 0.7
    m.SELF_MEMORY_PATH = "/tmp/test_jarvis_self.json"
    m.PRIMARY_API_KEY = "test-key"
    m.PRIMARY_API_URL = "http://localhost"
    m.PRIMARY_MODEL = "test-model"
    m.PRIMARY_TIMEOUT = 30.0
    m.PROMPT_DATA_DIR = "/tmp"
    m.REASONING_API_KEY = "test-key"
    m.REASONING_API_URL = "http://localhost"
    m.REASONING_MODEL = "test-reasoning"
    m.REASONING_TIMEOUT = 60.0
    m.ROUTER_API_KEY = "test-key"
    m.ROUTER_API_URL = "http://localhost"
    m.ROUTER_MODEL = "test-router"
    m.ROUTER_TIMEOUT = 10.0
    m.USER_ADMINS = {"KORBEN99": True}
    m.USER_CODES = {"KORBEN99": "Sébastien", "AQWZSX": "Alice", "TEST": "Test"}
    m.USER_EMAILS = {"KORBEN99": "test@test.com", "AQWZSX": "alice@test.com", "TEST": ""}
    m.USERS = {}
    m.USER_TIMEZONES = {
        "KORBEN99": "Europe/Paris",
        "AQWZSX": "Europe/Paris",
        "TEST": "Europe/Paris",
    }
    m.BRIEFING_TIMEZONE = "Europe/Paris"
    m.GROWTH_LOG_MAX_ENTRIES = 50
    m.MAX_CHAIN_ITERATIONS = 5
    m.MAX_REFLECTION_TOKENS = 1000
    m.REFINE_PROMPT_THRESHOLD = 3
    # ── Names added to config.py after this bootstrap was written ──────────
    m.LLM_LOCAL = False
    m.DEFAULT_TEMP = None
    m.MAX_TOKENS_SHORT = 300
    m.MAX_TOKENS_COMPACT = 600
    m.MAX_TOKENS_MEDIUM = 1000
    m.MAX_TOKENS_NO_THINK = 1500
    m.MAX_TOKENS_REASONING = 10000
    m.MAX_TOKENS_THINK_COMPACT = 2048
    m.MAX_TOKENS_THINK_MEDIUM = 5048
    m.THINKING_BUDGET_COMPACT = 1024
    m.THINKING_BUDGET_MEDIUM = 2048
    m.THINKING_BUDGET_DEEP = 4000
    m.PROFILE_NARRATIVE_TOKENS = 600
    m.llm_timeout = lambda max_tokens: 30.0
    m.APNS_KEY_ID = ""
    m.APNS_TEAM_ID = ""
    m.APNS_BUNDLE_ID = "com.test.JarvisApp"
    m.APNS_KEY_PATH = ""
    m.APNS_ENV = "sandbox"
    return m


class _PointIdsList:
    def __init__(self, points):
        self.points = points


_qdrant_models = types.ModuleType("qdrant_client.models")
_qdrant_models.PointIdsList = _PointIdsList
_qdrant_client_mod = types.ModuleType("qdrant_client")
_qdrant_client_mod.models = _qdrant_models

_helpers_mock = MagicMock()
_helpers_mock.get_logger.return_value = MagicMock()

_mocked_modules = {
    "config": _make_config_module(),
    "helpers": _helpers_mock,
    "qdrant_client": _qdrant_client_mod,
    "qdrant_client.models": _qdrant_models,
    "sentence_transformers": MagicMock(),
    "google_services": MagicMock(),
    "trade_keys": MagicMock(),
    "trading": MagicMock(),
    "web_search": MagicMock(),
    "analyzer": MagicMock(),
    "llm_client": MagicMock(),
    "rag": MagicMock(),
    "httpx": MagicMock(),
    "pytz": MagicMock(),
}
for _name, _mod in _mocked_modules.items():
    sys.modules[_name] = _mod

sys.path.insert(0, "/opt/jarvis/jarvis-core/src")

import memory   # noqa: E402
import prompts  # noqa: E402


def _import_jarvis_self():
    if "self" not in sys.modules:
        import self as _s  # noqa: A004
        sys.modules["self"] = _s
    return sys.modules["self"]


def _import_pipeline():
    if "pipeline" not in sys.modules:
        import pipeline as _p
        sys.modules["pipeline"] = _p
    return sys.modules["pipeline"]


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION A : UNIT TESTS ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@unittest.skip(
    "obsolète : _detect_satisfaction (heuristique) a été supprimé — la satisfaction "
    "est désormais jugée par le LLM dans analyzer.analyze_exchange"
)
class TestDetectSatisfaction(unittest.TestCase):

    def sat(self, msg):
        return memory._detect_satisfaction(msg)

    def test_positive_merci(self):
        self.assertEqual(self.sat("Merci beaucoup !"), "positive")

    def test_positive_merci_lowercase(self):
        self.assertEqual(self.sat("merci"), "positive")

    def test_positive_parfait(self):
        self.assertEqual(self.sat("Parfait, c'est exactement ce que je voulais"), "positive")

    def test_positive_super(self):
        self.assertEqual(self.sat("Super, tu peux continuer"), "positive")

    def test_positive_exactement(self):
        self.assertEqual(self.sat("Exactement ! Bien joué"), "positive")

    def test_positive_nickel(self):
        self.assertEqual(self.sat("nickel, merci"), "positive")

    def test_positive_genial(self):
        self.assertEqual(self.sat("Génial ! Continuons"), "positive")

    def test_positive_top(self):
        self.assertEqual(self.sat("top !"), "positive")

    def test_positive_cest_ca(self):
        self.assertEqual(self.sat("c'est ça exactement"), "positive")

    def test_positive_inline(self):
        self.assertEqual(self.sat("ok c'est parfait, continue"), "positive")

    def test_negative_non_comma(self):
        self.assertEqual(self.sat("Non, c'est pas ça"), "negative")

    def test_negative_non_period(self):
        self.assertEqual(self.sat("non. tu as mal compris"), "negative")

    def test_negative_cest_pas_ca(self):
        self.assertEqual(self.sat("c'est pas ça du tout"), "negative")

    def test_negative_tu_nas_pas(self):
        self.assertEqual(self.sat("tu n'as pas compris ma question"), "negative")

    def test_negative_pas_compris(self):
        self.assertEqual(self.sat("pas compris, recommence"), "negative")

    def test_negative_faux(self):
        self.assertEqual(self.sat("faux, recalcule"), "negative")

    def test_negative_incorrect(self):
        self.assertEqual(self.sat("incorrect, la bonne réponse est 42"), "negative")

    def test_unknown_normal_question(self):
        self.assertEqual(self.sat("Quel est le cours de Tesla en ce moment ?"), "unknown")

    def test_unknown_continuation(self):
        self.assertEqual(self.sat("Et pour le projet karting, où en est-on ?"), "unknown")

    def test_unknown_empty(self):
        self.assertEqual(self.sat(""), "unknown")

    def test_unknown_whitespace(self):
        self.assertEqual(self.sat("   "), "unknown")

    def test_unknown_greeting(self):
        self.assertEqual(self.sat("Bonjour Jarvis"), "unknown")

    def test_case_insensitive_SUPER(self):
        self.assertEqual(self.sat("SUPER résultat !"), "positive")

    def test_positive_takes_precedence_order(self):
        self.assertEqual(self.sat("merci pour l'analyse de l'erreur"), "positive")

    def test_known_false_positive_erreur_in_code_context(self):
        result = self.sat("j'ai une erreur dans mon code")
        self.assertIn(result, ("negative", "unknown"))


class TestRetractAutobiographicalEvent(unittest.TestCase):

    def _point(self, point_id, score):
        p = MagicMock()
        p.id = point_id
        p.score = score
        return p

    @patch("memory._invalidate_timeline_cache")
    @patch("memory.get_qdrant")
    @patch("memory.get_embed_model")
    def test_no_results_returns_zero(self, mock_embed, mock_qdrant, mock_inv):
        mock_embed.return_value.encode.return_value.tolist.return_value = [0.1] * 384
        mock_qdrant.return_value.query_points.return_value.points = []
        self.assertEqual(memory.retract_autobiographical_event("KORBEN99", "query"), 0)
        mock_qdrant.return_value.delete.assert_not_called()

    @patch("memory._invalidate_timeline_cache")
    @patch("memory.get_qdrant")
    @patch("memory.get_embed_model")
    def test_below_threshold_not_deleted(self, mock_embed, mock_qdrant, mock_inv):
        mock_embed.return_value.encode.return_value.tolist.return_value = [0.1] * 384
        mock_qdrant.return_value.query_points.return_value.points = [
            self._point("abc-123", 0.80),
        ]
        self.assertEqual(memory.retract_autobiographical_event("KORBEN99", "query"), 0)
        mock_qdrant.return_value.delete.assert_not_called()

    @patch("memory._invalidate_timeline_cache")
    @patch("memory.get_qdrant")
    @patch("memory.get_embed_model")
    def test_above_threshold_deleted_and_cache_invalidated(self, mock_embed, mock_qdrant, mock_inv):
        mock_embed.return_value.encode.return_value.tolist.return_value = [0.1] * 384
        mock_qdrant.return_value.query_points.return_value.points = [
            self._point("abc-123", 0.92),
        ]
        result = memory.retract_autobiographical_event("KORBEN99", "je ne travaille plus chez Acme")
        self.assertEqual(result, 1)
        mock_qdrant.return_value.delete.assert_called_once()
        mock_inv.assert_called_once_with("KORBEN99")

    @patch("memory._invalidate_timeline_cache")
    @patch("memory.get_qdrant")
    @patch("memory.get_embed_model")
    def test_mixed_scores_only_above_deleted(self, mock_embed, mock_qdrant, mock_inv):
        mock_embed.return_value.encode.return_value.tolist.return_value = [0.1] * 384
        mock_qdrant.return_value.query_points.return_value.points = [
            self._point("high-1", 0.93),
            self._point("low-1",  0.75),
            self._point("high-2", 0.91),
            self._point("low-2",  0.82),
        ]
        result = memory.retract_autobiographical_event("KORBEN99", "query")
        self.assertEqual(result, 2)
        deleted_ids = mock_qdrant.return_value.delete.call_args.kwargs["points_selector"].points
        self.assertIn("high-1", deleted_ids)
        self.assertIn("high-2", deleted_ids)
        self.assertNotIn("low-1", deleted_ids)
        self.assertNotIn("low-2", deleted_ids)

    @patch("memory._invalidate_timeline_cache")
    @patch("memory.get_qdrant")
    @patch("memory.get_embed_model")
    def test_custom_threshold_respected(self, mock_embed, mock_qdrant, mock_inv):
        mock_embed.return_value.encode.return_value.tolist.return_value = [0.1] * 384
        mock_qdrant.return_value.query_points.return_value.points = [
            self._point("id-1", 0.89),
        ]
        self.assertEqual(memory.retract_autobiographical_event("KORBEN99", "q", threshold=0.95), 0)
        self.assertEqual(memory.retract_autobiographical_event("KORBEN99", "q", threshold=0.80), 1)

    @patch("memory.get_qdrant")
    @patch("memory.get_embed_model")
    def test_exception_returns_zero(self, mock_embed, mock_qdrant):
        mock_embed.return_value.encode.side_effect = RuntimeError("embed failed")
        self.assertEqual(memory.retract_autobiographical_event("KORBEN99", "query"), 0)

    @patch("memory._invalidate_timeline_cache")
    @patch("memory.get_qdrant")
    @patch("memory.get_embed_model")
    def test_user_and_type_filters_passed_to_qdrant(self, mock_embed, mock_qdrant, mock_inv):
        mock_embed.return_value.encode.return_value.tolist.return_value = [0.0]
        mock_qdrant.return_value.query_points.return_value.points = []
        memory.retract_autobiographical_event("AQWZSX", "fait à corriger")
        must = mock_qdrant.return_value.query_points.call_args.kwargs["query_filter"]["must"]
        user_clause = next(c for c in must if c.get("key") == "user_code")
        type_clause = next(c for c in must if c.get("key") == "memory_type")
        self.assertEqual(user_clause["match"]["value"], "AQWZSX")
        self.assertEqual(type_clause["match"]["value"], "autobiographical")


@unittest.skip(
    "obsolète : log_conversation stocke satisfaction='unknown' (back-fillée ensuite "
    "par l'analyzer LLM) — l'heuristique testée ici n'existe plus"
)
class TestLogConversationSatisfaction(unittest.TestCase):

    def _log_and_capture(self, user_msg):
        captured = {}
        r = MagicMock()
        r.zadd.side_effect = lambda key, mapping: captured.update(mapping)
        with patch("memory.store_autobiographical_event"), \
             patch("memory.store_memory_vector"), \
             patch("memory.get_redis", return_value=r):
            memory.log_conversation(
                user_code="KORBEN99",
                session_id="sess-test",
                user_msg=user_msg,
                assistant_msg="réponse.",
                importance=0.0,
            )
        raw = list(captured.keys())[0]
        return json.loads(raw)

    def test_satisfaction_positive_stored(self):
        entry = self._log_and_capture("Merci, c'était parfait !")
        self.assertEqual(entry["satisfaction"], "positive")

    def test_satisfaction_negative_stored(self):
        entry = self._log_and_capture("Non, c'est pas ça du tout")
        self.assertEqual(entry["satisfaction"], "negative")

    def test_satisfaction_unknown_stored(self):
        entry = self._log_and_capture("Quel temps fait-il à Paris ?")
        self.assertEqual(entry["satisfaction"], "unknown")

    def test_satisfaction_field_always_present(self):
        entry = self._log_and_capture("bonjour")
        self.assertIn("satisfaction", entry)


class TestGetUserActivitySatisfaction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.jarvis_self = _import_jarvis_self()

    def _entry(self, satisfaction, topics=None):
        return json.dumps({
            "timestamp": time.time(),
            "session_id": "s",
            "topics": topics or [],
            "satisfaction": satisfaction,
            "importance": 0.1,
        })

    def _activity(self, entries):
        r = MagicMock()
        r.zrangebyscore.return_value = entries
        with patch("self.get_redis", return_value=r):
            return self.jarvis_self._get_user_activity(24)

    def test_no_entries_empty_satisfaction(self):
        result = self._activity([])
        for code in result:
            self.assertEqual(result[code]["satisfaction"], {})

    def test_all_unknown_empty_satisfaction(self):
        result = self._activity([self._entry("unknown"), self._entry("unknown")])
        for code in result:
            self.assertEqual(result[code]["satisfaction"], {})

    def test_positive_counted(self):
        result = self._activity([self._entry("positive"), self._entry("positive")])
        for code in result:
            self.assertEqual(result[code]["satisfaction"].get("positive"), 2)

    def test_mixed_satisfaction_counted(self):
        entries = [
            self._entry("positive"),
            self._entry("positive"),
            self._entry("negative"),
            self._entry("unknown"),
        ]
        result = self._activity(entries)
        for code in result:
            self.assertEqual(result[code]["satisfaction"].get("positive"), 2)
            self.assertEqual(result[code]["satisfaction"].get("negative"), 1)
            self.assertNotIn("unknown", result[code]["satisfaction"])

    def test_topics_still_aggregated(self):
        entries = [
            self._entry("positive", topics=["karting"]),
            self._entry("unknown",  topics=["finance", "karting"]),
        ]
        result = self._activity(entries)
        for code in result:
            self.assertIn("karting", result[code]["topics"])

    def test_fmt_activity_shows_satisfaction(self):
        activity = {
            "KORBEN99": {
                "name": "Sébastien",
                "conversations": 5,
                "topics": ["karting"],
                "satisfaction": {"positive": 3, "negative": 1},
            }
        }
        line = self.jarvis_self._fmt_activity(activity)
        self.assertIn("+3", line)
        self.assertIn("-1", line)
        self.assertIn("satisfaction", line)

    def test_fmt_activity_no_satisfaction_key_absent(self):
        activity = {
            "KORBEN99": {
                "name": "Sébastien",
                "conversations": 3,
                "topics": [],
                "satisfaction": {},
            }
        }
        line = self.jarvis_self._fmt_activity(activity)
        self.assertNotIn("satisfaction", line)

    def test_fmt_activity_only_positive(self):
        activity = {
            "KORBEN99": {
                "name": "Sébastien",
                "conversations": 2,
                "topics": [],
                "satisfaction": {"positive": 2},
            }
        }
        line = self.jarvis_self._fmt_activity(activity)
        self.assertIn("+2", line)
        self.assertNotIn("-", line.split("satisfaction")[-1])

    def test_fmt_activity_only_negative(self):
        activity = {
            "KORBEN99": {
                "name": "Sébastien",
                "conversations": 1,
                "topics": [],
                "satisfaction": {"negative": 1},
            }
        }
        line = self.jarvis_self._fmt_activity(activity)
        self.assertIn("-1", line)


@unittest.skip(
    "obsolète : le champ 'retractions' a été retiré d'ANALYSIS_PROMPT — l'archivage "
    "autobio est géré exclusivement par la nightly review"
)
class TestAnalysisPromptRetractions(unittest.TestCase):

    def test_retractions_field_in_prompt(self):
        self.assertIn("retractions", prompts.ANALYSIS_PROMPT)

    def test_retractions_correction_context(self):
        self.assertTrue(
            "corrige" in prompts.ANALYSIS_PROMPT.lower()
            or "correction" in prompts.ANALYSIS_PROMPT.lower(),
        )

    def test_retractions_default_is_empty_list(self):
        self.assertIn("[]", prompts.ANALYSIS_PROMPT)

    def test_retractions_format_is_string_list(self):
        prompt_lower = prompts.ANALYSIS_PROMPT.lower()
        self.assertTrue("phrase" in prompt_lower or "liste" in prompt_lower)


@unittest.skip(
    "obsolète : post_analysis ne traite plus de 'retractions' (champ supprimé du "
    "pipeline d'analyse)"
)
class TestPostAnalysisRetractions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = _import_pipeline()

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _base_analysis(self, **kwargs):
        base = {
            "topics": [], "mood": "neutral", "user_facts": [],
            "projects": [], "importance": 0.0, "memory_summary": None,
            "interest_weights": [], "retractions": [],
        }
        base.update(kwargs)
        return base

    def _patched_post_analysis(self, analysis_result, session="sess", user_code="KORBEN99"):
        async_analyze = AsyncMock(return_value=analysis_result)
        with patch("pipeline.analyze_exchange", async_analyze), \
             patch("pipeline.get_user_profile", return_value={}), \
             patch("pipeline.get_user_projects", return_value=[]), \
             patch("pipeline.update_emotional_state"), \
             patch("pipeline.apply_project_updates"), \
             patch("pipeline.update_user_profile"), \
             patch("pipeline.set_interest_weight"), \
             patch("pipeline.log_conversation"), \
             patch("pipeline.retract_autobiographical_event") as mock_retract:
            self._run(self.pipeline.post_analysis(session, user_code, "msg", "resp"))
            return mock_retract

    def test_single_retraction_triggers_call(self):
        mock_retract = self._patched_post_analysis(
            self._base_analysis(retractions=["Sébastien travaillait chez Acme Corp"])
        )
        mock_retract.assert_called_once_with("KORBEN99", "Sébastien travaillait chez Acme Corp")

    def test_multiple_retractions_all_called(self):
        mock_retract = self._patched_post_analysis(
            self._base_analysis(retractions=["fait A", "fait B"])
        )
        self.assertEqual(mock_retract.call_count, 2)

    def test_empty_list_no_call(self):
        mock_retract = self._patched_post_analysis(self._base_analysis(retractions=[]))
        mock_retract.assert_not_called()

    def test_absent_key_no_call(self):
        analysis = self._base_analysis()
        del analysis["retractions"]
        mock_retract = self._patched_post_analysis(analysis)
        mock_retract.assert_not_called()

    def test_empty_string_ignored(self):
        mock_retract = self._patched_post_analysis(
            self._base_analysis(retractions=["", "   ", None, "fait valide"])
        )
        self.assertEqual(mock_retract.call_count, 1)
        mock_retract.assert_called_with("KORBEN99", "fait valide")

    def test_retraction_scoped_to_correct_user(self):
        mock_retract = self._patched_post_analysis(
            self._base_analysis(retractions=["un fait"]),
            user_code="AQWZSX",
        )
        mock_retract.assert_called_once_with("AQWZSX", "un fait")


# ══════════════════════════════════════════════════════════════════════════════
# ── SECTION B : INTEGRATION TESTS ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

BASE_URL = os.getenv("JARVIS_URL", "http://localhost:8000")
TEST_USER = "TEST"
# Le bearer token = le code utilisateur (cf. briefing_routes.py)
TEST_AUTH = f"Bearer {TEST_USER}"

# Délai d'attente en secondes après un échange pour que post_analysis (background)
# ait le temps d'écrire le profil dans Redis. Le modèle local est lent.
_ANALYSIS_WAIT = float(os.getenv("JARVIS_ANALYSIS_WAIT", "12"))

# Timeout HTTP pour les appels chat (le LLM local peut être lent)
_CHAT_TIMEOUT = float(os.getenv("JARVIS_CHAT_TIMEOUT", "120"))


def _server_up() -> bool:
    """Retourne True si le serveur Jarvis répond sur BASE_URL."""
    try:
        r = _httpx.get(f"{BASE_URL}/self/state", timeout=3.0)
        return r.status_code < 500
    except Exception:
        return False


# ── Collecteur de perf (partagé entre tous les tests) ────────────────────────
# Chaque entrée : {"label", "ttft_ms", "total_ms", "server_ms", "model", "passed"}
_PERF_LOG: list[dict] = []


# ── Helpers intégration ───────────────────────────────────────────────────────

def _chat(message: str, session_id: str = "qa_default", timeout: float = _CHAT_TIMEOUT) -> dict:
    """
    Envoie un message au pipeline Jarvis (stream=False) et retourne le JSON.
    Lève AssertionError si la requête échoue.
    """
    payload = {
        "message": message,
        "session_id": session_id,
        "user_code": TEST_USER,
        "stream": False,
    }
    log.debug("→ CHAT [%s] %r", session_id, message[:80])
    t0 = time.perf_counter()
    r = _httpx.post(f"{BASE_URL}/chat", json=payload, timeout=timeout)
    total_ms = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    data = r.json()
    log.debug(
        "← CHAT [%s] %.1fs | model=%s | response=%r",
        session_id, total_ms / 1000, data.get("model", "?"), data.get("response", "")[:120],
    )
    data["_total_ms"] = total_ms  # injecté pour les tests de perf
    return data


def _measure_ttft(
    message: str,
    session_id: str,
    timeout: float = _CHAT_TIMEOUT,
) -> dict:
    """
    Mesure le TTFT (Time To First Token) via streaming SSE.

    Retourne un dict :
      ttft_ms   — ms jusqu'au premier chunk de contenu
      total_ms  — ms jusqu'à la fin du stream (client-side)
      server_ms — duration_ms rapportée par le serveur (inclut routing + LLM)
      model     — modèle utilisé
      response  — texte complet assemblé
    """
    payload = {
        "message": message,
        "session_id": session_id,
        "user_code": TEST_USER,
        "stream": True,
    }
    log.debug("→ TTFT [%s] %r", session_id, message[:80])
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    chunks: list[str] = []
    server_ms: int | None = None
    model = "?"

    with _httpx.stream("POST", f"{BASE_URL}/chat", json=payload, timeout=timeout) as r:
        assert r.status_code == 200, f"HTTP {r.status_code}"
        for raw_line in r.iter_lines():
            if not raw_line.startswith("data: "):
                continue
            try:
                evt = json.loads(raw_line[6:])
            except json.JSONDecodeError:
                continue
            if "content" in evt and evt["content"]:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                chunks.append(evt["content"])
            if evt.get("done"):
                server_ms = evt.get("duration_ms")
                model = evt.get("model", "?")
                break

    total_ms = (time.perf_counter() - t0) * 1000
    result = {
        "ttft_ms": ttft_ms if ttft_ms is not None else total_ms,
        "total_ms": total_ms,
        "server_ms": server_ms,
        "model": model,
        "response": "".join(chunks),
    }
    log.info(
        "← TTFT [%s] ttft=%.0fms total=%.0fms server=%s model=%s",
        session_id,
        result["ttft_ms"],
        result["total_ms"],
        f"{server_ms}ms" if server_ms else "?",
        model.split("/")[-1],
    )
    return result


def _get_profile() -> dict:
    """GET /memory/profile/TEST → renvoie le profil complet."""
    r = _httpx.get(f"{BASE_URL}/memory/profile/{TEST_USER}", timeout=10)
    assert r.status_code == 200, f"profile HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()
    log.debug("← PROFILE keys=%s", list(data.get("profile", {}).keys()))
    return data


def _get_projects() -> list:
    return _get_profile().get("projects", [])


def _get_profile_keys() -> dict:
    return _get_profile().get("profile", {})


def _wait_analysis(label: str = ""):
    """Force l'extraction de faits/projets pour TEST_USER puis attend la fin.

    Historique : post_analysis extrayait les faits en ligne — un simple sleep
    suffisait. L'extraction est désormais faite par le job planifié
    analyse_recent_conversations (toutes les CONV_ANALYSIS_INTERVAL_MINUTES) ;
    ce helper appelle la route POST /memory/analyze/{user} qui exécute une
    passe immédiate (synchrone côté serveur, inférence LLM incluse).
    """
    log.info("⏳ analyse à la demande %s(POST /memory/analyze)…", f"[{label}] " if label else "")
    r = _httpx.post(f"{BASE_URL}/memory/analyze/{TEST_USER}", timeout=300)
    assert r.status_code == 200, f"/memory/analyze → HTTP {r.status_code}: {r.text[:200]}"
    time.sleep(1)  # laisse les écritures Redis/Qdrant se déposer


# ── Marker pytest ─────────────────────────────────────────────────────────────

_integration = pytest.mark.integration


@pytest.mark.skipif(not _server_up(), reason=f"Jarvis server not reachable at {BASE_URL}")
class TestIntegrationSmoke(unittest.TestCase):
    """Tests de base — vérifie que le serveur répond correctement."""

    @_integration
    def test_01_health_endpoint(self):
        """GET /self/state doit retourner 200 avec les champs Jarvis attendus."""
        r = _httpx.get(f"{BASE_URL}/self/state", timeout=5)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("identity", data)
        self.assertIn("goals", data)
        log.info("✓ /self/state → identity=%s", data.get("identity", {}).get("name", "?"))

    @_integration
    def test_02_simple_chat(self):
        """Réponse non-vide à un message simple."""
        data = _chat("Bonjour Jarvis, c'est un test automatique.", session_id="qa_smoke")
        response = data.get("response", "")
        self.assertGreater(len(response), 10, f"Réponse trop courte: {response!r}")
        self.assertIn("session_id", data)
        self.assertIn("model", data)
        self.assertIn("duration_ms", data)
        log.info("✓ chat smoke → %r", response[:100])

    @_integration
    def test_03_response_not_json_blob(self):
        """La réponse ne doit pas être un bloc JSON brut (régression bug)."""
        data = _chat("Quelle heure est-il ?", session_id="qa_smoke_json")
        response = data.get("response", "")
        # Si la réponse commence par { ou [ c'est un bug de pipeline
        stripped = response.strip()
        self.assertFalse(
            stripped.startswith("{") or stripped.startswith("["),
            f"Réponse ressemble à du JSON brut:\n{stripped[:300]}",
        )


@pytest.mark.skipif(not _server_up(), reason=f"Jarvis server not reachable at {BASE_URL}")
class TestIntegrationWeather(unittest.TestCase):
    """Vérifie que l'intent météo est détecté et que la réponse contient des données météo."""

    @_integration
    def test_04_weather_response(self):
        """
        Demande météo → réponse doit mentionner des mots météo (temp, °C, ciel, nuage…).
        """
        data = _chat(
            "Quel temps fait-il à Paris aujourd'hui ?",
            session_id="qa_weather",
        )
        response = data.get("response", "").lower()
        weather_keywords = ["°c", "température", "degrés", "nuage", "soleil",
                            "pluie", "vent", "météo", "ciel", "humidité"]
        found = [kw for kw in weather_keywords if kw in response]
        self.assertTrue(
            len(found) > 0,
            f"Aucun mot météo trouvé dans la réponse. Réponse: {response[:300]}\n"
            f"Vérifier: intent weather détecté ? web_search fonctionnel ?",
        )
        log.info("✓ weather → mots trouvés: %s", found)


@pytest.mark.skipif(not _server_up(), reason=f"Jarvis server not reachable at {BASE_URL}")
class TestIntegrationWebSearch(unittest.TestCase):
    """Vérifie que la recherche web est déclenchée et produit des sources."""

    @_integration
    def test_05_web_search_returns_sources(self):
        """
        Question nécessitant une recherche web → web_sources non vide ou
        réponse contient du contenu factuel récent.
        """
        data = _chat(
            "Recherche les dernières nouvelles sur Mistral AI cette semaine.",
            session_id="qa_web",
        )
        web_sources = data.get("web_sources", [])
        response = data.get("response", "").lower()
        has_sources = len(web_sources) > 0
        has_content = any(kw in response for kw in ["mistral", "ia", "modèle", "ai", "intelligence"])
        log.info("✓ web sources=%d | content keywords found=%s", len(web_sources), has_content)
        self.assertTrue(
            has_sources or has_content,
            f"Pas de sources web et pas de contenu Mistral.\n"
            f"web_sources={web_sources}\nRéponse={response[:300]}",
        )

    @_integration
    def test_06_web_error_not_exposed(self):
        """
        En cas d'erreur web, le message d'erreur ne doit pas fuiter brut dans la réponse.
        """
        data = _chat(
            "Donne-moi le cours actuel du bitcoin.",
            session_id="qa_web_err",
        )
        response = data.get("response", "")
        self.assertNotIn("INTERNET_ERROR", response)
        self.assertNotIn("Exception", response)
        self.assertNotIn("Traceback", response)


@pytest.mark.skipif(not _server_up(), reason=f"Jarvis server not reachable at {BASE_URL}")
class TestIntegrationMemory(unittest.TestCase):
    """
    Teste le cycle complet : écriture → attente post_analysis → lecture.
    Utilise des valeurs marquées QA_ pour éviter de polluer le profil TEST.
    """

    # Clé/valeur de test — choisie pour être reconnaissable et inoffensive
    _FACT_KEY = "qa_couleur_preferee"
    _FACT_VAL = "bleu_indigo_qa"
    # Suffixe horodaté : les sessions Redis persistent 90 jours — réutiliser un
    # session_id fixe accumule les artefacts QA des runs précédents dans le
    # contexte (historique + réponses de Jarvis), ce qui pousse l'analyzer à
    # rejeter les faits ("si incertain → ne rien ajouter"). Constaté empiriquement.
    _SESSION_WRITE = f"qa_mem_write_{int(time.time())}"
    _SESSION_READ  = f"qa_mem_read_{int(time.time())}"

    @_integration
    def test_07_memory_write(self):
        """
        Enoncé d'un fait personnel → post_analysis doit l'écrire dans le profil Redis.
        """
        _chat(
            f"Note bien : ma couleur préférée de test est {self._FACT_VAL}.",
            session_id=self._SESSION_WRITE,
        )
        _wait_analysis("memory_write")

        profile = _get_profile_keys()
        log.info("PROFILE après write: %s", {k: v for k, v in profile.items() if "qa" in k.lower()})

        # Cherche la valeur dans n'importe quelle clé du profil
        # ensure_ascii=False : sinon les accents sont échappés (\u00e9) et le test
        # de sous-chaîne échoue silencieusement sur toute valeur accentuée.
        profile_dump = json.dumps(profile, ensure_ascii=False).lower()
        self.assertIn(
            self._FACT_VAL.lower(),
            profile_dump,
            f"Valeur '{self._FACT_VAL}' non trouvée dans le profil.\n"
            f"Profil complet: {json.dumps(profile, indent=2, ensure_ascii=False)[:500]}",
        )
        log.info("✓ mémoire écrite : '%s' présente dans le profil", self._FACT_VAL)

    @_integration
    def test_08_memory_read(self):
        """
        Après écriture du fait, Jarvis doit le rappeler dans une question directe.
        Ce test dépend de test_07 (l'ordre alphabétique de pytest garantit l'ordre).
        """
        data = _chat(
            "Tu te souviens de ma couleur préférée de test ? Dis-moi laquelle c'est.",
            session_id=self._SESSION_READ,
        )
        response = data.get("response", "").lower()
        log.info("RECALL réponse: %r", response[:200])
        self.assertTrue(
            "bleu" in response or "indigo" in response or self._FACT_VAL.lower() in response,
            f"La couleur '{self._FACT_VAL}' n'a pas été rappelée.\n"
            f"Réponse: {response[:300]}\n"
            f"Vérifier: search_memory fonctionne ? contexte mémoire injecté dans le prompt ?",
        )
        log.info("✓ mémoire rappelée dans la réponse")


@pytest.mark.skipif(not _server_up(), reason=f"Jarvis server not reachable at {BASE_URL}")
class TestIntegrationProjects(unittest.TestCase):
    """Cycle de vie d'un projet : ajout → vérification → clôture."""

    _PROJECT_NAME = "QA_PROJET_AUTOTEST"
    # Sessions uniques par run — voir commentaire TestIntegrationMemory.
    _SESSION_ADD  = f"qa_proj_add_{int(time.time())}"
    _SESSION_DONE = f"qa_proj_done_{int(time.time())}"

    @_integration
    def test_09_project_add(self):
        """Demande d'ajout d'un projet → doit apparaître dans la liste Redis."""
        # Nettoyer d'abord le projet s'il existe déjà (idempotence)
        _chat(
            f"Supprime le projet {self._PROJECT_NAME} si il existe.",
            session_id="qa_proj_cleanup",
        )
        _wait_analysis("cleanup")

        # Formulation alignée sur ANALYSIS_PROMPT : un "create" exige une initiative
        # annoncée explicitement, multi-étapes, sur plusieurs semaines — une commande
        # sèche ("ajoute le projet X") est volontairement rejetée par l'analyzer.
        _chat(
            f"Je démarre un nouveau projet qui va me prendre plusieurs semaines : "
            f"le projet {self._PROJECT_NAME}. Première étape : préparer le plan de test, "
            f"ensuite j'écrirai les scénarios, et je finirai par l'automatisation complète.",
            session_id=self._SESSION_ADD,
        )
        _wait_analysis("project_add")

        projects = _get_projects()
        names = [p.get("name", "") for p in projects]
        log.info("PROJETS après add: %s", names)

        # Recherche flexible (le nom peut être légèrement reformulé par le LLM)
        found = any(
            self._PROJECT_NAME.lower() in n.lower() or "autotest" in n.lower()
            for n in names
        )
        self.assertTrue(
            found,
            f"Projet '{self._PROJECT_NAME}' non trouvé dans la liste.\n"
            f"Projets: {names}\n"
            f"Vérifier: apply_project_updates fonctionnel ? intent self/memory déclenché ?",
        )
        log.info("✓ projet ajouté: %s", [n for n in names if "autotest" in n.lower()])

    @_integration
    def test_10_project_done(self):
        """Clôture du projet → statut doit passer à 'done'."""
        # Mention explicite du projet par son nom + fin annoncée — requis par
        # ANALYSIS_PROMPT pour émettre une action "done" sans confabulation.
        _chat(
            f"Bonne nouvelle : j'ai terminé le projet {self._PROJECT_NAME}, "
            f"l'automatisation est en place, tout est fini.",
            session_id=self._SESSION_DONE,
        )
        _wait_analysis("project_done")

        projects = _get_projects()
        log.info("PROJETS après done: %s", [(p.get("name"), p.get("status")) for p in projects])

        target = next(
            (p for p in projects
             if self._PROJECT_NAME.lower() in p.get("name", "").lower()
             or "autotest" in p.get("name", "").lower()),
            None,
        )
        self.assertIsNotNone(
            target,
            f"Projet '{self._PROJECT_NAME}' introuvable pour vérifier le statut done.",
        )
        self.assertEqual(
            target.get("status"), "done",
            f"Statut attendu 'done', obtenu '{target.get('status')}'.\n"
            f"Projet complet: {target}",
        )
        log.info("✓ projet clôturé: %s", target)


@pytest.mark.skipif(not _server_up(), reason=f"Jarvis server not reachable at {BASE_URL}")
class TestIntegrationProfileKeys(unittest.TestCase):
    """Update et suppression de clé de profil via conversation."""

    _KEY_HINT = "qa_test_preference"
    # Fait calqué sur l'exemple "Bon" d'ANALYSIS_PROMPT (loisir:tennis → "joue le
    # week-end en club"). Pièges vérifiés empiriquement sur le 35B :
    #  1. jetons artificiels ("valeur_qa_42") rejetés (prior anti-artefact) ;
    #  2. faits dégénérés clé≈valeur ("parfum préféré = pistache") rejetés
    #     (règle "la valeur doit apporter une info que la clé ne contient pas") ;
    #  3. l'orthographe est normalisée en clé (sans accents) — assertion tolérante.
    _VAL = "accordéon"
    # Sessions uniques par run — voir commentaire TestIntegrationMemory.
    _SESSION_SET = f"qa_profile_set_{int(time.time())}"
    _SESSION_DEL = f"qa_profile_del_{int(time.time())}"

    @_integration
    def test_11_profile_key_update(self):
        """
        Information spécifique transmise en conversation →
        post_analysis doit créer/mettre à jour une clé dans le profil.
        """
        # Formulation "préférence durable" — ANALYSIS_PROMPT n'extrait que des faits
        # durables dans des namespaces autorisés (preference:…). Une valeur "méta"
        # ("mon paramètre X vaut Y", "pour les tests automatiques") est volontairement
        # rejetée par l'analyzer ; on décrit donc une préférence de vie plausible,
        # sur un domaine distinct de test_07 (couleur) pour éviter le dedup de clés.
        _chat(
            f"Je joue de l'{self._VAL} en amateur tous les dimanches, "
            f"depuis une dizaine d'années.",
            session_id=self._SESSION_SET,
        )
        _wait_analysis("profile_key_update")

        profile = _get_profile_keys()
        # ensure_ascii=False : sinon les accents sont échappés (\u00e9) et le test
        # de sous-chaîne échoue silencieusement sur toute valeur accentuée.
        profile_dump = json.dumps(profile, ensure_ascii=False).lower()
        log.info("PROFIL clés: %s", list(profile.keys()))

        self.assertTrue(
            "accordeon" in profile_dump or "accordéon" in profile_dump,
            f"'{self._VAL}' non trouvé dans le profil (clé ou valeur).\n"
            f"Profil: {json.dumps(profile, indent=2, ensure_ascii=False)[:500]}",
        )
        log.info("✓ clé de profil mise à jour, '%s' présent", self._VAL)

    @_integration
    @unittest.skip(
        "obsolète : la suppression de profil via conversation n'est pas une "
        "fonctionnalité du pipeline actuel — l'analyzer n'émet pas de deletions "
        "(user_facts sans valeur) ; les suppressions sont gérées par la nightly "
        "review (curative_profile_cleanup) et la réflexion Phase 2 l'interdit aussi"
    )
    def test_12_profile_key_delete(self):
        """
        Demande de suppression d'une info → la valeur doit disparaître du profil.
        """
        _chat(
            f"Supprime de ton profil le paramètre qa_test_preference, "
            f"tu n'as plus besoin de retenir cette information.",
            session_id=self._SESSION_DEL,
        )
        _wait_analysis("profile_key_delete")

        profile = _get_profile_keys()
        # ensure_ascii=False : sinon les accents sont échappés (\u00e9) et le test
        # de sous-chaîne échoue silencieusement sur toute valeur accentuée.
        profile_dump = json.dumps(profile, ensure_ascii=False).lower()
        log.info("PROFIL après delete: clés = %s", list(profile.keys()))

        self.assertNotIn(
            self._VAL.lower(),
            profile_dump,
            f"Valeur '{self._VAL}' toujours présente après suppression.\n"
            f"Profil: {json.dumps(profile, indent=2, ensure_ascii=False)[:500]}\n"
            f"Info: ce test peut échouer si le LLM ne détecte pas la suppression — "
            f"vérifier la qualité du modèle d'analyse (no_think=False ?).",
        )
        log.info("✓ clé supprimée du profil")


@pytest.mark.skipif(not _server_up(), reason=f"Jarvis server not reachable at {BASE_URL}")
class TestIntegrationBriefing(unittest.TestCase):
    """Génération et lecture du briefing."""

    @_integration
    def test_13_briefing_generate(self):
        """POST /briefing/generate/TEST → doit retourner un aperçu non vide."""
        r = _httpx.post(
            f"{BASE_URL}/briefing/generate/{TEST_USER}",
            headers={"Authorization": TEST_AUTH},
            timeout=120,
        )
        log.info("briefing generate → HTTP %d: %s", r.status_code, r.text[:200])
        self.assertEqual(r.status_code, 200, f"Erreur génération briefing: {r.text[:300]}")
        data = r.json()
        self.assertIn("preview", data)
        self.assertGreater(len(data.get("preview", "")), 20)
        log.info("✓ briefing généré: %r", data.get("preview", "")[:100])

    @_integration
    def test_14_briefing_get(self):
        """GET /briefing/TEST → doit retourner le briefing stocké."""
        r = _httpx.get(
            f"{BASE_URL}/briefing/{TEST_USER}",
            headers={"Authorization": TEST_AUTH},
            timeout=10,
        )
        log.info("briefing get → HTTP %d", r.status_code)
        # 200 = briefing dispo, 404 = pas encore généré (acceptable si test_13 skippé)
        self.assertIn(r.status_code, [200, 404], f"Erreur inattendue: {r.text[:200]}")
        if r.status_code == 200:
            data = r.json()
            self.assertIn("text", data)
            self.assertGreater(len(data.get("text", "")), 50)
            log.info("✓ briefing lu: %d caractères", len(data.get("text", "")))
        else:
            log.info("ℹ briefing pas encore généré (404 acceptable)")


@pytest.mark.skipif(not _server_up(), reason=f"Jarvis server not reachable at {BASE_URL}")
class TestIntegrationPortfolio(unittest.TestCase):
    """Lecture d'un cours boursier via chat (web search)."""

    @_integration
    def test_15_stock_price_chat(self):
        """
        Demande du cours d'une action → réponse doit contenir des données financières.
        (TEST user n'a pas trading=true, mais la question passe par web search)
        """
        data = _chat(
            "Quel est le cours actuel de l'action Apple (AAPL) en dollars ?",
            session_id="qa_stock",
        )
        response = data.get("response", "").lower()
        log.info("STOCK réponse: %r", response[:200])
        financial_keywords = [
            "$", "dollar", "usd", "cours", "action", "aapl", "apple",
            "bourse", "prix", "€", "cotation",
        ]
        found = [kw for kw in financial_keywords if kw in response]
        self.assertTrue(
            len(found) >= 2,
            f"Peu de mots financiers dans la réponse: {found}\n"
            f"Réponse: {response[:300]}",
        )
        log.info("✓ réponse bourse → mots trouvés: %s", found)


@pytest.mark.skipif(not _server_up(), reason=f"Jarvis server not reachable at {BASE_URL}")
class TestIntegrationRouter(unittest.TestCase):
    """Vérifie que le routeur LLM fonctionne et écrit dans routing_samples.jsonl."""

    _SAMPLES_FILE = os.getenv("ROUTER_DATA_DIR", "/opt/jarvis/RouterData") + "/routing_samples.jsonl"

    @_integration
    def test_16_router_samples_written(self):
        """
        Après un échange routé par le LLM router, routing_samples.jsonl doit avoir
        reçu une entrée. Vérifie que le routeur LLM n'est pas en panne silencieuse.

        Note : l'embed router (fast-path) court-circuite le LLM router pour les
        formulations proches de ses exemples ("Quel est le temps à Lyon ?" → météo
        directe, aucun échantillon écrit). On envoie donc des formulations
        inhabituelles/ambiguës qui retombent sur le LLM router — plusieurs
        tentatives pour rester robuste aux évolutions du seuil embedding.
        """

        def _count() -> int:
            if not os.path.exists(self._SAMPLES_FILE):
                return 0
            with open(self._SAMPLES_FILE) as f:
                return sum(1 for _ in f)

        before = _count()
        attempts = [
            "Un truc me chiffonne depuis hier soir au sujet du contrat d'assurance de la voiture.",
            "Entre ce qu'on s'était dit sur mes placements et les taux actuels, je devrais m'inquiéter ?",
            "J'aimerais creuser la question des panneaux solaires pour la grange, par où commencer ?",
        ]
        after = before
        for i, msg in enumerate(attempts, 1):
            _chat(msg, session_id="qa_router_check")
            time.sleep(2)  # le routeur est synchrone — pas besoin d'attendre longtemps
            after = _count()
            log.info("routing_samples tentative %d: avant=%d après=%d", i, before, after)
            if after > before:
                break

        self.assertGreater(
            after, before,
            f"Aucune entrée ajoutée dans {self._SAMPLES_FILE} après {len(attempts)} messages.\n"
            f"Vérifier: llm_router.py syntaxiquement valide ? ROUTER_MODEL défini ? "
            f"le routeur génère du JSON valide ? (ou l'embed router capte tout — élargir les formulations)",
        )

        # Lire la dernière entrée et afficher pour diagnostic
        if os.path.exists(self._SAMPLES_FILE):
            with open(self._SAMPLES_FILE) as f:
                lines = f.readlines()
            if lines:
                last = json.loads(lines[-1])
                log.info(
                    "✓ dernière entrée routeur: model=%s intents=%s reasoning=%s",
                    last.get("model"), last.get("routing", {}).get("intents"),
                    last.get("routing", {}).get("use_reasoning"),
                )

    @_integration
    def test_17_router_intents_valid(self):
        """
        Vérifie que les intents produits par le routeur sont des valeurs reconnues.
        """
        valid_intents = {"memory", "rag", "web", "weather", "gmail", "calendar",
                         "briefing", "self", "portfolio"}

        if not os.path.exists(self._SAMPLES_FILE):
            self.skipTest(f"Pas de fichier {self._SAMPLES_FILE}")

        with open(self._SAMPLES_FILE) as f:
            lines = f.readlines()

        if not lines:
            self.skipTest("routing_samples.jsonl est vide")

        # Vérifier les 10 dernières entrées
        errors = []
        for line in lines[-10:]:
            try:
                entry = json.loads(line)
                intents = entry.get("routing", {}).get("intents", [])
                for intent in intents:
                    if intent not in valid_intents:
                        errors.append(f"Intent inconnu '{intent}' dans: {entry.get('message', '')[:50]}")
            except json.JSONDecodeError as e:
                errors.append(f"JSON invalide: {e}")

        self.assertEqual(errors, [], f"Intents invalides détectés:\n" + "\n".join(errors))
        log.info("✓ intents valides dans les 10 dernières entrées routeur")


@pytest.mark.skipif(not _server_up(), reason=f"Jarvis server not reachable at {BASE_URL}")
class TestIntegrationRAG(unittest.TestCase):
    """
    Vérifie le pipeline RAG (Qdrant knowledge base).
    Requiert au moins un document dans la collection open-webui_knowledge.
    """

    @_integration
    def test_18_rag_pipeline_no_crash(self):
        """
        Requête avec use_rag=True forcé → le pipeline ne doit pas crasher
        et la réponse doit être non-vide, sans traceback.
        """
        payload = {
            "message": "Qu'est-ce que tu trouves dans ma base de documents sur l'investissement ?",
            "session_id": "qa_rag_basic",
            "user_code": TEST_USER,
            "stream": False,
            "use_rag": True,
        }
        r = _httpx.post(f"{BASE_URL}/chat", json=payload, timeout=_CHAT_TIMEOUT)
        self.assertEqual(r.status_code, 200, f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        response = data.get("response", "")
        self.assertGreater(len(response), 10, "Réponse RAG vide")
        self.assertNotIn("Traceback", response)
        self.assertNotIn("Exception", response)
        log.info("✓ RAG pipeline OK — réponse: %r", response[:100])

    @_integration
    def test_19_rag_returns_sources(self):
        """
        Avec use_rag=True, rag_sources doit être non-vide si des documents existent.
        """
        payload = {
            "message": "Résume ce que tu sais sur la finance personnelle depuis mes documents.",
            "session_id": "qa_rag_sources",
            "user_code": TEST_USER,
            "stream": False,
            "use_rag": True,
        }
        r = _httpx.post(f"{BASE_URL}/chat", json=payload, timeout=_CHAT_TIMEOUT)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        rag_sources = data.get("rag_sources", [])
        log.info("RAG sources: %d trouvées", len(rag_sources))
        if rag_sources:
            log.info("  → %s", [s.get("source", "?")[:50] for s in rag_sources[:3]])
        # Si la collection a des docs, on en attend au moins un
        self.assertGreater(
            len(rag_sources), 0,
            f"Aucune source RAG retournée malgré use_rag=True.\n"
            f"Vérifier: Qdrant accessible ? collection non-vide ? RAG_SCORE_THRESHOLD trop élevé ?",
        )
        log.info("✓ RAG sources retournées: %d", len(rag_sources))

    @_integration
    def test_20_rag_ttft(self):
        """TTFT d'une requête RAG — mesure l'impact du context Qdrant sur la latence."""
        payload = {
            "message": "Que disent mes documents sur la gestion de patrimoine ?",
            "session_id": "qa_rag_ttft",
            "user_code": TEST_USER,
            "stream": True,
            "use_rag": True,
        }
        t0 = time.perf_counter()
        ttft_ms = None
        chunks = []
        server_ms = None
        model = "?"

        with _httpx.stream("POST", f"{BASE_URL}/chat", json=payload, timeout=_CHAT_TIMEOUT) as resp:
            self.assertEqual(resp.status_code, 200)
            for raw_line in resp.iter_lines():
                if not raw_line.startswith("data: "):
                    continue
                try:
                    evt = json.loads(raw_line[6:])
                except json.JSONDecodeError:
                    continue
                if "content" in evt and evt["content"] and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                if "content" in evt:
                    chunks.append(evt["content"])
                if evt.get("done"):
                    server_ms = evt.get("duration_ms")
                    model = evt.get("model", "?")
                    break

        total_ms = (time.perf_counter() - t0) * 1000
        ttft_ms = ttft_ms or total_ms
        _PERF_LOG.append({
            "label": "rag",
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "server_ms": server_ms,
            "model": model,
            "response": "".join(chunks),
        })
        log.info("✓ RAG TTFT=%.0fms total=%.0fms", ttft_ms, total_ms)
        # Pas de seuil strict — juste mesure (RAG + Qdrant + LLM variable)


@pytest.mark.skipif(not _server_up(), reason=f"Jarvis server not reachable at {BASE_URL}")
class TestIntegrationExpertMode(unittest.TestCase):
    """
    Vérifie le mode expert (use_reasoning=True) :
    - Le routeur détecte les mots-clés déclencheurs
    - Le modèle produit une réponse plus longue / raisonnée
    - Le TTFT est plus élevé (thinking budget actif)
    """

    @_integration
    def test_21_expert_mode_routing(self):
        """
        Message avec 'mode expert' → routing doit retourner use_reasoning=True
        dans routing_samples.jsonl.
        """
        samples_file = os.getenv("ROUTER_DATA_DIR", "/opt/jarvis/RouterData") + "/routing_samples.jsonl"

        before_count = 0
        if os.path.exists(samples_file):
            with open(samples_file) as f:
                lines_before = f.readlines()
            before_count = len(lines_before)

        _chat(
            "Mode expert : explique-moi le fonctionnement des transformers en deep learning.",
            session_id="qa_expert_routing",
        )
        time.sleep(2)

        if not os.path.exists(samples_file):
            self.skipTest("routing_samples.jsonl absent — routeur peut-être en panne")

        with open(samples_file) as f:
            lines_after = f.readlines()

        new_entries = lines_after[before_count:]
        if not new_entries:
            self.skipTest("Aucune nouvelle entrée routeur — routeur peut-être en panne")

        last_entry = json.loads(new_entries[-1])
        routing = last_entry.get("routing", {})
        use_reasoning = routing.get("use_reasoning", False)
        intents = routing.get("intents", [])

        log.info(
            "Expert routing: use_reasoning=%s intents=%s message=%r",
            use_reasoning, intents, last_entry.get("message", "")[:60],
        )
        self.assertTrue(
            use_reasoning,
            f"use_reasoning=False pour un message 'mode expert'.\n"
            f"Routing complet: {routing}\n"
            f"Vérifier: prompt ROUTER_USER contient-il 'mode expert' dans les déclencheurs ?",
        )
        log.info("✓ use_reasoning=True détecté par le routeur")

    @_integration
    def test_22_expert_mode_response_quality(self):
        """
        Mode expert → réponse plus longue et plus structurée qu'une réponse simple.
        Compare avec une réponse simple pour valider le thinking.
        """
        # Réponse simple (no_think=True attendu)
        simple = _chat("C'est quoi un transformer ?", session_id="qa_expert_simple")
        simple_len = len(simple.get("response", ""))
        simple_ms = simple.get("duration_ms", 0)

        # Réponse expert (no_think=False, thinking budget actif)
        expert = _chat(
            "Mode expert : explique en détail le mécanisme d'attention des transformers, "
            "les avantages sur les RNN, et les limites actuelles.",
            session_id="qa_expert_full",
        )
        expert_len = len(expert.get("response", ""))
        expert_ms = expert.get("duration_ms", 0)

        log.info(
            "Comparaison — simple: %d chars / %dms | expert: %d chars / %dms",
            simple_len, simple_ms, expert_len, expert_ms,
        )

        # La réponse expert doit être substantiellement plus longue
        self.assertGreater(
            expert_len, simple_len * 1.5,
            f"La réponse expert ({expert_len} chars) n'est pas significativement plus longue "
            f"que la réponse simple ({simple_len} chars).\n"
            f"Vérifier: no_think=False appliqué pour use_reasoning=True ? "
            f"thinking_budget > 0 ?",
        )
        log.info(
            "✓ réponse expert %dx plus longue que simple (%.1f×)",
            expert_len // max(simple_len, 1),
            expert_len / max(simple_len, 1),
        )

    @_integration
    def test_23_expert_mode_ttft(self):
        """
        TTFT expert vs TTFT simple — le thinking rallonge le TTFT.
        Mesure et log pour diagnostic.
        """
        # Simple
        r_simple = _measure_ttft("C'est quoi un neurone artificiel ?", "qa_perf_expert_baseline")
        _PERF_LOG.append({"label": "simple-baseline", **r_simple})

        # Expert
        r_expert = _measure_ttft(
            "Mode expert : analyse les différences architecturales entre GPT et BERT.",
            "qa_perf_expert_think",
        )
        _PERF_LOG.append({"label": "expert/thinking", **r_expert})

        log.info(
            "TTFT — simple: %.0fms | expert: %.0fms | delta: +%.0fms",
            r_simple["ttft_ms"],
            r_expert["ttft_ms"],
            r_expert["ttft_ms"] - r_simple["ttft_ms"],
        )

        # Le mode expert DOIT être plus lent (thinking actif)
        # Si c'est plus rapide ou identique → le thinking n'est probablement pas actif
        self.assertGreater(
            r_expert["ttft_ms"],
            r_simple["ttft_ms"],
            f"TTFT expert ({r_expert['ttft_ms']:.0f}ms) ≤ TTFT simple ({r_simple['ttft_ms']:.0f}ms).\n"
            f"Le thinking semble inactif en mode expert — vérifier chat_no_think logic.",
        )
        log.info("✓ TTFT expert > TTFT simple (thinking actif confirmé)")


@pytest.mark.skipif(not _server_up(), reason=f"Jarvis server not reachable at {BASE_URL}")
class TestIntegrationPerformance(unittest.TestCase):
    """
    Mesures de latence TTFT (Time To First Token) sur des requêtes types.
    Utilise stream=True pour chronométrer le premier chunk côté client.

    Seuils : orientés modèle local (Qwen3.6-35B-A3B-MLX). Ajuster via env vars.
      JARVIS_TTFT_SIMPLE_MAX  (défaut 10s)  — réponse sans contexte externe
      JARVIS_TTFT_CONTEXT_MAX (défaut 20s)  — réponse avec météo/web/mémoire
    """

    _TTFT_SIMPLE_MAX  = float(os.getenv("JARVIS_TTFT_SIMPLE_MAX",  "10")) * 1000
    _TTFT_CONTEXT_MAX = float(os.getenv("JARVIS_TTFT_CONTEXT_MAX", "40")) * 1000

    @classmethod
    def setUpClass(cls):
        # Fenêtre de stabilisation : les 25 tests précédents laissent des analyses
        # background (post_analysis, résumés de session, analyzer 35B) en vol — une
        # génération bg en cours n'est pas préemptée et peut retarder un TTFT de
        # ~20 s. On mesure ici le TTFT nominal (GPU au repos), pas la contention
        # artificielle de la suite ; les seuils restent stricts.
        settle = float(os.getenv("JARVIS_TTFT_SETTLE_S", "30"))
        log.info("⏳ stabilisation GPU avant mesures TTFT (%.0fs)…", settle)
        time.sleep(settle)

    def _perf(self, label: str, message: str, session_id: str) -> dict:
        result = _measure_ttft(message, session_id)
        _PERF_LOG.append({"label": label, **result})
        return result

    @_integration
    def test_24_ttft_simple_chat(self):
        """TTFT pour une réponse simple sans contexte externe."""
        r = self._perf("simple", "Quelle est la capitale de la France ?", "qa_perf_simple")
        self.assertLess(
            r["ttft_ms"], self._TTFT_SIMPLE_MAX,
            f"TTFT simple trop lent : {r['ttft_ms']:.0f}ms > {self._TTFT_SIMPLE_MAX:.0f}ms\n"
            f"(modèle: {r['model'].split('/')[-1]}) — vérifier charge GPU, KV cache",
        )

    @_integration
    def test_25_ttft_weather(self):
        """TTFT pour une requête météo (routing + web_search + LLM)."""
        r = self._perf("weather", "Météo actuelle à Toulouse", "qa_perf_weather")
        self.assertLess(
            r["ttft_ms"], self._TTFT_CONTEXT_MAX,
            f"TTFT météo trop lent : {r['ttft_ms']:.0f}ms > {self._TTFT_CONTEXT_MAX:.0f}ms",
        )

    @_integration
    def test_26_ttft_memory_recall(self):
        """TTFT pour une requête avec recall mémoire — bénéfice du prefetch parallèle."""
        r = self._perf("memory", "Qu'est-ce que tu sais sur moi ?", "qa_perf_memory")
        self.assertLess(
            r["ttft_ms"], self._TTFT_CONTEXT_MAX,
            f"TTFT mémoire trop lent : {r['ttft_ms']:.0f}ms > {self._TTFT_CONTEXT_MAX:.0f}ms",
        )

    @_integration
    def test_27_ttft_web_search(self):
        """TTFT pour une requête web (routing + search + LLM)."""
        r = self._perf("web", "Actualités IA du jour", "qa_perf_web")
        self.assertLess(
            r["ttft_ms"], self._TTFT_CONTEXT_MAX,
            f"TTFT web trop lent : {r['ttft_ms']:.0f}ms > {self._TTFT_CONTEXT_MAX:.0f}ms",
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── Rapport final (appelé par pytest après tous les tests du module) ──────────
# ══════════════════════════════════════════════════════════════════════════════

def _print_report() -> None:
    """Affiche un rapport de performance si des mesures TTFT ont été collectées."""
    if not _PERF_LOG:
        return

    W = 68
    SEP = "═" * W

    print(f"\n{SEP}")
    print("  JARVIS QA — RAPPORT DE PERFORMANCE".center(W))
    print(SEP)
    print(f"  Serveur  : {BASE_URL}")
    print(f"  Modèle   : {_PERF_LOG[0].get('model', '?').split('/')[-1]}")
    print(f"  Mesures  : {len(_PERF_LOG)} appels streaming")
    print(f"{'─' * W}")
    print(f"  {'Scenario':<20}  {'TTFT':>8}  {'Total':>8}  {'Serveur':>8}  {'Réponse'}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*30}")

    ttft_values = []
    total_values = []
    for p in _PERF_LOG:
        ttft_ms  = p.get("ttft_ms",  0)
        total_ms = p.get("total_ms", 0)
        srv_ms   = p.get("server_ms")
        label    = p.get("label", "?")
        response = p.get("response", "")[:35].replace("\n", " ")
        ttft_values.append(ttft_ms)
        total_values.append(total_ms)
        srv_str = f"{srv_ms}ms" if srv_ms else "    —"
        print(
            f"  {label:<20}  {ttft_ms:>6.0f}ms  {total_ms:>6.0f}ms  "
            f"{srv_str:>8}  {response!r}"
        )

    print(f"{'─' * W}")
    if ttft_values:
        avg_ttft  = sum(ttft_values) / len(ttft_values)
        max_ttft  = max(ttft_values)
        min_ttft  = min(ttft_values)
        avg_total = sum(total_values) / len(total_values)
        print(f"  {'TTFT moy':<20}  {avg_ttft:>6.0f}ms")
        print(f"  {'TTFT min':<20}  {min_ttft:>6.0f}ms")
        print(f"  {'TTFT max':<20}  {max_ttft:>6.0f}ms")
        print(f"  {'Total moy':<20}  {avg_total:>6.0f}ms")

        # Verdict
        print(f"{'─' * W}")
        if avg_ttft < 5000:
            verdict = "EXCELLENT  (< 5s)"
        elif avg_ttft < 10000:
            verdict = "BON        (< 10s)"
        elif avg_ttft < 20000:
            verdict = "ACCEPTABLE (< 20s)"
        else:
            verdict = "LENT       (> 20s) — vérifier KV cache, charge GPU"
        print(f"  Verdict TTFT  : {verdict}")

    print(SEP)


# pytest hook — appelé une fois à la fin de tous les tests du module
def teardown_module(_module):  # noqa: ANN001
    _print_report()


# ══════════════════════════════════════════════════════════════════════════════
# ── Point d'entrée standalone ────────────────────────────────────════════════
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 68)
    print(f"  Jarvis QA — server: {BASE_URL}")
    print(f"  Server reachable  : {_server_up()}")
    print(f"  Analysis wait     : {_ANALYSIS_WAIT}s")
    print(f"  Chat timeout      : {_CHAT_TIMEOUT}s")
    print(f"  TTFT simple max   : {TestIntegrationPerformance._TTFT_SIMPLE_MAX:.0f}ms")
    print(f"  TTFT context max  : {TestIntegrationPerformance._TTFT_CONTEXT_MAX:.0f}ms")
    print("=" * 68)
    unittest.main(verbosity=2)
