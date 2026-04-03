"""
tests/test_quality.py — Jarvis quality tests
=============================================
Covers:
  1. _detect_satisfaction        — pure function, 20+ cases
  2. retract_autobiographical_event — Qdrant interactions (mocked)
  3. log_conversation satisfaction  — field written to convlog
  4. _get_user_activity / _fmt_activity satisfaction aggregation (self.py)
  5. ANALYSIS_PROMPT retractions field (prompts.py)
  6. post_analysis retractions pipeline (pipeline.py, async)

Run:
    cd /opt/jarvis && source venv/bin/activate
    python -m pytest jarvis-core/tests/test_quality.py -v
"""

import asyncio
import json
import sys
import time
import types
import unittest
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, call, patch


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap: mock all external deps before importing project modules.
# IMPORTANT: all mocks must be in place before any `import` of project code.
# ──────────────────────────────────────────────────────────────────────────────

def _make_config_module():
    m = types.ModuleType("config")
    # memory.py
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
    # self.py + pipeline.py + shared
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
    m.USER_CODES = {"KORBEN99": "Sébastien", "AQWZSX": "Alice"}
    m.USER_EMAILS = {"KORBEN99": "test@test.com", "AQWZSX": "alice@test.com"}
    m.USERS = {}  # required by self.py
    m.USER_TIMEZONES = {"KORBEN99": "Europe/Paris", "AQWZSX": "Europe/Paris"}  # required by pipeline.py
    m.GROWTH_LOG_MAX_ENTRIES = 50
    m.MAX_CHAIN_ITERATIONS = 5
    m.MAX_REFLECTION_TOKENS = 1000
    m.REFINE_PROMPT_THRESHOLD = 3
    return m


# PointIdsList stub — constructed by retract_autobiographical_event
class _PointIdsList:
    def __init__(self, points):
        self.points = points


_qdrant_models = types.ModuleType("qdrant_client.models")
_qdrant_models.PointIdsList = _PointIdsList

_qdrant_client_mod = types.ModuleType("qdrant_client")
_qdrant_client_mod.models = _qdrant_models

_helpers_mock = MagicMock()
_helpers_mock.get_logger.return_value = MagicMock()

_sentence_transformers_mock = MagicMock()

# Third-party modules needed by self.py / pipeline.py
_google_services_mock = MagicMock()
_trade_keys_mock = MagicMock()
_trading_mock = MagicMock()
_web_search_mock = MagicMock()
_analyzer_mock = MagicMock()
_llm_client_mock = MagicMock()
_rag_mock = MagicMock()

for _name, _mod in [
    ("config",              _make_config_module()),
    ("helpers",             _helpers_mock),
    ("qdrant_client",       _qdrant_client_mod),
    ("qdrant_client.models", _qdrant_models),
    ("sentence_transformers", _sentence_transformers_mock),
    ("google_services",     _google_services_mock),
    ("trade_keys",          _trade_keys_mock),
    ("trading",             _trading_mock),
    ("web_search",          _web_search_mock),
    ("analyzer",            _analyzer_mock),
    ("llm_client",          _llm_client_mock),
    ("rag",                 _rag_mock),
    ("httpx",               MagicMock()),
    ("pytz",                MagicMock()),
]:
    sys.modules[_name] = _mod

sys.path.insert(0, "/opt/jarvis/jarvis-core/src")

# Import project modules now that deps are mocked
import memory       # noqa: E402
import prompts      # noqa: E402


# ── Deferred imports for modules with heavier dep trees ──────────────────────
# self.py and pipeline.py import from memory at module level; memory is already
# loaded above. We import them here so they share the same sys.modules state.

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


# ──────────────────────────────────────────────────────────────────────────────
# 1. _detect_satisfaction — pure function, no mocking needed
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectSatisfaction(unittest.TestCase):

    def sat(self, msg):
        return memory._detect_satisfaction(msg)

    # ── Positive signals ──────────────────────────────────────────────────────

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
        """Positive keyword mid-sentence."""
        self.assertEqual(self.sat("ok c'est parfait, continue"), "positive")

    # ── Negative signals ──────────────────────────────────────────────────────

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

    # ── Unknown / neutral ─────────────────────────────────────────────────────

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

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_case_insensitive_SUPER(self):
        self.assertEqual(self.sat("SUPER résultat !"), "positive")

    def test_positive_takes_precedence_order(self):
        """'merci' (positive) appears before 'erreur' (negative) — positive wins."""
        self.assertEqual(self.sat("merci pour l'analyse de l'erreur"), "positive")

    def test_known_false_positive_erreur_in_code_context(self):
        """
        Known limitation: 'erreur' in a neutral code context triggers negative.
        This is an acceptable trade-off (satisfaction is a lagged proxy signal).
        Test documents current behaviour so a future refactor doesn't silently change it.
        """
        result = self.sat("j'ai une erreur dans mon code")
        # Currently returns "negative" — document as known behaviour
        self.assertIn(result, ("negative", "unknown"))  # allow fix without breaking test


# ──────────────────────────────────────────────────────────────────────────────
# 2. retract_autobiographical_event — Qdrant interactions
# ──────────────────────────────────────────────────────────────────────────────

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
        mock_inv.assert_not_called()

    @patch("memory._invalidate_timeline_cache")
    @patch("memory.get_qdrant")
    @patch("memory.get_embed_model")
    def test_below_threshold_not_deleted(self, mock_embed, mock_qdrant, mock_inv):
        mock_embed.return_value.encode.return_value.tolist.return_value = [0.1] * 384
        mock_qdrant.return_value.query_points.return_value.points = [
            self._point("abc-123", 0.80),  # below default threshold 0.88
        ]

        self.assertEqual(memory.retract_autobiographical_event("KORBEN99", "query"), 0)
        mock_qdrant.return_value.delete.assert_not_called()
        mock_inv.assert_not_called()

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
        """Query must be scoped to user_code + memory_type=autobiographical."""
        mock_embed.return_value.encode.return_value.tolist.return_value = [0.0]
        mock_qdrant.return_value.query_points.return_value.points = []

        memory.retract_autobiographical_event("AQWZSX", "fait à corriger")

        must = mock_qdrant.return_value.query_points.call_args.kwargs["query_filter"]["must"]
        user_clause = next(c for c in must if c.get("key") == "user_code")
        type_clause = next(c for c in must if c.get("key") == "memory_type")
        self.assertEqual(user_clause["match"]["value"], "AQWZSX")
        self.assertEqual(type_clause["match"]["value"], "autobiographical")


# ──────────────────────────────────────────────────────────────────────────────
# 3. log_conversation — satisfaction field stored in convlog entry
# ──────────────────────────────────────────────────────────────────────────────

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
        """Every convlog entry must carry the satisfaction key."""
        entry = self._log_and_capture("bonjour")
        self.assertIn("satisfaction", entry)


# ──────────────────────────────────────────────────────────────────────────────
# 4. _get_user_activity / _fmt_activity — satisfaction aggregation (self.py)
# ──────────────────────────────────────────────────────────────────────────────

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
        """Run _get_user_activity with mocked Redis returning given entries."""
        r = MagicMock()
        r.zrangebyscore.return_value = entries
        # get_redis is imported directly in self.py: patch the module-level name
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
        self.assertNotIn("-", line.split("satisfaction")[-1])  # no negative shown

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


# ──────────────────────────────────────────────────────────────────────────────
# 5. ANALYSIS_PROMPT — retractions field present and documented
# ──────────────────────────────────────────────────────────────────────────────

class TestAnalysisPromptRetractions(unittest.TestCase):

    def test_retractions_field_in_prompt(self):
        self.assertIn("retractions", prompts.ANALYSIS_PROMPT)

    def test_retractions_correction_context(self):
        """Prompt must explain WHEN to use retractions (explicit user correction)."""
        self.assertTrue(
            "corrige" in prompts.ANALYSIS_PROMPT.lower()
            or "correction" in prompts.ANALYSIS_PROMPT.lower(),
            "ANALYSIS_PROMPT should mention fact correction context for retractions"
        )

    def test_retractions_default_is_empty_list(self):
        """Prompt should indicate default is [] (no retractions by default)."""
        self.assertIn("[]", prompts.ANALYSIS_PROMPT)

    def test_retractions_format_is_string_list(self):
        """Instruction should mention a phrase / short description format."""
        prompt_lower = prompts.ANALYSIS_PROMPT.lower()
        self.assertTrue(
            "phrase" in prompt_lower or "liste" in prompt_lower,
            "Retractions field should describe expected list format"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6. post_analysis — retractions processed (pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

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
        """Helper: run post_analysis with all heavy deps mocked."""
        # analyze_exchange is async — must use AsyncMock so it can be awaited
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
        mock_retract = self._patched_post_analysis(
            self._base_analysis(retractions=[])
        )
        mock_retract.assert_not_called()

    def test_absent_key_no_call(self):
        """Backward compat: old analyzer output without 'retractions' key."""
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
