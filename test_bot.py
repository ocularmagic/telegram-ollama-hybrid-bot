import os
import unittest
from unittest.mock import patch

import bot


class BotHelpersTest(unittest.TestCase):
    def setUp(self):
        bot.CHAT_MEMORY.clear()
        bot.CHAT_LOCKS.clear()
        bot.SEARCH_CACHE.clear()
        bot.SEARCH_STATS.clear()
        self._original_cache_path = bot.SEARCH_CACHE_PATH
        self._original_stats_path = bot.SEARCH_STATS_PATH
        self._test_cache_path = os.path.join(os.getcwd(), "test_search_cache.json")
        self._test_stats_path = os.path.join(os.getcwd(), "test_search_stats.json")
        if os.path.exists(self._test_cache_path):
            os.remove(self._test_cache_path)
        if os.path.exists(self._test_stats_path):
            os.remove(self._test_stats_path)
        bot.SEARCH_CACHE_PATH = self._test_cache_path
        bot.SEARCH_STATS_PATH = self._test_stats_path

    def tearDown(self):
        bot.SEARCH_CACHE_PATH = self._original_cache_path
        bot.SEARCH_STATS_PATH = self._original_stats_path
        bot.SEARCH_CACHE.clear()
        bot.SEARCH_STATS.clear()
        if os.path.exists(self._test_cache_path):
            os.remove(self._test_cache_path)
        if os.path.exists(self._test_stats_path):
            os.remove(self._test_stats_path)

    def test_choose_fetch_candidates_round_robins_across_queries(self):
        query_buckets = [
            [{"url": "https://a/1"}, {"url": "https://a/2"}],
            [{"url": "https://b/1"}, {"url": "https://b/2"}],
            [{"url": "https://c/1"}],
        ]

        chosen = bot.choose_fetch_candidates(query_buckets, 4)

        self.assertEqual(
            [item["url"] for item in chosen],
            ["https://a/1", "https://b/1", "https://c/1", "https://a/2"],
        )

    def test_build_shared_search_pool_falls_back_to_ollama_when_gemini_search_fails_generically(self):
        plan = {
            "search_objective": "Test objective",
            "queries": [
                {"query": "good query", "purpose": "works"},
                {"query": "bad query", "purpose": "fails"},
            ],
        }

        def fake_search(question, recent_chat_context, search_plan):
            self.assertEqual(search_plan, plan)
            if question == "question":
                raise RuntimeError("boom")

        with patch("bot.gemini_grounded_search", side_effect=fake_search), patch(
            "bot.ollama_search_pool",
            return_value={
                "summary": "Fallback retrieval from Ollama web search.",
                "results": [{"title": "Fallback Result", "url": "https://example.com/fallback"}],
                "executed_queries": ["good query", "bad query"],
                "errors": [],
            },
        ):
            search_result = bot.build_shared_search_pool("question", "None", plan)
            shared_pool = search_result["pool_text"]

        self.assertIn("Gemini grounded search failed. Fell back to Ollama web search.", search_result["notices"])
        self.assertIn("Ollama web search fallback", shared_pool)
        self.assertIn("Fallback Result", shared_pool)
        self.assertEqual(search_result["metrics"]["planned_query_count"], 2)
        self.assertEqual(search_result["metrics"]["executed_query_count"], 2)
        self.assertEqual(search_result["metrics"]["unique_executed_query_count"], 2)

    def test_build_shared_search_pool_collects_results_from_single_grounded_call(self):
        plan = {
            "search_objective": "Test objective",
            "queries": [
                {"query": "query one", "purpose": "first angle"},
                {"query": "query two", "purpose": "second angle"},
            ],
        }

        with patch(
            "bot.gemini_grounded_search",
            return_value={
                "summary": "Combined grounded summary",
                "results": [{"title": "Good Result", "url": "https://example.com/good"}],
                "executed_queries": ["query one", "query two"],
            },
        ):
            search_result = bot.build_shared_search_pool("question", "None", plan)
            shared_pool = search_result["pool_text"]

        self.assertIn("Good Result", shared_pool)
        self.assertIn("Combined grounded summary", shared_pool)
        self.assertIn("Google queries: query one, query two", shared_pool)
        self.assertEqual(search_result["metrics"]["planned_query_count"], 2)
        self.assertEqual(search_result["metrics"]["executed_query_count"], 2)
        self.assertEqual(search_result["metrics"]["unique_executed_query_count"], 2)

    def test_build_shared_search_pool_falls_back_to_ollama_when_gemini_quota_exhausted(self):
        plan = {
            "search_objective": "Test objective",
            "queries": [{"query": "query one", "purpose": "first angle"}],
        }

        with patch(
            "bot.gemini_grounded_search",
            side_effect=bot.GeminiQuotaExceededError("429 RESOURCE_EXHAUSTED"),
        ), patch(
            "bot.ollama_search_pool",
            return_value={
                "summary": "Fallback retrieval from Ollama web search.",
                "results": [{"title": "Fallback Result", "url": "https://example.com/fallback"}],
                "executed_queries": ["query one"],
                "errors": [],
            },
        ):
            search_result = bot.build_shared_search_pool("question", "None", plan)

        self.assertIn("Gemini search quota exhausted. Fell back to Ollama web search.", search_result["notices"])
        self.assertIn("Ollama web search fallback", search_result["pool_text"])
        self.assertIn("Fallback Result", search_result["pool_text"])
        self.assertEqual(search_result["metrics"]["planned_query_count"], 1)
        self.assertEqual(search_result["metrics"]["executed_query_count"], 1)

    def test_build_shared_search_pool_falls_back_to_ollama_when_gemini_search_returns_503(self):
        plan = {
            "search_objective": "Test objective",
            "queries": [{"query": "query one", "purpose": "first angle"}],
        }

        with patch(
            "bot.gemini_grounded_search",
            side_effect=bot.GeminiSearchUnavailableError("503 UNAVAILABLE"),
        ), patch(
            "bot.ollama_search_pool",
            return_value={
                "summary": "Fallback retrieval from Ollama web search.",
                "results": [{"title": "Fallback Result", "url": "https://example.com/fallback"}],
                "executed_queries": ["query one"],
                "errors": [],
            },
        ):
            search_result = bot.build_shared_search_pool("question", "None", plan)

        self.assertIn("Gemini grounded search returned 503. Fell back to Ollama web search.", search_result["notices"])
        self.assertIn("Ollama web search fallback", search_result["pool_text"])
        self.assertIn("Fallback Result", search_result["pool_text"])
        self.assertEqual(search_result["metrics"]["planned_query_count"], 1)
        self.assertEqual(search_result["metrics"]["executed_query_count"], 1)

    def test_format_search_metrics_notice_lists_executed_queries(self):
        notice = bot.format_search_metrics_notice(
            {
                "retrieval_mode": "Gemini grounded search",
                "cache_hit": False,
                "planned_query_count": 3,
                "executed_query_count": 5,
                "unique_executed_query_count": 4,
                "candidate_count": 7,
                "executed_queries": ["query one", "query two"],
            }
        )

        self.assertIn("Search cost", notice)
        self.assertIn("Cache hit: no", notice)
        self.assertIn("Planned queries: 3", notice)
        self.assertIn("Executed queries: 5 total (4 unique)", notice)
        self.assertIn("1. query one", notice)

    def test_build_shared_search_pool_uses_cache_on_repeat_request(self):
        plan = {
            "search_objective": "Test objective",
            "queries": [{"query": "query one", "purpose": "first angle"}],
        }

        with patch(
            "bot.gemini_grounded_search",
            return_value={
                "summary": "Combined grounded summary",
                "results": [{"title": "Good Result", "url": "https://example.com/good"}],
                "executed_queries": ["query one"],
            },
        ) as mock_search:
            first_result = bot.build_shared_search_pool("question", "None", plan)
            second_result = bot.build_shared_search_pool("question", "None", plan)

        self.assertEqual(mock_search.call_count, 1)
        self.assertFalse(first_result["metrics"]["cache_hit"])
        self.assertTrue(second_result["metrics"]["cache_hit"])
        self.assertEqual(second_result["metrics"]["executed_query_count"], 1)

    def test_ollama_web_search_uses_cache_for_repeated_query(self):
        with patch(
            "bot.ollama_api_post",
            return_value={"results": [{"title": "Result", "url": "https://example.com"}]},
        ) as mock_post:
            first_response = bot.ollama_web_search("query one", max_results=5)
            second_response = bot.ollama_web_search(" Query   One ", max_results=5)

        self.assertEqual(mock_post.call_count, 1)
        self.assertFalse(first_response["cache_hit"])
        self.assertTrue(second_response["cache_hit"])

    def test_record_search_metrics_counts_google_and_cache_usage(self):
        bot.record_search_metrics(
            {
                "retrieval_mode": "Gemini grounded search",
                "cache_hit": False,
                "executed_query_count": 3,
            }
        )
        bot.record_search_metrics(
            {
                "retrieval_mode": "Gemini grounded search (cached)",
                "cache_hit": True,
                "executed_query_count": 3,
            }
        )
        bot.record_search_metrics(
            {
                "retrieval_mode": "Ollama web search fallback",
                "cache_hit": False,
                "executed_query_count": 2,
            }
        )

        stats = bot.get_today_search_stats()

        self.assertEqual(stats["ask_count"], 3)
        self.assertEqual(stats["google_uncached_asks"], 1)
        self.assertEqual(stats["google_executed_queries"], 3)
        self.assertEqual(stats["cache_hit_asks"], 1)
        self.assertEqual(stats["fallback_asks"], 1)

    def test_format_today_search_stats_includes_estimate(self):
        bot.record_search_metrics(
            {
                "retrieval_mode": "Gemini grounded search",
                "cache_hit": False,
                "executed_query_count": 4,
            }
        )

        text = bot.format_today_search_stats()

        self.assertIn("Google executed queries today: 4/", text)
        self.assertIn("Avg Google queries per uncached ask: 4.00", text)
        self.assertIn("Estimated uncached asks remaining before limit:", text)

    def test_save_chat_turn_trims_history(self):
        for index in range(bot.MAX_HISTORY_TURNS + 2):
            bot.save_chat_turn(123, f"q{index}", f"a{index}")

        history = bot.get_chat_history(123)

        self.assertEqual(len(history), bot.MAX_HISTORY_TURNS)
        self.assertEqual(history[0]["question"], "q2")

    def test_get_chat_lock_reuses_lock_per_chat(self):
        self.assertIs(bot.get_chat_lock(1), bot.get_chat_lock(1))
        self.assertIsNot(bot.get_chat_lock(1), bot.get_chat_lock(2))


if __name__ == "__main__":
    unittest.main()
