import unittest
from unittest.mock import patch

import bot


class BotHelpersTest(unittest.TestCase):
    def setUp(self):
        bot.CHAT_MEMORY.clear()
        bot.CHAT_LOCKS.clear()

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

    def test_build_shared_search_pool_survives_partial_search_failure(self):
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

        with patch("bot.gemini_grounded_search", side_effect=fake_search):
            search_result = bot.build_shared_search_pool("question", "None", plan)
            shared_pool = search_result["pool_text"]

        self.assertIn("SEARCH WARNINGS", shared_pool)
        self.assertIn("Grounded search failed: boom", shared_pool)
        self.assertIn("No search results collected.", shared_pool)
        self.assertIn("GROUNDED QUERY SUMMARIES", shared_pool)

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
