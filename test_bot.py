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

        def fake_search(question, recent_chat_context, query_text, purpose):
            if query_text == "bad query":
                raise RuntimeError("boom")
            return {
                "summary": "Example snippet",
                "results": [{"title": "Good Result", "url": "https://example.com/good"}],
                "executed_queries": ["good query"],
            }

        with patch("bot.gemini_grounded_search", side_effect=fake_search):
            shared_pool = bot.build_shared_search_pool("question", "None", plan)

        self.assertIn("Good Result", shared_pool)
        self.assertIn("SEARCH WARNINGS", shared_pool)
        self.assertIn("Q2 failed: boom", shared_pool)
        self.assertIn("GROUNDED QUERY SUMMARIES", shared_pool)

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
