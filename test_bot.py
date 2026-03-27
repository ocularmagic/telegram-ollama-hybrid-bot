import os
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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

    def test_user_requested_no_search_detects_explicit_opt_out(self):
        self.assertTrue(bot.user_requested_no_search("Answer this, but do not search the internet."))
        self.assertTrue(bot.user_requested_no_search("Please help without browsing the web."))
        self.assertTrue(bot.user_requested_no_search("No internet search, just answer from what you know."))
        self.assertFalse(bot.user_requested_no_search("Should I search the internet for this?"))

    def test_usage_text_mentions_ask_no_search_command(self):
        self.assertIn("/asknosearch your question here", bot.ASK_USAGE)
        self.assertNotIn("--no-search", bot.ASK_USAGE)
        self.assertIn("Use /asknosearch to answer without internet search.", bot.START_TEXT)
        self.assertIn("Use /fast for a concise live-search answer.", bot.START_TEXT)

    def test_fast_usage_mentions_fast_command(self):
        self.assertIn("/fast your question here", bot.FAST_USAGE)
        self.assertIn("Costco", bot.FAST_USAGE)

    def test_build_no_search_pool_marks_search_as_skipped(self):
        result = bot.build_no_search_pool("Explain TLS handshakes.", "None")

        self.assertIn("Search skipped by user request", result["pool_text"])
        self.assertIn("No external search was performed.", result["pool_text"])
        self.assertEqual(result["metrics"]["retrieval_mode"], "Search skipped by user request")
        self.assertEqual(result["metrics"]["planned_query_count"], 0)
        self.assertEqual(result["metrics"]["executed_query_count"], 0)

    def test_build_current_date_context_uses_absolute_dates(self):
        now = datetime(2026, 3, 26, 15, 30, tzinfo=timezone.utc)

        context = bot.build_current_date_context(now)

        self.assertIn("Today is Thursday, March 26, 2026.", context)
        self.assertIn("Yesterday was Wednesday, March 25, 2026.", context)
        self.assertIn("Tomorrow is Friday, March 27, 2026.", context)

    def test_build_local_prompt_includes_current_date_context(self):
        with patch("bot.build_current_date_context", return_value="CURRENT LOCAL DATE CONTEXT\n- Today is Thursday, March 26, 2026."):
            prompt = bot.build_local_prompt(
                question="What is the weather today?",
                recent_chat_context="None",
                shared_pool="Weather results.",
            )

        self.assertIn("CURRENT LOCAL DATE CONTEXT", prompt)
        self.assertIn("Today is Thursday, March 26, 2026.", prompt)

    def test_build_shared_search_pool_includes_current_date_context(self):
        plan = {
            "search_objective": "Test objective",
            "queries": [{"query": "weather today", "purpose": "direct angle"}],
        }

        with patch("bot.build_current_date_context", return_value="CURRENT LOCAL DATE CONTEXT\n- Today is Thursday, March 26, 2026."), patch(
            "bot.tavily_search_pool",
            return_value={
                "summaries": [{"query": "weather today", "purpose": "direct angle", "topic": "news", "summary": "Weather summary"}],
                "results": [{"title": "Forecast", "url": "https://example.com/weather"}],
                "executed_queries": ["weather today"],
                "errors": [],
                "credits_used": 1.0,
            },
        ):
            search_result = bot.build_shared_search_pool("What is the weather today?", "None", plan)

        self.assertIn("CURRENT LOCAL DATE CONTEXT", search_result["pool_text"])
        self.assertIn("Today is Thursday, March 26, 2026.", search_result["pool_text"])

    def test_split_answer_into_segments_keeps_aligned_plaintext_as_text(self):
        answer = (
            "Hours today  8:00 AM - 10:00 PM\n"
            "Phone  (206) 555-1212\n"
            "Address  123 Pike St"
        )

        segments = bot.split_answer_into_segments(answer)

        self.assertEqual(segments, [("text", answer)])

    def test_split_answer_into_segments_keeps_markdown_tables_as_table(self):
        answer = (
            "| Item | Price |\n"
            "| --- | --- |\n"
            "| Soup | $9 |\n"
            "| Salad | $11 |"
        )

        segments = bot.split_answer_into_segments(answer)

        self.assertEqual(segments, [("table", answer)])

    def test_reflow_text_segment_merges_wrapped_prose_into_full_paragraphs(self):
        answer = (
            "**Overview:** This response was wrapped\n"
            "too early and should use the full width\n"
            "of the Telegram bubble.\n\n"
            "- Keep bullets on their own lines.\n"
            "- Preserve list formatting."
        )

        reformatted = bot.reflow_text_segment(answer)

        self.assertEqual(
            reformatted,
            (
                "**Overview:** This response was wrapped too early and should use the full width "
                "of the Telegram bubble.\n\n"
                "- Keep bullets on their own lines.\n"
                "- Preserve list formatting."
            ),
        )

    def test_reflow_text_segment_converts_markdown_headings_and_drops_dividers(self):
        answer = (
            "### Overview\n"
            "This should look like normal Telegram text.\n"
            "---\n"
            "More detail here."
        )

        reformatted = bot.reflow_text_segment(answer)

        self.assertEqual(
            reformatted,
            "**Overview**\nThis should look like normal Telegram text.\n\nMore detail here.",
        )

    def test_render_text_chunk_as_html_converts_markdown_bold_without_literal_asterisks(self):
        rendered = bot.render_text_chunk_as_html("**Overview:** Wider text & safer formatting.")

        self.assertEqual(rendered, "<b>Overview:</b> Wider text &amp; safer formatting.")

    def test_build_shared_search_pool_falls_back_to_ollama_when_tavily_search_fails_generically(self):
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

        with patch("bot.tavily_search_pool", side_effect=fake_search), patch(
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

        self.assertIn("Tavily search failed. Fell back to Ollama web search.", search_result["notices"])
        self.assertIn("Ollama web search fallback", shared_pool)
        self.assertIn("Fallback Result", shared_pool)
        self.assertEqual(search_result["metrics"]["planned_query_count"], 2)
        self.assertEqual(search_result["metrics"]["executed_query_count"], 2)
        self.assertEqual(search_result["metrics"]["unique_executed_query_count"], 2)

    def test_build_shared_search_pool_collects_results_from_tavily_search(self):
        plan = {
            "search_objective": "Test objective",
            "queries": [
                {"query": "query one", "purpose": "first angle"},
                {"query": "query two", "purpose": "second angle"},
            ],
        }

        with patch(
            "bot.tavily_search_pool",
            return_value={
                "summaries": [
                    {"query": "query one", "purpose": "first angle", "topic": "general", "summary": "Combined Tavily summary"}
                ],
                "results": [{"title": "Good Result", "url": "https://example.com/good"}],
                "executed_queries": ["query one", "query two"],
                "errors": [],
                "credits_used": 2.0,
            },
        ):
            search_result = bot.build_shared_search_pool("question", "None", plan)
            shared_pool = search_result["pool_text"]

        self.assertIn("Good Result", shared_pool)
        self.assertIn("Combined Tavily summary", shared_pool)
        self.assertIn("Query: query one", shared_pool)
        self.assertEqual(search_result["metrics"]["planned_query_count"], 2)
        self.assertEqual(search_result["metrics"]["executed_query_count"], 2)
        self.assertEqual(search_result["metrics"]["unique_executed_query_count"], 2)
        self.assertEqual(search_result["metrics"]["credits_used"], 2.0)

    def test_build_shared_search_pool_falls_back_to_ollama_when_tavily_search_errors(self):
        plan = {
            "search_objective": "Test objective",
            "queries": [{"query": "query one", "purpose": "first angle"}],
        }

        with patch(
            "bot.tavily_search_pool",
            side_effect=bot.TavilySearchError("429 rate limit"),
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

        self.assertIn("Tavily search failed. Fell back to Ollama web search.", search_result["notices"])
        self.assertIn("Ollama web search fallback", search_result["pool_text"])
        self.assertIn("Fallback Result", search_result["pool_text"])
        self.assertEqual(search_result["metrics"]["planned_query_count"], 1)
        self.assertEqual(search_result["metrics"]["executed_query_count"], 1)

    def test_format_search_metrics_notice_lists_executed_queries(self):
        notice = bot.format_search_metrics_notice(
            {
                "retrieval_mode": "Tavily search",
                "cache_hit": False,
                "planned_query_count": 3,
                "executed_query_count": 5,
                "unique_executed_query_count": 4,
                "candidate_count": 7,
                "executed_queries": ["query one", "query two"],
                "credits_used": 2.0,
            }
        )

        self.assertIn("Search cost", notice)
        self.assertIn("Cache hit: no", notice)
        self.assertIn("Planned queries: 3", notice)
        self.assertIn("Executed queries: 5 total (4 unique)", notice)
        self.assertIn("Tavily credits used: 2.00", notice)
        self.assertIn("1. query one", notice)

    def test_build_shared_search_pool_uses_cache_on_repeat_request(self):
        plan = {
            "search_objective": "Test objective",
            "queries": [{"query": "query one", "purpose": "first angle"}],
        }

        with patch(
            "bot.tavily_search_pool",
            return_value={
                "summaries": [{"query": "query one", "purpose": "first angle", "topic": "general", "summary": "Combined Tavily summary"}],
                "results": [{"title": "Good Result", "url": "https://example.com/good"}],
                "executed_queries": ["query one"],
                "errors": [],
                "credits_used": 1.0,
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

    def test_record_search_metrics_counts_tavily_and_cache_usage(self):
        bot.record_search_metrics(
            {
                "retrieval_mode": "Tavily search",
                "cache_hit": False,
                "executed_query_count": 3,
                "credits_used": 1.5,
            }
        )
        bot.record_search_metrics(
            {
                "retrieval_mode": "Tavily search (cached)",
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
        self.assertEqual(stats["tavily_uncached_asks"], 1)
        self.assertEqual(stats["tavily_executed_queries"], 3)
        self.assertEqual(stats["tavily_credits_used"], 1.5)
        self.assertEqual(stats["cache_hit_asks"], 1)
        self.assertEqual(stats["fallback_asks"], 1)

    def test_format_today_search_stats_includes_estimate(self):
        original_limit = bot.TAVILY_DAILY_CREDIT_LIMIT
        bot.TAVILY_DAILY_CREDIT_LIMIT = 10.0
        bot.record_search_metrics(
            {
                "retrieval_mode": "Tavily search",
                "cache_hit": False,
                "executed_query_count": 4,
                "credits_used": 2.0,
            }
        )

        try:
            text = bot.format_today_search_stats()
        finally:
            bot.TAVILY_DAILY_CREDIT_LIMIT = original_limit

        self.assertIn("Tavily executed queries today: 4", text)
        self.assertIn("Avg Tavily queries per uncached ask: 4.00", text)
        self.assertIn("Tavily credits used today: 2.00/10.00", text)
        self.assertIn("Estimated uncached asks remaining before credit budget:", text)

    def test_save_chat_turn_trims_history(self):
        for index in range(bot.MAX_HISTORY_TURNS + 2):
            bot.save_chat_turn(123, f"q{index}", f"a{index}")

        history = bot.get_chat_history(123)

        self.assertEqual(len(history), bot.MAX_HISTORY_TURNS)
        self.assertEqual(history[0]["question"], "q2")

    def test_get_chat_lock_reuses_lock_per_chat(self):
        self.assertIs(bot.get_chat_lock(1), bot.get_chat_lock(1))
        self.assertIsNot(bot.get_chat_lock(1), bot.get_chat_lock(2))

    def test_cloud_final_system_prompt_prefers_detailed_synthesis(self):
        self.assertIn("Default to a thorough answer unless the user explicitly asks for something brief.", bot.CLOUD_FINAL_SYSTEM_PROMPT)
        self.assertIn("Treat the local model answers as inputs, not as the finished product.", bot.CLOUD_FINAL_SYSTEM_PROMPT)
        self.assertIn("Prioritize correctness over speed or confidence.", bot.CLOUD_FINAL_SYSTEM_PROMPT)
        self.assertIn("Be explicit about uncertainty.", bot.CLOUD_FINAL_SYSTEM_PROMPT)
        self.assertIn("For factual claims based on the shared pool, cite the source inline", bot.CLOUD_FINAL_SYSTEM_PROMPT)
        self.assertIn("Do not use Markdown headings like #, ##, or ###.", bot.CLOUD_FINAL_SYSTEM_PROMPT)
        self.assertIn("Output only the final user-facing answer.", bot.CLOUD_FINAL_SYSTEM_PROMPT)

    def test_build_final_prompt_requests_richer_final_response(self):
        prompt = bot.build_final_prompt(
            question="How should I pick a laptop?",
            recent_chat_context="User wants help choosing a work machine.",
            shared_pool="Battery life, thermals, RAM, repairability.",
            local_answer_1="Pick the lightest option.",
            local_answer_2="Pick the fastest option.",
        )

        self.assertIn("SYNTHESIS GOAL:", prompt)
        self.assertIn("Produce a richer final response than the local model drafts.", prompt)
        self.assertIn("Do not just average the two local answers.", prompt)

    def test_fast_final_system_prompt_prefers_concise_direct_answers(self):
        self.assertIn("answer the user's question directly and concisely", bot.FAST_FINAL_SYSTEM_PROMPT)
        self.assertIn("prioritize concrete, current, high-signal facts", bot.FAST_FINAL_SYSTEM_PROMPT)
        self.assertIn("do not add long analysis", bot.FAST_FINAL_SYSTEM_PROMPT)
        self.assertIn("cite the source inline for factual claims", bot.FAST_FINAL_SYSTEM_PROMPT)
        self.assertIn("use measured confidence, not absolute certainty", bot.FAST_FINAL_SYSTEM_PROMPT)

    def test_build_fast_prompt_requests_direct_live_answer(self):
        prompt = bot.build_fast_prompt(
            question="What time does Costco close today?",
            recent_chat_context="None",
            shared_pool="Store hours and location details.",
        )

        self.assertIn("FAST ANSWER GOAL:", prompt)
        self.assertIn("Use the live search pool to answer quickly and directly.", prompt)
        self.assertIn("Store hours and location details.", prompt)


class BotFormattingAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_send_formatted_answer_sends_tables_as_regular_messages_not_pre_blocks(self):
        update = MagicMock()
        update.message.reply_text = AsyncMock()

        answer = (
            "| Item | Price |\n"
            "| --- | --- |\n"
            "| Soup | $9 |"
        )

        await bot.send_formatted_answer(update, answer)

        first_call = update.message.reply_text.await_args_list[0]
        self.assertEqual(first_call.kwargs["parse_mode"], "HTML")
        self.assertNotIn("<pre>", first_call.args[0])
        self.assertIn("| Item | Price |", first_call.args[0])


if __name__ == "__main__":
    unittest.main()
