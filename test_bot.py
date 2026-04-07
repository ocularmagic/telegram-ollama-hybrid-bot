import os
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import bot


class BotHelpersTest(unittest.TestCase):
    def setUp(self):
        bot.CHAT_MEMORY.clear()
        bot.CHAT_LOCKS.clear()
        bot.CHAT_LAST_REQUEST_TIMINGS.clear()
        bot.SEARCH_CACHE.clear()
        bot.SEARCH_STATS.clear()
        self._original_retrieval_model = bot.SEARCH_RETRIEVAL_MODEL
        self._original_exa_search_type = bot.EXA_SEARCH_TYPE
        self._original_local_candidate_limit = bot.LOCAL_CANDIDATE_LIMIT
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
        bot.SEARCH_RETRIEVAL_MODEL = "tavily-search"
        bot.EXA_SEARCH_TYPE = "auto"
        bot.LOCAL_CANDIDATE_LIMIT = 10

    def tearDown(self):
        bot.SEARCH_RETRIEVAL_MODEL = self._original_retrieval_model
        bot.EXA_SEARCH_TYPE = self._original_exa_search_type
        bot.LOCAL_CANDIDATE_LIMIT = self._original_local_candidate_limit
        bot.SEARCH_CACHE_PATH = self._original_cache_path
        bot.SEARCH_STATS_PATH = self._original_stats_path
        bot.SEARCH_CACHE.clear()
        bot.SEARCH_STATS.clear()
        bot.CHAT_LAST_REQUEST_TIMINGS.clear()
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
        self.assertTrue(bot.user_requested_no_search("Without an internet search, explain TCP vs UDP."))
        self.assertFalse(bot.user_requested_no_search("Should I search the internet for this?"))

    def test_decide_search_behavior_uses_search_for_current_questions(self):
        decision = bot.decide_search_behavior("What are the top 5 news headlines from the last 72 hours?")

        self.assertEqual(decision["decision"], bot.SEARCH_DECISION_USE_SEARCH)

    def test_decide_search_behavior_uses_search_for_currently_wording(self):
        decision = bot.decide_search_behavior("Is California currently the only state that requires a boater card?")

        self.assertEqual(decision["decision"], bot.SEARCH_DECISION_USE_SEARCH)

    def test_decide_search_behavior_skips_search_for_explanations(self):
        decision = bot.decide_search_behavior("Explain TLS handshakes.")

        self.assertEqual(decision["decision"], bot.SEARCH_DECISION_SKIP_SEARCH)

    def test_decide_search_behavior_asks_user_for_borderline_factual_lookup(self):
        decision = bot.decide_search_behavior("When was Gemini 2.5 Flash released?")

        self.assertEqual(decision["decision"], bot.SEARCH_DECISION_ASK_USER)

    def test_parse_yes_no_reply_accepts_search_confirmation_words(self):
        self.assertEqual(bot.parse_yes_no_reply("yes"), bot.SEARCH_DECISION_USE_SEARCH)
        self.assertEqual(bot.parse_yes_no_reply("no search"), bot.SEARCH_DECISION_SKIP_SEARCH)
        self.assertIsNone(bot.parse_yes_no_reply("maybe"))

    def test_usage_text_mentions_answer_commands(self):
        self.assertIn("/ask your question here", bot.ASK_USAGE)
        self.assertIn("/asksearch your question here", bot.ASK_USAGE)
        self.assertIn("/asknosearch your question here", bot.ASK_USAGE)
        self.assertNotIn("--no-search", bot.ASK_USAGE)
        self.assertIn("/ask <question> - Auto-decide whether live search is needed.", bot.START_TEXT)
        self.assertIn("/asksearch <question> - Force the full live-search workflow.", bot.START_TEXT)
        self.assertIn("/asknosearch <question> - Force an answer without internet search.", bot.START_TEXT)
        self.assertIn("/image <prompt> - Generate an image locally with ComfyUI.", bot.START_TEXT)
        self.assertIn("/fast <question> - Use live search, skip local review, and return a concise answer.", bot.START_TEXT)

    def test_fast_usage_mentions_fast_command(self):
        self.assertIn("/fast your question here", bot.FAST_USAGE)
        self.assertIn("Costco", bot.FAST_USAGE)

    def test_image_usage_mentions_image_command(self):
        self.assertIn("/image your image prompt here", bot.IMAGE_USAGE)
        self.assertIn("tabby cat", bot.IMAGE_USAGE)

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

    def test_default_search_plan_compacts_long_prompt_like_questions(self):
        question = (
            "You are a used boat expert. I am a customer looking for a boat that I can have stored on the coast "
            "of southern CA, at a port close to Yucaipa CA. The boat has to meet these requirements. Has living "
            "quarters below deck, an electrical hookup to charge when in port, a way to empty waste, enough fuel "
            "range for Catalina Island, and broad parts availability. Give me a list of 10 options, estimated "
            "ownership costs, and any fees I might be missing. Be objective and informative. Take your time."
        )

        plan = bot.default_search_plan(question)

        self.assertTrue(plan["queries"])
        for item in plan["queries"]:
            self.assertLessEqual(len(item["query"]), bot.TAVILY_MAX_QUERY_CHARS)
            self.assertNotIn("Take your time", item["query"])

        self.assertIn("Catalina", plan["queries"][0]["query"])
        self.assertLessEqual(len(plan["queries"]), 2)

    def test_search_query_budget_uses_one_for_simple_and_two_for_complex(self):
        self.assertEqual(bot.search_query_budget("What time does Costco close today?"), 1)
        self.assertEqual(
            bot.search_query_budget(
                "Compare 10 used boats for Southern California Catalina trips, overnight stays, shore power, "
                "pump-out needs, realistic costs, and parts availability."
            ),
            2,
        )
        self.assertEqual(
            bot.search_query_budget("Is California the only state that requires a boater card?"),
            2,
        )

    def test_generate_search_plan_uses_planner_model_json_output(self):
        original_planner_model = bot.SEARCH_PLANNER_MODEL
        bot.SEARCH_PLANNER_MODEL = "planner-model"

        try:
            with patch(
                "bot.call_ollama_model",
                return_value=json.dumps(
                    {
                        "search_objective": "Research used liveaboard-capable coastal boats.",
                        "queries": [
                            {
                                "query": "used cabin cruiser southern california catalina island shore power pump-out",
                                "purpose": "direct boat fit",
                            },
                            {
                                "query": "used boat ownership costs southern california slip fees maintenance",
                                "purpose": "budget reality",
                            },
                        ],
                    }
                ),
            ) as mock_call:
                plan = bot.generate_search_plan(
                    "Compare used cabin boats for Southern California Catalina trips, overnight stays, and realistic ownership costs.",
                    "None",
                )
        finally:
            bot.SEARCH_PLANNER_MODEL = original_planner_model

        self.assertEqual(mock_call.call_count, 1)
        self.assertEqual(plan["search_objective"], "Research used liveaboard-capable coastal boats.")
        self.assertEqual(len(plan["queries"]), 2)
        self.assertEqual(plan["queries"][0]["purpose"], "direct boat fit")

    def test_generate_search_plan_uses_built_in_fallback_when_configured(self):
        original_planner_model = bot.SEARCH_PLANNER_MODEL
        bot.SEARCH_PLANNER_MODEL = "built-in"

        try:
            with patch("bot.call_ollama_model") as mock_call:
                plan = bot.generate_search_plan(
                    "I need a used cabin boat for Catalina trips with shore power and sleeping quarters.",
                    "None",
                )
        finally:
            bot.SEARCH_PLANNER_MODEL = original_planner_model

        self.assertEqual(mock_call.call_count, 0)
        self.assertTrue(plan["queries"])

    def test_normalize_search_plan_dedupes_and_compacts_model_queries(self):
        noisy_plan = {
            "search_objective": "Boat research",
            "queries": [
                {
                    "query": (
                        "You are a used boat expert. Give me a detailed answer about used cabin cruisers in Southern "
                        "California with shore power, pump-out, liveaboard amenities, Catalina Island range, and "
                        "realistic costs. "
                    ) * 8,
                    "purpose": "direct fit",
                },
                {
                    "query": (
                        "You are a used boat expert. Give me a detailed answer about used cabin cruisers in Southern "
                        "California with shore power, pump-out, liveaboard amenities, Catalina Island range, and "
                        "realistic costs. "
                    ) * 8,
                    "purpose": "duplicate",
                },
            ],
        }

        plan = bot.normalize_search_plan(noisy_plan, "fallback question")

        self.assertEqual(len(plan["queries"]), 1)
        self.assertLessEqual(len(plan["queries"][0]["query"]), bot.TAVILY_MAX_QUERY_CHARS)
        self.assertNotIn("Give me a detailed answer", plan["queries"][0]["query"])

    def test_reinforce_query_with_question_intent_preserves_only_state_logic(self):
        query = bot.reinforce_query_with_question_intent(
            "Is California the only state that requires a boater card?",
            "California boater card requirement",
            bot.TAVILY_MAX_QUERY_CHARS,
        )

        self.assertIn("California", query)
        self.assertTrue("only state" in query.lower() or "which states" in query.lower())

    def test_default_search_plan_builds_comparison_query_for_only_state_question(self):
        plan = bot.default_search_plan("Is California the only state that requires a boater card?")

        self.assertEqual(len(plan["queries"]), 2)
        self.assertTrue("only state" in plan["queries"][0]["query"].lower() or "which states" in plan["queries"][0]["query"].lower())
        self.assertIn("which states", plan["queries"][1]["query"].lower())
        self.assertIn("boater card", plan["queries"][1]["query"].lower())

    def test_normalize_search_plan_repairs_model_query_that_loses_exclusivity(self):
        plan = bot.normalize_search_plan(
            {
                "search_objective": "Check uniqueness",
                "queries": [{"query": "California boater card requirement", "purpose": "direct"}],
            },
            "Is California the only state that requires a boater card?",
        )

        self.assertEqual(len(plan["queries"]), 1)
        self.assertTrue("only state" in plan["queries"][0]["query"].lower() or "which states" in plan["queries"][0]["query"].lower())

    def test_normalize_search_plan_caps_complex_questions_at_two_queries(self):
        noisy_plan = {
            "search_objective": "Complex comparison",
            "queries": [
                {"query": "query one", "purpose": "first"},
                {"query": "query two", "purpose": "second"},
                {"query": "query three", "purpose": "third"},
            ],
        }

        plan = bot.normalize_search_plan(
            noisy_plan,
            "Compare used boats for Catalina trips, overnight stays, maintenance costs, and parts availability.",
        )

        self.assertEqual(len(plan["queries"]), 2)

    def test_tavily_search_pool_shortens_queries_before_api_call(self):
        plan = {
            "queries": [
                {
                    "query": (
                        "You are a used boat expert. I need a used cabin boat for Southern California with living "
                        "quarters, shore power, pump-out, Catalina Island range, and realistic ownership costs. "
                    ) * 10,
                    "purpose": "direct query",
                }
            ]
        }

        with patch(
            "bot.tavily_api_post",
            return_value={"answer": "", "results": [], "usage": {"credits": 1.0}},
        ) as mock_post:
            result = bot.tavily_search_pool("question", "None", plan)

        sent_query = mock_post.call_args.args[1]["query"]
        self.assertLessEqual(len(sent_query), bot.TAVILY_MAX_QUERY_CHARS)
        self.assertEqual(result["executed_queries"], [sent_query])

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

    def test_build_shared_search_pool_collects_results_from_exa_search(self):
        bot.SEARCH_RETRIEVAL_MODEL = "exa-search"
        plan = {
            "search_objective": "Test objective",
            "queries": [{"query": "query one", "purpose": "first angle"}],
        }

        with patch(
            "bot.exa_search_pool",
            return_value={
                "summaries": [{"query": "query one", "purpose": "first angle", "topic": "news", "summary": "Exa summary"}],
                "results": [{"title": "Exa Result", "url": "https://example.com/exa"}],
                "executed_queries": ["query one"],
                "errors": [],
                "cost_estimate_usd": 0.007,
            },
        ):
            search_result = bot.build_shared_search_pool("question", "None", plan)

        self.assertIn("Exa search", search_result["pool_text"])
        self.assertIn("Exa Result", search_result["pool_text"])
        self.assertEqual(search_result["metrics"]["cost_estimate_usd"], 0.007)

    def test_build_shared_search_pool_creates_trimmed_local_pool(self):
        bot.LOCAL_CANDIDATE_LIMIT = 1
        plan = {
            "search_objective": "Test objective",
            "queries": [{"query": "query one", "purpose": "first angle"}],
        }

        with patch(
            "bot.tavily_search_pool",
            return_value={
                "summaries": [],
                "results": [
                    {"title": "Result One", "url": "https://example.com/one", "snippet": "Snippet one"},
                    {"title": "Result Two", "url": "https://example.com/two", "snippet": "Snippet two"},
                ],
                "executed_queries": ["query one"],
                "errors": [],
                "credits_used": 1.0,
            },
        ):
            search_result = bot.build_shared_search_pool("question", "None", plan)

        self.assertIn("Result One", search_result["local_pool_text"])
        self.assertNotIn("Result Two", search_result["local_pool_text"])
        self.assertIn("Result Two", search_result["pool_text"])

    def test_build_shared_search_pool_falls_back_from_exa_to_tavily(self):
        bot.SEARCH_RETRIEVAL_MODEL = "exa-search"
        plan = {
            "search_objective": "Test objective",
            "queries": [{"query": "query one", "purpose": "first angle"}],
        }

        with patch(
            "bot.exa_search_pool",
            side_effect=bot.ExaSearchError("missing key"),
        ), patch(
            "bot.tavily_search_pool",
            return_value={
                "summaries": [{"query": "query one", "purpose": "first angle", "topic": "general", "summary": "Tavily fallback summary"}],
                "results": [{"title": "Fallback Tavily Result", "url": "https://example.com/fallback"}],
                "executed_queries": ["query one"],
                "errors": [],
                "credits_used": 1.0,
            },
        ):
            search_result = bot.build_shared_search_pool("question", "None", plan)

        self.assertIn("Exa search failed. Fell back to Tavily web search.", search_result["notices"])
        self.assertIn("Tavily search", search_result["pool_text"])
        self.assertIn("Fallback Tavily Result", search_result["pool_text"])

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

    def test_format_search_metrics_notice_lists_exa_cost(self):
        notice = bot.format_search_metrics_notice(
            {
                "retrieval_mode": "Exa search",
                "cache_hit": False,
                "planned_query_count": 1,
                "executed_query_count": 1,
                "unique_executed_query_count": 1,
                "candidate_count": 5,
                "executed_queries": ["query one"],
                "cost_estimate_usd": 0.007,
            }
        )

        self.assertIn("Exa estimated cost: $0.0070", notice)

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

    def test_apply_comfyui_prompt_updates_configured_node(self):
        original_node_id = bot.COMFYUI_PROMPT_NODE_ID
        bot.COMFYUI_PROMPT_NODE_ID = "6"

        try:
            workflow = bot.apply_comfyui_prompt(
                {
                    "6": {
                        "class_type": "CLIPTextEncode",
                        "inputs": {"text": "{{prompt}}"},
                    }
                },
                "a cat",
            )
        finally:
            bot.COMFYUI_PROMPT_NODE_ID = original_node_id

        self.assertEqual(workflow["6"]["inputs"]["text"], "a cat")

    def test_apply_comfyui_prompt_infers_primitive_string_node(self):
        original_node_id = bot.COMFYUI_PROMPT_NODE_ID
        bot.COMFYUI_PROMPT_NODE_ID = ""

        try:
            workflow = bot.apply_comfyui_prompt(
                {
                    "54": {
                        "class_type": "CLIPTextEncode",
                        "inputs": {"text": ["557", 0]},
                    },
                    "557": {
                        "class_type": "PrimitiveStringMultiline",
                        "inputs": {"value": "old prompt"},
                    },
                },
                "a cat",
            )
        finally:
            bot.COMFYUI_PROMPT_NODE_ID = original_node_id

        self.assertEqual(workflow["557"]["inputs"]["value"], "a cat")

    def test_build_image_prompt_adds_prefix_and_suffix(self):
        original_prefix = bot.IMAGE_PROMPT_PREFIX
        original_suffix = bot.IMAGE_PROMPT_SUFFIX
        bot.IMAGE_PROMPT_PREFIX = "score_9, best quality"
        bot.IMAGE_PROMPT_SUFFIX = "cinematic lighting"

        try:
            prompt = bot.build_image_prompt("a cat")
        finally:
            bot.IMAGE_PROMPT_PREFIX = original_prefix
            bot.IMAGE_PROMPT_SUFFIX = original_suffix

        self.assertEqual(prompt, "score_9, best quality, a cat, cinematic lighting")

    def test_find_comfyui_output_images_returns_all_images(self):
        images = bot.find_comfyui_output_images(
            {
                "outputs": {
                    "9": {
                        "images": [
                            {
                                "filename": "ComfyUI_00001_.png",
                                "subfolder": "",
                                "type": "output",
                            },
                            {
                                "filename": "ComfyUI_00002_.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                }
            }
        )

        self.assertEqual([image["filename"] for image in images], ["ComfyUI_00001_.png", "ComfyUI_00002_.png"])

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

    def test_record_search_metrics_counts_exa_usage(self):
        bot.record_search_metrics(
            {
                "retrieval_mode": "Exa search",
                "cache_hit": False,
                "executed_query_count": 2,
                "cost_estimate_usd": 0.014,
            }
        )

        stats = bot.get_today_search_stats()

        self.assertEqual(stats["exa_uncached_asks"], 1)
        self.assertEqual(stats["exa_executed_queries"], 2)
        self.assertEqual(stats["exa_estimated_cost_usd"], 0.014)

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
        self.assertIn("Do not just average the local model drafts.", prompt)

    def test_build_final_prompt_omits_second_model_section_when_disabled(self):
        original_local_model_2 = bot.LOCAL_MODEL_2
        bot.LOCAL_MODEL_2 = ""
        try:
            prompt = bot.build_final_prompt(
                question="How should I pick a laptop?",
                recent_chat_context="User wants help choosing a work machine.",
                shared_pool="Battery life, thermals, RAM, repairability.",
                local_answer_1="Pick the lightest option.",
                local_answer_2="Unused answer.",
            )
        finally:
            bot.LOCAL_MODEL_2 = original_local_model_2

        self.assertIn(f"LOCAL MODEL 1 ({bot.LOCAL_MODEL_1}) ANSWER:", prompt)
        self.assertNotIn("LOCAL MODEL 2", prompt)

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

    def test_record_stage_duration_and_format_summary(self):
        stage_state = {"durations": {}}

        bot.record_stage_duration(stage_state, "planning", 1.234)
        bot.record_stage_duration(stage_state, "final", 9.876)

        summary = bot.format_stage_timing_summary(stage_state)

        self.assertIn("planning=1.23s", summary)
        self.assertIn("final=9.88s", summary)

    def test_record_prompt_metrics_and_format_summary(self):
        stage_state = {}

        bot.record_prompt_metrics(stage_state, "shared_pool", "alpha\nbeta")
        bot.record_prompt_metrics(stage_state, "local_prompt", "gamma")

        summary = bot.format_prompt_metrics_summary(stage_state)

        self.assertIn("shared_pool=10c/2l", summary)
        self.assertIn("local_prompt=5c/1l", summary)

    def test_format_last_request_timing_for_chat(self):
        stage_state = {
            "durations": {"planning": 1.2, "final": 3.4},
            "prompt_metrics": {"shared_pool": {"chars": 120, "lines": 6}},
        }

        bot.save_last_request_timing(123, "full", 5.6, stage_state)
        text = bot.format_last_request_timing(123)

        self.assertIn("Most recent request timing", text)
        self.assertIn("Mode: full", text)
        self.assertIn("Total elapsed: 5.60s", text)
        self.assertIn("planning=1.20s", text)
        self.assertIn("shared_pool=120c/6l", text)

    def test_clear_chat_history_clears_last_timing(self):
        bot.save_last_request_timing(123, "fast", 2.5, {"durations": {"final": 2.5}})

        bot.clear_chat_history(123)

        self.assertEqual(bot.format_last_request_timing(123), "Most recent request timing: none yet.")


class BotFormattingAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_safe_edit_status_message_ignores_telegram_timeout(self):
        status_message = MagicMock()
        status_message.text = "old"
        status_message.edit_text = AsyncMock(side_effect=bot.TimedOut())

        await bot.safe_edit_status_message(status_message, "new")

        self.assertEqual(status_message.edit_text.await_count, 1)

    async def test_reply_text_in_chunks_splits_long_plaintext_messages(self):
        message = MagicMock()
        message.reply_text = AsyncMock()

        long_line = "x" * (bot.MAX_TELEGRAM_CHUNK + 50)

        await bot.reply_text_in_chunks(message, long_line)

        self.assertGreaterEqual(message.reply_text.await_count, 2)

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

    async def test_status_command_includes_available_commands(self):
        update = MagicMock()
        update.effective_chat.id = 123
        update.message = MagicMock()
        context = MagicMock()

        with patch("bot.reply_text_in_chunks", new=AsyncMock()) as mock_reply:
            await bot.status_command(update, context)

        status_text = mock_reply.await_args.args[1]
        self.assertIn("Available commands:", status_text)
        self.assertIn("/ask <question>", status_text)
        self.assertIn("/asksearch <question>", status_text)
        self.assertIn("/asknosearch <question>", status_text)
        self.assertIn("/image <prompt>", status_text)
        self.assertIn("/fast <question>", status_text)

    async def test_handle_ask_request_prompts_when_search_decision_is_unclear(self):
        update = MagicMock()
        update.message = MagicMock()
        update.effective_chat.id = 123
        context = MagicMock()
        context.args = ["When", "was", "Gemini", "2.5", "Flash", "released?"]
        context.chat_data = {}

        with patch("bot.reply_text_in_chunks", new=AsyncMock()) as mock_reply, patch(
            "bot.execute_question_request",
            new=AsyncMock(),
        ) as mock_execute:
            await bot.handle_ask_request(update, context, search_policy="auto")

        self.assertIn(bot.PENDING_SEARCH_DECISION_KEY, context.chat_data)
        self.assertEqual(mock_execute.await_count, 0)
        self.assertEqual(mock_reply.await_count, 1)

    async def test_pending_search_decision_reply_runs_saved_question(self):
        update = MagicMock()
        update.effective_message = MagicMock()
        update.effective_message.text = "yes"
        context = MagicMock()
        context.chat_data = {
            bot.PENDING_SEARCH_DECISION_KEY: {
                "question": "When was Gemini 2.5 Flash released?"
            }
        }

        with patch("bot.execute_question_request", new=AsyncMock()) as mock_execute:
            await bot.pending_search_decision_reply(update, context)

        self.assertNotIn(bot.PENDING_SEARCH_DECISION_KEY, context.chat_data)
        self.assertEqual(mock_execute.await_count, 1)
        self.assertFalse(mock_execute.await_args.kwargs["force_no_search"])

    async def test_handle_ask_request_auto_search_adds_post_answer_note(self):
        update = MagicMock()
        update.message = MagicMock()
        update.effective_chat.id = 123
        context = MagicMock()
        context.args = ["What", "are", "the", "top", "5", "news", "headlines", "today?"]
        context.chat_data = {}

        with patch("bot.execute_question_request", new=AsyncMock()) as mock_execute:
            await bot.handle_ask_request(update, context, search_policy="auto")

        self.assertEqual(mock_execute.await_count, 1)
        self.assertEqual(
            mock_execute.await_args.kwargs["post_answer_note"],
            "Search used: yes (auto-decided).",
        )

    async def test_handle_ask_request_auto_no_search_adds_post_answer_note(self):
        update = MagicMock()
        update.message = MagicMock()
        update.effective_chat.id = 123
        context = MagicMock()
        context.args = ["Explain", "TLS", "handshakes."]
        context.chat_data = {}

        with patch("bot.execute_question_request", new=AsyncMock()) as mock_execute:
            await bot.handle_ask_request(update, context, search_policy="auto")

        self.assertEqual(mock_execute.await_count, 1)
        self.assertEqual(
            mock_execute.await_args.kwargs["post_answer_note"],
            "Search used: no (auto-decided).",
        )

    async def test_ask_no_search_command_forces_no_search(self):
        update = MagicMock()
        context = MagicMock()

        with patch("bot.handle_ask_request", new=AsyncMock()) as mock_handle:
            await bot.ask_no_search_command(update, context)

        self.assertEqual(mock_handle.await_count, 1)
        self.assertEqual(mock_handle.await_args.kwargs["search_policy"], "force_no_search")

    async def test_image_command_sends_all_generated_photos(self):
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        update.message.reply_photo = AsyncMock()
        status_message = MagicMock()
        status_message.text = "Generating image..."
        status_message.edit_text = AsyncMock()
        update.message.reply_text.return_value = status_message
        context = MagicMock()
        context.args = ["a", "cat"]

        with patch("bot.generate_comfyui_images", return_value=[b"fake-png-1", b"fake-png-2"]):
            await bot.image_command(update, context)

        self.assertEqual(update.message.reply_photo.await_count, 2)
        first_photo = update.message.reply_photo.await_args_list[0].kwargs["photo"]
        second_photo = update.message.reply_photo.await_args_list[1].kwargs["photo"]
        self.assertEqual(first_photo.getvalue(), b"fake-png-1")
        self.assertEqual(second_photo.getvalue(), b"fake-png-2")


if __name__ == "__main__":
    unittest.main()
