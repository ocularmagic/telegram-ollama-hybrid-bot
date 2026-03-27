import os
import re
import json
import html
import time
import logging
import asyncio
import threading
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from ollama import chat
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

LOCAL_MODEL_1 = os.getenv("LOCAL_MODEL_1", "qwen3:14b")
LOCAL_MODEL_2 = os.getenv("LOCAL_MODEL_2", "gemma3:12b")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "kimi-k2.5:cloud")
SEARCH_PLANNER_MODEL = os.getenv("SEARCH_PLANNER_MODEL", "built-in")
SEARCH_RETRIEVAL_MODEL = os.getenv("SEARCH_RETRIEVAL_MODEL", "tavily-search")
BOT_USERNAME = os.getenv("BOT_USERNAME", "your_bot_username")
TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "basic")
TAVILY_DAILY_CREDIT_LIMIT = float(os.getenv("TAVILY_DAILY_CREDIT_LIMIT", "0"))

MAX_TELEGRAM_CHUNK = 3500
MAX_PRE_CHUNK = 3000
HEARTBEAT_SECONDS = int(os.getenv("HEARTBEAT_SECONDS", "15"))

EVIDENCE_TIMEOUT_SECONDS = int(os.getenv("EVIDENCE_TIMEOUT_SECONDS", "300"))
LOCAL_MODEL_TIMEOUT_SECONDS = int(os.getenv("LOCAL_MODEL_TIMEOUT_SECONDS", "300"))
FINAL_TIMEOUT_SECONDS = int(os.getenv("FINAL_TIMEOUT_SECONDS", "600"))

SEARCH_QUERY_LIMIT = int(os.getenv("SEARCH_QUERY_LIMIT", "3"))
SEARCH_RESULTS_PER_QUERY = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "5"))
TOTAL_CANDIDATE_LIMIT = int(os.getenv("TOTAL_CANDIDATE_LIMIT", "40"))
SEARCH_SNIPPET_LIMIT = int(os.getenv("SEARCH_SNIPPET_LIMIT", "280"))
FETCH_CONTENT_LIMIT = int(os.getenv("FETCH_CONTENT_LIMIT", "900"))
SEARCH_CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "43200"))
SEARCH_CACHE_PATH = os.getenv("SEARCH_CACHE_PATH", "search_cache.json")
SEARCH_STATS_PATH = os.getenv("SEARCH_STATS_PATH", "search_stats.json")

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "3"))
MEMORY_QUESTION_CHARS = int(os.getenv("MEMORY_QUESTION_CHARS", "1200"))
MEMORY_ANSWER_CHARS = int(os.getenv("MEMORY_ANSWER_CHARS", "5000"))

CHAT_MEMORY = {}
CHAT_LOCKS = {}
SEARCH_CACHE = {}
SEARCH_CACHE_LOCK = threading.Lock()
SEARCH_STATS = {}
SEARCH_STATS_LOCK = threading.Lock()
PROGRESS_STAGES = [
    ("starting", "Starting request"),
    ("planning", "Planning search"),
    ("search_pool", "Gathering evidence"),
    ("local_model_1", f"Running {LOCAL_MODEL_1}"),
    ("local_model_2", f"Running {LOCAL_MODEL_2}"),
    ("final", f"Running {CLOUD_MODEL}"),
    ("sending", "Sending answer"),
]

LOCAL_SYSTEM_PROMPT = """You are a helpful technical assistant.

You will receive:
- the user's current question
- recent chat context from this Telegram chat
- a broad shared search pool gathered centrally

Your job:
- answer the user's question directly
- use your own judgment when selecting, ranking, filtering, or prioritizing items from the pool
- use the recent chat context only when it is relevant
- do not mention other models
- if the question asks for \"top\", \"best\", \"most important\", or similar, make your own independent choice from the pool
- if evidence is mixed or incomplete, say what is uncertain

Keep the answer concise but useful.
"""

CLOUD_FINAL_SYSTEM_PROMPT = """You are a precise, high-trust AI assistant producing the final synthesis answer.

You will receive:
- the user's current question
- recent chat context from this Telegram chat
- a broad shared search pool
- an answer from local model 1
- an answer from local model 2

Your tasks:
- Give your own full final answer, using the broad shared pool, recent chat context when relevant, and both local answers.

Rules:
- Prioritize correctness over speed or confidence.
- Be direct, clear, and useful.
- Use the shared pool directly, not just the local summaries.
- Prefer the strongest evidence in the pool.
- Prefer official or primary sources when the pool includes them.
- If the question asks for ranking or selection, make your own judgment.
- If comparing options, normalize the comparison criteria before concluding and pick a winner when the evidence supports it.
- Do not mention internal implementation details.
- Treat the local model answers as inputs, not as the finished product.
- Synthesize all useful information into one polished response instead of merely aggregating or compressing what the local models said.
- Default to a thorough answer unless the user explicitly asks for something brief.
- Add helpful context, explanation, caveats, examples, or practical next steps only when they improve the user's understanding.
- If the evidence supports it, fill in important missing context that neither local model explained well.
- Be explicit about uncertainty. State what is known, what is uncertain, and what would verify it when that distinction matters.
- Separate facts, inferences, and recommendations when doing so improves clarity.
- For factual claims based on the shared pool, cite the source inline in a short form such as "(Source: example.com)".
- Never invent citations, data, events, or capabilities.
- Do not claim you verified something live unless it is supported by the shared pool.
- Use measured confidence, not absolute certainty.
- Start with the conclusion or direct answer when that improves clarity.
- Use clean, natural formatting that reads well in Telegram.
- Do not use Markdown headings like #, ##, or ###.
- Do not use code fences, ASCII divider lines, or decorative punctuation.
- If section labels help, make them short and natural so they can be rendered as bold text.
- Avoid unnecessary padding, but do not be sparse.
- Avoid filler, hype, exaggerated praise, and motivational language.

Output only the final user-facing answer.
"""

FAST_FINAL_SYSTEM_PROMPT = """You are a precise, high-trust AI assistant producing a fast answer.

You will receive:
- the user's current question
- recent chat context from this Telegram chat
- a broad shared search pool gathered from live web retrieval

Your job:
- answer the user's question directly and concisely
- prioritize concrete, current, high-signal facts from the shared pool
- for operational questions like hours, menu items, prices, addresses, availability, or contact details, lead with the exact answer the user likely wants
- cite the source inline for factual claims from the shared pool using a short form such as "(Source: example.com)"
- mention uncertainty briefly if the search pool is ambiguous, stale, or conflicting
- say what would verify the answer if the evidence is incomplete or conflicting
- never invent citations, facts, or verification steps
- do not add long analysis, local-model summaries, or unnecessary background
- do not mention internal implementation details
- use measured confidence, not absolute certainty

Output only the user-facing answer.
"""

START_TEXT = (
    "Bot is working.\n\n"
    "Use /ask followed by your question.\n"
    "Use /fast for a concise live-search answer.\n"
    f"In groups, use /ask@{BOT_USERNAME} your question.\n"
    "Use /asknosearch to answer without internet search.\n"
    "Use /clear to clear this chat's rolling memory."
)

ASK_USAGE = (
    "Usage:\n"
    "/ask your question here\n"
    "/asknosearch your question here\n\n"
    "Example:\n"
    "/ask what are the top 5 news headlines from the last 72 hours?\n"
    "/asknosearch explain TLS handshakes from general knowledge"
)

FAST_USAGE = (
    "Usage:\n"
    "/fast your question here\n\n"
    "Example:\n"
    "/fast what are the hours for Costco in Seattle today?"
)


class TavilySearchError(RuntimeError):
    pass


def truncate_text(text: str, max_len: int) -> str:
    text = text or ""
    return text if len(text) <= max_len else text[:max_len]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_query_for_cache(query: str) -> str:
    return normalize_whitespace((query or "").lower())


def format_absolute_date(dt: datetime) -> str:
    return dt.strftime("%A, %B %d, %Y")


def build_current_date_context(now: datetime | None = None) -> str:
    local_now = now.astimezone() if now else datetime.now().astimezone()
    today = local_now.date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    return (
        "CURRENT LOCAL DATE CONTEXT\n"
        f"- Today is {format_absolute_date(datetime.combine(today, datetime.min.time(), tzinfo=local_now.tzinfo))}.\n"
        f"- Yesterday was {format_absolute_date(datetime.combine(yesterday, datetime.min.time(), tzinfo=local_now.tzinfo))}.\n"
        f"- Tomorrow is {format_absolute_date(datetime.combine(tomorrow, datetime.min.time(), tzinfo=local_now.tzinfo))}.\n"
        "- Use these absolute dates to resolve phrases like today, yesterday, tomorrow, this week, and last week.\n"
        "- Do not second-guess these dates unless the user explicitly gives a different date reference."
    )


def user_requested_no_search(question: str) -> bool:
    text = normalize_whitespace(question).lower()
    if not text:
        return False

    patterns = [
        r"\bdo not search (the )?(internet|web|online)\b",
        r"\bdon'?t search (the )?(internet|web|online)\b",
        r"\bno (internet|web|online) search\b",
        r"\bwithout (searching|browsing) (the )?(internet|web|online)\b",
        r"\bdo not browse (the )?(internet|web|online)\b",
        r"\bdon'?t browse (the )?(internet|web|online)\b",
        r"\bno web browsing\b",
        r"\bno browsing\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def load_search_cache() -> None:
    global SEARCH_CACHE

    try:
        if not os.path.exists(SEARCH_CACHE_PATH):
            SEARCH_CACHE = {}
            return

        with open(SEARCH_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        SEARCH_CACHE = data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Failed to load search cache from %s: %s", SEARCH_CACHE_PATH, e)
        SEARCH_CACHE = {}


def persist_search_cache() -> None:
    try:
        directory = os.path.dirname(SEARCH_CACHE_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)

        temp_path = f"{SEARCH_CACHE_PATH}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(SEARCH_CACHE, f, ensure_ascii=True, separators=(",", ":"))
        os.replace(temp_path, SEARCH_CACHE_PATH)
    except Exception as e:
        logger.warning("Failed to persist search cache to %s: %s", SEARCH_CACHE_PATH, e)


def prune_expired_search_cache(now: int | None = None) -> None:
    now = int(now or time.time())
    expired_keys = []

    for key, entry in SEARCH_CACHE.items():
        expires_at = entry.get("expires_at", 0)
        if not isinstance(expires_at, int) or expires_at <= now:
            expired_keys.append(key)

    for key in expired_keys:
        SEARCH_CACHE.pop(key, None)


def get_search_cache_entry(cache_key: str) -> dict | None:
    with SEARCH_CACHE_LOCK:
        prune_expired_search_cache()
        entry = SEARCH_CACHE.get(cache_key)
        if not entry:
            return None

        value = entry.get("value")
        if not isinstance(value, dict):
            SEARCH_CACHE.pop(cache_key, None)
            persist_search_cache()
            return None

        return value


def set_search_cache_entry(cache_key: str, value: dict) -> None:
    with SEARCH_CACHE_LOCK:
        prune_expired_search_cache()
        SEARCH_CACHE[cache_key] = {
            "expires_at": int(time.time()) + SEARCH_CACHE_TTL_SECONDS,
            "value": value,
        }
        persist_search_cache()


def build_search_pool_cache_key(question: str, recent_chat_context: str, plan: dict) -> str:
    payload = {
        "kind": "search_pool",
        "model": SEARCH_RETRIEVAL_MODEL,
        "question": normalize_query_for_cache(question),
        "recent_chat_context": normalize_query_for_cache(recent_chat_context),
        "queries": [
            normalize_query_for_cache(item.get("query", ""))
            for item in (plan.get("queries", []) or [])[:SEARCH_QUERY_LIMIT]
            if normalize_query_for_cache(item.get("query", ""))
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_web_search_cache_key(query: str, max_results: int) -> str:
    payload = {
        "kind": "web_search",
        "query": normalize_query_for_cache(query),
        "max_results": int(max_results),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_search_stats() -> None:
    global SEARCH_STATS

    try:
        if not os.path.exists(SEARCH_STATS_PATH):
            SEARCH_STATS = {}
            return

        with open(SEARCH_STATS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        SEARCH_STATS = data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Failed to load search stats from %s: %s", SEARCH_STATS_PATH, e)
        SEARCH_STATS = {}


def persist_search_stats() -> None:
    try:
        directory = os.path.dirname(SEARCH_STATS_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)

        temp_path = f"{SEARCH_STATS_PATH}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(SEARCH_STATS, f, ensure_ascii=True, separators=(",", ":"))
        os.replace(temp_path, SEARCH_STATS_PATH)
    except Exception as e:
        logger.warning("Failed to persist search stats to %s: %s", SEARCH_STATS_PATH, e)


def today_key() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def default_daily_search_stats() -> dict:
    return {
        "ask_count": 0,
        "tavily_uncached_asks": 0,
        "tavily_executed_queries": 0,
        "tavily_credits_used": 0.0,
        "cache_hit_asks": 0,
        "fallback_asks": 0,
    }


def record_search_metrics(metrics: dict) -> None:
    if not metrics:
        return

    with SEARCH_STATS_LOCK:
        day = today_key()
        entry = SEARCH_STATS.get(day)
        if not isinstance(entry, dict):
            entry = default_daily_search_stats()
            SEARCH_STATS[day] = entry

        entry["ask_count"] = int(entry.get("ask_count", 0)) + 1

        retrieval_mode = metrics.get("retrieval_mode", "")
        cache_hit = bool(metrics.get("cache_hit"))

        if cache_hit:
            entry["cache_hit_asks"] = int(entry.get("cache_hit_asks", 0)) + 1

        if retrieval_mode == "Ollama web search fallback":
            entry["fallback_asks"] = int(entry.get("fallback_asks", 0)) + 1

        if retrieval_mode == "Tavily search" and not cache_hit:
            entry["tavily_uncached_asks"] = int(entry.get("tavily_uncached_asks", 0)) + 1
            entry["tavily_executed_queries"] = int(entry.get("tavily_executed_queries", 0)) + int(
                metrics.get("executed_query_count", 0)
            )
            entry["tavily_credits_used"] = float(entry.get("tavily_credits_used", 0.0)) + float(
                metrics.get("credits_used", 0.0)
            )

        persist_search_stats()


def get_today_search_stats() -> dict:
    with SEARCH_STATS_LOCK:
        day = today_key()
        entry = SEARCH_STATS.get(day)
        if not isinstance(entry, dict):
            entry = default_daily_search_stats()
            SEARCH_STATS[day] = entry
        return dict(entry)


def format_today_search_stats() -> str:
    stats = get_today_search_stats()
    ask_count = int(stats.get("ask_count", 0))
    tavily_uncached_asks = int(stats.get("tavily_uncached_asks", 0))
    tavily_executed_queries = int(stats.get("tavily_executed_queries", 0))
    tavily_credits_used = float(stats.get("tavily_credits_used", 0.0))
    cache_hit_asks = int(stats.get("cache_hit_asks", 0))
    fallback_asks = int(stats.get("fallback_asks", 0))
    remaining_tavily_credits = max(0.0, TAVILY_DAILY_CREDIT_LIMIT - tavily_credits_used)

    if tavily_uncached_asks > 0:
        avg_queries_per_uncached_ask = tavily_executed_queries / tavily_uncached_asks
        avg_credits_per_uncached_ask = tavily_credits_used / tavily_uncached_asks
        if TAVILY_DAILY_CREDIT_LIMIT > 0 and avg_credits_per_uncached_ask > 0:
            estimated_remaining_uncached_asks = int(remaining_tavily_credits / avg_credits_per_uncached_ask)
        else:
            estimated_remaining_uncached_asks = 0
        avg_text = f"{avg_queries_per_uncached_ask:.2f}"
        avg_credits_text = f"{avg_credits_per_uncached_ask:.2f}"
    else:
        estimated_remaining_uncached_asks = 0
        avg_text = "0.00"
        avg_credits_text = "0.00"

    cache_hit_rate = (cache_hit_asks / ask_count * 100.0) if ask_count > 0 else 0.0
    if TAVILY_DAILY_CREDIT_LIMIT > 0:
        budget_line = f"Tavily credits used today: {tavily_credits_used:.2f}/{TAVILY_DAILY_CREDIT_LIMIT:.2f}"
        remaining_line = f"Estimated uncached asks remaining before credit budget: {estimated_remaining_uncached_asks}"
    else:
        budget_line = f"Tavily credits used today: {tavily_credits_used:.2f} (no daily credit budget configured)"
        remaining_line = "Estimated uncached asks remaining before credit budget: not configured"

    return (
        f"Daily search stats ({today_key()})\n"
        f"Total asks today: {ask_count}\n"
        f"Tavily uncached asks today: {tavily_uncached_asks}\n"
        f"Tavily executed queries today: {tavily_executed_queries}\n"
        f"{budget_line}\n"
        f"Cache-hit asks today: {cache_hit_asks} ({cache_hit_rate:.1f}%)\n"
        f"Ollama fallback asks today: {fallback_asks}\n"
        f"Avg Tavily queries per uncached ask: {avg_text}\n"
        f"Avg Tavily credits per uncached ask: {avg_credits_text}\n"
        f"{remaining_line}"
    )


def domain_from_url(url: str) -> str:
    try:
        netloc = urllib.parse.urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or "unknown"
    except Exception:
        return "unknown"


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_json_object(text: str):
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            pass

    raise ValueError("Could not parse JSON from planner response.")


def default_search_plan(question: str) -> dict:
    return {
        "search_objective": "Build a broad, multi-angle search pool for the user question.",
        "queries": [
            {"query": question, "purpose": "direct query"},
            {"query": f"{question} official documentation OR official site", "purpose": "official/primary sources"},
            {"query": f"{question} latest OR current OR recent", "purpose": "freshness/current status"},
            {"query": f"{question} comparison OR overview OR analysis", "purpose": "broader explanatory angle"},
        ],
    }


def build_no_search_pool(question: str, recent_chat_context: str) -> dict:
    current_date_context = build_current_date_context()
    lines = [
        "SEARCH OBJECTIVE",
        "- Answer the user without performing an internet search.",
        "",
        current_date_context,
        "",
        "RETRIEVAL MODE",
        "- Search skipped by user request",
        "",
        "RECENT CHAT CONTEXT",
        recent_chat_context or "None",
        "",
        "CURRENT USER QUESTION",
        f"- {question}",
        "",
        "ANSWERING CONSTRAINTS",
        "- The user explicitly asked not to perform an internet or web search.",
        "- Answer from existing model knowledge and recent chat context only.",
        "- Do not imply that live browsing, Tavily retrieval, or fresh web verification was performed.",
        "- If the answer may depend on current information, say that it was not verified on the web.",
        "",
        "CANDIDATE RESULT POOL",
        "- None. No external search was performed.",
        "",
        "SEARCH QUERY SUMMARIES",
        "- None. Search was intentionally skipped.",
    ]

    return {
        "pool_text": "\n".join(lines).strip(),
        "notices": ["Internet search skipped because the user explicitly requested no web search."],
        "metrics": {
            "retrieval_mode": "Search skipped by user request",
            "cache_hit": False,
            "planned_query_count": 0,
            "planned_queries": [],
            "executed_query_count": 0,
            "unique_executed_query_count": 0,
            "executed_queries": [],
            "candidate_count": 0,
        },
    }


def tavily_api_post(path: str, payload: dict) -> dict:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set.")

    url = f"https://api.tavily.com/{path}"
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise TavilySearchError(f"Tavily API HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise TavilySearchError(f"Tavily API connection error: {e}")


def ollama_api_post(path: str, payload: dict) -> dict:
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_API_KEY is not set.")

    url = f"https://ollama.com/api/{path}"
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama API HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama API connection error: {e}")


def ollama_web_search(query: str, max_results: int = SEARCH_RESULTS_PER_QUERY) -> dict:
    max_results = max(1, min(max_results, 10))
    cache_key = build_web_search_cache_key(query, max_results)
    cached_response = get_search_cache_entry(cache_key)
    if cached_response:
        return {
            "results": cached_response.get("results", []),
            "cache_hit": True,
        }

    response = ollama_api_post(
        "web_search",
        {
            "query": query,
            "max_results": max_results,
        },
    )
    set_search_cache_entry(
        cache_key,
        {
            "results": response.get("results", []) or [],
        },
    )
    response["cache_hit"] = False
    return response


def get_chat_history(chat_id: int):
    return CHAT_MEMORY.get(chat_id, [])


def get_chat_lock(chat_id: int) -> asyncio.Lock:
    lock = CHAT_LOCKS.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        CHAT_LOCKS[chat_id] = lock
    return lock


def save_chat_turn(chat_id: int, question: str, final_answer: str) -> None:
    history = CHAT_MEMORY.setdefault(chat_id, [])
    history.append(
        {
            "question": truncate_text(question.strip(), MEMORY_QUESTION_CHARS),
            "final_answer": truncate_text(final_answer.strip(), MEMORY_ANSWER_CHARS),
            "timestamp": int(time.time()),
        }
    )
    if len(history) > MAX_HISTORY_TURNS:
        CHAT_MEMORY[chat_id] = history[-MAX_HISTORY_TURNS:]


def clear_chat_history(chat_id: int) -> None:
    CHAT_MEMORY.pop(chat_id, None)
    CHAT_LOCKS.pop(chat_id, None)


def format_recent_chat_context(chat_id: int) -> str:
    history = get_chat_history(chat_id)
    if not history:
        return "None"

    lines = []
    for idx, turn in enumerate(history[-MAX_HISTORY_TURNS:], start=1):
        lines.append(f"TURN {idx}")
        lines.append(f"User question: {turn.get('question', '')}")
        lines.append("Final bot answer:")
        lines.append(turn.get("final_answer", ""))
        lines.append("")

    return "\n".join(lines).strip()


def generate_search_plan(question: str, recent_chat_context: str) -> dict:
    _ = recent_chat_context
    plan = default_search_plan(question)
    cleaned_queries = []

    for item in plan.get("queries", []):
        query = normalize_whitespace(item.get("query", ""))
        purpose = normalize_whitespace(item.get("purpose", ""))
        if query:
            cleaned_queries.append(
                {
                    "query": query,
                    "purpose": purpose or "general angle",
                }
            )

    return {
        "search_objective": normalize_whitespace(plan.get("search_objective", "")) or "Build a broad, multi-angle search pool.",
        "queries": cleaned_queries[:SEARCH_QUERY_LIMIT],
    }


def is_freshness_sensitive_query(text: str) -> bool:
    normalized = normalize_query_for_cache(text)
    freshness_terms = [
        "today",
        "latest",
        "current",
        "recent",
        "news",
        "headline",
        "this week",
        "last week",
        "last 24 hours",
        "last 48 hours",
        "last 72 hours",
        "breaking",
    ]
    return any(term in normalized for term in freshness_terms)


def infer_tavily_topic(question: str, query: str) -> str:
    return "news" if is_freshness_sensitive_query(question) or is_freshness_sensitive_query(query) else "general"


def tavily_search_pool(question: str, recent_chat_context: str, plan: dict) -> dict:
    _ = recent_chat_context
    all_candidates = []
    deduped_by_url = {}
    search_errors = []
    executed_queries = []
    query_summaries = []
    credits_used = 0.0

    for idx, query_item in enumerate(plan.get("queries", [])[:SEARCH_QUERY_LIMIT], start=1):
        query_text = normalize_whitespace(query_item.get("query", ""))
        purpose = normalize_whitespace(query_item.get("purpose", "")) or "general angle"
        if not query_text:
            continue

        executed_queries.append(query_text)
        topic = infer_tavily_topic(question, query_text)

        try:
            response = tavily_api_post(
                "search",
                {
                    "query": query_text,
                    "search_depth": TAVILY_SEARCH_DEPTH,
                    "topic": topic,
                    "max_results": max(1, min(SEARCH_RESULTS_PER_QUERY, 20)),
                    "include_answer": "basic",
                    "include_usage": True,
                },
            )
        except Exception as e:
            logger.warning("Tavily search failed for %r: %s", query_text, e)
            search_errors.append(f"Q{idx} failed: {e}")
            continue

        usage = response.get("usage", {}) or {}
        credits_used += float(usage.get("credits", 0.0) or 0.0)

        summary_text = truncate_text(normalize_whitespace(response.get("answer", "")), FETCH_CONTENT_LIMIT)
        if summary_text:
            query_summaries.append(
                {
                    "query": query_text,
                    "purpose": purpose,
                    "topic": topic,
                    "summary": summary_text,
                }
            )

        for rank, result in enumerate(response.get("results", []) or [], start=1):
            title = normalize_whitespace(result.get("title", ""))
            url = normalize_whitespace(result.get("url", ""))
            snippet = truncate_text(normalize_whitespace(result.get("content", "")), SEARCH_SNIPPET_LIMIT)

            if not url:
                continue

            candidate = {
                "query_index": idx,
                "query_text": query_text,
                "purpose": purpose,
                "rank_within_query": rank,
                "title": title or "Untitled result",
                "url": url,
                "source": domain_from_url(url),
                "snippet": snippet,
            }

            if url not in deduped_by_url and len(all_candidates) < TOTAL_CANDIDATE_LIMIT:
                deduped_by_url[url] = candidate
                all_candidates.append(candidate)

    return {
        "summaries": query_summaries,
        "results": all_candidates,
        "executed_queries": executed_queries,
        "errors": search_errors,
        "credits_used": credits_used,
    }


def ollama_search_pool(question: str, plan: dict) -> dict:
    all_candidates = []
    deduped_by_url = {}
    search_errors = []
    executed_queries = []

    for idx, query_item in enumerate(plan.get("queries", [])[:SEARCH_QUERY_LIMIT], start=1):
        query_text = normalize_whitespace(query_item.get("query", ""))
        if not query_text:
            continue

        executed_queries.append(query_text)
        try:
            response = ollama_web_search(query_text, SEARCH_RESULTS_PER_QUERY)
        except Exception as e:
            logger.warning("Ollama fallback search failed for %r: %s", query_text, e)
            search_errors.append(f"Q{idx} failed: {e}")
            continue

        for rank, result in enumerate(response.get("results", []) or [], start=1):
            title = normalize_whitespace(result.get("title", ""))
            url = normalize_whitespace(result.get("url", ""))
            snippet = truncate_text(normalize_whitespace(result.get("content", "")), SEARCH_SNIPPET_LIMIT)

            if not url:
                continue

            candidate = {
                "rank_within_query": rank,
                "title": title or "Untitled result",
                "url": url,
                "source": domain_from_url(url),
                "snippet": snippet,
            }

            if url not in deduped_by_url and len(all_candidates) < TOTAL_CANDIDATE_LIMIT:
                deduped_by_url[url] = candidate
                all_candidates.append(candidate)

    return {
        "summary": "Fallback retrieval from Ollama web search.",
        "results": all_candidates,
        "executed_queries": executed_queries,
        "errors": search_errors,
    }


def choose_fetch_candidates(query_buckets: list, fetch_limit: int) -> list:
    chosen = []
    seen_urls = set()
    index = 0

    while len(chosen) < fetch_limit:
        added_this_round = False

        for bucket in query_buckets:
            if index < len(bucket):
                item = bucket[index]
                url = item.get("url", "")
                if url and url not in seen_urls:
                    chosen.append(item)
                    seen_urls.add(url)
                    added_this_round = True
                    if len(chosen) >= fetch_limit:
                        break

        if not added_this_round:
            break

        index += 1

    return chosen


def build_shared_search_pool(question: str, recent_chat_context: str, plan: dict) -> dict:
    queries = plan.get("queries", [])[:SEARCH_QUERY_LIMIT]
    current_date_context = build_current_date_context()
    notices = []
    retrieval_label = "Tavily search"
    cache_hit = False
    credits_used = 0.0

    def fallback_to_ollama(search_error: Exception, notice_message: str, fallback_unavailable_notice: str, fallback_error_prefix: str):
        nonlocal retrieval_label
        retrieval_label = "Ollama web search fallback"
        try:
            fallback_response = ollama_search_pool(question, plan)
            notices.append(notice_message)
            return fallback_response
        except Exception as fallback_error:
            logger.warning("Ollama fallback search failed after Tavily error: %s", fallback_error)
            notices.append(fallback_unavailable_notice)
            return {
                "summaries": [],
                "results": [],
                "executed_queries": [],
                "errors": [
                    f"{fallback_error_prefix}: {search_error}",
                    f"Fallback search failed: {fallback_error}",
                ],
                "credits_used": 0.0,
            }

    search_pool_cache_key = build_search_pool_cache_key(question, recent_chat_context, plan)
    cached_search_response = get_search_cache_entry(search_pool_cache_key)
    if cached_search_response:
        search_response = cached_search_response
        retrieval_label = "Tavily search (cached)"
        cache_hit = True
    else:
        try:
            search_response = tavily_search_pool(question, recent_chat_context, plan)
            set_search_cache_entry(
                search_pool_cache_key,
                {
                    "summaries": search_response.get("summaries", []) or [],
                    "results": search_response.get("results", []) or [],
                    "executed_queries": search_response.get("executed_queries", []) or [],
                    "errors": search_response.get("errors", []) or [],
                    "credits_used": search_response.get("credits_used", 0.0) or 0.0,
                },
            )
        except Exception as e:
            logger.warning("Tavily search failed for question %r: %s", question, e)
            search_response = fallback_to_ollama(
                search_error=e,
                notice_message="Tavily search failed. Fell back to Ollama web search.",
                fallback_unavailable_notice=(
                    "Tavily search failed, and Ollama web-search fallback was unavailable. "
                    "Add OLLAMA_API_KEY or check TAVILY_API_KEY."
                ),
                fallback_error_prefix="Tavily search failed",
            )

    search_summaries = search_response.get("summaries", []) or []
    if not search_summaries and search_response.get("summary"):
        search_summaries = [
            {
                "query": "Combined fallback search",
                "purpose": "fallback retrieval",
                "topic": "",
                "summary": truncate_text(normalize_whitespace(search_response.get("summary", "")), FETCH_CONTENT_LIMIT),
            }
        ]
    all_candidates = search_response.get("results", []) or []
    search_errors = search_response.get("errors", []) or []
    credits_used = float(search_response.get("credits_used", 0.0) or 0.0)

    lines = []
    lines.append("SEARCH OBJECTIVE")
    lines.append(f"- {plan.get('search_objective', 'Build a broad search pool.')}")
    lines.append("")
    lines.append(current_date_context)
    lines.append("")
    lines.append("RETRIEVAL MODE")
    lines.append(f"- {retrieval_label}")
    lines.append("")
    lines.append("RECENT CHAT CONTEXT")
    lines.append(recent_chat_context or "None")
    lines.append("")
    lines.append("CURRENT USER QUESTION")
    lines.append(f"- {question}")
    lines.append("")
    lines.append("SEARCH QUERIES")
    for idx, query_item in enumerate(queries, start=1):
        lines.append(f"- Q{idx}: {query_item.get('query', '')}")
        lines.append(f"  Purpose: {query_item.get('purpose', '')}")
    lines.append("")
    if search_errors:
        lines.append("SEARCH WARNINGS")
        for item in search_errors:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("CANDIDATE RESULT POOL")
    if all_candidates:
        for idx, item in enumerate(all_candidates, start=1):
            source = item.get("source") or domain_from_url(item.get("url", ""))
            lines.append(
                f"[{idx:02d}] Source={source} | Title={item.get('title', 'Untitled result')}"
            )
            lines.append(f"URL: {item.get('url', '')}")
            if item.get("snippet"):
                lines.append(f"Snippet: {item['snippet']}")
            lines.append("")
    else:
        lines.append("- No search results collected.")
        lines.append("")

    lines.append("SEARCH QUERY SUMMARIES")
    if search_summaries:
        for idx, item in enumerate(search_summaries, start=1):
            lines.append(f"[{idx:02d}] {retrieval_label}")
            lines.append(f"Query: {item.get('query', '')}")
            lines.append(f"Purpose: {item.get('purpose', '')}")
            if item.get("topic"):
                lines.append(f"Topic: {item.get('topic', '')}")
            lines.append(f"Summary: {item['summary']}")
            lines.append("")
    else:
        lines.append("- No query summaries collected.")
        lines.append("")

    executed_queries = search_response.get("executed_queries", []) or []

    unique_executed_queries = []
    seen_executed_queries = set()
    for query in executed_queries:
        normalized_query = normalize_query_for_cache(query)
        if not normalized_query or normalized_query in seen_executed_queries:
            continue
        seen_executed_queries.add(normalized_query)
        unique_executed_queries.append(normalize_whitespace(query))

    metrics = {
        "retrieval_mode": retrieval_label,
        "cache_hit": cache_hit,
        "planned_query_count": len(queries),
        "planned_queries": [
            normalize_whitespace(item.get("query", ""))
            for item in queries
            if normalize_whitespace(item.get("query", ""))
        ],
        "executed_query_count": len(executed_queries),
        "unique_executed_query_count": len(unique_executed_queries),
        "executed_queries": unique_executed_queries,
        "candidate_count": len(all_candidates),
        "credits_used": credits_used,
    }

    return {
        "pool_text": "\n".join(lines).strip(),
        "notices": notices,
        "metrics": metrics,
    }


def format_search_metrics_notice(metrics: dict) -> str:
    if not metrics:
        return "Search cost: unavailable."

    executed_queries = metrics.get("executed_queries", []) or []
    lines = [
        "Search cost",
        f"Retrieval mode: {metrics.get('retrieval_mode', 'unknown')}",
        f"Cache hit: {'yes' if metrics.get('cache_hit') else 'no'}",
        f"Planned queries: {metrics.get('planned_query_count', 0)}",
        (
            f"Executed queries: {metrics.get('executed_query_count', 0)} total "
            f"({metrics.get('unique_executed_query_count', 0)} unique)"
        ),
        f"Candidate results kept: {metrics.get('candidate_count', 0)}",
    ]

    credits_used = metrics.get("credits_used")
    if credits_used is not None:
        lines.append(f"Tavily credits used: {float(credits_used):.2f}")

    if executed_queries:
        lines.append("Executed search queries:")
        for idx, query in enumerate(executed_queries, start=1):
            lines.append(f"{idx}. {query}")

    return "\n".join(lines)


def call_ollama_model(model_name: str, user_text: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    response = chat(
        model=model_name,
        messages=messages,
    )

    return (response.message.content or "").strip() or "No response."


def build_local_prompt(question: str, recent_chat_context: str, shared_pool: str) -> str:
    current_date_context = build_current_date_context()
    return f"""Answer the user's current question using the broad shared search pool below.

Use your own judgment.
If the question asks for ranking, selection, prioritization, or \"top\" items, make your own independent choice from the pool.
Use recent chat context only when it helps resolve a follow-up reference.

{current_date_context}

RECENT CHAT CONTEXT:
{recent_chat_context}

CURRENT USER QUESTION:
{question}

BROAD SHARED SEARCH POOL:
{shared_pool}

Write one direct answer for the user.
Do not mention other models.
"""


def build_final_prompt(question: str, recent_chat_context: str, shared_pool: str, local_answer_1: str, local_answer_2: str) -> str:
    current_date_context = build_current_date_context()
    return f"""{current_date_context}

RECENT CHAT CONTEXT:
{recent_chat_context}

CURRENT USER QUESTION:
{question}

SYNTHESIS GOAL:
Produce a richer final response than the local model drafts. Combine the strongest evidence, resolve gaps or weak spots, and expand with useful explanation when it helps the user. Do not just average the two local answers.

BROAD SHARED SEARCH POOL:
{shared_pool}

LOCAL MODEL 1 ({LOCAL_MODEL_1}) ANSWER:
{local_answer_1}

LOCAL MODEL 2 ({LOCAL_MODEL_2}) ANSWER:
{local_answer_2}
"""


def build_fast_prompt(question: str, recent_chat_context: str, shared_pool: str) -> str:
    current_date_context = build_current_date_context()
    return f"""{current_date_context}

RECENT CHAT CONTEXT:
{recent_chat_context}

CURRENT USER QUESTION:
{question}

FAST ANSWER GOAL:
Use the live search pool to answer quickly and directly. Prefer the specific operational detail the user is likely asking for over broad analysis.

BROAD SHARED SEARCH POOL:
{shared_pool}
"""


def build_fallback_answer(
    question: str,
    shared_pool: str,
    local_answer_1: str,
    local_answer_2: str,
    final_error: str,
) -> str:
    return (
        "Local model 1 summary:\n"
        f"{LOCAL_MODEL_1} returned a draft answer.\n\n"
        "Local model 2 summary:\n"
        f"{LOCAL_MODEL_2} returned a draft answer.\n\n"
        "Differences:\n"
        "Cloud synthesis failed, so this is a fallback response.\n\n"
        "Final answer:\n"
        f"I could not complete the cloud synthesis step.\n"
        f"Cloud error: {final_error}\n\n"
        f"Question:\n{question}\n\n"
        f"Broad shared search pool:\n{shared_pool}\n\n"
        f"{LOCAL_MODEL_1} answer:\n{local_answer_1}\n\n"
        f"{LOCAL_MODEL_2} answer:\n{local_answer_2}"
    )


def build_fast_fallback_answer(question: str, shared_pool: str, final_error: str) -> str:
    return (
        "I could not complete the fast cloud answer step.\n"
        f"Cloud error: {final_error}\n\n"
        f"Question:\n{question}\n\n"
        "Live search pool:\n"
        f"{shared_pool}"
    )


def format_progress_text(stage_state: dict) -> str:
    elapsed = int(time.monotonic() - stage_state["started_at"])
    current_stage = stage_state["stage"]
    completed = stage_state.get("completed_stages", set())
    skipped = stage_state.get("skipped_stages", set())
    detail = stage_state.get("detail", "")

    lines = [
        "Working on it...",
        f"Elapsed: {elapsed}s",
        "",
    ]

    for stage_key, label in PROGRESS_STAGES:
        if stage_key in completed:
            prefix = "[x]"
        elif stage_key in skipped:
            prefix = "[-]"
        elif stage_key == current_stage:
            prefix = "[>]"
        else:
            prefix = "[ ]"
        lines.append(f"{prefix} {label}")

    if detail:
        lines.extend(["", f"Now: {detail}"])

    return "\n".join(lines)


def is_markdown_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def is_table_separator_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "|" in stripped and re.fullmatch(r"\|?[\s:\-|\+]+\|?", stripped):
        return True
    return False


def is_aligned_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    if stripped.startswith(("- ", "* ", "> ")):
        return False
    if re.match(r"^\d+\.\s", stripped):
        return False

    parts = [part for part in re.split(r"\s{2,}", stripped) if part]
    return len(parts) >= 2


def is_table_line(line: str) -> bool:
    return (
        is_markdown_table_line(line)
        or is_table_separator_line(line)
    )


def split_answer_into_segments(answer: str):
    lines = answer.splitlines()
    segments = []

    current_lines = []
    current_kind = None

    def flush_block():
        nonlocal current_lines, current_kind

        if not current_lines:
            return

        text = "\n".join(current_lines).strip("\n")
        if not text.strip():
            current_lines = []
            current_kind = None
            return

        kind = current_kind or "text"

        if kind == "table":
            strong_lines = sum(
                1 for line in current_lines
                if is_markdown_table_line(line) or is_table_separator_line(line)
            )
            general_lines = sum(1 for line in current_lines if is_table_line(line))

            if strong_lines == 0 and general_lines < 2:
                kind = "text"

        segments.append((kind, text))
        current_lines = []
        current_kind = None

    for line in lines:
        if not line.strip():
            if current_lines:
                current_lines.append(line)
            continue

        line_kind = "table" if is_table_line(line) else "text"

        if current_kind is None:
            current_kind = line_kind
            current_lines.append(line)
        elif line_kind == current_kind:
            current_lines.append(line)
        else:
            flush_block()
            current_kind = line_kind
            current_lines.append(line)

    flush_block()

    if not segments and answer.strip():
        return [("text", answer.strip())]

    return segments


def split_text_by_lines(text: str, max_len: int):
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""

    for line in text.splitlines(keepends=True):
        if len(line) > max_len:
            if current.strip():
                chunks.append(current.rstrip())
                current = ""

            remaining = line
            while len(remaining) > max_len:
                chunks.append(remaining[:max_len].rstrip())
                remaining = remaining[max_len:]
            current = remaining
            continue

        if len(current) + len(line) > max_len and current:
            chunks.append(current.rstrip())
            current = line
        else:
            current += line

    if current.strip():
        chunks.append(current.rstrip())

    return chunks




def render_text_chunk_as_html(text: str) -> str:
    parts = []
    cursor = 0

    for match in re.finditer(r"\*\*(.+?)\*\*", text, flags=re.DOTALL):
        parts.append(html.escape(text[cursor:match.start()]))
        parts.append(f"<b>{html.escape(match.group(1).strip())}</b>")
        cursor = match.end()

    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def is_list_item_line(line: str) -> bool:
    return bool(re.match(r"^([-*]|\d+\.)\s", line))


def is_separator_line(line: str) -> bool:
    return bool(re.fullmatch(r"[-*_=]{3,}", line.strip()))


def normalize_display_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if is_separator_line(stripped):
        return ""

    markdown_heading = re.match(r"^#{1,6}\s+(.+?)\s*$", stripped)
    if markdown_heading:
        return f"**{markdown_heading.group(1).strip()}**"

    return stripped


def is_standalone_emphasis_line(line: str) -> bool:
    return bool(re.fullmatch(r"\*\*[^*\n][^*\n]{0,120}\*\*", line.strip()))


def should_preserve_text_line(line: str) -> bool:
    return (
        is_list_item_line(line)
        or line.startswith("> ")
        or is_standalone_emphasis_line(line)
    )


def reflow_text_segment(text: str) -> str:
    output_lines = []
    paragraph_parts = []

    def flush_paragraph():
        if paragraph_parts:
            output_lines.append(" ".join(paragraph_parts))
            paragraph_parts.clear()

    for raw_line in text.splitlines():
        line = normalize_display_line(raw_line)

        if not line:
            flush_paragraph()
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            continue

        if should_preserve_text_line(line):
            flush_paragraph()
            output_lines.append(line)
            continue

        paragraph_parts.append(line)

    flush_paragraph()

    while output_lines and output_lines[-1] == "":
        output_lines.pop()

    return "\n".join(output_lines)


async def send_formatted_answer(update: Update, answer: str) -> None:
    segments = split_answer_into_segments(answer)

    for kind, text in segments:
        normalized_text = text if kind == "table" else reflow_text_segment(text)
        for chunk in split_text_by_lines(normalized_text, MAX_TELEGRAM_CHUNK):
            await update.message.reply_text(
                render_text_chunk_as_html(chunk),
                parse_mode="HTML",
            )


async def safe_edit_status_message(status_message, text: str) -> None:
    try:
        current_text = getattr(status_message, "text", None)
        if current_text == text:
            return
        await status_message.edit_text(text)
    except BadRequest as e:
        if "message is not modified" in str(e).lower():
            return
        logger.warning("Could not edit status message: %s", e)
    except Exception:
        logger.exception("Unexpected error while editing status message.")
        return


async def set_stage(status_message, stage_state: dict, stage_text: str) -> None:
    previous_stage = stage_state.get("stage")
    if previous_stage and previous_stage != stage_text:
        stage_state.setdefault("completed_stages", set()).add(previous_stage)
    stage_state["stage"] = stage_text
    await safe_edit_status_message(
        status_message,
        format_progress_text(stage_state),
    )


async def progress_heartbeat(status_message, stage_state: dict, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=HEARTBEAT_SECONDS)
        except asyncio.TimeoutError:
            await safe_edit_status_message(
                status_message,
                format_progress_text(stage_state),
            )


async def run_search_plan_step(status_message, stage_state: dict, question: str, recent_chat_context: str, timeout_seconds: int):
    stage_state["detail"] = "Building multiple search angles from your question."
    await set_stage(status_message, stage_state, "planning")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(generate_search_plan, question, recent_chat_context),
            timeout=timeout_seconds,
        )
        return result, None
    except asyncio.TimeoutError:
        return None, f"Search planning timed out after {timeout_seconds} seconds."
    except Exception as e:
        return None, f"Search planning failed: {e}"


async def run_search_pool_step(status_message, stage_state: dict, question: str, recent_chat_context: str, plan: dict, timeout_seconds: int):
    stage_state["detail"] = "Running Tavily web searches and collecting cited sources."
    await set_stage(status_message, stage_state, "search_pool")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(build_shared_search_pool, question, recent_chat_context, plan),
            timeout=timeout_seconds,
        )
        return result, None
    except asyncio.TimeoutError:
        return {"pool_text": "", "notices": []}, f"Shared search pool step timed out after {timeout_seconds} seconds."
    except Exception as e:
        return {"pool_text": "", "notices": []}, f"Shared search pool step failed: {e}"


async def run_ollama_step(
    status_message,
    stage_state: dict,
    stage_key: str,
    stage_label: str,
    model_name: str,
    user_text: str,
    system_prompt: str,
    timeout_seconds: int,
):
    stage_state["detail"] = stage_label
    await set_stage(status_message, stage_state, stage_key)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                call_ollama_model,
                model_name,
                user_text,
                system_prompt,
            ),
            timeout=timeout_seconds,
        )
        return result, None
    except asyncio.TimeoutError:
        return "", f"{stage_label} timed out after {timeout_seconds} seconds."
    except Exception as e:
        return "", f"{stage_label} failed: {e}"


async def prepare_shared_pool(
    question: str,
    recent_chat_context: str,
    status_message,
    stage_state: dict,
    allow_no_search: bool = False,
    force_no_search: bool = False,
):
    if allow_no_search and (force_no_search or user_requested_no_search(question)):
        stage_state["detail"] = "Skipping web search because the user explicitly asked not to browse."
        stage_state["completed_stages"].update({"planning", "search_pool"})
        search_result = build_no_search_pool(question, recent_chat_context)
        shared_pool = search_result.get("pool_text", "")
        notices = list(search_result.get("notices", []))
        search_metrics = search_result.get("metrics", {})
    else:
        plan, plan_error = await run_search_plan_step(
            status_message=status_message,
            stage_state=stage_state,
            question=question,
            recent_chat_context=recent_chat_context,
            timeout_seconds=EVIDENCE_TIMEOUT_SECONDS,
        )

        if plan_error or not plan:
            if plan_error:
                logger.warning("Falling back to default search plan: %s", plan_error)
            plan = default_search_plan(question)

        search_result, pool_error = await run_search_pool_step(
            status_message=status_message,
            stage_state=stage_state,
            question=question,
            recent_chat_context=recent_chat_context,
            plan=plan,
            timeout_seconds=EVIDENCE_TIMEOUT_SECONDS,
        )
        shared_pool = search_result.get("pool_text", "")
        notices = list(search_result.get("notices", []))
        search_metrics = search_result.get("metrics", {})

        if pool_error:
            shared_pool = (
                "SEARCH OBJECTIVE\n"
                "- Build a broad shared search pool.\n\n"
                "RECENT CHAT CONTEXT\n"
                f"{recent_chat_context}\n\n"
                "CURRENT USER QUESTION\n"
                f"- {question}\n\n"
                "SEARCH QUERIES\n"
                + "\n".join(f"- {q['query']}" for q in plan.get("queries", []))
                + "\n\nCANDIDATE RESULT POOL\n"
                f"- Search pool generation failed: {pool_error}\n\n"
                "SEARCH QUERY SUMMARIES\n"
                "- None"
            )
            notices = []
            search_metrics = {
                "retrieval_mode": "Search pool unavailable",
                "cache_hit": False,
                "planned_query_count": len(plan.get("queries", [])),
                "planned_queries": [
                    normalize_whitespace(item.get("query", ""))
                    for item in plan.get("queries", [])
                    if normalize_whitespace(item.get("query", ""))
                ],
                "executed_query_count": 0,
                "unique_executed_query_count": 0,
                "executed_queries": [],
                "candidate_count": 0,
            }

    notices.append(format_search_metrics_notice(search_metrics))
    record_search_metrics(search_metrics)
    logger.info(
        "Search metrics | mode=%s planned=%s executed_total=%s executed_unique=%s candidates=%s queries=%s",
        search_metrics.get("retrieval_mode", "unknown"),
        search_metrics.get("planned_query_count", 0),
        search_metrics.get("executed_query_count", 0),
        search_metrics.get("unique_executed_query_count", 0),
        search_metrics.get("candidate_count", 0),
        search_metrics.get("executed_queries", []),
    )
    return shared_pool, notices


async def orchestrate_answer(question: str, recent_chat_context: str, status_message, force_no_search: bool = False):
    stage_state = {
        "stage": "starting",
        "started_at": time.monotonic(),
        "completed_stages": set(),
        "skipped_stages": set(),
        "detail": "Request received. Initializing pipeline.",
    }
    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(progress_heartbeat(status_message, stage_state, stop_event))

    try:
        shared_pool, notices = await prepare_shared_pool(
            question=question,
            recent_chat_context=recent_chat_context,
            status_message=status_message,
            stage_state=stage_state,
            allow_no_search=True,
            force_no_search=force_no_search,
        )

        local_prompt = build_local_prompt(question, recent_chat_context, shared_pool)

        local_answer_1, local_error_1 = await run_ollama_step(
            status_message=status_message,
            stage_state=stage_state,
            stage_key="local_model_1",
            stage_label=f"Asking {LOCAL_MODEL_1} to review the shared evidence.",
            model_name=LOCAL_MODEL_1,
            user_text=local_prompt,
            system_prompt=LOCAL_SYSTEM_PROMPT,
            timeout_seconds=LOCAL_MODEL_TIMEOUT_SECONDS,
        )
        if local_error_1:
            local_answer_1 = f"{LOCAL_MODEL_1} failed: {local_error_1}"

        local_answer_2, local_error_2 = await run_ollama_step(
            status_message=status_message,
            stage_state=stage_state,
            stage_key="local_model_2",
            stage_label=f"Asking {LOCAL_MODEL_2} to review the shared evidence.",
            model_name=LOCAL_MODEL_2,
            user_text=local_prompt,
            system_prompt=LOCAL_SYSTEM_PROMPT,
            timeout_seconds=LOCAL_MODEL_TIMEOUT_SECONDS,
        )
        if local_error_2:
            local_answer_2 = f"{LOCAL_MODEL_2} failed: {local_error_2}"

        final_prompt = build_final_prompt(
            question=question,
            recent_chat_context=recent_chat_context,
            shared_pool=shared_pool,
            local_answer_1=local_answer_1,
            local_answer_2=local_answer_2,
        )

        final_answer, final_error = await run_ollama_step(
            status_message=status_message,
            stage_state=stage_state,
            stage_key="final",
            stage_label=f"Asking {CLOUD_MODEL} to produce the final answer.",
            model_name=CLOUD_MODEL,
            user_text=final_prompt,
            system_prompt=CLOUD_FINAL_SYSTEM_PROMPT,
            timeout_seconds=FINAL_TIMEOUT_SECONDS,
        )

        if final_error:
            logger.warning("Cloud synthesis failed, building fallback answer: %s", final_error)
            final_answer = build_fallback_answer(
                question=question,
                shared_pool=shared_pool,
                local_answer_1=local_answer_1,
                local_answer_2=local_answer_2,
                final_error=final_error,
            )

        stage_state["detail"] = "Formatting the response for Telegram."
        await set_stage(status_message, stage_state, "sending")
        stage_state["completed_stages"].add("sending")
        return final_answer, notices

    finally:
        stop_event.set()
        await heartbeat_task


async def orchestrate_fast_answer(question: str, recent_chat_context: str, status_message):
    stage_state = {
        "stage": "starting",
        "started_at": time.monotonic(),
        "completed_stages": set(),
        "skipped_stages": {"local_model_1", "local_model_2"},
        "detail": "Request received. Preparing fast live-search answer.",
    }
    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(progress_heartbeat(status_message, stage_state, stop_event))

    try:
        shared_pool, notices = await prepare_shared_pool(
            question=question,
            recent_chat_context=recent_chat_context,
            status_message=status_message,
            stage_state=stage_state,
            allow_no_search=False,
            force_no_search=False,
        )

        fast_prompt = build_fast_prompt(question, recent_chat_context, shared_pool)
        stage_state["detail"] = "Skipping local model debate for a concise fast answer."

        final_answer, final_error = await run_ollama_step(
            status_message=status_message,
            stage_state=stage_state,
            stage_key="final",
            stage_label=f"Asking {CLOUD_MODEL} for a concise fast answer.",
            model_name=CLOUD_MODEL,
            user_text=fast_prompt,
            system_prompt=FAST_FINAL_SYSTEM_PROMPT,
            timeout_seconds=FINAL_TIMEOUT_SECONDS,
        )

        if final_error:
            logger.warning("Fast cloud answer failed, building fallback answer: %s", final_error)
            final_answer = build_fast_fallback_answer(
                question=question,
                shared_pool=shared_pool,
                final_error=final_error,
            )

        stage_state["detail"] = "Formatting the response for Telegram."
        await set_stage(status_message, stage_state, "sending")
        stage_state["completed_stages"].add("sending")
        return final_answer, notices

    finally:
        stop_event.set()
        await heartbeat_task


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(START_TEXT)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    history_count = len(get_chat_history(chat_id))

    await update.message.reply_text(
        "Status: bot is running.\n"
        f"Search planner model: {SEARCH_PLANNER_MODEL}\n"
        f"Search retrieval model: {SEARCH_RETRIEVAL_MODEL}\n"
        f"Tavily search depth: {TAVILY_SEARCH_DEPTH}\n"
        f"Local model 1: {LOCAL_MODEL_1}\n"
        f"Local model 2: {LOCAL_MODEL_2}\n"
        f"Cloud model: {CLOUD_MODEL}\n"
        f"Search query limit: {SEARCH_QUERY_LIMIT}\n"
        f"Search results per query: {SEARCH_RESULTS_PER_QUERY}\n"
        f"Total candidate limit: {TOTAL_CANDIDATE_LIMIT}\n"
        f"Rolling memory turns kept per chat: {MAX_HISTORY_TURNS}\n"
        f"Current chat memory count: {history_count}\n"
        f"Heartbeat interval: {HEARTBEAT_SECONDS}s\n"
        f"Evidence timeout: {EVIDENCE_TIMEOUT_SECONDS}s\n"
        f"Local model timeout: {LOCAL_MODEL_TIMEOUT_SECONDS}s\n"
        f"Final timeout: {FINAL_TIMEOUT_SECONDS}s\n\n"
        f"{format_today_search_stats()}"
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    clear_chat_history(chat_id)
    await update.message.reply_text("Cleared this chat's rolling memory.")


async def handle_ask_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    force_no_search: bool = False,
    fast_mode: bool = False,
    usage_text: str = ASK_USAGE,
) -> None:
    user_text = normalize_whitespace(" ".join(context.args))

    if not user_text:
        await update.message.reply_text(usage_text)
        return

    chat_id = update.effective_chat.id
    recent_chat_context = format_recent_chat_context(chat_id)
    chat_lock = get_chat_lock(chat_id)

    if chat_lock.locked():
        await update.message.reply_text(
            "A previous request is still running for this chat. Wait for it to finish or use /clear after it completes."
        )
        return

    status_message = await update.message.reply_text(
        "Working on it...\nStage: Starting...\nElapsed: 0s"
    )

    async with chat_lock:
        try:
            if fast_mode:
                answer, notices = await orchestrate_fast_answer(
                    user_text,
                    recent_chat_context,
                    status_message,
                )
            else:
                answer, notices = await orchestrate_answer(
                    user_text,
                    recent_chat_context,
                    status_message,
                    force_no_search=force_no_search,
                )
            if not answer:
                answer = "I didn't get a usable response."
        except Exception as e:
            logger.exception("Unhandled error while answering request.")
            await safe_edit_status_message(status_message, "Failed.")
            answer = f"Error: {e}"
            notices = []

        for notice in notices:
            await update.message.reply_text(notice)
        save_chat_turn(chat_id, user_text, answer)
        await send_formatted_answer(update, answer)


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_ask_request(update, context)


async def ask_no_search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_ask_request(update, context, force_no_search=True)


async def fast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_ask_request(update, context, fast_mode=True, usage_text=FAST_USAGE)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    load_search_cache()
    load_search_stats()

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("ask", ask_command))
    app.add_handler(CommandHandler("asknosearch", ask_no_search_command))
    app.add_handler(CommandHandler("fast", fast_command))

    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
