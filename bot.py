import os
import re
import json
import html
import time
import io
import uuid
import logging
import asyncio
import threading
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta

from telegram import Update
from telegram.error import BadRequest, TimedOut
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
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

LOCAL_MODEL_1 = os.getenv("LOCAL_MODEL_1", "ministral-3:8b").strip()
LOCAL_MODEL_2 = os.getenv("LOCAL_MODEL_2", "").strip()
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "kimi-k2.5:cloud")
SEARCH_PLANNER_MODEL = os.getenv("SEARCH_PLANNER_MODEL", LOCAL_MODEL_1)
SEARCH_RETRIEVAL_MODEL = os.getenv("SEARCH_RETRIEVAL_MODEL", "tavily-search")
BOT_USERNAME = os.getenv("BOT_USERNAME", "your_bot_username")
TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "basic")
TAVILY_DAILY_CREDIT_LIMIT = float(os.getenv("TAVILY_DAILY_CREDIT_LIMIT", "0"))
EXA_SEARCH_TYPE = os.getenv("EXA_SEARCH_TYPE", "auto")
EXA_HIGHLIGHTS_MAX_CHARS = int(os.getenv("EXA_HIGHLIGHTS_MAX_CHARS", "4000"))
COMFYUI_BASE_URL = os.getenv("COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
COMFYUI_WORKFLOW_PATH = os.getenv("COMFYUI_WORKFLOW_PATH", "comfyui_workflow_api.json")
COMFYUI_PROMPT_NODE_ID = os.getenv("COMFYUI_PROMPT_NODE_ID", "")
COMFYUI_PROMPT_INPUT = os.getenv("COMFYUI_PROMPT_INPUT", "text")
COMFYUI_CLIENT_ID = os.getenv("COMFYUI_CLIENT_ID", f"telegram-bot-{uuid.uuid4()}")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "ComfyUI workflow")
IMAGE_TIMEOUT_SECONDS = int(os.getenv("IMAGE_TIMEOUT_SECONDS", "600"))
IMAGE_PROMPT_CHARS = int(os.getenv("IMAGE_PROMPT_CHARS", "2000"))
IMAGE_PROMPT_PREFIX = os.getenv("IMAGE_PROMPT_PREFIX", "")
IMAGE_PROMPT_SUFFIX = os.getenv("IMAGE_PROMPT_SUFFIX", "")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.20-multi-agent-0309")
XAI_TIMEOUT_SECONDS = int(os.getenv("XAI_TIMEOUT_SECONDS", "1200"))

MAX_TELEGRAM_CHUNK = 3500
MAX_PRE_CHUNK = 3000
HEARTBEAT_SECONDS = int(os.getenv("HEARTBEAT_SECONDS", "15"))
TAVILY_MAX_QUERY_CHARS = int(os.getenv("TAVILY_MAX_QUERY_CHARS", "400"))

EVIDENCE_TIMEOUT_SECONDS = int(os.getenv("EVIDENCE_TIMEOUT_SECONDS", "300"))
LOCAL_MODEL_TIMEOUT_SECONDS = int(os.getenv("LOCAL_MODEL_TIMEOUT_SECONDS", "600"))
FINAL_TIMEOUT_SECONDS = int(os.getenv("FINAL_TIMEOUT_SECONDS", "600"))

SEARCH_QUERY_LIMIT = int(os.getenv("SEARCH_QUERY_LIMIT", "2"))
SEARCH_RESULTS_PER_QUERY = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "20"))
TOTAL_CANDIDATE_LIMIT = int(os.getenv("TOTAL_CANDIDATE_LIMIT", "40"))
LOCAL_CANDIDATE_LIMIT = int(os.getenv("LOCAL_CANDIDATE_LIMIT", "10"))
SEARCH_SNIPPET_LIMIT = int(os.getenv("SEARCH_SNIPPET_LIMIT", "280"))
FETCH_CONTENT_LIMIT = int(os.getenv("FETCH_CONTENT_LIMIT", "900"))
SEARCH_CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "43200"))
SEARCH_CACHE_PATH = os.getenv("SEARCH_CACHE_PATH", "search_cache.json")
SEARCH_STATS_PATH = os.getenv("SEARCH_STATS_PATH", "search_stats.json")

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "3"))
MEMORY_QUESTION_CHARS = int(os.getenv("MEMORY_QUESTION_CHARS", "1200"))
MEMORY_ANSWER_CHARS = int(os.getenv("MEMORY_ANSWER_CHARS", "5000"))

CHAT_MEMORY = {}
GROK_MEMORY = {}
CHAT_LOCKS = {}
CHAT_LAST_REQUEST_TIMINGS = {}
SEARCH_CACHE = {}
SEARCH_CACHE_LOCK = threading.Lock()
SEARCH_STATS = {}
SEARCH_STATS_LOCK = threading.Lock()
LOCAL_MODEL_STAGES = [("local_model_1", LOCAL_MODEL_1)]
if LOCAL_MODEL_2:
    LOCAL_MODEL_STAGES.append(("local_model_2", LOCAL_MODEL_2))

PROGRESS_STAGES = [
    ("starting", "Starting request"),
    ("planning", "Planning search"),
    ("search_pool", "Gathering evidence"),
    *[(stage_key, f"Running {model_name}") for stage_key, model_name in LOCAL_MODEL_STAGES],
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
- one or more local model answers

Your tasks:
- Give your own full final answer, using the broad shared pool, recent chat context when relevant, and any local model answers that are provided.

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

AVAILABLE_COMMANDS_TEXT = (
    "Available commands:\n"
    "/start - Show this help summary.\n"
    "/status - Show config, limits, recent timings, usage, and this command list.\n"
    "/ask <question> - Auto-decide whether live search is needed.\n"
    "/asksearch <question> - Force the full live-search workflow.\n"
    "/asknosearch <question> - Force an answer without internet search.\n"
    "/image <prompt> - Generate an image locally with ComfyUI.\n"
    "/grok <question> - Ask Grok without search tools.\n"
    "/groksearch <question> - Ask Grok with xAI web search tools enabled.\n"
    "/fast <question> - Use live search, skip local review, and return a concise answer.\n"
    "/clear - Clear this chat's rolling memory and pending search decision."
)

START_TEXT = (
    "Bot is working.\n\n"
    f"{AVAILABLE_COMMANDS_TEXT}\n\n"
    f"In groups, use /ask@{BOT_USERNAME}, /asksearch@{BOT_USERNAME}, or /asknosearch@{BOT_USERNAME}."
)

ASK_USAGE = (
    "Usage:\n"
    "/ask your question here\n"
    "/asksearch your question here\n\n"
    "/asknosearch your question here\n\n"
    "Example:\n"
    "/ask explain TLS handshakes\n"
    "/asksearch what are the top 5 news headlines from the last 72 hours?\n"
    "/asknosearch explain TCP vs UDP from general knowledge"
)

FAST_USAGE = (
    "Usage:\n"
    "/fast your question here\n\n"
    "Example:\n"
    "/fast what are the hours for Costco in Seattle today?"
)

IMAGE_USAGE = (
    "Usage:\n"
    "/image your image prompt here\n\n"
    "Example:\n"
    "/image a photorealistic orange tabby cat wearing tiny aviator goggles, cinematic lighting"
)

GROK_USAGE = (
    "Usage:\n"
    "/grok your question here\n"
    "/groksearch your question here\n\n"
    "Example:\n"
    "/grok explain why TLS handshakes matter\n"
    "/groksearch what are the top AI headlines today?"
)

PENDING_SEARCH_DECISION_KEY = "pending_search_decision"
SEARCH_DECISION_USE_SEARCH = "use_search"
SEARCH_DECISION_SKIP_SEARCH = "skip_search"
SEARCH_DECISION_ASK_USER = "ask_user"


class TavilySearchError(RuntimeError):
    pass


class ExaSearchError(RuntimeError):
    pass


class ImageGenerationError(RuntimeError):
    pass


class XaiApiError(RuntimeError):
    pass


def truncate_text(text: str, max_len: int) -> str:
    text = text or ""
    return text if len(text) <= max_len else text[:max_len]


def get_object_field(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


QUERY_STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "do", "does", "for", "from", "get", "give", "had", "has", "have",
    "help", "here", "how", "i", "if", "in", "into", "is", "it", "its", "just", "let",
    "like", "make", "me", "might", "my", "need", "not", "of", "on", "or", "our", "so",
    "take", "that", "the", "their", "them", "there", "these", "this", "those", "to",
    "too", "was", "we", "welcome", "what", "when", "where", "which", "while", "who",
    "will", "with", "would", "you", "your",
}

QUERY_FILLER_PATTERNS = [
    r"^you are\b",
    r"^i am\b",
    r"\bgive me\b",
    r"\btake your time\b",
    r"\bas lengthy an answer as necessary\b",
    r"\bif i am missing anything\b",
    r"\bmake me aware\b",
    r"\bbe objective\b",
    r"\bbe informative\b",
    r"\bi welcome\b",
]

QUERY_INTENT_PATTERNS = [
    (r"\bwhich states?\b", "which states"),
    (r"\bwhich countries?\b", "which countries"),
    (r"\bwhich cities?\b", "which cities"),
    (r"\bwhich counties?\b", "which counties"),
    (r"\bwhich companies?\b", "which companies"),
    (r"\bwhich models?\b", "which models"),
    (r"\bonly state\b", "only state"),
    (r"\bonly states\b", "only states"),
    (r"\bonly country\b", "only country"),
    (r"\bonly countries\b", "only countries"),
    (r"\bonly city\b", "only city"),
    (r"\bonly cities\b", "only cities"),
    (r"\bonly company\b", "only company"),
    (r"\bonly companies\b", "only companies"),
    (r"\bonly model\b", "only model"),
    (r"\bonly models\b", "only models"),
    (r"\bdifference between\b", "difference between"),
    (r"\bcompare\b", "compare"),
    (r"\bcomparison\b", "comparison"),
    (r"\bversus\b", "versus"),
    (r"\bvs\.?\b", "vs"),
    (r"\bunique\b", "unique"),
    (r"\bexclusive\b", "exclusive"),
]


SEARCH_PLANNER_SYSTEM_PROMPT = """You create compact web search plans for a retrieval pipeline.

Return JSON only with this exact schema:
{
  "search_objective": "short sentence",
  "queries": [
    {"query": "compact web query", "purpose": "why this angle helps"}
  ]
}

Rules:
- Produce at most 2 queries.
- Use exactly 1 query for simple questions.
- Use 2 queries only when the question is genuinely complex, multi-constraint, or benefits from a second angle.
- Each query must be a short search-engine query, not a sentence addressed to an assistant.
- Keep each query under 400 characters.
- Focus on high-signal nouns, places, product names, dates, and constraints.
- Preserve the user's logical operators and comparison scope when they matter, especially words or phrases like "only", "unique", "which states", "difference between", "compare", "versus", and "vs".
- Remove filler such as "take your time", "be objective", "give me", or role-play framing.
- Vary the angles when useful: direct answer, official/primary sources, current/fresh status, comparison/background.
- If the user asks for current or recent information, include a freshness-oriented query.
- If the question is already simple, keep the queries simple.
- Do not include Markdown or explanation outside the JSON.
"""


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_query_for_cache(query: str) -> str:
    return normalize_whitespace((query or "").lower())


def split_into_query_clauses(text: str) -> list[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+|[\n;]+", text or "")
    clauses = []
    for part in raw_parts:
        cleaned = normalize_whitespace(part)
        if cleaned:
            clauses.append(cleaned)
    return clauses


def is_query_filler_clause(clause: str) -> bool:
    lowered = normalize_query_for_cache(clause)
    return any(re.search(pattern, lowered) for pattern in QUERY_FILLER_PATTERNS)


def extract_query_intent_phrases(text: str) -> list[str]:
    lowered = normalize_query_for_cache(text)
    phrases = []
    seen = set()

    for pattern, phrase in QUERY_INTENT_PATTERNS:
        if re.search(pattern, lowered) and phrase not in seen:
            phrases.append(phrase)
            seen.add(phrase)

    return phrases


def question_has_comparison_or_exclusivity(text: str) -> bool:
    return bool(extract_query_intent_phrases(text))


def compress_clause_to_keywords(clause: str, max_terms: int = 10) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'./+-]*", clause or "")
    kept = []
    seen = set()

    for token in tokens:
        normalized = token.strip(".,:;!?()[]{}\"'")
        lowered = normalized.lower()
        if not normalized or lowered in QUERY_STOPWORDS:
            continue
        if len(lowered) <= 2 and not re.fullmatch(r"(ca|uk|us|eu|\d+)", lowered):
            continue
        if lowered in seen:
            continue
        kept.append(normalized)
        seen.add(lowered)
        if len(kept) >= max_terms:
            break

    return " ".join(kept)


def compact_search_query(text: str, max_len: int) -> str:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= max_len:
        return cleaned

    clauses = [clause for clause in split_into_query_clauses(cleaned) if not is_query_filler_clause(clause)]
    if not clauses:
        clauses = split_into_query_clauses(cleaned)

    joined = "; ".join(clauses)
    if joined and len(joined) <= max_len:
        return joined

    compressed_clauses = []
    for clause in clauses:
        compressed = compress_clause_to_keywords(clause)
        if compressed:
            compressed_clauses.append(compressed)

    compacted = "; ".join(compressed_clauses)
    if compacted:
        if len(compacted) <= max_len:
            return compacted
        compacted = compacted[:max_len].rstrip(" ,;")
        if compacted:
            return compacted

    return truncate_text(cleaned, max_len).rstrip(" ,;")


def reinforce_query_with_question_intent(question: str, query: str, max_len: int) -> str:
    compacted_query = compact_search_query(query, max_len)
    intent_phrases = extract_query_intent_phrases(question)
    if not intent_phrases:
        return compacted_query

    lowered_query = normalize_query_for_cache(compacted_query)
    missing_phrases = [phrase for phrase in intent_phrases if phrase not in lowered_query]
    if not missing_phrases:
        return compacted_query

    prioritized_prefixes = [phrase for phrase in missing_phrases if phrase.startswith("which ")]
    prioritized_suffixes = [phrase for phrase in missing_phrases if phrase not in prioritized_prefixes]

    combined = normalize_whitespace(
        " ".join(prioritized_prefixes[:1] + [compacted_query] + prioritized_suffixes[:2])
    )
    if len(combined) <= max_len:
        return combined

    allowed_query_len = max(1, max_len - len(" ".join(prioritized_prefixes[:1] + prioritized_suffixes[:2])) - 2)
    shortened_query = compact_search_query(compacted_query, allowed_query_len)
    return normalize_whitespace(
        " ".join(prioritized_prefixes[:1] + [shortened_query] + prioritized_suffixes[:2])
    )[:max_len].rstrip(" ,;")


def derive_comparison_followup_query(question: str, base_query: str, max_len: int) -> str | None:
    normalized_question = normalize_whitespace(question).rstrip(" ?")

    require_match = re.match(
        r"^is\s+.+?\bthe only\s+(state|country|city|county|company|model)\s+that\s+requires\s+(.+)$",
        normalized_question,
        flags=re.IGNORECASE,
    )
    if require_match:
        group = require_match.group(1).lower()
        target = normalize_whitespace(require_match.group(2))
        plural_group = f"{group}s" if not group.endswith("y") else f"{group[:-1]}ies"
        return compact_search_query(f"which {plural_group} require {target}", max_len)

    has_match = re.match(
        r"^is\s+.+?\bthe only\s+(state|country|city|county|company|model)\s+that\s+has\s+(.+)$",
        normalized_question,
        flags=re.IGNORECASE,
    )
    if has_match:
        group = has_match.group(1).lower()
        target = normalize_whitespace(has_match.group(2))
        plural_group = f"{group}s" if not group.endswith("y") else f"{group[:-1]}ies"
        return compact_search_query(f"which {plural_group} have {target}", max_len)

    with_match = re.match(
        r"^is\s+.+?\bthe only\s+(state|country|city|county|company|model)\s+with\s+(.+)$",
        normalized_question,
        flags=re.IGNORECASE,
    )
    if with_match:
        group = with_match.group(1).lower()
        target = normalize_whitespace(with_match.group(2))
        plural_group = f"{group}s" if not group.endswith("y") else f"{group[:-1]}ies"
        return compact_search_query(f"which {plural_group} have {target}", max_len)

    if any(phrase.startswith("which ") for phrase in extract_query_intent_phrases(question)):
        return reinforce_query_with_question_intent(question, base_query, max_len)

    return None


def append_query_terms(base_query: str, suffix: str, max_len: int) -> str:
    base = compact_search_query(base_query, max_len)
    suffix = normalize_whitespace(suffix)
    if not suffix:
        return base

    combined = normalize_whitespace(f"{base} {suffix}")
    if len(combined) <= max_len:
        return combined

    allowed_base_len = max(1, max_len - len(suffix) - 1)
    shortened_base = compact_search_query(base, allowed_base_len)
    return normalize_whitespace(f"{shortened_base} {suffix}")[:max_len].rstrip(" ,;")


def search_query_budget(question: str) -> int:
    text = normalize_whitespace(question)
    if not text:
        return 1

    if question_has_comparison_or_exclusivity(question):
        return max(1, min(SEARCH_QUERY_LIMIT, 2))

    lowered = normalize_query_for_cache(text)
    complexity_score = 0

    if len(text) >= 140:
        complexity_score += 1
    if text.count(",") >= 2 or "\n" in text or ";" in text or "(" in text:
        complexity_score += 1
    if len(re.findall(r"\b(and|or|with|including|requires?|must|compare|versus|vs\.?|options?|cost|budget|fees)\b", lowered)) >= 2:
        complexity_score += 1
    if any(token in lowered for token in ["best", "top", "list", "compare", "recommend", "pros and cons", "tradeoff"]):
        complexity_score += 1
    recommended_budget = 2 if complexity_score >= 2 else 1
    return max(1, min(SEARCH_QUERY_LIMIT, recommended_budget))


def choose_secondary_query_suffix(question: str) -> str:
    lowered = normalize_query_for_cache(question)
    if is_freshness_sensitive_query(question):
        return "latest OR current OR recent"
    if any(term in lowered for term in ["official", "documentation", "docs", "api", "policy", "rules", "law", "regulation"]):
        return "official documentation OR official site"
    return "comparison OR overview OR analysis"


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
        r"\bwithout (an? )?(internet|web|online) search\b",
        r"\bwithout (searching|browsing) (the )?(internet|web|online)\b",
        r"\bdo not browse (the )?(internet|web|online)\b",
        r"\bdon'?t browse (the )?(internet|web|online)\b",
        r"\bno web browsing\b",
        r"\bno browsing\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def parse_yes_no_reply(text: str) -> str | None:
    normalized = normalize_whitespace(text).lower()
    if normalized in {"yes", "y", "search", "use search", "yes search", "web search", "browse"}:
        return SEARCH_DECISION_USE_SEARCH
    if normalized in {"no", "n", "no search", "skip search", "don't search", "do not search"}:
        return SEARCH_DECISION_SKIP_SEARCH
    return None


def decide_search_behavior(question: str, recent_chat_context: str = "") -> dict:
    text = normalize_whitespace(question)
    lowered = text.lower()

    if not text:
        return {"decision": SEARCH_DECISION_SKIP_SEARCH, "reason": "Empty prompt."}

    if user_requested_no_search(text):
        return {"decision": SEARCH_DECISION_SKIP_SEARCH, "reason": "User explicitly asked not to browse."}

    explicit_search_patterns = [
        r"\b(search|look up|lookup|browse|check online|find online|verify online)\b",
        r"\b(latest|most recent|recent|current|currently|today|today's|tonight|yesterday|tomorrow|now|right now|this week|last week|up to date|up-to-date)\b",
        r"\bas of (today|now|this week|this month|this year|20\d{2})\b",
        r"^\bis\b.+\bstill\b",
        r"\b(news|headline|weather|forecast|score|scores|standings|schedule|stock price|share price|exchange rate)\b",
        r"\b(open now|hours|closing time|phone number|address|menu|availability|price today)\b",
        r"\b(quote|quotes|citation|citations|source|sources|link|links)\b",
        r"\b(recommend|best|top|compare|comparison|vs\.?|versus)\b",
        r"\b(buy|purchase|worth it|budget|cost to own|costs to own|which should i buy)\b",
        r"\b(medical|legal|tax|taxes|financial|finance|investment|insurance|laws?|regulations?)\b",
    ]
    if any(re.search(pattern, lowered) for pattern in explicit_search_patterns):
        return {"decision": SEARCH_DECISION_USE_SEARCH, "reason": "Question looks current, comparative, sourced, or high-stakes."}

    stable_knowledge_patterns = [
        r"^(what is|what are)\b",
        r"^(explain|define|summarize|translate|rewrite)\b",
        r"^(how does|how do|why does|why do)\b",
        r"\bfrom general knowledge\b",
        r"\bwithout internet search\b",
        r"\bno search\b",
        r"\bpython\b|\bjavascript\b|\btypescript\b|\breact\b|\bsql\b|\bregex\b|\btls\b|\bhttp\b",
    ]
    if any(re.search(pattern, lowered) for pattern in stable_knowledge_patterns):
        return {"decision": SEARCH_DECISION_SKIP_SEARCH, "reason": "Question looks like evergreen explanation or writing help."}

    factual_lookup_patterns = [
        r"^(who is|who was|when was|when did|where is|which is|is [a-z0-9].+\?)",
        r"\b(released|release date|founded|founded by|ceo|president|governor|capital)\b",
    ]
    if any(re.search(pattern, lowered) for pattern in factual_lookup_patterns):
        return {"decision": SEARCH_DECISION_ASK_USER, "reason": "This looks like a factual lookup that may or may not need live verification."}

    if len(lowered.split()) <= 10:
        return {"decision": SEARCH_DECISION_ASK_USER, "reason": "Short prompt without clear freshness or explanation signals."}

    return {"decision": SEARCH_DECISION_SKIP_SEARCH, "reason": "No strong signal that live web retrieval is necessary."}


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
        "exa_uncached_asks": 0,
        "exa_executed_queries": 0,
        "exa_estimated_cost_usd": 0.0,
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

        if retrieval_mode == "Exa search" and not cache_hit:
            entry["exa_uncached_asks"] = int(entry.get("exa_uncached_asks", 0)) + 1
            entry["exa_executed_queries"] = int(entry.get("exa_executed_queries", 0)) + int(
                metrics.get("executed_query_count", 0)
            )
            entry["exa_estimated_cost_usd"] = float(entry.get("exa_estimated_cost_usd", 0.0)) + float(
                metrics.get("cost_estimate_usd", 0.0)
            )

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
    exa_uncached_asks = int(stats.get("exa_uncached_asks", 0))
    exa_executed_queries = int(stats.get("exa_executed_queries", 0))
    exa_estimated_cost_usd = float(stats.get("exa_estimated_cost_usd", 0.0))
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
        f"Exa uncached asks today: {exa_uncached_asks}\n"
        f"Exa executed queries today: {exa_executed_queries}\n"
        f"Exa estimated cost today: ${exa_estimated_cost_usd:.4f}\n"
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


def normalize_search_plan(plan: dict, question: str) -> dict:
    default_plan = default_search_plan(question)
    query_budget = search_query_budget(question)
    queries = []
    seen_queries = set()

    for item in (plan or {}).get("queries", []):
        raw_query = item.get("query", "") if isinstance(item, dict) else ""
        raw_purpose = item.get("purpose", "") if isinstance(item, dict) else ""
        query = reinforce_query_with_question_intent(question, raw_query, TAVILY_MAX_QUERY_CHARS)
        purpose = normalize_whitespace(raw_purpose) or "general angle"
        cache_key = normalize_query_for_cache(query)
        if not query or not cache_key or cache_key in seen_queries:
            continue
        seen_queries.add(cache_key)
        queries.append({"query": query, "purpose": purpose})
        if len(queries) >= query_budget:
            break

    if not queries:
        queries = default_plan.get("queries", [])[:query_budget]

    return {
        "search_objective": normalize_whitespace((plan or {}).get("search_objective", "")) or default_plan["search_objective"],
        "queries": queries,
    }


def default_search_plan(question: str) -> dict:
    base_query = reinforce_query_with_question_intent(
        question,
        question,
        max(80, TAVILY_MAX_QUERY_CHARS - 64),
    )
    query_budget = search_query_budget(question)
    queries = [{"query": base_query, "purpose": "direct query"}]

    if query_budget >= 2:
        secondary_query = derive_comparison_followup_query(question, base_query, TAVILY_MAX_QUERY_CHARS)
        if not secondary_query:
            secondary_query = append_query_terms(
                base_query,
                choose_secondary_query_suffix(question),
                TAVILY_MAX_QUERY_CHARS,
            )
        queries.append(
            {
                "query": secondary_query,
                "purpose": "secondary angle",
            }
        )

    return {
        "search_objective": "Build a broad, multi-angle search pool for the user question.",
        "queries": queries,
    }


def build_search_planner_prompt(question: str, recent_chat_context: str) -> str:
    current_date_context = build_current_date_context()
    return (
        "Create a compact web search plan for this user request.\n\n"
        f"{current_date_context}\n\n"
        "RECENT CHAT CONTEXT\n"
        f"{recent_chat_context or 'None'}\n\n"
        "CURRENT USER QUESTION\n"
        f"{question}\n\n"
        "Return JSON only."
    )


def generate_search_plan_with_model(question: str, recent_chat_context: str, planner_model: str) -> dict:
    planner_prompt = build_search_planner_prompt(question, recent_chat_context)
    planner_response = call_ollama_model(
        planner_model,
        planner_prompt,
        SEARCH_PLANNER_SYSTEM_PROMPT,
    )
    parsed_plan = extract_json_object(planner_response)
    if not isinstance(parsed_plan, dict):
        raise ValueError("Planner response was not a JSON object.")
    return normalize_search_plan(parsed_plan, question)


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


def get_exa_client():
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ExaSearchError("EXA_API_KEY is not set.")

    try:
        from exa_py import Exa
    except ImportError as e:
        raise ExaSearchError("exa-py is not installed. Run `pip install exa-py`.") from e

    return Exa(api_key=api_key)


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


def extract_xai_response_text(response: dict) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts = []
    for output_item in response.get("output", []) or []:
        if not isinstance(output_item, dict):
            continue
        content_items = output_item.get("content", [])
        if isinstance(content_items, str):
            parts.append(content_items)
            continue
        if not isinstance(content_items, list):
            continue
        for content_item in content_items:
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

    if parts:
        return "\n\n".join(parts).strip()

    choices = response.get("choices", [])
    if choices and isinstance(choices, list):
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def call_xai_model(question: str, grok_context: str = "None", use_search: bool = False) -> str:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise XaiApiError("XAI_API_KEY is not set.")

    system_prompt = (
        "You are a precise, helpful assistant inside a Telegram bot. "
        "Answer directly and naturally. If search tools are available, use them when helpful for current or factual claims. "
        "Do not mention internal implementation details."
    )
    payload = {
        "model": XAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "RECENT GROK-ONLY CONTEXT:\n"
                    f"{grok_context or 'None'}\n\n"
                    "CURRENT USER QUESTION:\n"
                    f"{question}"
                ),
            },
        ],
        "store": False,
    }
    if use_search:
        payload["tools"] = [{"type": "web_search"}]

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{XAI_BASE_URL}/responses",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=XAI_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise XaiApiError(f"xAI API HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise XaiApiError(f"xAI API connection error: {e}") from e

    try:
        response = json.loads(body)
    except json.JSONDecodeError as e:
        raise XaiApiError(f"xAI API returned invalid JSON: {e}") from e

    answer = extract_xai_response_text(response)
    if not answer:
        raise XaiApiError("xAI API returned no usable text.")
    return answer


def load_comfyui_workflow() -> dict:
    if not COMFYUI_WORKFLOW_PATH:
        raise ImageGenerationError("COMFYUI_WORKFLOW_PATH is not set.")
    if not os.path.exists(COMFYUI_WORKFLOW_PATH):
        raise ImageGenerationError(
            f"ComfyUI workflow file not found: {COMFYUI_WORKFLOW_PATH}. "
            "Export an API-format workflow from ComfyUI and set COMFYUI_WORKFLOW_PATH."
        )

    try:
        with open(COMFYUI_WORKFLOW_PATH, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        raise ImageGenerationError(f"Could not load ComfyUI workflow: {e}") from e

    if not isinstance(workflow, dict):
        raise ImageGenerationError("ComfyUI workflow must be a JSON object in API format.")
    return workflow


def replace_prompt_placeholders(value, prompt: str):
    if isinstance(value, str):
        return value.replace("{{prompt}}", prompt)
    if isinstance(value, list):
        return [replace_prompt_placeholders(item, prompt) for item in value]
    if isinstance(value, dict):
        return {key: replace_prompt_placeholders(item, prompt) for key, item in value.items()}
    return value


def infer_comfyui_prompt_target(workflow: dict) -> tuple[str, str] | None:
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        if node.get("class_type") == "PrimitiveStringMultiline" and isinstance(inputs.get("value"), str):
            return str(node_id), "value"

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        if node.get("class_type") == "CLIPTextEncode" and isinstance(inputs.get("text"), str):
            return str(node_id), "text"

    return None


def apply_comfyui_prompt(workflow: dict, prompt: str) -> dict:
    workflow = replace_prompt_placeholders(workflow, prompt)

    prompt_node_id = COMFYUI_PROMPT_NODE_ID
    prompt_input = COMFYUI_PROMPT_INPUT
    if not prompt_node_id:
        inferred_target = infer_comfyui_prompt_target(workflow)
        if inferred_target:
            prompt_node_id, prompt_input = inferred_target

    if prompt_node_id:
        node = workflow.get(prompt_node_id)
        if not isinstance(node, dict):
            raise ImageGenerationError(f"ComfyUI prompt node {prompt_node_id!r} was not found in the workflow.")
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            raise ImageGenerationError(f"Workflow node {prompt_node_id!r} does not have an inputs object.")
        inputs[prompt_input] = prompt

    return workflow


def comfyui_api_json(path: str, payload: dict | None = None, timeout: int = 60) -> dict:
    url = f"{COMFYUI_BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise ImageGenerationError(f"ComfyUI API HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise ImageGenerationError(f"ComfyUI API connection error: {e}") from e
    except json.JSONDecodeError as e:
        raise ImageGenerationError(f"ComfyUI API returned invalid JSON: {e}") from e


def queue_comfyui_prompt(workflow: dict) -> str:
    response = comfyui_api_json(
        "/prompt",
        {
            "prompt": workflow,
            "client_id": COMFYUI_CLIENT_ID,
        },
    )
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise ImageGenerationError(f"ComfyUI did not return a prompt_id: {response}")
    return prompt_id


def build_image_prompt(user_prompt: str) -> str:
    parts = [
        normalize_whitespace(IMAGE_PROMPT_PREFIX).strip(),
        normalize_whitespace(user_prompt).strip(),
        normalize_whitespace(IMAGE_PROMPT_SUFFIX).strip(),
    ]
    return truncate_text(", ".join(part for part in parts if part), IMAGE_PROMPT_CHARS).strip()


def find_comfyui_output_images(history_item: dict) -> list[dict]:
    outputs = history_item.get("outputs", {})
    if not isinstance(outputs, dict):
        return []

    output_images = []
    for output in outputs.values():
        images = output.get("images", []) if isinstance(output, dict) else []
        for image in images:
            if isinstance(image, dict) and image.get("filename"):
                output_images.append(image)
    return output_images


def fetch_comfyui_image(image_info: dict) -> bytes:
    params = urllib.parse.urlencode(
        {
            "filename": image_info.get("filename", ""),
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        }
    )
    url = f"{COMFYUI_BASE_URL}/view?{params}"

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise ImageGenerationError(f"ComfyUI image fetch HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise ImageGenerationError(f"ComfyUI image fetch connection error: {e}") from e


def generate_comfyui_images(prompt: str) -> list[bytes]:
    clean_prompt = build_image_prompt(prompt)
    if not clean_prompt:
        raise ImageGenerationError("Image prompt is empty.")

    workflow = apply_comfyui_prompt(load_comfyui_workflow(), clean_prompt)
    prompt_id = queue_comfyui_prompt(workflow)
    deadline = time.monotonic() + IMAGE_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        history = comfyui_api_json(f"/history/{urllib.parse.quote(prompt_id)}", timeout=30)
        history_item = history.get(prompt_id)
        if isinstance(history_item, dict):
            image_infos = find_comfyui_output_images(history_item)
            if image_infos:
                return [fetch_comfyui_image(image_info) for image_info in image_infos]
        time.sleep(1)

    raise ImageGenerationError(f"ComfyUI image generation timed out after {IMAGE_TIMEOUT_SECONDS}s.")


def get_chat_history(chat_id: int):
    return CHAT_MEMORY.get(chat_id, [])


def get_grok_history(chat_id: int):
    return GROK_MEMORY.get(chat_id, [])


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


def save_grok_turn(chat_id: int, question: str, final_answer: str) -> None:
    history = GROK_MEMORY.setdefault(chat_id, [])
    history.append(
        {
            "question": truncate_text(question.strip(), MEMORY_QUESTION_CHARS),
            "final_answer": truncate_text(final_answer.strip(), MEMORY_ANSWER_CHARS),
            "timestamp": int(time.time()),
        }
    )
    if len(history) > MAX_HISTORY_TURNS:
        GROK_MEMORY[chat_id] = history[-MAX_HISTORY_TURNS:]


def clear_chat_history(chat_id: int) -> None:
    CHAT_MEMORY.pop(chat_id, None)
    GROK_MEMORY.pop(chat_id, None)
    CHAT_LOCKS.pop(chat_id, None)
    CHAT_LAST_REQUEST_TIMINGS.pop(chat_id, None)


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


def format_recent_grok_context(chat_id: int) -> str:
    history = get_grok_history(chat_id)
    if not history:
        return "None"

    lines = []
    for idx, turn in enumerate(history[-MAX_HISTORY_TURNS:], start=1):
        lines.append(f"GROK TURN {idx}")
        lines.append(f"User question: {turn.get('question', '')}")
        lines.append("Grok answer:")
        lines.append(turn.get("final_answer", ""))
        lines.append("")

    return "\n".join(lines).strip()


def save_last_request_timing(chat_id: int, mode: str, total_elapsed: float, stage_state: dict) -> None:
    CHAT_LAST_REQUEST_TIMINGS[chat_id] = {
        "mode": mode,
        "total_elapsed": max(0.0, float(total_elapsed)),
        "stage_summary": format_stage_timing_summary(stage_state),
        "prompt_summary": format_prompt_metrics_summary(stage_state),
        "recorded_at": int(time.time()),
    }


def format_last_request_timing(chat_id: int) -> str:
    timing = CHAT_LAST_REQUEST_TIMINGS.get(chat_id)
    if not isinstance(timing, dict):
        return "Most recent request timing: none yet."

    mode = timing.get("mode", "unknown")
    total_elapsed = float(timing.get("total_elapsed", 0.0))
    stage_summary = timing.get("stage_summary", "no stage timings recorded")
    prompt_summary = timing.get("prompt_summary", "no prompt metrics recorded")
    recorded_at = int(timing.get("recorded_at", 0))
    recorded_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(recorded_at)) if recorded_at else "unknown"

    return (
        "Most recent request timing\n"
        f"Mode: {mode}\n"
        f"Total elapsed: {total_elapsed:.2f}s\n"
        f"Stages: {stage_summary}\n"
        f"Prompts: {prompt_summary}\n"
        f"Recorded at: {recorded_text}"
    )


def generate_search_plan(question: str, recent_chat_context: str) -> dict:
    planner_model = normalize_whitespace(SEARCH_PLANNER_MODEL)
    if not planner_model or planner_model.lower() == "built-in":
        return normalize_search_plan(default_search_plan(question), question)

    return generate_search_plan_with_model(question, recent_chat_context, planner_model)


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


def infer_exa_category(question: str, query: str) -> str | None:
    combined = normalize_query_for_cache(f"{question} {query}")
    if any(term in combined for term in ["news", "headline", "breaking", "today", "latest", "current", "recent"]):
        return "news"
    if any(term in combined for term in ["paper", "research", "study", "arxiv", "publication", "journal"]):
        return "research paper"
    return None


def exa_search_pool(question: str, recent_chat_context: str, plan: dict) -> dict:
    _ = recent_chat_context
    exa = get_exa_client()

    all_candidates = []
    deduped_by_url = {}
    search_errors = []
    executed_queries = []
    query_summaries = []
    cost_estimate_usd = 0.0

    for idx, query_item in enumerate(plan.get("queries", [])[:SEARCH_QUERY_LIMIT], start=1):
        query_text = compact_search_query(query_item.get("query", ""), TAVILY_MAX_QUERY_CHARS)
        purpose = normalize_whitespace(query_item.get("purpose", "")) or "general angle"
        if not query_text:
            continue

        executed_queries.append(query_text)
        payload = {
            "query": query_text,
            "type": normalize_whitespace(EXA_SEARCH_TYPE) or "auto",
            "num_results": max(1, min(SEARCH_RESULTS_PER_QUERY, 100)),
            "contents": {
                "highlights": {
                    "max_characters": max(200, EXA_HIGHLIGHTS_MAX_CHARS),
                }
            },
        }

        category = infer_exa_category(question, query_text)
        if category:
            payload["category"] = category

        try:
            response = exa.search(**payload)
        except Exception as e:
            logger.warning("Exa search failed for %r: %s", query_text, e)
            search_errors.append(f"Q{idx} failed: {e}")
            continue

        cost_dollars = get_object_field(response, "cost_dollars")
        if cost_dollars is None:
            cost_dollars = get_object_field(response, "costDollars")
        cost_total = get_object_field(cost_dollars, "total", 0.0)
        cost_estimate_usd += float(cost_total or 0.0)

        output = get_object_field(response, "output")
        output_content_raw = get_object_field(output, "content", "")
        if isinstance(output_content_raw, (dict, list)):
            output_content_raw = json.dumps(output_content_raw, ensure_ascii=True)
        output_content = normalize_whitespace(str(output_content_raw))
        if output_content:
            query_summaries.append(
                {
                    "query": query_text,
                    "purpose": purpose,
                    "topic": category or "",
                    "summary": truncate_text(output_content, FETCH_CONTENT_LIMIT),
                }
            )

        for rank, result in enumerate(get_object_field(response, "results", []) or [], start=1):
            title = normalize_whitespace(get_object_field(result, "title", ""))
            url = normalize_whitespace(get_object_field(result, "url", ""))
            highlights = get_object_field(result, "highlights", []) or []
            highlight_text = normalize_whitespace(" ".join(normalize_whitespace(item) for item in highlights if item))
            summary_text = normalize_whitespace(get_object_field(result, "summary", ""))
            text_snippet = normalize_whitespace(get_object_field(result, "text", ""))
            snippet = truncate_text(highlight_text or summary_text or text_snippet, SEARCH_SNIPPET_LIMIT)

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

    if executed_queries and search_errors and len(search_errors) >= len(executed_queries) and not all_candidates and not query_summaries:
        raise ExaSearchError("; ".join(search_errors))

    return {
        "summaries": query_summaries,
        "results": all_candidates,
        "executed_queries": executed_queries,
        "errors": search_errors,
        "cost_estimate_usd": cost_estimate_usd,
    }


def tavily_search_pool(question: str, recent_chat_context: str, plan: dict) -> dict:
    _ = recent_chat_context
    if not os.getenv("TAVILY_API_KEY"):
        raise TavilySearchError("TAVILY_API_KEY is not set.")

    all_candidates = []
    deduped_by_url = {}
    search_errors = []
    executed_queries = []
    query_summaries = []
    credits_used = 0.0

    for idx, query_item in enumerate(plan.get("queries", [])[:SEARCH_QUERY_LIMIT], start=1):
        query_text = compact_search_query(query_item.get("query", ""), TAVILY_MAX_QUERY_CHARS)
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

    if executed_queries and search_errors and len(search_errors) >= len(executed_queries) and not all_candidates and not query_summaries:
        raise TavilySearchError("; ".join(search_errors))

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


def build_search_pool_text(
    *,
    question: str,
    recent_chat_context: str,
    plan: dict,
    retrieval_label: str,
    current_date_context: str,
    search_summaries: list,
    all_candidates: list,
    search_errors: list,
    candidate_limit: int,
    snippet_limit: int = SEARCH_SNIPPET_LIMIT,
) -> str:
    queries = plan.get("queries", [])[:SEARCH_QUERY_LIMIT]
    limited_candidates = list(all_candidates[:max(0, candidate_limit)])

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
    if limited_candidates:
        for idx, item in enumerate(limited_candidates, start=1):
            source = item.get("source") or domain_from_url(item.get("url", ""))
            lines.append(
                f"[{idx:02d}] Source={source} | Title={item.get('title', 'Untitled result')}"
            )
            lines.append(f"URL: {item.get('url', '')}")
            if item.get("snippet"):
                lines.append(f"Snippet: {truncate_text(item['snippet'], snippet_limit)}")
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

    return "\n".join(lines).strip()


def build_shared_search_pool(question: str, recent_chat_context: str, plan: dict) -> dict:
    current_date_context = build_current_date_context()
    notices = []
    configured_retrieval_model = normalize_query_for_cache(SEARCH_RETRIEVAL_MODEL)
    retrieval_label = "Exa search" if configured_retrieval_model == "exa-search" else "Tavily search"
    cache_hit = False
    credits_used = 0.0
    cost_estimate_usd = 0.0

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
                "cost_estimate_usd": 0.0,
            }

    def fallback_to_tavily(search_error: Exception, notice_message: str, fallback_unavailable_notice: str, fallback_error_prefix: str):
        nonlocal retrieval_label
        retrieval_label = "Tavily search"
        try:
            notices.append(notice_message)
            return tavily_search_pool(question, recent_chat_context, plan)
        except Exception as fallback_error:
            logger.warning("Tavily fallback search failed after Exa error: %s", fallback_error)
            notices.append(fallback_unavailable_notice)
            return fallback_to_ollama(
                search_error=search_error,
                notice_message="Exa and Tavily search both failed. Fell back to Ollama web search.",
                fallback_unavailable_notice=(
                    "Exa search failed, Tavily fallback failed, and Ollama web-search fallback was unavailable. "
                    "Check EXA_API_KEY, TAVILY_API_KEY, or OLLAMA_API_KEY."
                ),
                fallback_error_prefix=fallback_error_prefix,
            )

    search_pool_cache_key = build_search_pool_cache_key(question, recent_chat_context, plan)
    cached_search_response = get_search_cache_entry(search_pool_cache_key)
    if cached_search_response:
        search_response = cached_search_response
        cached_model = normalize_query_for_cache(cached_search_response.get("provider", configured_retrieval_model))
        retrieval_label = "Exa search (cached)" if cached_model == "exa-search" else "Tavily search (cached)"
        cache_hit = True
    else:
        if configured_retrieval_model == "exa-search":
            try:
                search_response = exa_search_pool(question, recent_chat_context, plan)
                set_search_cache_entry(
                    search_pool_cache_key,
                    {
                        "provider": "exa-search",
                        "summaries": search_response.get("summaries", []) or [],
                        "results": search_response.get("results", []) or [],
                        "executed_queries": search_response.get("executed_queries", []) or [],
                        "errors": search_response.get("errors", []) or [],
                        "cost_estimate_usd": search_response.get("cost_estimate_usd", 0.0) or 0.0,
                    },
                )
            except Exception as e:
                logger.warning("Exa search failed for question %r: %s", question, e)
                search_response = fallback_to_tavily(
                    search_error=e,
                    notice_message="Exa search failed. Fell back to Tavily web search.",
                    fallback_unavailable_notice="Exa search failed, and Tavily fallback was unavailable. Trying Ollama web-search fallback.",
                    fallback_error_prefix="Exa search failed",
                )
        else:
            try:
                search_response = tavily_search_pool(question, recent_chat_context, plan)
                set_search_cache_entry(
                    search_pool_cache_key,
                    {
                        "provider": "tavily-search",
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
    cost_estimate_usd = float(search_response.get("cost_estimate_usd", 0.0) or 0.0)

    pool_text = build_search_pool_text(
        question=question,
        recent_chat_context=recent_chat_context,
        plan=plan,
        retrieval_label=retrieval_label,
        current_date_context=current_date_context,
        search_summaries=search_summaries,
        all_candidates=all_candidates,
        search_errors=search_errors,
        candidate_limit=TOTAL_CANDIDATE_LIMIT,
    )
    local_pool_text = build_search_pool_text(
        question=question,
        recent_chat_context=recent_chat_context,
        plan=plan,
        retrieval_label=retrieval_label,
        current_date_context=current_date_context,
        search_summaries=search_summaries,
        all_candidates=all_candidates,
        search_errors=search_errors,
        candidate_limit=min(LOCAL_CANDIDATE_LIMIT, TOTAL_CANDIDATE_LIMIT),
        snippet_limit=min(180, SEARCH_SNIPPET_LIMIT),
    )

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
        "planned_query_count": len(plan.get("queries", [])[:SEARCH_QUERY_LIMIT]),
        "planned_queries": [
            normalize_whitespace(item.get("query", ""))
            for item in plan.get("queries", [])[:SEARCH_QUERY_LIMIT]
            if normalize_whitespace(item.get("query", ""))
        ],
        "executed_query_count": len(executed_queries),
        "unique_executed_query_count": len(unique_executed_queries),
        "executed_queries": unique_executed_queries,
        "candidate_count": len(all_candidates),
        "credits_used": credits_used,
        "cost_estimate_usd": cost_estimate_usd,
    }

    return {
        "pool_text": pool_text,
        "local_pool_text": local_pool_text,
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

    cost_estimate_usd = metrics.get("cost_estimate_usd")
    if cost_estimate_usd is not None and float(cost_estimate_usd) > 0:
        lines.append(f"Exa estimated cost: ${float(cost_estimate_usd):.4f}")

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
    local_sections = [
        f"LOCAL MODEL 1 ({LOCAL_MODEL_1}) ANSWER:\n{local_answer_1}",
    ]
    if LOCAL_MODEL_2:
        local_sections.append(f"LOCAL MODEL 2 ({LOCAL_MODEL_2}) ANSWER:\n{local_answer_2}")

    return f"""{current_date_context}

RECENT CHAT CONTEXT:
{recent_chat_context}

CURRENT USER QUESTION:
{question}

SYNTHESIS GOAL:
Produce a richer final response than the local model drafts. Combine the strongest evidence, resolve gaps or weak spots, and expand with useful explanation when it helps the user. Do not just average the local model drafts.

BROAD SHARED SEARCH POOL:
{shared_pool}

{"\n\n".join(local_sections)}
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
    sections = [
        "Local model 1 summary:\n"
        f"{LOCAL_MODEL_1} returned a draft answer."
    ]
    if LOCAL_MODEL_2:
        sections.append(
            "Local model 2 summary:\n"
            f"{LOCAL_MODEL_2} returned a draft answer."
        )
    sections.extend(
        [
            "Differences:\n"
            "Cloud synthesis failed, so this is a fallback response.",
            "Final answer:\n"
            f"I could not complete the cloud synthesis step.\n"
            f"Cloud error: {final_error}",
            f"Question:\n{question}",
            f"Broad shared search pool:\n{shared_pool}",
            f"{LOCAL_MODEL_1} answer:\n{local_answer_1}",
        ]
    )
    if LOCAL_MODEL_2:
        sections.append(f"{LOCAL_MODEL_2} answer:\n{local_answer_2}")
    return "\n\n".join(sections)


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


def record_stage_duration(stage_state: dict, stage_key: str, elapsed_seconds: float) -> None:
    stage_state.setdefault("durations", {})[stage_key] = max(0.0, float(elapsed_seconds))


def format_stage_timing_summary(stage_state: dict) -> str:
    durations = stage_state.get("durations", {}) or {}
    parts = []
    for stage_key, label in PROGRESS_STAGES:
        if stage_key in durations:
            parts.append(f"{stage_key}={durations[stage_key]:.2f}s")
    return ", ".join(parts) if parts else "no stage timings recorded"


def record_prompt_metrics(stage_state: dict, key: str, text: str) -> None:
    metrics = stage_state.setdefault("prompt_metrics", {})
    content = text or ""
    metrics[key] = {
        "chars": len(content),
        "lines": content.count("\n") + 1 if content else 0,
    }


def format_prompt_metrics_summary(stage_state: dict) -> str:
    metrics = stage_state.get("prompt_metrics", {}) or {}
    ordered_keys = [
        "shared_pool",
        "local_shared_pool",
        "local_prompt",
        "final_prompt",
        "fast_prompt",
        "local_answer_1",
        "local_answer_2",
    ]
    parts = []

    for key in ordered_keys:
        item = metrics.get(key)
        if not isinstance(item, dict):
            continue
        parts.append(f"{key}={int(item.get('chars', 0))}c/{int(item.get('lines', 0))}l")

    return ", ".join(parts) if parts else "no prompt metrics recorded"


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


async def reply_text_in_chunks(message, text: str, parse_mode: str | None = None, max_len: int = MAX_TELEGRAM_CHUNK) -> None:
    normalized = (text or "").strip()
    if not normalized:
        return

    for chunk in split_text_by_lines(normalized, max_len):
        await message.reply_text(chunk, parse_mode=parse_mode)




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
        for chunk in split_text_by_lines(normalized_text, MAX_PRE_CHUNK):
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
    except TimedOut:
        logger.warning("Status message edit timed out; keeping the previous progress message.")
        return
    except BadRequest as e:
        error_text = str(e).lower()
        if "message is not modified" in error_text:
            return
        if "message is too long" in error_text:
            shortened = truncate_text(text, MAX_TELEGRAM_CHUNK - 16).rstrip() + "\n\n[truncated]"
            if shortened != text:
                try:
                    await status_message.edit_text(shortened)
                    return
                except Exception:
                    logger.warning("Could not edit overlong status message even after truncation.")
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
    started_at = time.monotonic()

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(generate_search_plan, question, recent_chat_context),
            timeout=timeout_seconds,
        )
        elapsed = time.monotonic() - started_at
        record_stage_duration(stage_state, "planning", elapsed)
        logger.info("Stage timing | stage=planning elapsed=%.2fs", elapsed)
        return result, None
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - started_at
        record_stage_duration(stage_state, "planning", elapsed)
        logger.warning("Stage timing | stage=planning outcome=timeout elapsed=%.2fs", elapsed)
        return None, f"Search planning timed out after {timeout_seconds} seconds."
    except Exception as e:
        elapsed = time.monotonic() - started_at
        record_stage_duration(stage_state, "planning", elapsed)
        logger.warning("Stage timing | stage=planning outcome=error elapsed=%.2fs error=%s", elapsed, e)
        return None, f"Search planning failed: {e}"


async def run_search_pool_step(status_message, stage_state: dict, question: str, recent_chat_context: str, plan: dict, timeout_seconds: int):
    stage_state["detail"] = "Running Tavily web searches and collecting cited sources."
    await set_stage(status_message, stage_state, "search_pool")
    started_at = time.monotonic()

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(build_shared_search_pool, question, recent_chat_context, plan),
            timeout=timeout_seconds,
        )
        elapsed = time.monotonic() - started_at
        record_stage_duration(stage_state, "search_pool", elapsed)
        logger.info("Stage timing | stage=search_pool elapsed=%.2fs", elapsed)
        return result, None
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - started_at
        record_stage_duration(stage_state, "search_pool", elapsed)
        logger.warning("Stage timing | stage=search_pool outcome=timeout elapsed=%.2fs", elapsed)
        return {"pool_text": "", "notices": []}, f"Shared search pool step timed out after {timeout_seconds} seconds."
    except Exception as e:
        elapsed = time.monotonic() - started_at
        record_stage_duration(stage_state, "search_pool", elapsed)
        logger.warning("Stage timing | stage=search_pool outcome=error elapsed=%.2fs error=%s", elapsed, e)
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
    started_at = time.monotonic()

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
        elapsed = time.monotonic() - started_at
        record_stage_duration(stage_state, stage_key, elapsed)
        logger.info("Stage timing | stage=%s model=%s elapsed=%.2fs", stage_key, model_name, elapsed)
        return result, None
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - started_at
        record_stage_duration(stage_state, stage_key, elapsed)
        logger.warning("Stage timing | stage=%s model=%s outcome=timeout elapsed=%.2fs", stage_key, model_name, elapsed)
        return "", f"{stage_label} timed out after {timeout_seconds} seconds."
    except Exception as e:
        elapsed = time.monotonic() - started_at
        record_stage_duration(stage_state, stage_key, elapsed)
        logger.warning("Stage timing | stage=%s model=%s outcome=error elapsed=%.2fs error=%s", stage_key, model_name, elapsed, e)
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
        local_shared_pool = search_result.get("local_pool_text", shared_pool)
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
        local_shared_pool = search_result.get("local_pool_text", shared_pool)
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
            local_shared_pool = shared_pool
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
    return shared_pool, local_shared_pool, notices


async def orchestrate_answer(question: str, recent_chat_context: str, status_message, force_no_search: bool = False):
    stage_state = {
        "stage": "starting",
        "started_at": time.monotonic(),
        "completed_stages": set(),
        "skipped_stages": {"local_model_2"} if not LOCAL_MODEL_2 else set(),
        "detail": "Request received. Initializing pipeline.",
    }
    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(progress_heartbeat(status_message, stage_state, stop_event))

    try:
        shared_pool, local_shared_pool, notices = await prepare_shared_pool(
            question=question,
            recent_chat_context=recent_chat_context,
            status_message=status_message,
            stage_state=stage_state,
            allow_no_search=True,
            force_no_search=force_no_search,
        )
        record_prompt_metrics(stage_state, "shared_pool", shared_pool)
        record_prompt_metrics(stage_state, "local_shared_pool", local_shared_pool)

        local_prompt = build_local_prompt(question, recent_chat_context, local_shared_pool)
        record_prompt_metrics(stage_state, "local_prompt", local_prompt)
        logger.info(
            "Prompt metrics | mode=full shared_pool=%sc local_shared_pool=%sc local_prompt=%sc",
            len(shared_pool or ""),
            len(local_shared_pool or ""),
            len(local_prompt or ""),
        )

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
        record_prompt_metrics(stage_state, "local_answer_1", local_answer_1)

        if LOCAL_MODEL_2:
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
        else:
            local_answer_2 = "Local model 2 is disabled."
        record_prompt_metrics(stage_state, "local_answer_2", local_answer_2)

        final_prompt = build_final_prompt(
            question=question,
            recent_chat_context=recent_chat_context,
            shared_pool=shared_pool,
            local_answer_1=local_answer_1,
            local_answer_2=local_answer_2,
        )
        record_prompt_metrics(stage_state, "final_prompt", final_prompt)
        logger.info(
            "Prompt metrics | mode=full final_prompt=%sc local_answer_1=%sc local_answer_2=%sc",
            len(final_prompt or ""),
            len(local_answer_1 or ""),
            len(local_answer_2 or ""),
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
        return final_answer, notices, stage_state

    finally:
        stop_event.set()
        await heartbeat_task
        total_elapsed = time.monotonic() - stage_state["started_at"]
        logger.info(
            "Request timing | mode=full total=%.2fs | %s | prompts=%s",
            total_elapsed,
            format_stage_timing_summary(stage_state),
            format_prompt_metrics_summary(stage_state),
        )


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
        shared_pool, _local_shared_pool, notices = await prepare_shared_pool(
            question=question,
            recent_chat_context=recent_chat_context,
            status_message=status_message,
            stage_state=stage_state,
            allow_no_search=False,
            force_no_search=False,
        )
        record_prompt_metrics(stage_state, "shared_pool", shared_pool)

        fast_prompt = build_fast_prompt(question, recent_chat_context, shared_pool)
        record_prompt_metrics(stage_state, "fast_prompt", fast_prompt)
        logger.info(
            "Prompt metrics | mode=fast shared_pool=%sc fast_prompt=%sc",
            len(shared_pool or ""),
            len(fast_prompt or ""),
        )
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
        return final_answer, notices, stage_state

    finally:
        stop_event.set()
        await heartbeat_task
        total_elapsed = time.monotonic() - stage_state["started_at"]
        logger.info(
            "Request timing | mode=fast total=%.2fs | %s | prompts=%s",
            total_elapsed,
            format_stage_timing_summary(stage_state),
            format_prompt_metrics_summary(stage_state),
        )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await reply_text_in_chunks(update.message, START_TEXT)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    history_count = len(get_chat_history(chat_id))
    timing_text = format_last_request_timing(chat_id)

    await reply_text_in_chunks(
        update.message,
        "Status: bot is running.\n"
        f"Search planner model: {SEARCH_PLANNER_MODEL}\n"
        f"Search retrieval model: {SEARCH_RETRIEVAL_MODEL}\n"
        f"Tavily search depth: {TAVILY_SEARCH_DEPTH}\n"
        f"Exa search type: {EXA_SEARCH_TYPE}\n"
        f"Local model 1: {LOCAL_MODEL_1}\n"
        f"Local model 2: {LOCAL_MODEL_2 or 'disabled'}\n"
        f"Cloud model: {CLOUD_MODEL}\n"
        f"xAI model: {XAI_MODEL}\n"
        f"Search query limit: {SEARCH_QUERY_LIMIT}\n"
        f"Search results per query: {SEARCH_RESULTS_PER_QUERY}\n"
        f"Total candidate limit: {TOTAL_CANDIDATE_LIMIT}\n"
        f"Local candidate limit: {LOCAL_CANDIDATE_LIMIT}\n"
        f"Rolling memory turns kept per chat: {MAX_HISTORY_TURNS}\n"
        f"Current chat memory count: {history_count}\n"
        f"Heartbeat interval: {HEARTBEAT_SECONDS}s\n"
        f"Evidence timeout: {EVIDENCE_TIMEOUT_SECONDS}s\n"
        f"Local model timeout: {LOCAL_MODEL_TIMEOUT_SECONDS}s\n"
        f"Final timeout: {FINAL_TIMEOUT_SECONDS}s\n\n"
        f"{timing_text}\n\n"
        f"{format_today_search_stats()}\n\n"
        f"{AVAILABLE_COMMANDS_TEXT}"
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    clear_chat_history(chat_id)
    context.chat_data.pop(PENDING_SEARCH_DECISION_KEY, None)
    await reply_text_in_chunks(update.message, "Cleared this chat's rolling memory.")


async def execute_question_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_text: str,
    force_no_search: bool = False,
    fast_mode: bool = False,
    post_answer_note: str | None = None,
) -> None:
    chat_id = update.effective_chat.id
    chat_lock = get_chat_lock(chat_id)

    if chat_lock.locked():
        await reply_text_in_chunks(
            update.message,
            "A previous request is still running for this chat. Wait for it to finish or use /clear after it completes."
        )
        return

    status_message = await update.message.reply_text(
        "Working on it...\nStage: Starting...\nElapsed: 0s"
    )

    async with chat_lock:
        try:
            if fast_mode:
                answer, notices, stage_state = await orchestrate_fast_answer(
                    user_text,
                    recent_chat_context,
                    status_message,
                )
            else:
                answer, notices, stage_state = await orchestrate_answer(
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
            stage_state = None

        for notice in notices:
            await reply_text_in_chunks(update.message, notice)
        if stage_state:
            total_elapsed = time.monotonic() - stage_state["started_at"]
            save_last_request_timing(
                chat_id,
                "fast" if fast_mode else "full",
                total_elapsed,
                stage_state,
            )
        save_chat_turn(chat_id, user_text, answer)
        await send_formatted_answer(update, answer)
        if post_answer_note:
            await reply_text_in_chunks(update.message, post_answer_note)


async def handle_ask_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    search_policy: str = "auto",
    fast_mode: bool = False,
    usage_text: str = ASK_USAGE,
) -> None:
    user_text = normalize_whitespace(" ".join(context.args))

    if not user_text:
        await reply_text_in_chunks(update.message, usage_text)
        return

    context.chat_data.pop(PENDING_SEARCH_DECISION_KEY, None)

    if fast_mode:
        await execute_question_request(update, context, user_text, fast_mode=True)
        return

    if search_policy == "force_search":
        await execute_question_request(update, context, user_text, force_no_search=False)
        return

    if search_policy == "force_no_search":
        await execute_question_request(update, context, user_text, force_no_search=True)
        return

    chat_id = update.effective_chat.id
    recent_chat_context = format_recent_chat_context(chat_id)
    search_decision = decide_search_behavior(user_text, recent_chat_context)
    decision = search_decision.get("decision", SEARCH_DECISION_SKIP_SEARCH)

    if decision == SEARCH_DECISION_USE_SEARCH:
        await execute_question_request(
            update,
            context,
            user_text,
            force_no_search=False,
            post_answer_note="Search used: yes (auto-decided).",
        )
        return

    if decision == SEARCH_DECISION_SKIP_SEARCH:
        await execute_question_request(
            update,
            context,
            user_text,
            force_no_search=True,
            post_answer_note="Search used: no (auto-decided).",
        )
        return

    context.chat_data[PENDING_SEARCH_DECISION_KEY] = {
        "question": user_text,
        "created_at": int(time.time()),
        "reason": search_decision.get("reason", ""),
    }
    await reply_text_in_chunks(
        update.message,
        "I’m not sure whether this question needs live web search.\n"
        f"Reason: {search_decision.get('reason', 'The request could reasonably go either way.')}\n\n"
        "Reply with `yes` to use live search or `no` to answer from model knowledge only."
    )


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_ask_request(update, context, search_policy="auto")


async def ask_search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_ask_request(update, context, search_policy="force_search")


async def ask_no_search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_ask_request(update, context, search_policy="force_no_search")


async def fast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_ask_request(update, context, fast_mode=True, usage_text=FAST_USAGE)


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = normalize_whitespace(" ".join(context.args))
    if not prompt:
        await reply_text_in_chunks(update.message, IMAGE_USAGE)
        return

    chat_id = update.effective_chat.id
    chat_lock = get_chat_lock(chat_id)
    if chat_lock.locked():
        await reply_text_in_chunks(
            update.message,
            "A previous request is still running for this chat. Wait for it to finish or use /clear after it completes.",
        )
        return

    status_message = await update.message.reply_text(
        f"Generating image with ComfyUI...\nWorkflow: {COMFYUI_WORKFLOW_PATH}\nThis can take a while."
    )

    async with chat_lock:
        started_at = time.monotonic()
        try:
            image_blobs = await asyncio.wait_for(
                asyncio.to_thread(generate_comfyui_images, prompt),
                timeout=IMAGE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            await safe_edit_status_message(
                status_message,
                f"Image generation timed out after {IMAGE_TIMEOUT_SECONDS}s.",
            )
            return
        except Exception as e:
            logger.warning("Image generation failed: %s", e)
            await safe_edit_status_message(status_message, "Image generation failed.")
            await reply_text_in_chunks(
                update.message,
                "I couldn't generate the image.\n"
                f"Error: {e}\n\n"
                "Make sure ComfyUI is running, the Z-Image Turbo workflow works in ComfyUI, "
                "and COMFYUI_WORKFLOW_PATH points to an API-format workflow JSON.",
            )
            return

        elapsed = time.monotonic() - started_at
        await safe_edit_status_message(
            status_message,
            f"Generated {len(image_blobs)} image(s) in {elapsed:.1f}s. Sending...",
        )
        caption = truncate_text(f"Generated with ComfyUI: {prompt}", 1000)
        for index, image_bytes in enumerate(image_blobs, start=1):
            image_file = io.BytesIO(image_bytes)
            image_file.name = f"generated-image-{index}.png"
            image_caption = caption if index == 1 else f"Generated with ComfyUI ({index}/{len(image_blobs)})"
            await update.message.reply_photo(photo=image_file, caption=image_caption)
        await safe_edit_status_message(
            status_message,
            f"Generated and sent {len(image_blobs)} image(s) in {elapsed:.1f}s.",
        )


async def handle_grok_request(update: Update, context: ContextTypes.DEFAULT_TYPE, use_search: bool = False) -> None:
    user_text = normalize_whitespace(" ".join(context.args))
    if not user_text:
        await reply_text_in_chunks(update.message, GROK_USAGE)
        return

    chat_id = update.effective_chat.id
    grok_context = format_recent_grok_context(chat_id)
    chat_lock = get_chat_lock(chat_id)
    if chat_lock.locked():
        await reply_text_in_chunks(
            update.message,
            "A previous request is still running for this chat. Wait for it to finish or use /clear after it completes.",
        )
        return

    status_message = await update.message.reply_text(
        f"Asking {XAI_MODEL}{' with web search' if use_search else ''}..."
    )

    async with chat_lock:
        started_at = time.monotonic()
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(call_xai_model, user_text, grok_context, use_search),
                timeout=XAI_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            await safe_edit_status_message(
                status_message,
                f"xAI request timed out after {XAI_TIMEOUT_SECONDS}s.",
            )
            return
        except Exception as e:
            logger.warning("xAI request failed: %s", e)
            await safe_edit_status_message(status_message, "xAI request failed.")
            await reply_text_in_chunks(
                update.message,
                "I couldn't get a Grok response.\n"
                f"Error: {e}\n\n"
                "Make sure XAI_API_KEY is set in .env and your xAI account has access to the configured model.",
            )
            return

        elapsed = time.monotonic() - started_at
        await safe_edit_status_message(
            status_message,
            f"Grok response completed in {elapsed:.1f}s.",
        )
        save_grok_turn(chat_id, user_text, answer)
        await send_formatted_answer(update, answer)


async def grok_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_grok_request(update, context, use_search=False)


async def grok_search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_grok_request(update, context, use_search=True)


async def pending_search_decision_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pending = context.chat_data.get(PENDING_SEARCH_DECISION_KEY)
    message = getattr(update, "effective_message", None)
    if not pending or message is None:
        return

    decision = parse_yes_no_reply(message.text or "")
    if decision is None:
        await reply_text_in_chunks(
            message,
            "I still have a pending `/ask` search decision for your last question. Reply `yes` to use live search or `no` to answer without it."
        )
        return

    question = pending.get("question", "")
    context.chat_data.pop(PENDING_SEARCH_DECISION_KEY, None)
    await execute_question_request(
        update,
        context,
        question,
        force_no_search=(decision == SEARCH_DECISION_SKIP_SEARCH),
    )


async def application_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    error = getattr(context, "error", None)
    exc_info = None
    if error is not None:
        exc_info = (type(error), error, error.__traceback__)
    logger.error("Unhandled application error: %s", error, exc_info=exc_info)

    message = getattr(update, "effective_message", None)
    if message is None:
        return

    try:
        await reply_text_in_chunks(
            message,
            "Something went wrong while processing that request. Please try again.",
        )
    except Exception:
        logger.exception("Failed to send application error notice to Telegram.")


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
    app.add_handler(CommandHandler("asksearch", ask_search_command))
    app.add_handler(CommandHandler("asknosearch", ask_no_search_command))
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(CommandHandler("grok", grok_command))
    app.add_handler(CommandHandler("groksearch", grok_search_command))
    app.add_handler(CommandHandler("fast", fast_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, pending_search_decision_reply))
    app.add_error_handler(application_error_handler)

    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
