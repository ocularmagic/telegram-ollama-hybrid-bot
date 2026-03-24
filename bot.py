import os
import re
import json
import html
import time
import logging
import asyncio
import urllib.parse

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from ollama import chat
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOCAL_MODEL_1 = os.getenv("LOCAL_MODEL_1", "qwen3:14b")
LOCAL_MODEL_2 = os.getenv("LOCAL_MODEL_2", "gemma3:12b")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "kimi-k2.5:cloud")
SEARCH_PLANNER_MODEL = os.getenv("SEARCH_PLANNER_MODEL", "gemini-2.5-flash")
SEARCH_RETRIEVAL_MODEL = os.getenv("SEARCH_RETRIEVAL_MODEL", SEARCH_PLANNER_MODEL)
BOT_USERNAME = os.getenv("BOT_USERNAME", "your_bot_username")

MAX_TELEGRAM_CHUNK = 3500
MAX_PRE_CHUNK = 3000
HEARTBEAT_SECONDS = int(os.getenv("HEARTBEAT_SECONDS", "15"))

EVIDENCE_TIMEOUT_SECONDS = int(os.getenv("EVIDENCE_TIMEOUT_SECONDS", "300"))
LOCAL_MODEL_TIMEOUT_SECONDS = int(os.getenv("LOCAL_MODEL_TIMEOUT_SECONDS", "300"))
FINAL_TIMEOUT_SECONDS = int(os.getenv("FINAL_TIMEOUT_SECONDS", "600"))

SEARCH_QUERY_LIMIT = int(os.getenv("SEARCH_QUERY_LIMIT", "6"))
SEARCH_RESULTS_PER_QUERY = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "8"))
TOTAL_CANDIDATE_LIMIT = int(os.getenv("TOTAL_CANDIDATE_LIMIT", "40"))
SEARCH_SNIPPET_LIMIT = int(os.getenv("SEARCH_SNIPPET_LIMIT", "280"))
FETCH_CONTENT_LIMIT = int(os.getenv("FETCH_CONTENT_LIMIT", "900"))

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "3"))
MEMORY_QUESTION_CHARS = int(os.getenv("MEMORY_QUESTION_CHARS", "1200"))
MEMORY_ANSWER_CHARS = int(os.getenv("MEMORY_ANSWER_CHARS", "5000"))

CHAT_MEMORY = {}
CHAT_LOCKS = {}

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

CLOUD_FINAL_SYSTEM_PROMPT = """You are the final synthesis model.

You will receive:
- the user's current question
- recent chat context from this Telegram chat
- a broad shared search pool
- an answer from local model 1
- an answer from local model 2

Your tasks:
1. Write a very brief summary of local model 1.
2. Write a very brief summary of local model 2.
3. Briefly note any meaningful disagreement. If there is no meaningful disagreement, say so.
4. Give your own full final answer, using the broad shared pool, recent chat context when relevant, and both local answers.

Rules:
- Use the shared pool directly, not just the local summaries.
- Prefer the strongest evidence in the pool.
- If the question asks for ranking or selection, make your own judgment.
- Do not mention internal implementation details.
- Keep the summaries short and the final answer useful.

Output exactly in this format:

Local model 1 summary:
...

Local model 2 summary:
...

Differences:
...

Final answer:
...
"""

START_TEXT = (
    "Bot is working.\n\n"
    "Use /ask followed by your question.\n"
    f"In groups, use /ask@{BOT_USERNAME} your question.\n"
    "Use /clear to clear this chat's rolling memory."
)

ASK_USAGE = (
    "Usage:\n"
    "/ask your question here\n\n"
    "Example:\n"
    "/ask what are the top 5 news headlines from the last 72 hours?"
)


def truncate_text(text: str, max_len: int) -> str:
    text = text or ""
    return text if len(text) <= max_len else text[:max_len]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


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

    raise ValueError("Could not parse JSON from Gemini response.")


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


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return genai.Client()


def get_grounding_tool():
    return types.Tool(google_search=types.GoogleSearch())


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
    client = get_gemini_client()

    prompt = f"""You are a search-planning model.

Create a broad search plan for the user's current question.
Use the recent chat context only when it helps interpret a follow-up question.
Do NOT answer the question.
Return VALID JSON ONLY with this exact shape:

{{
  \"search_objective\": \"short description\",
  \"queries\": [
    {{
      \"query\": \"search query text\",
      \"purpose\": \"why this query exists\"
    }}
  ]
}}

Rules:
- Create 4 to 6 diverse queries.
- Prefer breadth over a single narrow query.
- Include an official/primary-source query when relevant.
- Include a freshness/current-status query when relevant.
- Include alternate wording or alternate angles when helpful.
- If the current question is a follow-up, use recent chat context to resolve references like \"that\", \"it\", or \"you mentioned\".
- Keep queries practical for web search.
- No markdown.
- No explanation outside JSON.

RECENT CHAT CONTEXT:
{recent_chat_context}

CURRENT USER QUESTION:
{question}
"""

    response = client.models.generate_content(
        model=SEARCH_PLANNER_MODEL,
        contents=prompt,
    )

    text = (response.text or "").strip()
    plan = extract_json_object(text)

    queries = plan.get("queries", [])
    cleaned_queries = []

    for item in queries:
        if not isinstance(item, dict):
            continue
        query = normalize_whitespace(item.get("query", ""))
        purpose = normalize_whitespace(item.get("purpose", ""))
        if query:
            cleaned_queries.append(
                {
                    "query": query,
                    "purpose": purpose or "general angle",
                }
            )

    if not cleaned_queries:
        return default_search_plan(question)

    return {
        "search_objective": normalize_whitespace(plan.get("search_objective", "")) or "Build a broad, multi-angle search pool.",
        "queries": cleaned_queries[:SEARCH_QUERY_LIMIT],
    }


def gemini_grounded_search(question: str, recent_chat_context: str, query_text: str, purpose: str) -> dict:
    client = get_gemini_client()
    prompt = f"""Use Google Search grounding to gather web evidence for one search angle.

Do not answer the user fully yet.
Focus on collecting useful facts and relevant web sources for this search angle.
Keep the response concise and factual.

RECENT CHAT CONTEXT:
{recent_chat_context}

CURRENT USER QUESTION:
{question}

SEARCH ANGLE:
{query_text}

PURPOSE:
{purpose}
"""

    response = client.models.generate_content(
        model=SEARCH_RETRIEVAL_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(tools=[get_grounding_tool()]),
    )

    candidates = getattr(response, "candidates", None) or []
    grounding_metadata = None
    if candidates:
        grounding_metadata = getattr(candidates[0], "grounding_metadata", None)

    web_search_queries = []
    grounding_chunks = []
    if grounding_metadata:
        web_search_queries = getattr(grounding_metadata, "web_search_queries", None) or []
        grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []

    results = []
    for chunk in grounding_chunks:
        web = getattr(chunk, "web", None)
        if not web:
            continue

        url = normalize_whitespace(getattr(web, "uri", "") or "")
        if not url:
            continue

        results.append(
            {
                "title": normalize_whitespace(getattr(web, "title", "") or "") or "Untitled result",
                "url": url,
                "source": domain_from_url(url),
            }
        )

    return {
        "summary": truncate_text(normalize_whitespace(response.text or ""), FETCH_CONTENT_LIMIT),
        "results": results[:SEARCH_RESULTS_PER_QUERY],
        "executed_queries": [normalize_whitespace(item) for item in web_search_queries if normalize_whitespace(item)],
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


def build_shared_search_pool(question: str, recent_chat_context: str, plan: dict) -> str:
    queries = plan.get("queries", [])[:SEARCH_QUERY_LIMIT]

    all_candidates = []
    deduped_by_url = {}
    search_errors = []
    grounded_summaries = []

    for idx, query_item in enumerate(queries, start=1):
        query_text = normalize_whitespace(query_item.get("query", ""))
        purpose = normalize_whitespace(query_item.get("purpose", "")) or "general angle"

        if not query_text:
            continue

        try:
            search_response = gemini_grounded_search(question, recent_chat_context, query_text, purpose)
        except Exception as e:
            error_text = f"Q{idx} failed: {e}"
            logger.warning("Search query failed for %r: %s", query_text, e)
            search_errors.append(error_text)
            continue

        if search_response.get("summary"):
            grounded_summaries.append(
                {
                    "query_index": idx,
                    "query_text": query_text,
                    "purpose": purpose,
                    "summary": search_response["summary"],
                    "executed_queries": search_response.get("executed_queries", []),
                }
            )

        raw_results = search_response.get("results", []) or []
        for rank, result in enumerate(raw_results, start=1):
            title = normalize_whitespace(result.get("title", ""))
            url = normalize_whitespace(result.get("url", ""))

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
                "snippet": truncate_text(
                    search_response.get("summary", ""),
                    SEARCH_SNIPPET_LIMIT,
                ),
            }

            if url not in deduped_by_url and len(all_candidates) < TOTAL_CANDIDATE_LIMIT:
                deduped_by_url[url] = candidate
                all_candidates.append(candidate)

    lines = []
    lines.append("SEARCH OBJECTIVE")
    lines.append(f"- {plan.get('search_objective', 'Build a broad search pool.')}")
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
            lines.append(
                f"[{idx:02d}] Query=Q{item['query_index']} | Source={item['source']} | Title={item['title']}"
            )
            lines.append(f"URL: {item['url']}")
            if item["snippet"]:
                lines.append(f"Snippet: {item['snippet']}")
            lines.append("")
    else:
        lines.append("- No search results collected.")
        lines.append("")

    lines.append("GROUNDED QUERY SUMMARIES")
    if grounded_summaries:
        for idx, item in enumerate(grounded_summaries, start=1):
            lines.append(f"[{idx:02d}] Query=Q{item['query_index']} | Purpose={item['purpose']}")
            if item["executed_queries"]:
                lines.append(f"Google queries: {', '.join(item['executed_queries'])}")
            lines.append(f"Summary: {item['summary']}")
            lines.append("")
    else:
        lines.append("- No grounded summaries collected.")
        lines.append("")

    return "\n".join(lines).strip()


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
    return f"""Answer the user's current question using the broad shared search pool below.

Use your own judgment.
If the question asks for ranking, selection, prioritization, or \"top\" items, make your own independent choice from the pool.
Use recent chat context only when it helps resolve a follow-up reference.

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
    return f"""RECENT CHAT CONTEXT:
{recent_chat_context}

CURRENT USER QUESTION:
{question}

BROAD SHARED SEARCH POOL:
{shared_pool}

LOCAL MODEL 1 ({LOCAL_MODEL_1}) ANSWER:
{local_answer_1}

LOCAL MODEL 2 ({LOCAL_MODEL_2}) ANSWER:
{local_answer_2}
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


def format_progress_text(stage: str, started_at: float) -> str:
    elapsed = int(time.monotonic() - started_at)
    return (
        "Working on it...\n"
        f"Stage: {stage}\n"
        f"Elapsed: {elapsed}s"
    )


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
        or is_aligned_table_line(line)
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


async def send_formatted_answer(update: Update, answer: str) -> None:
    segments = split_answer_into_segments(answer)

    for kind, text in segments:
        if kind == "table":
            for chunk in split_text_by_lines(text, MAX_PRE_CHUNK):
                await update.message.reply_text(
                    f"<pre>{html.escape(chunk)}</pre>",
                    parse_mode="HTML",
                )
        else:
            for chunk in split_text_by_lines(text, MAX_TELEGRAM_CHUNK):
                await update.message.reply_text(chunk)


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
    stage_state["stage"] = stage_text
    await safe_edit_status_message(
        status_message,
        format_progress_text(stage_text, stage_state["started_at"]),
    )


async def progress_heartbeat(status_message, stage_state: dict, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=HEARTBEAT_SECONDS)
        except asyncio.TimeoutError:
            await safe_edit_status_message(
                status_message,
                format_progress_text(stage_state["stage"], stage_state["started_at"]),
            )


async def run_search_plan_step(status_message, stage_state: dict, question: str, recent_chat_context: str, timeout_seconds: int):
    await set_stage(status_message, stage_state, "Planning broad search...")

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
    await set_stage(status_message, stage_state, "Collecting shared search pool...")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(build_shared_search_pool, question, recent_chat_context, plan),
            timeout=timeout_seconds,
        )
        return result, None
    except asyncio.TimeoutError:
        return "", f"Shared search pool step timed out after {timeout_seconds} seconds."
    except Exception as e:
        return "", f"Shared search pool step failed: {e}"


async def run_ollama_step(
    status_message,
    stage_state: dict,
    stage_label: str,
    model_name: str,
    user_text: str,
    system_prompt: str,
    timeout_seconds: int,
):
    await set_stage(status_message, stage_state, stage_label)

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


async def orchestrate_answer(question: str, recent_chat_context: str, status_message) -> str:
    stage_state = {
        "stage": "Starting...",
        "started_at": time.monotonic(),
    }
    stop_event = asyncio.Event()
    heartbeat_task = asyncio.create_task(progress_heartbeat(status_message, stage_state, stop_event))

    try:
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

        shared_pool, pool_error = await run_search_pool_step(
            status_message=status_message,
            stage_state=stage_state,
            question=question,
            recent_chat_context=recent_chat_context,
            plan=plan,
            timeout_seconds=EVIDENCE_TIMEOUT_SECONDS,
        )

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
                "GROUNDED QUERY SUMMARIES\n"
                "- None"
            )

        local_prompt = build_local_prompt(question, recent_chat_context, shared_pool)

        local_answer_1, local_error_1 = await run_ollama_step(
            status_message=status_message,
            stage_state=stage_state,
            stage_label="Asking local model 1...",
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
            stage_label="Asking local model 2...",
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
            stage_label="Preparing final answer...",
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

        await set_stage(status_message, stage_state, "Done. Sending answer...")
        return final_answer

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
        f"Final timeout: {FINAL_TIMEOUT_SECONDS}s"
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    clear_chat_history(chat_id)
    await update.message.reply_text("Cleared this chat's rolling memory.")


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = " ".join(context.args).strip()

    if not user_text:
        await update.message.reply_text(ASK_USAGE)
        return

    chat_id = update.effective_chat.id
    recent_chat_context = format_recent_chat_context(chat_id)
    chat_lock = get_chat_lock(chat_id)

    if chat_lock.locked():
        await update.message.reply_text(
            "A previous /ask is still running for this chat. Wait for it to finish or use /clear after it completes."
        )
        return

    status_message = await update.message.reply_text(
        "Working on it...\nStage: Starting...\nElapsed: 0s"
    )

    async with chat_lock:
        try:
            answer = await orchestrate_answer(user_text, recent_chat_context, status_message)
            if not answer:
                answer = "I didn't get a usable response."
        except Exception as e:
            logger.exception("Unhandled error while answering /ask.")
            await safe_edit_status_message(status_message, "Failed.")
            answer = f"Error: {e}"

        save_chat_turn(chat_id, user_text, answer)
        await send_formatted_answer(update, answer)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("ask", ask_command))

    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
