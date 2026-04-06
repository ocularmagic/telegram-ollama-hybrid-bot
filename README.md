# Telegram Hybrid Research Bot with Ollama + Tavily

A command-only Telegram bot that separates search planning, retrieval, local model judgment, and final synthesis into a single practical workflow, with both an auto-deciding default mode and explicit live-search commands.

This repo is for builders who want to learn how a hybrid local + cloud system behaves in the real world, not just how to call one model once.

## What it does

When a user runs `/asksearch ...`, the bot:

1. Uses a **local planner model** to generate multiple search angles.
2. Uses **Exa or Tavily live web search** to collect a broad shared candidate pool.
3. Captures per-query summaries and cited web sources for extra context.
4. Sends a trimmed local-only evidence pool to **one local model by default**.
5. Sends the broad evidence pool plus the local answer to **one cloud model**.
6. Returns a final answer with staged progress updates in Telegram.

When a user runs `/ask ...`, the bot decides whether live search is needed. It searches automatically for clearly current, comparative, sourced, or high-stakes questions, skips search for clearly evergreen explanations, and asks for a yes/no confirmation when the request is ambiguous.

When a user runs `/fast ...`, the bot keeps live search, skips the local-model review step, and asks Kimi for a concise answer.

By default, the repo is configured as:

- **Search planner:** `ministral-3:8b`
- **Search retrieval:** `exa-search`
- **Local model 1:** `ministral-3:8b`
- **Local model 2:** disabled
- **Final synthesis:** `kimi-k2.5:cloud`

## Why this project is useful

This is a good learning repo if you want hands-on experience with:

- local vs. cloud model orchestration
- retrieval separated from reasoning
- broad shared evidence pools
- follow-up memory in chat workflows
- latency, timeout, and UX tradeoffs
- Telegram bot command handling

## High-level architecture

```text
Telegram
  |
Command-only bot (/ask, /asksearch, /fast, /status, /clear)
  |
Local search planner model
  |
Exa or Tavily live web search
  |
Broad shared evidence pool
  |
ministral-3:8b
  |
Local answer
  |
kimi-k2.5:cloud
  |
Final synthesis
```

## Requirements

- Python 3.11+
- Ollama installed locally
- A Telegram bot token from BotFather
- An Exa API key or Tavily API key
- Enough local hardware to run your chosen local model

If you keep the default final model as `kimi-k2.5:cloud`, sign in locally with:

```bash
ollama signin
```

## Quick start

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd telegram-ollama-hybrid-bot
```

### 2. Create and activate a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your environment file

Copy `.env.example` to `.env` and fill in your values.

Minimum required variables:

- `TELEGRAM_BOT_TOKEN`
- `BOT_USERNAME`
- at least one of `EXA_API_KEY` or `TAVILY_API_KEY`

The bot now loads `.env` automatically at startup.

Optional variables:

- `OLLAMA_API_KEY` if you want Ollama web-search fallback when Tavily is unavailable
- `EXA_API_KEY` if you want Exa as the primary retrieval provider
- `TAVILY_API_KEY` if you want Tavily as primary or as fallback behind Exa

### 5. Pull the default local model once

```bash
ollama pull ministral-3:8b
```

You only need to do this the first time on a machine, or later if you want to update or replace models.

### 6. Start Ollama each time you want to run the bot

Make sure the Ollama app is running, or start the server manually:

```bash
ollama serve
```

### 7. Run the bot

```bash
python bot.py
```

## Commands

### `/start`
Shows a short help message.

### `/status`
Shows model assignments, timeout settings, search-pool limits, and memory status.

### `/ask your question`
Auto-decides whether live search is needed. If the bot is unsure, it asks you to reply `yes` or `no`.

### `/asksearch your question`
Runs the full hybrid workflow with live search.

### `/fast your question`
Runs live search, skips the local-model debate, and returns a concise answer.

### `/clear`
Clears rolling memory for the current Telegram chat.

## Group usage

Recommended pattern in group chats:

```text
/asksearch@your_bot_username what are the top 5 news stories from the last 72 hours?
/fast@your_bot_username what are the hours for Pike Place Chowder today?
/ask@your_bot_username explain TCP vs UDP
```

Keep Telegram **Privacy Mode ON** unless you intentionally want different bot behavior in groups.

## Memory behavior

The bot stores a short rolling history per Telegram chat so follow-up questions work better.

Example:

```text
/asksearch what are the top 5 market stories this week?
/fast what time does Costco close today?
/ask give me more detail on the gold-price story you mentioned
/ask explain the last answer without internet search
```

This is in-memory only. If the process restarts, memory resets.

## Important implementation notes

### Search is centralized on purpose

This repo uses a **shared-evidence** design:

- The bot plans the search angles locally.
- The bot gathers evidence centrally.
- The local model reviews a trimmed local-only pool.
- The final model sees the broad pool plus any local-model answer that was produced.

That makes it easier to compare model behavior without introducing too many moving parts at once.

If you intentionally want to bypass that retrieval layer, add "without internet search" to `/ask`.
If you want live data without the full multi-model analysis, use `/fast`.

### Search breadth is configurable

Useful knobs in `bot.py`:

- `SEARCH_QUERY_LIMIT`
- `SEARCH_RESULTS_PER_QUERY`
- `TOTAL_CANDIDATE_LIMIT`
- `LOCAL_CANDIDATE_LIMIT`
- `SEARCH_SNIPPET_LIMIT`
- `FETCH_CONTENT_LIMIT`
- `SEARCH_CACHE_TTL_SECONDS`
- `SEARCH_CACHE_PATH`
- `SEARCH_STATS_PATH`
- `TAVILY_DAILY_CREDIT_LIMIT`

The bot also keeps a persistent search cache on disk. Repeated normalized searches can reuse the same retrieval result for 12 hours by default instead of spending another Tavily or web-search call.
It also keeps daily search stats on disk so `/status` can show recent Exa/Tavily usage and estimate how many uncached Tavily asks remain before the configured daily credit budget.

### Timeouts are intentionally generous

Default values:

- evidence/search stage: `300s`
- local model stage: `600s`
- final synthesis stage: `600s`

That gives hard prompts room to finish, while the heartbeat updates keep Telegram users informed.

## Known limitations

This repo is intentionally simple and not production-hardened.

Current limitations include:

- no persistent database
- no auth/rate-limit layer beyond Telegram itself
- in-memory chat history only
- long-running requests can still feel slow
- search quality depends heavily on the search plan and evidence-pool limits
- the default local model may still be too heavy for smaller machines

## Repo contents

```text
README.md
bot.py
docs/
  ARCHITECTURE.md
  GITHUB_PUBLISHING.md
  SETUP.md
test_bot.py
requirements.txt
.env.example
```

## Suggested next improvements

Good next steps if you want to keep building:

- add persistent chat history
- log structured run metadata for debugging
- expose model and timeout settings through commands
- add citation formatting in final answers
- move configuration into a dedicated settings module
- add a small web dashboard or admin view
