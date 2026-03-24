# Telegram Hybrid Research Bot with Ollama + Gemini

A command-only Telegram bot that separates search planning, retrieval, local model judgment, and final synthesis into a single practical workflow.

This repo is for builders who want to learn how a **hybrid local + cloud system** behaves in the real world, not just how to call one model once.

## What it does

When a user runs `/ask ...`, the bot:

1. Uses **Gemini 2.5 Flash** to plan multiple search angles.
2. Uses **Ollama web search** to collect a broad shared candidate pool.
3. Fetches a small number of pages for extra context.
4. Sends the same evidence pool to **two local models**.
5. Sends the evidence pool plus both local answers to **one cloud model**.
6. Returns a final answer with staged progress updates in Telegram.

By default, the repo is configured as:

- **Search planner:** `gemini-2.5-flash`
- **Local model 1:** `qwen3:14b`
- **Local model 2:** `gemma3:12b`
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
  ↓
Command-only bot (/ask, /status, /clear)
  ↓
Gemini 2.5 Flash search planner
  ↓
Ollama web_search + web_fetch
  ↓
Broad shared evidence pool
  ↓            ↓
qwen3:14b    gemma3:12b
  ↓            ↓
Independent local answers
  ↓
kimi-k2.5:cloud
  ↓
Final synthesis
```

## Requirements

- Python 3.11+
- Ollama installed locally
- A Telegram bot token from BotFather
- A Gemini API key
- An Ollama API key for Ollama web tools
- Enough local hardware to run your chosen local models

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
- `OLLAMA_API_KEY`
- `GEMINI_API_KEY`

The bot now loads `.env` automatically at startup.

### 5. Pull the default local models

```bash
ollama pull qwen3:14b
ollama pull gemma3:12b
```

### 6. Start Ollama

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
Runs the full hybrid workflow.

### `/clear`
Clears rolling memory for the current Telegram chat.

## Group usage

Recommended pattern in group chats:

```text
/ask@your_bot_username what are the top 5 news stories from the last 72 hours?
```

Keep Telegram **Privacy Mode ON** unless you intentionally want different bot behavior in groups.

## Memory behavior

The bot stores a short rolling history per Telegram chat so follow-up questions work better.

Example:

```text
/ask what are the top 5 market stories this week?
/ask give me more detail on the gold-price story you mentioned
```

This is in-memory only. If the process restarts, memory resets.

## Important implementation notes

### Search is centralized on purpose

This repo uses a **shared-evidence** design:

- Gemini plans the search angles.
- The bot gathers evidence centrally.
- Both local models judge the same pool.
- The final model sees the same pool plus both local answers.

That makes it easier to compare model behavior without introducing too many moving parts at once.

### Search breadth is configurable

Useful knobs in `bot.py`:

- `SEARCH_QUERY_LIMIT`
- `SEARCH_RESULTS_PER_QUERY`
- `TOTAL_CANDIDATE_LIMIT`
- `FETCH_URL_LIMIT`
- `SEARCH_SNIPPET_LIMIT`
- `FETCH_CONTENT_LIMIT`

### Timeouts are intentionally generous

Default values:

- evidence/search stage: `300s`
- local model stage: `300s`
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
- default local models may be too heavy for smaller machines

## Repo contents

```text
README.md
bot.py
docs/
  ARCHITECTURE.md
  GITHUB_PUBLISHING.md
  SETUP.md
article/
  x-thread.md
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
