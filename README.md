# Telegram Hybrid Research Bot with Ollama + Exa/Tavily

A command-only Telegram bot that separates search planning, retrieval, local model judgment, and final synthesis into a practical workflow, with an auto-deciding default mode and explicit live-search commands.

This repo is for builders who want to learn how a hybrid local + cloud system behaves in the real world, not just how to call one model once.

## What it does

The bot has three main answer paths:

- `/ask ...` auto-decides whether live search is needed.
- `/asksearch ...` always uses the full live-search workflow.
- `/asknosearch ...` always skips internet search.
- `/image ...` generates an image locally through a ComfyUI workflow.
- `/grok ...` uses xAI Grok without search tools for testing a flagship LLM path.
- `/groksearch ...` uses xAI Grok with the web search tool array enabled for testing a flagship LLM path.
- `/fast ...` always uses live search, but skips the local-model review step.

When a request uses the full live-search workflow, the bot:

1. Uses a **local planner model** to generate multiple search angles.
2. Uses **Exa or Tavily live web search** to collect a broad shared candidate pool.
3. Captures per-query summaries and cited web sources for extra context.
4. Sends a trimmed local-only evidence pool to **one local model by default**.
5. Sends the broad evidence pool plus the local answer to **one cloud model**.
6. Returns a final answer with staged progress updates in Telegram.

When a user runs `/ask ...`, the bot searches automatically for clearly current, comparative, sourced, or high-stakes questions; skips search for clearly evergreen explanations; and asks for a yes/no confirmation when the request is ambiguous. When it auto-decides without asking, it sends a short note under the answer saying whether search was used.

When a user runs `/fast ...`, the bot keeps live search, skips the local-model review step, and asks Kimi for a concise answer.

By default, the repo is configured as:

- **Search planner:** `ministral-3:8b`
- **Search retrieval:** `exa-search`
- **Local model 1:** `ministral-3:8b`
- **Local model 2:** disabled
- **Image generation:** local ComfyUI workflow, configured with `COMFYUI_WORKFLOW_PATH`
- **Grok commands:** `grok-4.20-multi-agent-0309` through xAI Responses API for testing a flagship LLM path
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
Command-only bot (/ask, /asksearch, /asknosearch, /image, /grok, /groksearch, /fast, /status, /clear)
  |
Auto search decision (/ask only)
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

For `/ask` questions that are auto-decided as no-search, the retrieval stage is replaced by a no-search context block and the bot still uses the local model plus Kimi for final synthesis. For `/fast`, the bot uses live retrieval plus Kimi and skips the local review step.

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
- `XAI_API_KEY` if you want to use `/grok` and `/groksearch`

### 5. Pull the default local model once

```bash
ollama pull ministral-3:8b
```

You only need to do this the first time on a machine, or later if you want to update or replace models.

For image generation, see [ComfyUI Image Setup](#comfyui-image-setup).

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
Shows model assignments, timeout settings, search-pool limits, memory status, recent Exa/Tavily usage, and the most recent request timing/prompt-size metrics.

### `/ask your question`
Auto-decides whether live search is needed. If the bot is unsure, it asks you to reply `yes` or `no`.

If `/ask` auto-decides without asking, the bot adds a note below the answer:

```text
Search used: yes (auto-decided).
```

or:

```text
Search used: no (auto-decided).
```

### `/asksearch your question`
Runs the full hybrid workflow with live search.

### `/asknosearch your question`
Forces an answer without internet search.

### `/image your image prompt`
Queues the prompt in your local ComfyUI workflow and sends all generated output images back to Telegram.

### `/grok your question`
Sends the prompt directly to xAI using `grok-4.20-multi-agent-0309` without search tools as a standalone flagship LLM test path. This command does not use the bot's search-decision logic, local model review, Kimi synthesis, or regular `/ask` memory. It only includes previous `/grok` and `/groksearch` turns as Grok-only context.

### `/groksearch your question`
Sends the prompt directly to xAI using `grok-4.20-multi-agent-0309` as a standalone flagship LLM test path and includes `tools: [{"type": "web_search"}]` on every request. It only includes previous `/grok` and `/groksearch` turns as Grok-only context; the response is returned directly to Telegram.

### `/fast your question`
Runs live search, skips the local-model review step, and returns a concise answer.

### `/clear`
Clears rolling memory for the current Telegram chat.

## Group usage

Recommended pattern in group chats:

```text
/asksearch@your_bot_username what are the top 5 news stories from the last 72 hours?
/fast@your_bot_username what are the hours for Pike Place Chowder today?
/ask@your_bot_username explain TCP vs UDP
/asknosearch@your_bot_username explain TCP vs UDP from general knowledge
/image@your_bot_username a photorealistic orange tabby cat wearing tiny aviator goggles
/grok@your_bot_username explain quantum tunneling in plain language
/groksearch@your_bot_username what are the latest AI headlines today?
```

Keep Telegram **Privacy Mode ON** unless you intentionally want different bot behavior in groups.

## Memory behavior

The bot stores a short rolling history per Telegram chat so follow-up questions work better.

Example:

```text
/asksearch what are the top 5 market stories this week?
/fast what time does Costco close today?
/ask give me more detail on the gold-price story you mentioned
/asknosearch explain the last answer without internet search
```

This is in-memory only. If the process restarts, memory resets.

## Important implementation notes

### Search is centralized on purpose

This repo uses a **shared-evidence** design:

- The bot plans search angles locally when search is needed.
- The bot gathers evidence centrally from Exa first, then Tavily, then Ollama web search fallback when configured.
- The local model reviews a trimmed local-only pool so it is not forced to read the full broad retrieval set.
- The final model sees the broad pool plus any local-model answer that was produced.

That makes it easier to compare model behavior without introducing too many moving parts at once.

If you intentionally want to bypass that retrieval layer, use `/asknosearch` or add "without internet search" to `/ask`.
If you want live data without the full multi-model analysis, use `/fast`.

### ComfyUI Image Setup

The `/image` command uses your locally running ComfyUI instance, not Ollama image generation. The bot talks to ComfyUI over its local HTTP API.

Current Windows desktop setup:

- `COMFYUI_BASE_URL=http://127.0.0.1:8000`
- `COMFYUI_WORKFLOW_PATH=comfyui_workflow_ui.json`
- `COMFYUI_PROMPT_NODE_ID=557`
- `COMFYUI_PROMPT_INPUT=value`

The `comfyui_workflow_ui.json` file in this repo is a ComfyUI API-format workflow. In that workflow, node `557` is a `PrimitiveStringMultiline` node that feeds the positive prompt into the rest of the graph. The bot replaces that node’s `value` input with the text from Telegram.

How `/image` works:

- It receives the Telegram prompt after `/image`.
- It optionally wraps the prompt with `IMAGE_PROMPT_PREFIX` and `IMAGE_PROMPT_SUFFIX`.
- It loads the workflow JSON from `COMFYUI_WORKFLOW_PATH`.
- It inserts the final prompt into the configured prompt node.
- It queues the workflow through ComfyUI’s `/prompt` endpoint.
- It polls ComfyUI’s `/history/{prompt_id}` endpoint.
- It fetches every generated output image from ComfyUI’s `/view` endpoint.
- It sends all generated images back to Telegram.

If the workflow generates two images, the bot sends two Telegram photos. If you change the ComfyUI workflow batch size or output nodes, the bot sends however many image outputs ComfyUI reports in history.

Prompt wrapping is configured with:

```env
IMAGE_PROMPT_PREFIX=
IMAGE_PROMPT_SUFFIX=
```

Example:

```env
IMAGE_PROMPT_PREFIX=score_9, score_8_up, best quality, highly detailed
IMAGE_PROMPT_SUFFIX=cinematic lighting, sharp focus
```

Then this Telegram command:

```text
/image a red fox sitting in a snowy forest
```

is sent to the workflow as:

```text
score_9, score_8_up, best quality, highly detailed, a red fox sitting in a snowy forest, cinematic lighting, sharp focus
```

If `/image` returns a connection refused error, open the ComfyUI UI in your browser and copy that base URL into `COMFYUI_BASE_URL`. The Windows desktop app may use `http://127.0.0.1:8000`, while portable/manual ComfyUI installs often use `http://127.0.0.1:8188`.

### Search breadth is configurable

Useful knobs in `bot.py`:

- `SEARCH_RETRIEVAL_MODEL`
- `EXA_SEARCH_TYPE`
- `EXA_HIGHLIGHTS_MAX_CHARS`
- `COMFYUI_BASE_URL`
- `COMFYUI_WORKFLOW_PATH`
- `COMFYUI_PROMPT_NODE_ID`
- `COMFYUI_PROMPT_INPUT`
- `IMAGE_MODEL`
- `IMAGE_TIMEOUT_SECONDS`
- `IMAGE_PROMPT_CHARS`
- `IMAGE_PROMPT_PREFIX`
- `IMAGE_PROMPT_SUFFIX`
- `XAI_BASE_URL`
- `XAI_MODEL`
- `XAI_TIMEOUT_SECONDS`
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

### Current command behavior

- `/ask` is the normal default. It decides whether search is needed, asks for clarification when ambiguous, and reports whether search was used when it chooses automatically.
- `/asksearch` bypasses that decision and always runs the full search workflow.
- `/asknosearch` bypasses that decision and always skips internet search.
- `/image` calls the local ComfyUI API, queues the configured workflow, fetches all generated images from ComfyUI history, and sends them back to Telegram.
- `/grok` calls xAI's Responses API without tools as a flagship LLM test path, includes only Grok-command history as context, and returns the response directly to Telegram.
- `/groksearch` calls xAI's Responses API with the `web_search` tool included in the `tools` array every time as a flagship LLM test path, includes only Grok-command history as context, and returns the response directly to Telegram.
- `/fast` always searches, skips the local model review, and asks Kimi for a shorter answer.
- `/clear` clears rolling chat memory and any pending `/ask` search decision for the chat.

## Changes Made

Recent project updates include:

- Switched from two local review models to one default local model: `ministral-3:8b`.
- Disabled `LOCAL_MODEL_2` by default while keeping the second local slot optional in code.
- Changed `/ask` from a no-search command into an auto-search command.
- Added `/asksearch` as the explicit full live-search command.
- Reintroduced `/asknosearch` as an explicit forced no-search command.
- Added `/image` for local ComfyUI image generation using a configurable API-format workflow.
- Added `IMAGE_PROMPT_PREFIX` and `IMAGE_PROMPT_SUFFIX` so the bot can wrap Telegram image prompts with fixed style text.
- Added `/grok` and `/groksearch` for testing a flagship xAI LLM path, with `/groksearch` sending the web search tool array on every request.
- Added yes/no follow-up handling when `/ask` is unsure whether live search is needed.
- Added the post-answer `Search used: yes/no (auto-decided).` note for silent `/ask` decisions.
- Added Exa as the primary retrieval provider, using the official `exa-py` SDK.
- Kept Tavily as fallback behind Exa and Ollama web search as an optional fallback.
- Increased search breadth to support up to two planned queries and up to 20 results per query.
- Added a trimmed local-only evidence pool so local models do not have to process the entire broad pool.
- Increased the local-model timeout to `600s`.
- Added per-stage timing and prompt-size diagnostics, visible in `/status`.
- Added safer Telegram message chunking and transient timeout handling for status-message edits.
- Added tests for search planning, Exa/Tavily fallback behavior, command routing, auto-search decisions, prompt metrics, and Telegram formatting.

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
