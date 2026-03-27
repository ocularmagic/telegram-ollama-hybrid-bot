# Setup Guide

This document turns the working prototype into a clean, repeatable setup.

## 1. Create the Telegram bot

Use **BotFather** in Telegram:

1. `/newbot`
2. choose a display name
3. choose a username ending in `bot`
4. save the token

## 2. Install Ollama

Install Ollama on the machine that will run the bot.

Then, if you plan to use the default cloud synthesis model, sign in locally:

```bash
ollama signin
```

## 3. Pull the local models once

```bash
ollama pull qwen3:14b
ollama pull gemma3:12b
```

You only need to do this the first time on a machine, or when you want to update or change models.

## 4. Get the required keys

You need:

- `TELEGRAM_BOT_TOKEN`
- `BOT_USERNAME`
- `TAVILY_API_KEY`

Optional:

- `OLLAMA_API_KEY` if you want Ollama web-search fallback when Tavily is unavailable

## 5. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 6. Create your `.env` file

Copy `.env.example` to `.env` and fill it in.

Example variables:

```env
TELEGRAM_BOT_TOKEN=...
BOT_USERNAME=...
TAVILY_API_KEY=...
OLLAMA_API_KEY=...
```

The bot loads `.env` automatically.

## 7. Start Ollama each time you want to run the bot

Make sure Ollama is running before you start the bot.

```bash
ollama serve
```

If you use Ollama Desktop, opening the app is usually enough.

## 8. Run the bot

```bash
python bot.py
```

## 9. Test the bot

Private chat:

```text
/status
/ask what is the latest version of Python?
/fast what time does Costco close today?
/asknosearch explain Python virtual environments from general knowledge
```

Group chat:

```text
/ask@your_bot_username what are the top 5 AI news stories this week?
/fast@your_bot_username what are the hours for the Apple Store in Bellevue today?
/asknosearch@your_bot_username explain transformers without browsing
```

## 10. Recommended Ollama context setting

If your hardware allows it, increase context length in Ollama for search-heavy prompts.

More context helps when the shared evidence pool gets large.

## 11. Optional Windows quality-of-life setup

- create `start_bot.bat`
- add a shortcut to the Startup folder using `shell:startup`

This makes the bot easier to relaunch at sign-in.
