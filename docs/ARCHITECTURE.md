# Architecture Notes

## Core design goal

Build a Telegram bot that:

1. gathers a broad evidence pool
2. lets two local models judge that pool independently
3. lets one stronger cloud model reconcile both judgments and answer

## Why not just use one model?

A single model can search and answer, but this project is meant to teach:

- orchestration
- separation of retrieval and reasoning
- local/cloud tradeoffs
- prompt specialization
- timeout and UX design

## Current pipeline

```text
Telegram /ask
  |
Recent per-chat memory lookup
  |
Gemini search-plan generation
  |
Gemini grounded Google Search across multiple queries
  |
Shared candidate pool + grounded summaries
  |
qwen3:14b local answer
  |
gemma3:12b local answer
  |
kimi-k2.5:cloud final synthesis
  |
Telegram response
```

## Why centralized broad retrieval?

This repo intentionally uses:

- **centralized retrieval**
- **decentralized local judgment**
- **centralized final synthesis**

That is a stronger default than letting every model search independently because it:

- reduces duplicated search effort
- makes the local comparison more meaningful
- keeps the system easier to debug

## Why Gemini is only the planner now

Earlier versions used Gemini as the evidence summarizer.

That was useful, but it compressed the search universe too early. The local models were often judging a summary of the world, not a larger pool of candidate sources.

The current version uses Gemini to plan the search angles instead:

- direct angle
- freshness/current angle
- official docs/primary sources angle
- comparison/overview angle
- follow-up resolution when prior context matters

## Why Gemini grounded search is in the loop

Gemini's grounded Google Search is the retrieval layer in this repo. The bot relies on it to build the shared evidence pool before any model answers, and it also exposes the web sources used for each grounded search pass.

## Memory model

The bot keeps a small rolling memory per chat:

- question
- final answer
- timestamp

This memory is:

- useful for follow-ups
- not persistent across restarts

## Output model roles

### Local Model 1
A local independent judge.

### Local Model 2
A second local independent judge from a different model family.

### Cloud Model
A final reconciler that sees:

- recent context
- the shared pool
- both local answers

## Important tradeoff

This system is intentionally slow but thoughtful.

Because the workflow is sequential, each `/ask` can run for several minutes under difficult conditions. The heartbeat/progress message is part of the design, not an afterthought.
