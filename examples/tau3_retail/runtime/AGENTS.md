# Tau 3 Retail SDK runtime

This directory is a standalone public benchmark sandbox. Use only the Trajectory SDK and the
explicitly injected key for the selected customer/judge provider: `OPENAI_API_KEY` by default
or `OPENROUTER_API_KEY` with the OpenRouter model override. Do not import repository services,
access Secret Manager, or log credentials. Keep Sierra's pinned retail data, official split,
and evaluator unchanged.
The only change to upstream grading is scoring malformed tool calls 0 in `agent.py`. The
simulated customer and judge default to Sierra's GPT-4.1 settings; an ingest-time model flag
can override their model while retaining upstream's temperature.
