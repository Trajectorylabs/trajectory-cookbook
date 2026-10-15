# Tau 3 Retail SDK runtime

This directory is a standalone public benchmark sandbox. Use only the Trajectory SDK and the
explicitly injected `OPENAI_API_KEY`; do not import repository services, access Secret Manager,
or log credentials. Keep Sierra's pinned retail data, official split, and evaluator unchanged; the only changes to
upstream's reward are in `agent.py`: the reasoning-leak gate and scoring malformed tool calls 0.
