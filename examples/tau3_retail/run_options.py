"""Rollout limits shared by the diagnostic, evaluation, and training commands."""

ROLLOUT_OPTIONS = {
    "reasoning_effort": "off",
    "max_output_tokens_per_step": 2048,
    "max_turns_per_trajectory": 200,
}
