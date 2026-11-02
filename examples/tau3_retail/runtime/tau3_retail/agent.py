"""Run one exact public Tau 3 Retail task through the Trajectory SDK."""

import json
import os
import sys

from tau3_retail.policy_provider import register_policy
from tau3_retail.protocol import (
    AGENT_MAX_TOKENS,
    DOMAIN,
    MAX_ERRORS,
    MAX_RETRIES,
    MAX_STEPS,
    SEED,
    USER_AND_JUDGE_MODEL,
    USER_AND_JUDGE_TEMPERATURE,
    configure_nl_assertion_judge,
    load_task,
)
from trajectory import Client


def termination_reason(reason: object, truncated: bool) -> str:
    if truncated:
        return "TRUNCATION"
    return {
        "AGENT_STOP": "ENV_DONE",
        "USER_STOP": "ENV_DONE",
        "MAX_STEPS": "MAX_STEPS",
        "TIMEOUT": "TIMEOUT",
        "CONTEXT_WINDOW_EXCEEDED": "TRUNCATION",
        "TOO_MANY_ERRORS": "ERROR",
        "AGENT_ERROR": "ERROR",
        "USER_ERROR": "ERROR",
        "INFRASTRUCTURE_ERROR": "ERROR",
        "UNEXPECTED_ERROR": "ERROR",
    }[reason.name]


def run_task(client: Client, task_id: str) -> str:
    from tau2.data_model.simulation import TextRunConfig  # noqa: PLC0415
    from tau2.evaluator.evaluator import EvaluationType  # noqa: PLC0415
    from tau2.runner.batch import run_single_task  # noqa: PLC0415

    user_and_judge_model = os.getenv(
        "TAU3_RETAIL_USER_JUDGE_MODEL", USER_AND_JUDGE_MODEL
    )
    configure_nl_assertion_judge(user_and_judge_model)
    task = load_task(task_id)
    trajectory_id = client.trajectories.create().tid
    provider = register_policy(client, trajectory_id)
    config = TextRunConfig(
        domain=DOMAIN,
        agent="llm_agent",
        llm_agent="trajectory_sdk/policy",
        llm_args_agent={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "max_tokens": AGENT_MAX_TOKENS,
            # The SDK retries one request under its idempotency key.
            "num_retries": 0,
        },
        user="user_simulator",
        llm_user=user_and_judge_model,
        llm_args_user={
            "temperature": USER_AND_JUDGE_TEMPERATURE,
            "num_retries": MAX_RETRIES,
        },
        max_steps=MAX_STEPS,
        max_errors=MAX_ERRORS,
        seed=SEED,
    )
    try:
        result = run_single_task(
            config,
            task,
            seed=SEED,
            evaluation_type=EvaluationType.ALL,
        )
    except json.JSONDecodeError:
        # Upstream raises; count malformed agent tool arguments as a failure.
        if not provider.malformed_tool_calls:
            raise
        result = None
    if result is None:
        upstream_reward = 0.0
        explanation = "The agent sent tool-call arguments that are not valid JSON"
        upstream_termination = "malformed_tool_call"
    elif result.reward_info is None:
        raise RuntimeError("The upstream Tau 3 evaluator returned no reward")
    else:
        upstream_reward = result.reward_info.reward
        explanation = result.reward_info.model_dump_json()
        upstream_termination = result.termination_reason.value

    reason = (
        "ENV_DONE"
        if result is None
        else termination_reason(result.termination_reason, provider.truncated)
    )
    # Tau 2 continues after a cut-off reply; its grade covers later turns.
    if result is None or (reason != "ERROR" and not provider.truncated):
        client.trajectories.log_reward(
            trajectory_id,
            name="reward_accuracy",
            value=upstream_reward,
            explanation=explanation,
        )
    client.trajectories.log_event(
        trajectory_id,
        event_id="tau3-retail-protocol",
        name="tau3_retail_protocol",
        payload={
            "task_id": task_id,
            "policy_calls": provider.calls,
            "reasoning_leaks": provider.reasoning_leaks,
            "malformed_tool_calls": provider.malformed_tool_calls,
            "upstream_reward": upstream_reward,
            "upstream_termination_reason": upstream_termination,
        },
    )
    completed = client.trajectories.complete(trajectory_id, termination_reason=reason)
    if completed.status not in ("completed", "finalizing"):
        raise RuntimeError(f"Trajectory completion returned {completed.status}")
    return trajectory_id


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tau3_retail.agent TASK_ID")
    with Client(max_retries=20, timeout=1200) as client:
        run_task(client, sys.argv[1])


if __name__ == "__main__":
    main()
