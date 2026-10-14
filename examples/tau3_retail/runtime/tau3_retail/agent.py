"""Run one exact public Tau 3 Retail task through the Trajectory SDK."""

import sys

from tau3_retail.policy_provider import register_policy
from tau3_retail.protocol import (
    AGENT_MAX_TOKENS,
    DOMAIN,
    EMPTY_USER_RESPONSE_MAX_ATTEMPTS,
    MAX_ERRORS,
    MAX_RETRIES,
    MAX_STEPS,
    SEED,
    SOURCE_COMMIT,
    SOURCE_SHA256,
    USER_AND_JUDGE_MODEL,
    USER_AND_JUDGE_REASONING_EFFORT,
    configure_nl_assertion_judge,
    load_task,
    verify_source,
)
from tau3_retail.user_simulator import USER_SIMULATOR, register_user_simulator
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

    verify_source()
    configure_nl_assertion_judge()
    task = load_task(task_id)
    trajectory_id = client.trajectories.create().tid
    try:
        provider = register_policy(client, trajectory_id)
        register_user_simulator()
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
                "num_retries": MAX_RETRIES,
            },
            user=USER_SIMULATOR,
            llm_user=USER_AND_JUDGE_MODEL,
            llm_args_user={
                "reasoning_effort": USER_AND_JUDGE_REASONING_EFFORT,
                "num_retries": MAX_RETRIES,
            },
            max_steps=MAX_STEPS,
            max_errors=MAX_ERRORS,
            seed=SEED,
        )
        result = run_single_task(
            config,
            task,
            seed=SEED,
            evaluation_type=EvaluationType.ALL,
        )
        if result.reward_info is None:
            raise RuntimeError("The upstream Tau 3 evaluator returned no reward")

        # Upstream grades only the database, so leaked reasoning would otherwise be rewarded.
        upstream_reward = result.reward_info.reward
        client.trajectories.log_reward(
            trajectory_id,
            name="reward_accuracy",
            value=0.0 if provider.reasoning_leaks else upstream_reward,
            explanation=result.reward_info.model_dump_json(),
        )
        client.trajectories.log_event(
            trajectory_id,
            event_id="tau3-retail-protocol",
            name="tau3_retail_protocol",
            payload={
                "task_id": task_id,
                "source_commit": SOURCE_COMMIT,
                "source_sha256": SOURCE_SHA256,
                "user_and_judge_model": USER_AND_JUDGE_MODEL,
                "user_and_judge_reasoning_effort": USER_AND_JUDGE_REASONING_EFFORT,
                "agent_max_tokens": AGENT_MAX_TOKENS,
                "max_steps": MAX_STEPS,
                "max_errors": MAX_ERRORS,
                "seed": SEED,
                "policy_calls": provider.calls,
                "reasoning_leaks": provider.reasoning_leaks,
                "upstream_reward": upstream_reward,
                "empty_user_response_max_attempts": EMPTY_USER_RESPONSE_MAX_ATTEMPTS,
                "upstream_termination_reason": result.termination_reason.value,
            },
        )
    except Exception:
        client.trajectories.complete(trajectory_id, termination_reason="ERROR")
        raise
    completed = client.trajectories.complete(
        trajectory_id,
        termination_reason=termination_reason(result.termination_reason, provider.truncated),
    )
    if completed.status != "completed":
        raise RuntimeError(f"Trajectory completion returned {completed.status}")
    return trajectory_id


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tau3_retail.agent TASK_ID")
    with Client(max_retries=0, timeout=1200) as client:
        run_task(client, sys.argv[1])


if __name__ == "__main__":
    main()
