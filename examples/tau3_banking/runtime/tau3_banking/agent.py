"""Run one exact public Tau 3 Banking task through the Trajectory SDK."""

import sys

from tau3_banking.policy_provider import register_policy
from tau3_banking.protocol import (
    DOMAIN,
    EMPTY_USER_RESPONSE_MAX_ATTEMPTS,
    MAX_ERRORS,
    MAX_RETRIES,
    MAX_STEPS,
    RETRIEVAL_CONFIG,
    SEED,
    SOURCE_COMMIT,
    TASKS_SHA256,
    USER_AND_JUDGE_MODEL,
    USER_AND_JUDGE_REASONING_EFFORT,
    configure_nl_assertion_judge,
    load_task,
    verify_source,
)
from tau3_banking.user_simulator import USER_SIMULATOR, register_user_simulator
from trajectory import Client


def _termination_reason(reason: object, truncated: bool) -> str:
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
    provider = register_policy(client, trajectory_id)
    register_user_simulator()
    config = TextRunConfig(
        domain=DOMAIN,
        agent="llm_agent",
        llm_agent="trajectory_sdk/policy",
        llm_args_agent={
            # Managed runs apply the run's output limit and sampling settings instead.
            "max_tokens": 131072,
            "num_retries": MAX_RETRIES,
        },
        user=USER_SIMULATOR,
        llm_user=USER_AND_JUDGE_MODEL,
        llm_args_user={
            "reasoning_effort": USER_AND_JUDGE_REASONING_EFFORT,
            "num_retries": MAX_RETRIES,
        },
        retrieval_config=RETRIEVAL_CONFIG,
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

    reason = _termination_reason(result.termination_reason, provider.truncated)
    # tau2 keeps running after a cut-off policy reply, so its final grade covers later turns.
    if reason != "ERROR" and not provider.truncated:
        client.trajectories.log_reward(
            trajectory_id,
            name="reward_accuracy",
            value=result.reward_info.reward,
            explanation=result.reward_info.model_dump_json(),
        )
    client.trajectories.log_event(
        trajectory_id,
        event_id="tau3-banking-protocol",
        name="tau3_banking_protocol",
        payload={
            "task_id": task_id,
            "source_commit": SOURCE_COMMIT,
            "tasks_sha256": TASKS_SHA256,
            "retrieval": RETRIEVAL_CONFIG,
            "user_and_judge_model": USER_AND_JUDGE_MODEL,
            "user_and_judge_reasoning_effort": USER_AND_JUDGE_REASONING_EFFORT,
            "max_steps": MAX_STEPS,
            "max_errors": MAX_ERRORS,
            "seed": SEED,
            "policy_calls": provider.calls,
            "empty_user_response_max_attempts": EMPTY_USER_RESPONSE_MAX_ATTEMPTS,
            "upstream_termination_reason": result.termination_reason.value,
            "reward_info": result.reward_info.model_dump(mode="json"),
        },
    )
    completed = client.trajectories.complete(trajectory_id, termination_reason=reason)
    if completed.status not in ("completed", "finalizing"):
        raise RuntimeError(f"Trajectory completion returned {completed.status}")
    return trajectory_id


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tau3_banking.agent TASK_ID")
    with Client(max_retries=20, timeout=1200) as client:
        run_task(client, sys.argv[1])


if __name__ == "__main__":
    main()
