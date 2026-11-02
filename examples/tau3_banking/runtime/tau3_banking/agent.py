"""Run one exact public Tau 3 Banking task through the Trajectory SDK."""

import os
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


def _termination_reason(reason: object) -> str:
    return {
        "AGENT_STOP": "ENV_DONE",
        "USER_STOP": "ENV_DONE",
        "MAX_STEPS": "MAX_STEPS",
        "TIMEOUT": "TIMEOUT",
        "CONTEXT_WINDOW_EXCEEDED": "TRUNCATION",
        "TOO_MANY_ERRORS": "ENV_DONE",
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
    user_and_judge_model = os.getenv(
        "TAU3_BANKING_USER_JUDGE_MODEL", USER_AND_JUDGE_MODEL
    )
    reasoning_effort = (
        USER_AND_JUDGE_REASONING_EFFORT
        if user_and_judge_model == USER_AND_JUDGE_MODEL
        else None
    )
    configure_nl_assertion_judge(user_and_judge_model)
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
            # The SDK client retries policy calls; LiteLLM retries would multiply them.
            "num_retries": 0,
        },
        user=USER_SIMULATOR,
        llm_user=user_and_judge_model,
        llm_args_user={
            **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
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

    reason = _termination_reason(result.termination_reason)
    # Tau 2 continues after an output-limited reply; its final grade covers later turns.
    if reason != "ERROR":
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
            "user_and_judge_model": user_and_judge_model,
            "user_and_judge_reasoning_effort": reasoning_effort,
            "max_steps": MAX_STEPS,
            "max_errors": MAX_ERRORS,
            "seed": SEED,
            "policy_calls": provider.calls,
            "output_limited_replies": provider.output_limited_replies,
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
