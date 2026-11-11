"""Run one pinned public tau2-bench retail or airline task through the Trajectory SDK."""

import os
import sys
from pathlib import Path

from tau2_domains.policy_provider import register_policy
from tau2_domains.protocol import (
    EMPTY_USER_RESPONSE_MAX_ATTEMPTS,
    MAX_ERRORS,
    MAX_RETRIES,
    MAX_STEPS,
    SEED,
    SOURCE_COMMIT,
    USER_AND_JUDGE_MODEL,
    USER_AND_JUDGE_REASONING_EFFORT,
    configure_nl_assertion_judge,
    load_task,
    task_split,
    verify_source,
)
from tau2_domains.user_simulator import USER_SIMULATOR, register_user_simulator
from trajectory import Client

DATA_ROOT = Path(os.getenv("TAU2_DATA_DIR", "/opt/tau2-source/data"))


def _termination_reason(reason: object) -> str:
    return {
        "AGENT_STOP": "ENV_DONE",
        "USER_STOP": "ENV_DONE",
        "MAX_STEPS": "TRUNCATION",
        "TIMEOUT": "TIMEOUT",
        "CONTEXT_WINDOW_EXCEEDED": "TRUNCATION",
        "TOO_MANY_ERRORS": "ENV_DONE",
        "AGENT_ERROR": "ERROR",
        "USER_ERROR": "ERROR",
        "INFRASTRUCTURE_ERROR": "ERROR",
        "UNEXPECTED_ERROR": "ERROR",
    }[reason.name]


def run_task(client: Client, domain: str, task_id: str) -> str:
    from tau2.data_model.simulation import TextRunConfig  # noqa: PLC0415
    from tau2.evaluator.evaluator import EvaluationType  # noqa: PLC0415
    from tau2.runner.batch import run_single_task  # noqa: PLC0415

    verify_source(domain, DATA_ROOT)
    split = task_split(domain, task_id, DATA_ROOT)
    user_and_judge_model = os.getenv("TAU2_USER_JUDGE_MODEL", USER_AND_JUDGE_MODEL)
    reasoning_effort = (
        USER_AND_JUDGE_REASONING_EFFORT if user_and_judge_model == USER_AND_JUDGE_MODEL else None
    )
    configure_nl_assertion_judge(user_and_judge_model)
    task = load_task(domain, task_id)
    trajectory_id = client.trajectories.create().tid
    provider = register_policy(client, trajectory_id)
    register_user_simulator()
    config = TextRunConfig(
        domain=domain,
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
        raise RuntimeError("The upstream tau2 evaluator returned no reward")

    reason = _termination_reason(result.termination_reason)
    # tau2 continues after an output-limited reply; its final grade covers later turns.
    if reason != "ERROR":
        client.trajectories.log_reward(
            trajectory_id,
            name="reward_accuracy",
            value=result.reward_info.reward,
            explanation=result.reward_info.model_dump_json(),
        )
    client.trajectories.log_event(
        trajectory_id,
        event_id="tau2-domain-protocol",
        name="tau2_domain_protocol",
        payload={
            "domain": domain,
            "task_id": task_id,
            "split": split,
            "source_commit": SOURCE_COMMIT,
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
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m tau2_domains.agent DOMAIN TASK_ID")
    with Client(max_retries=20, timeout=1200) as client:
        run_task(client, sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
