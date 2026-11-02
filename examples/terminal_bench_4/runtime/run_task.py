"""Run a pinned Terminal-Bench task with Harbor's original sandbox and verifier."""

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.models.agent.context import AgentContext
from harbor.models.task.task import Task
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial
from trajectory import BadRequestError, Client

ROOT = Path("/opt/terminal_bench_4")
TRIALS = Path("/tmp/terminal_bench_4_trials")
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in the task environment.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


class TrajectoryAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "trajectory-bash"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        pass

    async def run(self, instruction: str, environment, context: AgentContext) -> None:
        tid = os.environ["TRAJECTORY_TID"]
        deadline = time.monotonic() + self.timeout
        messages = [
            {
                "role": "system",
                "content": "Solve the task with the bash tool. Verify your work, then reply when finished.",
            },
            {"role": "user", "content": instruction},
        ]
        reason = "TIMEOUT"
        turn = 0
        with Client() as client:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                turn += 1
                try:
                    response = await asyncio.to_thread(
                        client.chat.completions.create,
                        model="terminal-bench-4",
                        messages=messages,
                        x_trajectory_id=tid,
                        extra_body={"tools": [BASH_TOOL]},
                        max_tokens=8192,
                        timeout=min(900, remaining),
                    )
                except BadRequestError as error:
                    if error.code != "context_length_exceeded":
                        raise
                    reason = "TRUNCATION"
                    break
                metadata = (response.model_extra or {}).get("metadata") or {}
                if (
                    metadata.get("trajectory_termination_reason")
                    == "trajectory_cancelled"
                ):
                    reason = "TRUNCATION"
                    break
                choice = response.choices[0]
                if choice.finish_reason in ("length", "max_tokens"):
                    reason = "TRUNCATION"
                    break
                calls = choice.message.tool_calls or []
                if not calls:
                    reason = "ENV_DONE"
                    break
                messages.append(
                    {
                        "role": "assistant",
                        "content": choice.message.content or "",
                        "tool_calls": [call.model_dump(mode="json") for call in calls],
                    }
                )
                for index, call in enumerate(calls):
                    try:
                        if call.function.name != "bash":
                            raise ValueError(
                                f"Unknown tool {call.function.name}; use bash"
                            )
                        command = json.loads(call.function.arguments)["command"]
                        if not isinstance(command, str):
                            raise TypeError("command must be a string")
                    except (ValueError, KeyError, TypeError) as error:
                        details = {
                            "exit_code": 2,
                            "output": f"Invalid bash call: {error}",
                            "truncated": False,
                        }
                    else:
                        result = await environment.exec(command)
                        output = (result.stdout or "") + (result.stderr or "")
                        details = {
                            "exit_code": result.return_code,
                            "output": output[-65536:],
                            "truncated": len(output) > 65536,
                        }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(details),
                        }
                    )
                    await asyncio.to_thread(
                        client.trajectories.log_event,
                        tid,
                        event_id=f"bash-{turn}-{index}",
                        name="bash",
                        payload={"tool_call_id": call.id, **details},
                    )
        context.metadata = {"termination_reason": reason}

    def __init__(self, *args, timeout: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = timeout


async def run_trial(task_path: Path, timeout_override: float | None):
    task = Task(task_dir=task_path)
    timeout = min(
        timeout_override or task.config.agent.timeout_sec, task.config.agent.timeout_sec
    )
    config = TrialConfig(
        task=TaskConfig(path=task_path),
        trials_dir=TRIALS,
        agent=AgentConfig(
            import_path=TrajectoryAgent.import_path(),
            override_timeout_sec=timeout,
            kwargs={"timeout": timeout},
        ),
        environment=EnvironmentConfig(type="docker"),
    )
    trial = await Trial.create(config)
    return await trial.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--agent-timeout", type=float)
    args = parser.parse_args()
    task_path = ROOT / "tasks" / args.task
    if not (task_path / "task.toml").is_file() or task_path.parent != ROOT / "tasks":
        parser.error("unknown task")
    if args.agent_timeout is not None and args.agent_timeout <= 0:
        parser.error("--agent-timeout must be positive")

    with Client() as client:
        tid = client.trajectories.create().tid
        os.environ["TRAJECTORY_TID"] = tid
        try:
            result = asyncio.run(run_trial(task_path, args.agent_timeout))
            if (
                result.exception_info
                and result.exception_info.exception_type != "AgentTimeoutError"
            ):
                raise RuntimeError(
                    f"Harbor trial failed: {result.exception_info.exception_type}"
                )
            if result.verifier_result is None:
                raise RuntimeError("Harbor verifier did not return a grade")
            reward = float(result.verifier_result.rewards["reward"])
            if not math.isfinite(reward):
                raise ValueError("Harbor verifier returned a non-finite reward")
            client.trajectories.log_reward(tid, name="reward", value=reward)
            reason = (
                (result.agent_result.metadata or {}).get(
                    "termination_reason", "TIMEOUT"
                )
                if result.agent_result
                else "TIMEOUT"
            )
            client.trajectories.complete(tid, termination_reason=reason)
            print(f"task={args.task} ending={reason} reward={reward}", flush=True)
        except Exception:
            client.trajectories.complete(tid, termination_reason="ERROR")
            raise


if __name__ == "__main__":
    main()
