"""Run official ExtractBench Harbor tasks with native Value F1 grading."""

import argparse
import base64
import hashlib
import io
import tarfile
import urllib.request
import asyncio
import json
import math
import os
import shlex
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
from trajectory import APITimeoutError, BadRequestError, Client

ROOT = Path("/opt/extractbench")
TRIALS = Path("/tmp/extractbench_trials")
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
        # This runs through the same container-only tool boundary as model commands.
        check = (
            "import os; from pathlib import Path; "
            "assert not any(Path(p).exists() for p in "
            "('/tests', '/solution', '/var/run/docker.sock')); "
            "assert not any(k.startswith('TRAJECTORY_') or k in "
            "('MODEL_ENDPOINT_ACCESS_TOKEN', 'MODEL_ENDPOINT_URL', 'MODEL_ENDPOINT_ID') "
            "for k in os.environ); print('actor-boundary-ok')"
        )
        result = await environment.exec("python -c " + shlex.quote(check), timeout_sec=30)
        if result.return_code != 0 or "actor-boundary-ok" not in (result.stdout or ""):
            raise RuntimeError("Actor can access private paths, credentials, or Docker socket")

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
                        model="extractbench",
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
                except APITimeoutError:
                    if deadline - time.monotonic() > 1:
                        raise
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
                        result = await environment.exec(command, timeout_sec=max(1, math.ceil(deadline - time.monotonic())))
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


async def run_trial(task_path: Path, timeout_override: float | None, verifier_smoke: bool = False):
    task = Task(task_dir=task_path)
    timeout = min(
        timeout_override or task.config.agent.timeout_sec, task.config.agent.timeout_sec
    )
    validate_task_policy(task)
    config = TrialConfig(
        task=TaskConfig(path=task_path),
        trials_dir=TRIALS,
        agent=(AgentConfig(name="nop") if verifier_smoke else AgentConfig(
            import_path=TrajectoryAgent.import_path(),
            override_timeout_sec=timeout,
            kwargs={"timeout": timeout},
        )),
        environment=EnvironmentConfig(type="docker"),
    )
    trial = await Trial.create(config)
    result = await trial.run()
    if verifier_smoke and result.exception_info is None:
        # Harbor's built-in NOP agent makes no model requests and returns an empty
        # context. A normal NOP completion is not an actor timeout.
        result.agent_result = AgentContext(metadata={
            "termination_reason": "ENV_DONE", "verifier_smoke": True,
        })
    return result, trial.paths


def validate_task_policy(task) -> None:
    """Do not weaken the source actor/private-verifier boundary."""
    config = task.config
    if (config.agent.network_mode != "allowlist" or config.agent.allowed_hosts != []
            or config.verifier.environment_mode != "separate"
            or config.verifier.environment is None
            or config.verifier.environment.network_mode != "no-network"):
        raise ValueError("Expected official ExtractBench actor and separate-verifier policies")


def native_grade(result, paths) -> tuple[float, dict]:
    if result.exception_info and result.exception_info.exception_type != "AgentTimeoutError":
        raise RuntimeError(f"Harbor trial failed: {result.exception_info.exception_type}")
    if result.verifier_result is None or result.verifier_result.rewards is None:
        raise RuntimeError("Harbor verifier did not return a grade")
    if result.verifier_environment_mode != "separate":
        raise RuntimeError("Harbor did not use the required separate verifier")
    reward = float(result.verifier_result.rewards["reward"])
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("Harbor verifier returned an invalid Value F1")
    report = json.loads((paths.verifier_dir / "metrics.json").read_text())
    if report.get("status") not in {"scored", "invalid_submission"}:
        raise ValueError("Unexpected native verifier status")
    if not math.isclose(float(report["value_f1"]), reward, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Native metrics and Harbor reward disagree")
    if report["status"] == "scored" and not math.isclose(
        float(report["metrics"]["extract_unified_value_f1"]), reward,
        rel_tol=0, abs_tol=1e-12,
    ):
        raise ValueError("Native metric and Harbor reward disagree")
    return reward, report


def save_evidence(client, tid: str, paths) -> None:
    """Retain only public predictions and verifier reports, never references."""
    candidates = {
        "output.json": paths.artifacts_dir / "app/output.json",
        "metrics.json": paths.verifier_dir / "metrics.json",
        "reward.txt": paths.reward_text_path,
        "test-stdout.txt": paths.test_stdout_path,
        "test-stderr.txt": paths.test_stderr_path,
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, path in candidates.items():
            if path.is_file():
                archive.add(path, arcname=name, recursive=False)
    content = buffer.getvalue()
    if len(content) > 16 * 1024 * 1024:
        raise ValueError("Diagnostic artifact exceeds 16 MiB")
    upload = client.trajectories.artifacts.create_upload(
        tid, media_type="application/gzip", size_bytes=len(content),
        md5=base64.b64encode(hashlib.md5(content, usedforsecurity=False).digest()).decode(),
    )
    request = urllib.request.Request(upload.upload_url, data=content, headers=upload.headers, method="PUT")
    with urllib.request.urlopen(request, timeout=120):
        pass
    artifact = client.trajectories.artifacts.complete_upload(tid, upload.artifact_id)
    client.trajectories.log_event(tid, event_id="harbor-evidence", name="harbor_evidence",
        payload={"artifact_id": artifact.artifact_id, "filename": "extractbench-evidence.tar.gz"})


def publish_result(client, tid: str, result, paths) -> tuple[str, float]:
    reward, report = native_grade(result, paths)
    reason = ((result.agent_result.metadata or {}).get("termination_reason", "TIMEOUT")
              if result.agent_result else "TIMEOUT")
    if reason not in {"ENV_DONE", "TIMEOUT", "TRUNCATION"}:
        raise ValueError("Unexpected actor termination reason")
    client.trajectories.log_event(tid, event_id="native-grade", name="native_grade",
        payload={"status": report["status"], "value_f1": reward,
                 "perfect": report.get("perfect"), "verifier_environment_mode": "separate",
                 "actor_termination_reason": reason})
    try:
        save_evidence(client, tid, paths)
    except Exception as error:
        # Evidence upload failure is distinct from an otherwise valid native grade.
        client.trajectories.log_event(tid, event_id="evidence-error", name="evidence_error",
            payload={"exception_type": type(error).__name__})
    client.trajectories.log_reward(tid, name="extract_unified_value_f1", value=reward)
    return reason, reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--agent-timeout", type=float)
    parser.add_argument("--verifier-smoke", action="store_true",
        help="Run Harbor's NOP agent without model calls to check native grading only")
    args = parser.parse_args()
    task_path = (ROOT / "tasks" / args.task).resolve()
    if not (task_path / "task.toml").is_file() or task_path.parent != (ROOT / "tasks").resolve():
        parser.error("unknown task")
    if args.agent_timeout is not None and args.agent_timeout <= 0:
        parser.error("--agent-timeout must be positive")

    with Client() as client:
        tid = client.trajectories.create().tid
        os.environ["TRAJECTORY_TID"] = tid
        try:
            result, paths = asyncio.run(run_trial(task_path, args.agent_timeout, args.verifier_smoke))
            reason, reward = publish_result(client, tid, result, paths)
        except Exception:
            client.trajectories.complete(tid, termination_reason="ERROR")
            raise
        completed = client.trajectories.complete(tid, termination_reason=reason)
        if completed.status not in ("completed", "finalizing"):
            raise RuntimeError(f"Trajectory completion returned {completed.status}")
        print(f"task={args.task} ending={reason} native_value_f1={reward}", flush=True)


if __name__ == "__main__":
    main()
