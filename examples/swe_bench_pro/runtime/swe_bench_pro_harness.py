"""Run one SWE-Bench Pro task: the agent works offline, and a fresh container grades its patch."""

import ast
import json
import platform
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, Template
from trajectory import Client

SOURCE_COMMIT = "66f92766bba642462d4bbe5479e83f91f9211862"
UPSTREAM = Path("/opt/swe_bench_pro/upstream/v2")
# mini-swe-agent's LocalEnvironment default; neither its mini.yaml nor mini_toolcall.yaml sets one.
COMMAND_TIMEOUT_SECONDS = 30
# mini-swe-agent's DefaultAgent ends a run after this many format errors in a row.
MAX_CONSECUTIVE_FORMAT_ERRORS = 3
PULL_TIMEOUT_SECONDS = 1800
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
# mini-swe-agent's bash tool, which the upstream prompt in mini_toolcall.yaml is written for.
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}
FIND_WORKDIR = "test -d /app && echo /app || echo /testbed"
# Upstream's patch capture in locked_mini_swe.py.
CAPTURE_PATCH = "git add -A && git diff --cached"
# The upstream patch-replay agent's apply command.
APPLY_PATCH = (
    "cd /app 2>/dev/null || cd /testbed; git apply --verbose /tmp/model.patch "
    "|| git apply --3way /tmp/model.patch || patch --fuzz=3 -p1 -i /tmp/model.patch"
)

Execute = Callable[[str, int], dict]


def render(template: str, **variables: object) -> str:
    return Template(template, undefined=StrictUndefined).render(**variables)


def load_prompts() -> dict:
    config = yaml.safe_load(
        (UPSTREAM / "tooling/configs/mini_toolcall.yaml").read_text()
    )
    return {**config["agent"], **config["model"], "env": config["environment"]["env"]}


def docker(*args: str, timeout: float, check: bool = True) -> str:
    result = subprocess.run(
        ["docker", *args],
        capture_output=True,
        check=False,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    if check and result.returncode:
        raise RuntimeError(f"docker {args[0]} failed: {result.stderr[-2000:]}")
    return result.stdout


def wait_for_docker() -> None:
    for _ in range(60):
        info = subprocess.run(["docker", "info"], capture_output=True, check=False)
        if info.returncode == 0:
            return
        time.sleep(2)
    raise RuntimeError("The Docker daemon did not start")


def start_container(image: str, *options: str) -> str:
    return docker(
        "run", "-d", *options, "--entrypoint", "sleep", image, "infinity", timeout=300
    ).strip()


def container_executor(container: str, workdir: str, env: dict[str, str]) -> Execute:
    def execute(command: str, timeout: int) -> dict:
        environment = [f"--env={key}={value}" for key, value in env.items()]
        args = ["docker", "exec", "-w", workdir, *environment, container]
        # LocalEnvironment runs commands with subprocess's shell=True, which is /bin/sh.
        args += ["timeout", "-k", "5", str(timeout), "sh", "-c", command]
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            errors="replace",
            timeout=timeout + 60,
        )
        timed_out = result.returncode == 124
        return {
            "output": result.stdout,
            "returncode": -1 if timed_out else result.returncode,
            "exception_info": "An error occurred while executing the command: "
            f"Command '{command}' timed out after {timeout} seconds"
            if timed_out
            else "",
        }

    return execute


def is_submission(output: dict) -> bool:
    lines = output["output"].lstrip().splitlines()
    return (
        bool(lines) and lines[0].strip() == SUBMIT_MARKER and output["returncode"] == 0
    )


def tool_call_error(tool_calls: list) -> str:
    """Return mini-swe-agent's error for a reply's tool calls, or "" if all are valid."""
    if not tool_calls:
        return "No tool calls found in the response. Every response MUST include at least one tool call."
    for call in tool_calls:
        error, arguments = "", {}
        try:
            arguments = json.loads(call.function.arguments)
        except (TypeError, ValueError) as e:
            error = f"Error parsing tool call arguments: {e}."
        if call.function.name != "bash":
            error += f"Unknown tool '{call.function.name}'."
        if not isinstance(arguments, dict) or "command" not in arguments:
            error += "Missing 'command' argument in bash tool call."
        if error:
            return error.strip()
    return ""


def run_agent(
    client: Client,
    tid: str,
    prompts: dict,
    instruction: str,
    execute: Execute,
    time_limit_seconds: float,
) -> tuple[str, int]:
    """Return the attempt's ending and the number of model calls."""
    deadline = time.monotonic() + time_limit_seconds
    system = platform.uname()._asdict()
    messages = [
        {"role": "system", "content": render(prompts["system_template"], **system)},
        {
            "role": "user",
            "content": render(prompts["instance_template"], task=instruction, **system),
        },
    ]
    # Upstream sets no step limit; the time budget bounds an attempt.
    turn = format_errors = 0
    while True:
        turn += 1
        if time.monotonic() >= deadline:
            return "TIMEOUT", turn - 1
        response = client.chat.completions.create(
            model="swe-bench-pro",  # Managed runs select the model.
            messages=messages,
            extra_body={"tools": [BASH_TOOL]},
            x_trajectory_id=tid,
        )
        metadata = (response.model_extra or {}).get("metadata") or {}
        if metadata.get("trajectory_termination_reason") == "trajectory_cancelled":
            # The platform stopped the attempt at one of its limits and keeps that ending.
            return "TRUNCATION", turn
        choice = response.choices[0]
        if choice.finish_reason in ("length", "max_tokens"):
            return "TRUNCATION", turn
        tool_calls = choice.message.tool_calls or []
        if error := tool_call_error(tool_calls):
            # As in mini-swe-agent, the reply leaves the history and none of its calls run.
            format_errors += 1
            if format_errors == MAX_CONSECUTIVE_FORMAT_ERRORS:
                return "ENV_DONE", turn
            content = render(
                prompts["format_error_template"],
                error=error,
                actions=[],
                has_tool_calls=bool(tool_calls),
                finish_reason=choice.finish_reason,
            )
            messages.append({"role": "user", "content": content})
            continue
        format_errors = 0
        messages.append(
            {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [call.model_dump(mode="json") for call in tool_calls],
            }
        )
        for call in tool_calls:
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                return "TIMEOUT", turn
            command = json.loads(call.function.arguments)["command"]
            output = execute(command, min(COMMAND_TIMEOUT_SECONDS, remaining))
            if is_submission(output):
                return "ENV_DONE", turn
            content = render(prompts["observation_template"], output=output)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )


def parse_test_list(value: str | list) -> list[str]:
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Some upstream configs store Python-literal lists.
        return ast.literal_eval(value)


def read_grade(logs: Path, config: dict) -> tuple[float, dict]:
    """Read the upstream verifier's reward and count the required tests that passed."""
    reward = float((logs / "reward.txt").read_text().strip())
    if reward not in (0.0, 1.0):
        raise ValueError(f"Unexpected verifier reward {reward}")
    output = logs / "output.json"
    passed = set()
    if output.exists():
        tests = json.loads(output.read_text()).get("tests", [])
        passed = {test.get("name") for test in tests if test.get("status") == "PASSED"}
    counts = {}
    for key in ("fail_to_pass", "pass_to_pass"):
        required = set(parse_test_list(config.get(key, [])))
        counts[key] = {"passed": len(required & passed), "total": len(required)}
    return reward, counts


def grade(
    image: str, task_dir: Path, patch: str, timeout_seconds: float
) -> tuple[float, dict]:
    with tempfile.TemporaryDirectory() as logs:
        container = start_container(
            image,
            f"--mount=type=bind,src={task_dir / 'tests'},dst=/tests,readonly",
            f"--mount=type=bind,src={logs},dst=/logs/verifier",
        )
        try:
            details = {"patch_applied": None}
            if patch:
                patch_file = Path(logs) / "model.patch"
                patch_file.write_text(patch)
                docker(
                    "cp", str(patch_file), f"{container}:/tmp/model.patch", timeout=120
                )
                applied = subprocess.run(
                    ["docker", "exec", container, "bash", "-c", APPLY_PATCH],
                    capture_output=True,
                    check=False,
                    text=True,
                    errors="replace",
                    timeout=300,
                )
                details["patch_applied"] = applied.returncode == 0
                details["patch_apply_output"] = (applied.stdout + applied.stderr)[
                    -4000:
                ]
            verifier = subprocess.run(
                ["docker", "exec", container, "bash", "/tests/test.sh"],
                capture_output=True,
                check=False,
                text=True,
                errors="replace",
                timeout=timeout_seconds,
            )
            details["verifier_output"] = (verifier.stdout + verifier.stderr)[-8000:]
            print(details["verifier_output"], flush=True)
            config = json.loads((task_dir / "tests/config.json").read_text())
            reward, counts = read_grade(Path(logs), config)
            return reward, {**details, **counts}
        finally:
            docker("rm", "-f", container, timeout=120, check=False)


def run_task(client: Client, tid: str, instance_id: str) -> str:
    task_dir = UPSTREAM / "tasks" / instance_id
    task = tomllib.loads((task_dir / "task.toml").read_text())
    image = task["environment"]["docker_image"]
    prompts = load_prompts()

    wait_for_docker()
    docker("pull", image, timeout=PULL_TIMEOUT_SECONDS)
    digest = docker(
        "image", "inspect", "--format={{index .RepoDigests 0}}", image, timeout=60
    )
    agent = start_container(image, "--network=none")
    try:
        workdir = docker("exec", agent, "sh", "-c", FIND_WORKDIR, timeout=60).strip()
        reason, turns = run_agent(
            client,
            tid,
            prompts,
            (task_dir / "instruction.md").read_text(),
            container_executor(agent, workdir, prompts["env"]),
            # locked_mini_swe.py ends the agent a minute before the task's budget.
            task["agent"]["timeout_sec"] - 60,
        )
        patch = docker(
            "exec",
            "-w",
            workdir,
            agent,
            "sh",
            "-c",
            CAPTURE_PATCH,
            timeout=300,
        )
    finally:
        docker("rm", "-f", agent, timeout=120, check=False)

    reward, details = grade(image, task_dir, patch, task["verifier"]["timeout_sec"])
    client.trajectories.log_reward(tid, name="resolved", value=reward)
    client.trajectories.log_event(
        tid,
        event_id="swe-bench-pro-result",
        name="swe_bench_pro_result",
        payload={
            "instance_id": instance_id,
            "source_commit": SOURCE_COMMIT,
            "image": digest.strip(),
            "ending": reason,
            "model_calls": turns,
            "resolved": reward,
            "patch": patch[:200_000],
            **details,
        },
    )
    print(
        f"instance={instance_id} ending={reason} model_calls={turns} resolved={reward}",
        flush=True,
    )
    return reason


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: swe_bench_pro_harness.py INSTANCE_ID")
    instance_id = sys.argv[1]
    if not (UPSTREAM / "tasks" / instance_id / "task.toml").is_file():
        raise SystemExit(f"Unknown SWE-Bench Pro instance: {instance_id}")
    client = Client(max_retries=20, timeout=1200)
    tid = client.trajectories.create().tid
    try:
        reason = run_task(client, tid, instance_id)
    except Exception:
        client.trajectories.complete(tid, termination_reason="ERROR")
        raise
    completed = client.trajectories.complete(tid, termination_reason=reason)
    if completed.status not in ("completed", "finalizing"):
        raise RuntimeError(f"Trajectory completion returned {completed.status}")


if __name__ == "__main__":
    main()
