# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.10.1"]
# ///
"""Run new ideas on a managed base model or checkpoint and print unedited drafts."""

import argparse
import json
import os
import re
import shlex
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from trajectory import BenchmarkSpec, Client, SecretRef, TaskSpec
from trajectory.lib import DockerfileBuild, start_push
from trajectory.lib.platform_links import PlatformLinks
from trajectory.types.benchmarks.task_spec import EnvResources

ROOT = Path(__file__).parent


def write_package_file(path: Path, content: str) -> None:
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary_path = handle.name
    os.replace(temporary_path, path)


def package_inputs(inputs: Path, voice_examples: Path, package_dir: Path) -> list[dict]:
    if package_dir.exists():
        raise FileExistsError(f"Choose a new package directory: {package_dir}")
    tasks = [
        json.loads(line) for line in inputs.read_text().splitlines() if line.strip()
    ]
    voice = [
        json.loads(line)
        for line in voice_examples.read_text().splitlines()
        if line.strip()
    ]
    ids = set()
    for task in tasks:
        if set(task) != {"task_id", "input"}:
            raise ValueError(
                "Draft inputs contain only task_id and input; no reference posts"
            )
        task_id = task["task_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id) or task_id in ids:
            raise ValueError("Task IDs must be unique letters, digits, _ or -")
        ids.add(task_id)
        content = task["input"]
        if set(content) != {
            "notes",
            "instructions",
            "audience",
            "facts",
            "constraints",
        }:
            raise ValueError(
                "Input requires notes, instructions, audience, facts and constraints"
            )
        if not content["notes"].strip() or not content["audience"].strip():
            raise ValueError("Notes and audience must not be empty")
        if not content["instructions"].strip():
            raise ValueError("Editing instructions must not be empty")
        if type(content["constraints"]["lowercase"]) is not bool:
            raise ValueError("lowercase must be a boolean")
        limit = content["constraints"]["max_characters"]
        if type(limit) is not int or not 1 <= limit <= 280:
            raise ValueError("max_characters must be an integer in [1, 280]")
    if not tasks or not voice:
        raise ValueError("Need new inputs and voice examples")
    package_dir.mkdir(parents=True, mode=0o700)
    for name in ("writing_harness.py",):
        write_package_file(package_dir / name, (ROOT / "runtime" / name).read_text())
    write_package_file(
        package_dir / "Dockerfile",
        "FROM python:3.12-slim\nWORKDIR /opt/writing_agent\n"
        'RUN pip install --no-cache-dir "trajectory-sdk>=0.10.1" "openai>=3.46,<4"\n'
        "COPY *.py tasks.jsonl voice_examples.jsonl /opt/writing_agent/\n"
        'CMD ["sleep", "infinity"]\n',
    )
    for name, rows in (("tasks.jsonl", tasks), ("voice_examples.jsonl", voice)):
        write_package_file(
            package_dir / name,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--voice-examples", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--judge-provider", choices=("openai", "openrouter"), default="openai"
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    tasks = package_inputs(args.inputs, args.voice_examples, args.package_dir)
    spec = BenchmarkSpec(
        name="writing-agent-draft-review",
        description="Fresh notes to tweets. Scorer v2 checks faithfulness, instruction leakage, voice and postability. Judge proxy, not calibrated author preference. Development inputs; no training tasks.",
        runtime=DockerfileBuild("Dockerfile"),
        tasks=[
            TaskSpec(
                name=f"draft/{task['task_id']}",
                split="test",
                run_command=(
                    "python -u /opt/writing_agent/writing_harness.py --task-id "
                    + shlex.quote(task["task_id"])
                    + " --judge-model "
                    + shlex.quote(args.judge_model)
                    + " --judge-provider "
                    + shlex.quote(args.judge_provider)
                ),
                env_vars={
                    "OPENAI_API_KEY": SecretRef(
                        secret_ref={
                            "openai": "OPENAI_API_KEY",
                            "openrouter": "OPENROUTER_API_KEY",
                        }[args.judge_provider]
                    )
                },
                env_resources=EnvResources(network_mode="public"),
            )
            for task in tasks
        ],
    )
    client = Client()
    links = PlatformLinks.from_client(client)
    operation = start_push(
        client,
        spec,
        agent_name=args.agent_name,
        root=args.package_dir,
        build_images=True,
    )
    print(f"ingestion_operation_id={operation.id}", flush=True)
    registered = operation.result(timeout=args.timeout_seconds)
    bench_id = registered.status.bench_id
    print(f"bench_id={bench_id}", flush=True)
    evaluation = client.evals.create(
        bench_id=bench_id,
        base_model_slug=args.model,
        parent_checkpoint_id=args.checkpoint_id,
        display_name="Fresh drafts — quality judge v2, author review pending",
        options={
            "val_num_tasks": len(tasks),
            "val_trajectories_per_task": 1,
            "val_max_concurrent_trajectories": 4,
            "reasoning_effort": "off",
            "max_output_tokens_per_turn": 512,
            "max_turns_per_trajectory": 1,
            "trajectory_timeout_seconds": 300,
        },
    )
    run_id = evaluation.eval_run_id
    print(f"evaluation={links.evaluation(run_id)}", flush=True)
    print(
        "Reward is a quality-judge proxy with explicit rejection gates. Author calibration is pending.",
        flush=True,
    )
    deadline = time.monotonic() + args.timeout_seconds
    previous = None
    while time.monotonic() < deadline:
        progress = client.evals.runs.progress(run_id)
        state = (progress.status, progress.terminal_rollouts)
        if state != previous:
            print(
                f"status={progress.status} terminal={progress.terminal_rollouts}/{progress.total_rollouts}",
                flush=True,
            )
            previous = state
        if progress.status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(5)
    if progress.status != "completed":
        raise RuntimeError(
            f"Draft run {run_id} status={progress.status}; inspect it before retrying"
        )
    for row in client.trajectories.list(eval_run_id=run_id):
        trajectory = client.trajectories.retrieve(row.trajectory_id, include_steps=True)
        messages = [message for step in trajectory.steps for message in step.messages]
        drafts = [
            message.content for message in messages if message.role == "assistant"
        ]
        print(
            json.dumps(
                {
                    "task_id": row.task_id,
                    "trajectory_id": row.trajectory_id,
                    "ending": trajectory.termination_reason,
                    "draft": drafts[-1] if drafts else None,
                    "url": links.evaluation(run_id, trajectory_id=row.trajectory_id),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
