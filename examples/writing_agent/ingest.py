# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.10.1"]
# ///
"""Package reviewed writing tasks, diagnose one task, or ingest the benchmark."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from trajectory import BenchmarkSpec, Client, SecretRef, TaskSpec
from trajectory.lib import DockerfileBuild, push, start_task_diagnostic
from trajectory.lib.platform_links import PlatformLinks
from trajectory.types.benchmarks.task_spec import EnvResources

ROOT = Path(__file__).parent
OPTIONS = {
    "reasoning_effort": "off",
    "max_output_tokens_per_turn": 512,
    "max_turns_per_trajectory": 1,
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def package_tasks(
    data_dir: Path,
    reviewed_tasks: Path,
    package_dir: Path,
    allow_agent_prepared: bool = False,
) -> list[dict]:
    if package_dir.exists():
        raise FileExistsError(f"Choose a new package directory: {package_dir}")
    voice = read_jsonl(data_dir / "voice_examples.jsonl")
    sources = {
        split: {row["id"]: row for row in read_jsonl(data_dir / f"{split}_seeds.jsonl")}
        for split in ("train", "test")
    }
    tasks = read_jsonl(reviewed_tasks)
    ids, used_sources = set(), set()
    voice_ids = {row["id"] for row in voice}
    for task in tasks:
        task_id, split = task["task_id"], task["split"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id) or task_id in ids:
            raise ValueError(
                "Task IDs must be unique and contain only letters, digits, _ or -"
            )
        ids.add(task_id)
        if split not in sources:
            raise ValueError(
                "Upload only train/test tasks; keep validation seeds offline"
            )
        grading = task["private_grading"]
        origin = grading["brief_origin"]
        agent_prepared = allow_agent_prepared and origin == "synthetic_agent_prepared"
        if grading["author_reviewed"] is not True and not agent_prepared:
            raise ValueError(f"Task {task_id} has not been author reviewed")
        source_ids = grading["source_post_ids"]
        if origin not in {
            "synthetic_author_reviewed",
            "synthetic_agent_prepared",
            "author_written",
        }:
            raise ValueError("Unknown brief origin")
        if origin.startswith("synthetic_") and not source_ids:
            raise ValueError("Synthetic tasks require a source post")
        for source_id in source_ids:
            if (
                source_id not in sources[split]
                or source_id in used_sources
                or source_id in voice_ids
            ):
                raise ValueError("Source post is reused or assigned to the wrong split")
            used_sources.add(source_id)
        content = task["input"]
        if not content["notes"].strip() or not content["audience"].strip():
            raise ValueError("Notes and audience must not be empty")
        if not content["instructions"].strip():
            raise ValueError("Editing instructions must not be empty")
        if type(content["constraints"]["lowercase"]) is not bool:
            raise ValueError("lowercase must be a boolean")
        limit = content["constraints"]["max_characters"]
        if type(limit) is not int or not 1 <= limit <= 280:
            raise ValueError("max_characters must be an integer in [1, 280]")
        if not isinstance(content["facts"], list) or not isinstance(
            grading["must_preserve"], list
        ):
            raise TypeError("facts and must_preserve must be lists")
        if not grading["reference_post"].strip():
            raise ValueError("A reference post is required")
        if set(content) != {
            "notes",
            "instructions",
            "audience",
            "facts",
            "constraints",
        }:
            raise ValueError(
                "Only notes, instructions, audience, facts and constraints belong in writer input"
            )
    if {task["split"] for task in tasks} != {"train", "test"} or not voice:
        raise ValueError(
            "Need voice examples and reviewed tasks in both train and test"
        )
    package_dir.mkdir(parents=True, mode=0o700)
    for name in ("Dockerfile", "writing_harness.py"):
        with NamedTemporaryFile(dir=package_dir, delete=False) as handle:
            temporary_path = handle.name
        shutil.copyfile(ROOT / "runtime" / name, temporary_path)
        shutil.copymode(ROOT / "runtime" / name, temporary_path)
        os.replace(temporary_path, package_dir / name)
    for name, rows in (("tasks.jsonl", tasks), ("voice_examples.jsonl", voice)):
        path = package_dir / name
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=package_dir, delete=False
        ) as handle:
            handle.write(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
            temporary_path = handle.name
        os.replace(temporary_path, path)
    return tasks


def build_benchmark(
    name: str,
    tasks: list[dict],
    judge_model: str,
    voice_sha256: str,
    judge_provider: str = "openai",
) -> BenchmarkSpec:
    judge_secrets = {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
    judge_secret = judge_secrets[judge_provider]
    return BenchmarkSpec(
        name=name,
        description=json.dumps(
            {
                "use_case": "new notes to author-voice posts",
                "scoring": "quality_proxy_v2",
                "scorer_sha256": hashlib.sha256(
                    (ROOT / "runtime" / "writing_harness.py").read_bytes()
                ).hexdigest(),
                "judge_model": judge_model,
                "judge_provider": judge_provider,
                "voice_sha256": voice_sha256,
            }
        ),
        runtime=DockerfileBuild("Dockerfile"),
        tasks=[
            TaskSpec(
                name=f"writing/{task['task_id']}",
                split=task["split"],
                run_command=(
                    "python -u /opt/writing_agent/writing_harness.py "
                    f"--task-id {shlex.quote(task['task_id'])} --judge-model {shlex.quote(judge_model)} "
                    f"--judge-provider {shlex.quote(judge_provider)}"
                ),
                env_vars={"OPENAI_API_KEY": SecretRef(secret_ref=judge_secret)},
                env_resources=EnvResources(network_mode="public"),
                tags=["writing-agent"],
            )
            for task in tasks
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=("package", "diagnose", "ingest"), default="package"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--name", default="writing-agent")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--judge-provider", choices=("openai", "openrouter"), default="openai"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--allow-agent-prepared",
        action="store_true",
        help="Allow explicitly labeled synthetic briefs for a technical workflow test",
    )
    args = parser.parse_args()
    tasks = package_tasks(
        args.data_dir, args.tasks, args.package_dir, args.allow_agent_prepared
    )
    benchmark = build_benchmark(
        args.name,
        tasks,
        args.judge_model,
        hashlib.sha256(
            (args.package_dir / "voice_examples.jsonl").read_bytes()
        ).hexdigest(),
        args.judge_provider,
    )
    print(
        f"package_dir={args.package_dir} tasks={len(tasks)} action={args.action}",
        flush=True,
    )
    if args.action == "package":
        return
    client = Client()
    links = PlatformLinks.from_client(client)
    agent_id = client.agents.retrieve_by_name(args.agent_name).agent_id
    if args.action == "ingest":
        result = push(
            client,
            benchmark,
            agent_name=args.agent_name,
            root=args.package_dir,
            build_images=True,
        )
        print(f"bench_id={result.bench_id} tasks={result.task_count}", flush=True)
        print(f"benchmark={links.spec(agent_id, result.bench_id)}", flush=True)
        return
    task = next(task for task in benchmark.tasks if task.split == "train")
    task.runtime = benchmark.runtime
    diagnostic = start_task_diagnostic(
        client,
        task,
        agent_name=args.agent_name,
        root=args.package_dir,
        base_model_slug=args.model,
        options=OPTIONS,
    )
    diagnostic_id = diagnostic.benchmark_diagnostic_id
    print(f"diagnostic_id={diagnostic_id}", flush=True)
    print(f"benchmark={links.spec(agent_id, diagnostic.bench_id)}", flush=True)
    previous = None
    while True:
        status = client.diagnostics.get_status(diagnostic_id).status
        if status != previous:
            print(f"diagnostic_status={status}", flush=True)
            previous = status
        if status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(5)
    report = client.diagnostics.get_diagnostics(diagnostic_id)
    print(f"evaluation={links.evaluation(report.eval_run_id)}", flush=True)
    print(f"status={report.status}", flush=True)
    for result in report.tasks:
        print(
            f"task={result.status} ending={result.termination_reason} reward={result.reward}",
            flush=True,
        )
    if (
        report.status != "completed"
        or report.failure
        or not report.tasks
        or any(
            task.status != "completed" or task.reward is None or task.failure
            for task in report.tasks
        )
    ):
        raise RuntimeError(
            "Writing diagnostic failed; inspect its report before ingestion"
        )


if __name__ == "__main__":
    main()
