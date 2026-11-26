# /// script
# requires-python = ">=3.12"
# dependencies = ["trajectory-sdk>=0.10.13", "huggingface-hub>=1.0,<2"]
# ///
"""Stage official ExtractBench Harbor tasks, upload, and diagnose Qwen 4B."""

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import time
import tomllib
import uuid
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, get_operation, start_push
from trajectory.lib.platform_links import PlatformLinks

SOURCE = "llamaindex/ExtractBench"
SOURCE_COMMIT = "f9b910906cd7e73cc9eb5379a8d5d935c67b92c4"
MANIFEST_SHA256 = "0452463e5b1be2c627c0b7d1a076531f0fd301b3297affdb7d9613735bc04840"
MODEL = "Qwen/Qwen3.5-4B"
EXAMPLE = Path(__file__).resolve().parent
OPTIONS = {
    "reasoning_effort": "off",
    "temperature": 0,
    "max_turns_per_trajectory": 300,
    "max_output_tokens_per_turn": 262142,
    "max_output_tokens_per_trajectory": 2097152,
    "trajectory_timeout_seconds": 4200,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_manifest() -> tuple[Path, dict]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(SOURCE, "harbor-manifest.json", repo_type="dataset", revision=SOURCE_COMMIT))
    if sha256(path) != MANIFEST_SHA256:
        raise ValueError("Pinned upstream manifest digest changed")
    manifest = json.loads(path.read_text())
    if len(manifest["tasks"]) != 370 or manifest["reward"] != "extract_unified_value_f1":
        raise ValueError("Unexpected source task/score contract")
    return path, manifest


def task_config(source: Path, name: str, record: dict) -> tuple[dict, list[Path]]:
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError("Unsafe source task name")
    root = source / "tasks" / name
    files = []
    for relative, expected in record["files"].items():
        # Reference solutions are never downloaded or packaged. The private verifier
        # reference remains outside the environment/ Docker build context.
        if relative.split("/")[0] in {"solution", "cheat"}:
            continue
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe source file path")
        if sha256(path) != expected:
            raise ValueError(f"Source file digest mismatch: {name}/{relative}")
        files.append(path)
    config = tomllib.loads((root / "task.toml").read_text())
    verifier = config["verifier"]
    if (
        config["artifacts"] != ["/app/output.json"]
        or config["agent"]["network_mode"] != "allowlist"
        or config["agent"]["allowed_hosts"] != []
        or config["agent"]["timeout_sec"] != 1800
        or verifier["environment_mode"] != "separate"
        or verifier["environment"]["network_mode"] != "no-network"
        or verifier["timeout_sec"] != 1800
        or config["metadata"]["source_case_id"] != record["case_id"]
    ):
        raise ValueError("Unexpected source isolation/deadline contract")
    return config, files


def stage(workspace: Path, selected: list[str], all_tasks: bool, source_dir: Path | None,
          verifier_smoke: bool = False) -> tuple[Path, BenchmarkSpec]:
    manifest_path, manifest = source_manifest()
    default_task = min(name for name in manifest["tasks"] if name.startswith("short--"))
    names = sorted(manifest["tasks"]) if all_tasks else sorted(set(selected or [default_task]))
    if any(name not in manifest["tasks"] for name in names):
        raise ValueError("Select published task names from harbor-manifest.json")
    if source_dir is None:
        from huggingface_hub import snapshot_download

        source_dir = Path(snapshot_download(
            SOURCE, repo_type="dataset", revision=SOURCE_COMMIT,
            local_dir=workspace / "source",
            allow_patterns=[f"tasks/{name}/{relative}" for name in names
                            for relative in manifest["tasks"][name]["files"]
                            if relative.split("/")[0] not in {"solution", "cheat"}],
        ))
    context = workspace / "context"
    context.mkdir(parents=True, exist_ok=True)
    for name in ("Dockerfile", "run_task.py"):
        shutil.copy2(EXAMPLE / "runtime" / name, context / name)
    shutil.copy2(EXAMPLE.parents[1] / "LICENSE", context / "LICENSE")
    (context / "NOTICE").write_text(
        f"Official ExtractBench Harbor export: https://huggingface.co/datasets/{SOURCE}\n"
        f"Dataset commit: {SOURCE_COMMIT}\nDataset card license: Apache-2.0.\n"
        "Source documents and third-party dependencies retain their original rights and licenses.\n"
    )
    shutil.copy2(manifest_path, context / "harbor-manifest.json")
    specs = []
    with tarfile.open(context / "tasks.tar.gz", "w:gz", compresslevel=1) as archive:
        for name in names:
            record = manifest["tasks"][name]
            config, files = task_config(source_dir, name, record)
            for path in files:
                archive.add(path, arcname=str(path.relative_to(source_dir)), recursive=False)
            actor, verifier = config["environment"], config["verifier"]["environment"]
            specs.append(TaskSpec(
                name=f"extractbench/{name}", split="test",
                run_command=f"python -u /opt/extractbench/run_task.py {name}" + (" --verifier-smoke" if verifier_smoke else ""),
                env_resources={
                    "docker_engine": True, "network_mode": "public",
                    "cpus": max(actor["cpus"], verifier["cpus"]) + 2,
                    "memory_mb": max(actor["memory_mb"], verifier["memory_mb"]) + 4096,
                },
                tags=["extractbench", config["metadata"]["source_split"], "evaluation-only"],
                spec={"dataset_revision": SOURCE_COMMIT, "source_task": name,
                      **config["metadata"], "evaluation_only": True, "do_not_train": True,
                      "verifier_smoke": verifier_smoke},
            ))
    return context, BenchmarkSpec(
        name="extractbench-harbor-verifier-check" if verifier_smoke else "extractbench-harbor",
        description="Official ExtractBench Harbor tasks; native Value F1, separate offline verifier; evaluation only.",
        runtime=DockerfileBuild("Dockerfile"), tasks=specs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name")
    parser.add_argument("--agent-id")
    parser.add_argument("--workspace", type=Path, default=Path(".extractbench"))
    parser.add_argument("--source-dir", type=Path, help="Reuse local files; every included file is checked against the pinned manifest")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", action="append", default=[])
    selection.add_argument("--all", action="store_true", help="Upload all 370 evaluation cases")
    parser.add_argument("--stage-only", action="store_true", help="Download and check locally; no SDK calls")
    parser.add_argument("--upload-only", action="store_true", help="Upload/build without starting a diagnostic")
    parser.add_argument("--verifier-smoke", action="store_true", help="Official Harbor nop agent: verify native zero without model calls, not extraction quality")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    if not args.stage_only and not args.agent_name:
        parser.error("--agent-name is required for SDK calls")
    if args.verifier_smoke and args.all:
        parser.error("--verifier-smoke checks one task, not a full extraction benchmark")
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    context, spec = stage(workspace, args.task, args.all, args.source_dir.resolve() if args.source_dir else None,
                          args.verifier_smoke)
    print(f"tasks={len(spec.tasks)} all_evaluation_only=True", flush=True)
    if args.stage_only:
        return
    identity = {"agent_name": args.agent_name, "agent_id": args.agent_id,
                "dataset_revision": SOURCE_COMMIT,
                "verifier_smoke": args.verifier_smoke,
                "tasks": [task.name for task in spec.tasks],
                "runtime_hashes": {name: sha256(context / name) for name in ("Dockerfile", "run_task.py")}}
    state_path = workspace / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("identity", identity) != identity:
        raise ValueError("Use a fresh --workspace for different tasks, runtime or agent")
    with Client(timeout=120, max_retries=0) as client:
        organization_key = client.organizations.retrieve_current()["organization_key"]
        if state.get("organization_key", organization_key) != organization_key:
            raise ValueError("Saved workspace belongs to another organization")
        state["organization_key"] = organization_key
        print("Organization:", organization_key, flush=True)
        links = PlatformLinks.from_client(client)
        if args.agent_id and client.agents.retrieve(args.agent_id).name != args.agent_name:
            raise ValueError("Agent ID/name mismatch")
        state["identity"] = identity
        state.setdefault("upload_key", str(uuid.uuid4()))
        save(state_path, state)
        if state.get("operation_id"):
            operation = get_operation(client, state["operation_id"])
        else:
            operation = start_push(client, spec,
                                   **({"agent_id": args.agent_id} if args.agent_id else {"agent_name": args.agent_name}),
                                   root=context, build_images=True, idempotency_key=state["upload_key"])
            state["operation_id"] = operation.id
            save(state_path, state)
        print(f"operation_id={operation.id}", flush=True)
        result = operation.result(timeout=2700)
        state["bench_id"] = result.status.bench_id
        save(state_path, state)
        agent_id = client.benchmarks.retrieve(state["bench_id"]).agent_id
        print("Spec:", state["bench_id"], links.spec(agent_id, state["bench_id"]), flush=True)
        if args.upload_only:
            return
        catalog = client.evals.list_options(bench_id=state["bench_id"], base_model_slug=args.model)
        supported = next(model for model in catalog.models if model.base_model_slug == args.model)
        bounds = supported.model_dump(mode="json")["options"]
        diagnostic_options = dict(OPTIONS)
        for name in ("max_output_tokens_per_turn", "max_output_tokens_per_trajectory"):
            diagnostic_options[name] = int(bounds[name]["max"])
        if state.get("diagnostic_model", args.model) != args.model:
            raise ValueError("This workspace already diagnosed another model")
        if state.get("diagnostic_options", diagnostic_options) != diagnostic_options:
            raise ValueError("This workspace already diagnosed different generation options")
        if not state.get("diagnostic_id"):
            state["diagnostic_model"] = args.model
            state["diagnostic_options"] = diagnostic_options
            state.setdefault("diagnostic_key", str(uuid.uuid4()))
            save(state_path, state)
            response = client.diagnostics.start_benchmark(
                bench_id=state["bench_id"], base_model_slug=args.model, options=diagnostic_options,
                extra_headers={"Idempotency-Key": state["diagnostic_key"]},
            )
            state["diagnostic_id"] = response.benchmark_diagnostic_id
            save(state_path, state)
        print("Diagnostic:", state["diagnostic_id"], flush=True)
        while client.diagnostics.get_status(state["diagnostic_id"]).status not in {"completed", "failed", "cancelled"}:
            time.sleep(10)
        report = client.diagnostics.get_diagnostics(state["diagnostic_id"])
        save(workspace / "diagnostic.json", report.model_dump(mode="json"))
        print("Inspect:", links.evaluation(report.eval_run_id), flush=True)
        print(report.to_json(), flush=True)
        if report.status != "completed" or report.failure or not report.tasks or any(
            task.status != "completed" or task.failure or task.termination_reason != "ENV_DONE" or task.reward is None
            for task in report.tasks
        ):
            raise RuntimeError("Diagnostic did not finish normally with native grades; inspect diagnostic.json")


if __name__ == "__main__":
    main()
