"""Pinned public tau2-bench retail and airline protocol and source-integrity checks."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

SOURCE_REPOSITORY = "https://github.com/sierra-research/tau2-bench"
SOURCE_TAG = "v1.0.1"
SOURCE_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"

# sha256 of tasks.json and split_tasks.json at SOURCE_COMMIT.
DOMAIN_FILE_SHA256 = {
    "retail": {
        "tasks.json": "8e03ebce7901bd6218e7a7dc3105faa9324091a68058f7fe61c65262868812e8",
        "split_tasks.json": "ed0580ec52575b63fbf76568af42490da6ee7783ecb4aa81af46961291358f20",
    },
    "airline": {
        "tasks.json": "ccd8ba737b4cc371415af70151187788f728d6108d0916e73bb4317b40542052",
        "split_tasks.json": "b22ced4d9a9850ac9aea31c53bdcb6d6009058140bd9acc7db37c1d36222ba8b",
    },
}

# Same simulated-user and judge settings as the Tau 3 Banking example.
USER_AND_JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"
USER_AND_JUDGE_REASONING_EFFORT = "medium"
MAX_STEPS = 200
MAX_ERRORS = 10
MAX_RETRIES = 30
EMPTY_USER_RESPONSE_MAX_ATTEMPTS = 3
SEED = 300


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_dir(domain: str, root: Path) -> Path:
    return root / "tau2" / "domains" / domain


def task_split(domain: str, task_id: str, root: Path) -> str:
    splits = json.loads((data_dir(domain, root) / "split_tasks.json").read_text())
    for split in ("train", "test"):
        if task_id in splits[split]:
            return split
    raise ValueError(f"Unknown {domain} task: {task_id}")


def _installed_commit() -> str | None:
    direct_url = importlib.metadata.distribution("tau2").read_text("direct_url.json")
    if direct_url is None:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def verify_source(domain: str, root: Path) -> None:
    """Refuse to run if the harness code or the domain's task bytes drifted."""
    if domain not in DOMAIN_FILE_SHA256:
        raise ValueError(f"Unsupported domain: {domain}")
    version = importlib.metadata.version("tau2")
    if version != "1.0.1":
        raise RuntimeError(f"Expected tau2 1.0.1, found {version}")
    commit = _installed_commit()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"Expected tau2 commit {SOURCE_COMMIT}, found {commit}")
    for name, expected in DOMAIN_FILE_SHA256[domain].items():
        actual = _sha256(data_dir(domain, root) / name)
        if actual != expected:
            raise RuntimeError(f"{domain}/{name} digest mismatch: {actual}")


def load_task(domain: str, task_id: str) -> object:
    from tau2.runner.helpers import get_tasks  # noqa: PLC0415

    return get_tasks(domain, task_ids=[task_id])[0]


def configure_nl_assertion_judge(model: str = USER_AND_JUDGE_MODEL) -> None:
    """Use the selected model for retail's natural-language assertions."""
    import tau2.config as config  # noqa: PLC0415
    import tau2.evaluator.evaluator_nl_assertions as evaluator  # noqa: PLC0415

    judge_args = {"num_retries": MAX_RETRIES}
    if model == USER_AND_JUDGE_MODEL:
        judge_args["reasoning_effort"] = USER_AND_JUDGE_REASONING_EFFORT
    config.DEFAULT_LLM_NL_ASSERTIONS = model
    config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = judge_args
    evaluator.DEFAULT_LLM_NL_ASSERTIONS = model
    evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS = judge_args
