"""Pinned public Tau 3 Banking protocol and source-integrity checks."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

SOURCE_REPOSITORY = "https://github.com/sierra-research/tau2-bench"
SOURCE_TAG = "v1.0.1"
SOURCE_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TASKS_SHA256 = "ca100722c17900aa62389dfd1058706a01cad166f1d13dabb94ac2268823f649"

DOMAIN = "banking_knowledge"
RETRIEVAL_CONFIG = "bm25_grep"
USER_AND_JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"
USER_AND_JUDGE_REASONING_EFFORT = "medium"
MAX_STEPS = 200
MAX_ERRORS = 10
MAX_RETRIES = 30
SEED = 300
REPEATS = 5

OMITTED_TASK_NUMBERS = frozenset({9, 11, 13, 30, 42})
TASK_IDS = tuple(
    f"task_{number:03d}"
    for number in range(1, 103)
    if number not in OMITTED_TASK_NUMBERS
)


def _task_digest(tasks_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(tasks_dir.glob("task_*.json")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _installed_commit() -> str | None:
    direct_url = importlib.metadata.distribution("tau2").read_text("direct_url.json")
    if direct_url is None:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def verify_source() -> None:
    """Refuse to run if either the harness code or task bytes drifted."""
    version = importlib.metadata.version("tau2")
    if version != "1.0.1":
        raise RuntimeError(f"Expected tau2 1.0.1, found {version}")
    commit = _installed_commit()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"Expected tau2 commit {SOURCE_COMMIT}, found {commit}")

    from tau2.domains.banking_knowledge.utils import (  # noqa: PLC0415
        KNOWLEDGE_TASK_SET_PATH,
    )

    tasks_dir = Path(KNOWLEDGE_TASK_SET_PATH)
    digest = _task_digest(tasks_dir)
    if digest != TASKS_SHA256:
        raise RuntimeError(f"Tau 3 Banking task digest mismatch: {digest}")
    installed_ids = tuple(path.stem for path in sorted(tasks_dir.glob("task_*.json")))
    if installed_ids != TASK_IDS:
        raise RuntimeError(
            "Tau 3 Banking task IDs do not match the published 97-task set"
        )


def load_task(task_id: str) -> object:
    if task_id not in TASK_IDS:
        raise ValueError(f"Unknown Tau 3 Banking task: {task_id}")
    from tau2.runner.helpers import get_tasks  # noqa: PLC0415

    return get_tasks(DOMAIN, task_ids=[task_id])[0]


def configure_nl_assertion_judge() -> None:
    """Pin the task_102 judge to the Artificial Analysis protocol."""
    import tau2.config as config  # noqa: PLC0415
    import tau2.evaluator.evaluator_nl_assertions as evaluator  # noqa: PLC0415

    judge_args = {
        "reasoning_effort": USER_AND_JUDGE_REASONING_EFFORT,
        "num_retries": MAX_RETRIES,
    }
    config.DEFAULT_LLM_NL_ASSERTIONS = USER_AND_JUDGE_MODEL
    config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = judge_args
    evaluator.DEFAULT_LLM_NL_ASSERTIONS = USER_AND_JUDGE_MODEL
    evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS = judge_args
