"""Pinned public Tau 3 Retail protocol and source-integrity checks."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

SOURCE_REPOSITORY = "https://github.com/sierra-research/tau2-bench"
SOURCE_TAG = "v1.0.1"
SOURCE_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
SOURCE_SHA256 = "39da8c56587dcbdfc299af1a512c43ac045f5e6fc1bb575b1741d4a9314b0371"
SOURCE_FILES = ("db.json", "policy.md", "split_tasks.json", "tasks.json")

DOMAIN = "retail"
USER_AND_JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"
USER_AND_JUDGE_REASONING_EFFORT = "medium"
AGENT_MAX_TOKENS = 2048
MAX_STEPS = 200
MAX_ERRORS = 10
MAX_RETRIES = 30
EMPTY_USER_RESPONSE_MAX_ATTEMPTS = 3
SEED = 300

# Upstream's official retail split (data/tau2/domains/retail/split_tasks.json).
TRAIN_IDS = (
    "0", "1", "2", "3", "4", "6", "7", "8", "10", "11", "13", "14", "15", "16", "19",
    "20", "21", "22", "23", "24", "25", "28", "29", "30", "31", "34", "35", "37", "41",
    "43", "44", "46", "47", "48", "50", "52", "54", "57", "58", "59", "63", "66", "67",
    "69", "72", "73", "75", "76", "78", "80", "81", "82", "83", "84", "85", "87", "88",
    "89", "91", "92", "93", "95", "96", "98", "99", "103", "104", "105", "106", "107",
    "109", "110", "112", "113",
)  # fmt: skip
TEST_IDS = (
    "5", "9", "12", "17", "18", "26", "27", "32", "33", "36", "38", "39", "40", "42",
    "45", "49", "51", "53", "55", "56", "60", "61", "62", "64", "65", "68", "70", "71",
    "74", "77", "79", "86", "90", "94", "97", "100", "101", "102", "108", "111",
)  # fmt: skip
TASK_IDS = tuple(sorted(TRAIN_IDS + TEST_IDS, key=int))


def source_digest(domain_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((domain_dir / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _installed_commit() -> str | None:
    direct_url = importlib.metadata.distribution("tau2").read_text("direct_url.json")
    if direct_url is None:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id")


def verify_source() -> None:
    """Refuse to run if the harness code, retail data, or official split drifted."""
    version = importlib.metadata.version("tau2")
    if version != "1.0.1":
        raise RuntimeError(f"Expected tau2 1.0.1, found {version}")
    commit = _installed_commit()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"Expected tau2 commit {SOURCE_COMMIT}, found {commit}")

    from tau2.domains.retail.utils import RETAIL_DATA_DIR  # noqa: PLC0415

    verify_domain_data(Path(RETAIL_DATA_DIR))


def verify_domain_data(domain_dir: Path) -> None:
    digest = source_digest(domain_dir)
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"Tau 3 Retail source digest mismatch: {digest}")
    split = json.loads((domain_dir / "split_tasks.json").read_text())
    if tuple(split["train"]) != TRAIN_IDS or tuple(split["test"]) != TEST_IDS:
        raise RuntimeError("Tau 3 Retail split does not match the pinned train/test IDs")


def load_task(task_id: str) -> object:
    if task_id not in TASK_IDS:
        raise ValueError(f"Unknown Tau 3 Retail task: {task_id}")
    from tau2.runner.helpers import get_tasks  # noqa: PLC0415

    return get_tasks(DOMAIN, task_ids=[task_id])[0]


def configure_nl_assertion_judge() -> None:
    """Pin the natural-language assertion judge used by retail rewards."""
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
