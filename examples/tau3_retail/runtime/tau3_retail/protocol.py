"""Pinned public Tau 3 Retail protocol."""

SOURCE_REPOSITORY = "https://github.com/sierra-research/tau2-bench"
SOURCE_TAG = "v1.0.1"
SOURCE_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"

DOMAIN = "retail"
USER_AND_JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"
USER_AND_JUDGE_REASONING_EFFORT = "medium"
AGENT_MAX_TOKENS = 2048
MAX_STEPS = 200
MAX_ERRORS = 10
MAX_RETRIES = 30
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
