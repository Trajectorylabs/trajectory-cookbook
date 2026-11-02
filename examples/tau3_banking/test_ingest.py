from ingest import build_benchmark
from runtime.tau3_banking.protocol import TASK_IDS, USER_AND_JUDGE_MODEL


def test_native_model_and_split():
    benchmark = build_benchmark("banking-test")
    assert len(benchmark.tasks) == len(TASK_IDS) == 97
    assert sum(task.split == "train" for task in benchmark.tasks) == 80
    task = benchmark.tasks[0]
    assert task.spec["user_and_judge_model"] == USER_AND_JUDGE_MODEL
    assert "OPENAI_API_KEY" in task.env_vars
    assert "OPENROUTER_API_KEY" not in task.env_vars


def test_one_model_override_routes_to_openrouter_secret():
    model = "openrouter/openai/gpt-6-luna"
    task = build_benchmark("banking-test", model).tasks[0]
    assert task.spec["user_and_judge_model"] == model
    assert task.spec["user_and_judge_reasoning_effort"] is None
    assert task.env_vars["TAU3_BANKING_USER_JUDGE_MODEL"] == model
    assert "OPENROUTER_API_KEY" in task.env_vars
    assert "OPENAI_API_KEY" not in task.env_vars
