from ingest import build_benchmark, task_spec
from runtime.tau3_retail.protocol import TEST_IDS, TRAIN_IDS


def test_source_split_and_default_model():
    benchmark = build_benchmark("retail-test")
    assert len(benchmark.tasks) == len(TRAIN_IDS) + len(TEST_IDS) == 114
    assert len({task.name for task in benchmark.tasks}) == 114
    assert sum(task.split == "train" for task in benchmark.tasks) == 74
    task = task_spec(TRAIN_IDS[0])
    assert task.spec["user_and_judge_model"] == "gpt-4.1-2025-04-14"
    assert "OPENAI_API_KEY" in task.env_vars
    assert "OPENROUTER_API_KEY" not in task.env_vars


def test_one_model_override_routes_to_openrouter_secret():
    model = "openrouter/openai/gpt-6-luna"
    task = task_spec(TRAIN_IDS[0], user_judge_model=model)
    assert task.spec["user_and_judge_model"] == model
    assert task.env_vars["TAU3_RETAIL_USER_JUDGE_MODEL"] == model
    assert "OPENROUTER_API_KEY" in task.env_vars
    assert "OPENAI_API_KEY" not in task.env_vars
