"""Offline checks of the uploaded task specs; no model calls or credentials required."""

import importlib.util
import sys
from pathlib import Path

from trajectory import SecretRef

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("tau3_retail_ingest", ROOT / "ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


def test_benchmark_preserves_upstream_split():
    benchmark = ingest.build_benchmark("test")
    splits = {task.name: task.split for task in benchmark.tasks}
    assert len(splits) == 114
    assert sum(split == "train" for split in splits.values()) == 74
    assert sum(split == "test" for split in splits.values()) == 40
    assert splits["retail/5"] == "test"
    assert splits["retail/0"] == "train"


def test_tasks_run_one_upstream_task_with_injected_secrets_only():
    for task in ingest.build_benchmark("test").tasks:
        task_id = task.name.removeprefix("retail/")
        assert task.run_command.endswith(f"-m tau3_retail.agent {task_id}")
        assert task.env_vars["OPENAI_API_KEY"] == SecretRef(secret_ref="OPENAI_API_KEY")
        assert not any(name.startswith("TRAJECTORY_") for name in task.env_vars)
        assert task.env_resources.network_mode == "public"
        assert task.spec["source_commit"] == "fc0055dc4e0a316c3f83133267fbd6faaa770992"


def test_smoketest_uses_one_task_per_split():
    benchmark = ingest.build_benchmark(
        "smoke", (ingest.TRAIN_IDS[0], ingest.TEST_IDS[0])
    )
    assert [task.split for task in benchmark.tasks] == ["train", "test"]


def test_upload_excludes_local_environments():
    ignored = (ROOT / ".dockerignore").read_text().split()
    assert "**/.venv/" in ignored
