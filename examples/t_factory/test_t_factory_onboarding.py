import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx2
import pytest
from trajectory import APIError, Client, TaskSpec
from trajectory.lib import DockerfileBuild
from trajectory.types.start_benchmark_diagnostics_response import StartBenchmarkDiagnosticsResponse

_ROOT = Path(__file__).parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ingest = load_module("t_factory_ingest", _ROOT / "ingest.py")
train = load_module("t_factory_train", _ROOT / "train.py")
harness = load_module("t_factory_harness", _ROOT / "runtime/t_factory_harness.py")

_TIMESTAMP = "2026-01-01T00:00:00Z"


def make_eval(reward: float | None) -> dict:
    return {
        "agent_id": "agent_math",
        "eval_run_id": "eval_math",
        "bench_id": "benchmark_math",
        "display_name": "Math diagnostic",
        "options": {},
        "status": "completed",
        "created_at": _TIMESTAMP,
        "updated_at": _TIMESTAMP,
        "reward_mean": reward,
    }


@pytest.mark.parametrize(
    "answer,reward", [("Tiny turtles travel", 1.0), ("Cats sleep", 0.0), ("Tiny cats sleep", 1 / 3)]
)
def test_first_task_records_computed_reward_on_model_trajectory(answer, reward, capsys):
    client = Mock()
    client.trajectories.create.return_value.tid = "task_first"
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=answer))
    ]
    client.trajectories.complete.return_value.status = "completed"
    harness.run_task(client, "Do you enjoy rainy weather? Why?", "openai/gpt-5.4-mini")
    request = client.chat.completions.create.call_args.kwargs
    assert request["x_trajectory_id"] == "task_first"
    assert request["messages"][1]["content"] == "Do you enjoy rainy weather? Why?"
    client.chat.completions.create.assert_called_once()
    client.trajectories.log_reward.assert_called_once_with(
        "task_first",
        reward_id="t-word-density",
        name="reward_t_word_density",
        value=reward,
    )
    client.trajectories.complete.assert_called_once_with(
        "task_first", termination_reason="ENV_DONE"
    )
    output = capsys.readouterr().out
    assert f"response={answer}" in output
    assert f"reward={reward:.6f}" in output
    assert "status=completed" in output
    client.training.create.assert_not_called()
    client.benchmarks.create.assert_not_called()


def test_first_task_reports_model_failure_without_inventing_reward():
    client = Mock()
    client.trajectories.create.return_value.tid = "task_first"
    error = APIError(
        "model failed", request=httpx2.Request("POST", "https://example.com"), body=None
    )
    client.chat.completions.create.side_effect = error
    with pytest.raises(APIError):
        harness.run_task(client, "Rain?", "openai/gpt-5.4-mini")
    client.trajectories.log_reward.assert_not_called()
    client.trajectories.complete.assert_called_once_with("task_first", termination_reason="ERROR")


def test_small_benchmark_keeps_distinct_train_and_test_tasks():
    small = ingest.build_benchmark("small", small=True)
    full = ingest.build_benchmark("full")
    assert [task.split for task in small.tasks] == ["train", "test"]
    assert len({task.env_vars["USER_PROMPT"] for task in small.tasks}) == 2
    assert small.tasks[0].env_vars == full.tasks[0].env_vars
    assert small.tasks[1].env_vars == full.tasks[128].env_vars
    assert len(full.tasks) == 192
    assert sum(task.split == "train" for task in full.tasks) == 128
    assert sum(task.split == "test" for task in full.tasks) == 64


@pytest.mark.parametrize(
    "run_status,task_status,failure,reward,error",
    [
        ("completed", "completed", None, 0.0, None),
        ("completed", "completed", None, 1.0, None),
        ("completed", "completed", None, None, "without a recorded reward"),
        ("completed", "failed", {"message": "Grader failed"}, None, "did not pass"),
        ("failed", "completed", {"message": "Run failed"}, None, "did not pass"),
        ("cancelled", "cancelled", None, None, "did not pass"),
    ],
)
def test_diagnostic_requires_execution_and_grade(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    run_status: str,
    task_status: str,
    failure: dict | None,
    reward: float | None,
    error: str | None,
) -> None:
    statuses = iter(["pending", "running", run_status])
    requests = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if request.url.path.endswith("/status"):
            data = {
                "benchmark_diagnostic_id": "diagnostic_math",
                "status": next(statuses),
            }
        elif request.url.path == "/api/v1/diagnostics/diagnostic_math":
            data = {
                "benchmark_diagnostic_id": "diagnostic_math",
                "bench_id": "benchmark_math",
                "eval_run_id": "eval_math",
                "base_model_slug": "Qwen/Qwen3.5-4B",
                "status": run_status,
                "created_at": _TIMESTAMP,
                "updated_at": _TIMESTAMP,
                "started_at": _TIMESTAMP,
                "completed_at": _TIMESTAMP,
                "failure": failure if run_status == "failed" else None,
                "task_counts": {
                    key: int(task_status == key) for key in ("completed", "failed", "cancelled")
                },
                "tasks": [
                    {
                        "task_id": "task_math",
                        "runtime_id": "runtime_math",
                        "status": task_status,
                        "failure": failure,
                        "created_at": _TIMESTAMP,
                        "updated_at": _TIMESTAMP,
                        "started_at": _TIMESTAMP,
                        "completed_at": _TIMESTAMP,
                    }
                ],
            }
        else:
            assert request.url.path == "/api/v1/eval/eval_math"
            data = make_eval(reward)
        return httpx2.Response(200, json=data)

    def start(
        client: Client,
        task: TaskSpec,
        agent_name: str,
        root: Path,
        timeout_seconds: int,
        base_model_slug: str,
    ) -> StartBenchmarkDiagnosticsResponse:
        assert agent_name == "math-example"
        assert task.runtime == DockerfileBuild("runtime/Dockerfile")
        assert task.run_command == "python -u /opt/t_factory/t_factory_harness.py"
        assert task.env_vars == {"USER_PROMPT": "Do you enjoy rainy weather? Why?"}
        assert task.split == "train"
        assert (root / "runtime" / "t_factory_harness.py").is_file()
        assert (root / "runtime" / "Dockerfile").is_file()
        assert base_model_slug == "Qwen/Qwen3.5-4B"
        return StartBenchmarkDiagnosticsResponse(
            bench_id="benchmark_math",
            benchmark_diagnostic_id="diagnostic_math",
            status="pending",
        )

    monkeypatch.setattr(ingest, "start_task_diagnostic", start)
    monkeypatch.setattr(ingest.time, "sleep", Mock())
    with Client(
        api_key="test",
        max_retries=0,
        _strict_response_validation=True,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
    ) as client:
        monkeypatch.setattr(ingest, "Client", lambda: client)
        if error:
            with pytest.raises(RuntimeError, match=error):
                ingest.diagnose("math-example")
        else:
            ingest.diagnose("math-example")
            assert f"diagnostic_reward={reward}" in capsys.readouterr().out
    assert sum(request.url.path.endswith("/status") for request in requests) == 3
    assert not any(request.method == "POST" for request in requests)


@pytest.mark.parametrize("small", [True, False])
@pytest.mark.parametrize("terminal", ["succeeded", "failed", "cancelled"])
def test_training_waits_for_progress_and_only_scores_success(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    terminal: str,
    small: bool,
) -> None:
    show_results = Mock()
    monkeypatch.setattr(train, "_print_task_results", show_results)
    task_count = 1 if small else 64
    states = iter([("pending", 0), ("running", 1), (terminal, 3)])
    state = ("pending", 0)
    requests = []
    evaluations = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        nonlocal state
        requests.append(request)
        path = request.url.path
        if path.endswith("/spec"):
            assert request.url.params["include_tasks"] == "true"
            data = {
                "bench_id": "benchmark_math",
                "task_count": task_count * 2,
                "tasks": [
                    {"task_id": f"task_{split}_{index}", "split": split}
                    for split in ("train", "test")
                    for index in range(task_count)
                ],
            }
        elif path == "/api/v1/eval":
            body = json.loads(request.content)
            assert body["bench_id"] == "benchmark_math"
            assert body["options"]["evaluation_max_samples"] == task_count
            if evaluations:
                assert state[0] == "succeeded"
                assert body["parent_checkpoint_id"] == "checkpoint_math"
            evaluations.append(body)
            data = {
                "bench_id": "benchmark_math",
                "eval_run_id": f"eval_math_{len(evaluations)}",
                "display_name": "Math evaluation",
            }
        elif path.startswith("/api/v1/eval/"):
            if path.endswith("/progress"):
                data = {
                    "eval_run_id": f"eval_math_{len(evaluations)}",
                    "status": "completed",
                    "terminal_rollouts": 16,
                    "total_rollouts": 16,
                }
            else:
                data = make_eval(0.0 if len(evaluations) == 1 else 0.5)
        elif path == "/api/v1/train":
            body = json.loads(request.content)
            assert body["bench_id"] == "benchmark_math"
            assert body["options"]["num_steps"] == 3
            assert body["options"]["train_batch_size"] == min(4, task_count)
            data = {
                "bench_id": "benchmark_math",
                "training_run_id": "training_math",
                "display_name": "Math training",
                "resolved_trajectory_options": None,
            }
        elif path == "/api/v1/train/training_math":
            state = next(states)
            data = {
                "agent_id": "agent_math",
                "training_run_id": "training_math",
                "pipeline_run_id": None,
                "display_name": "Math training",
                "status": state[0],
                "bench_id": "benchmark_math",
                "base_model_slug": "Qwen/Qwen3.5-4B",
                "options": {},
                "created_by": "test",
                "created_at": _TIMESTAMP,
                "has_training_execution": state[0] != "pending",
                "failure": None,
            }
        elif path.endswith("/progress"):
            data = {
                "training_run_id": "training_math",
                "status": state[0],
                "completed_steps": state[1],
                "total_steps": 3,
            }
        else:
            assert state[0] == "succeeded"
            assert path == "/api/v1/train/training_math/checkpoints/3"
            data = {
                "training_run_id": "training_math",
                "step_index": 3,
                "checkpoint_id": "checkpoint_math",
                "base_model_slug": "Qwen/Qwen3.5-4B",
            }
        return httpx2.Response(200, json=data)

    with Client(
        api_key="test",
        max_retries=0,
        _strict_response_validation=True,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
    ) as client:
        if terminal == "succeeded":
            result = train.train_and_evaluate(client, "benchmark_math", "Qwen/Qwen3.5-4B", 3, 0)
            assert (result[0].reward, result[1].reward) == (0.0, 0.5)
            show_results.assert_called_once_with(client, "eval_math_1", "eval_math_2")
            assert len(evaluations) == 2
        else:
            with pytest.raises(RuntimeError, match=f"status={terminal}"):
                train.train_and_evaluate(client, "benchmark_math", "Qwen/Qwen3.5-4B", 3, 0)
            assert len(evaluations) == 1
            show_results.assert_not_called()
            assert not any("/checkpoints/" in request.url.path for request in requests)
    output = capsys.readouterr().out
    assert "status=pending steps=0/3" in output
    assert "status=running steps=1/3" in output
