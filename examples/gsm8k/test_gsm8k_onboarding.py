import json

import httpx2
import pytest
import train
from trajectory import Client

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


@pytest.mark.parametrize("terminal", ["succeeded", "failed", "cancelled"])
def test_training_waits_for_progress_and_only_scores_success(
    capsys: pytest.CaptureFixture[str],
    terminal: str,
) -> None:
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
                "task_count": 2,
                "tasks": [
                    {"task_id": "task_train", "split": "train"},
                    {"task_id": "task_test", "split": "test"},
                ],
            }
        elif path == "/api/v1/eval":
            body = json.loads(request.content)
            assert body["bench_id"] == "benchmark_math"
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
            assert (result.baseline, result.final) == (0.0, 0.5)
            assert len(evaluations) == 2
        else:
            with pytest.raises(RuntimeError, match=f"status={terminal}"):
                train.train_and_evaluate(client, "benchmark_math", "Qwen/Qwen3.5-4B", 3, 0)
            assert len(evaluations) == 1
            assert not any("/checkpoints/" in request.url.path for request in requests)
    output = capsys.readouterr().out
    assert "status=pending steps=0/3" in output
    assert "status=running steps=1/3" in output
