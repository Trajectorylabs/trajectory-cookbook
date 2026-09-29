import json
import runpy
import sys
from collections import deque
from pathlib import Path

import httpx
import pytest
import trajectory

ROOT = Path(__file__).resolve().parents[1]


def test_upload_registers_owner_and_builds_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("trajectory.lib.benchmarks.time.sleep", lambda seconds: None)
    pending = deque(
        [
            (
                "POST",
                "/api/v1/benchmark-ingestion/sessions",
                {
                    "session_id": "operation_example",
                    "bench_id": "bench_example",
                    "accepted": True,
                },
            ),
            (
                "GET",
                "/api/v1/benchmark-ingestion/operations/operation_example",
                {
                    "operation_id": "operation_example",
                    "bench_id": "bench_example",
                    "status": "succeeded",
                    "stage": "registering",
                    "attempt": 1,
                    "max_attempts": 3,
                    "registered_tasks": 2,
                    "build_images": False,
                },
            ),
            (
                "POST",
                "/api/v1/benchmarks/bench_example/images/build",
                {
                    "bench_id": "bench_example",
                    "images": [
                        {
                            "task_id": "task_example",
                            "provider": "runloop",
                            "build_status": "building",
                        }
                    ],
                },
            ),
            (
                "GET",
                "/api/v1/benchmarks/bench_example/images",
                {
                    "bench_id": "bench_example",
                    "images": [
                        {
                            "task_id": "task_example",
                            "provider": "runloop",
                            "build_status": "ready",
                        }
                    ],
                },
            ),
        ]
    )
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        method, path, body = pending.popleft()
        assert (request.method, request.url.path) == (method, path)
        requests.append(request)
        return httpx.Response(200, json=body)

    with trajectory.Client(
        api_key="test-only",
        base_url="https://api.example.com",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        monkeypatch.setattr(trajectory, "Client", lambda: client)
        monkeypatch.setattr(sys, "argv", ["upload.py", "--agent-id", "agent_example"])
        runpy.run_path(str(ROOT / "upload.py"), run_name="__main__")
    assert not pending
    assert json.loads(requests[0].content)["agent_id"] == "agent_example"


def test_start_evaluates_one_task_once(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "eval_run_id": "eval_example",
                "bench_id": "bench_example",
                "status": "pending",
            },
        )

    with trajectory.Client(
        api_key="test-only",
        base_url="https://api.example.com",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        monkeypatch.setattr(trajectory, "Client", lambda: client)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate.py",
                "start",
                "--bench-id",
                "bench_example",
                "--model",
                "model_example",
            ],
        )
        runpy.run_path(str(ROOT / "evaluate.py"), run_name="__main__")
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body["model_slug"] == "model_example"
    assert body["eval_options"] == {"max_samples": 1, "pass_at_k": 1}


@pytest.mark.parametrize(
    ("answer", "model_status", "completion_status", "expected_reward"),
    [
        ("11", 200, 200, 1.0),
        ("12", 200, 200, 0.0),
        (None, 500, 200, None),
        (None, 500, 403, None),
    ],
)
def test_harness_grades_registered_task_and_preserves_model_failures(
    monkeypatch: pytest.MonkeyPatch,
    answer: str | None,
    model_status: int,
    completion_status: int,
    expected_reward: float | None,
) -> None:
    upload = runpy.run_path(str(ROOT / "upload.py"))
    task = next(
        task for task in upload["build_benchmark"]().tasks if task.split == "test"
    )
    for name, value in task.env_vars.items():
        monkeypatch.setenv(name, value)
    pending = deque(
        [
            ("/api/v1/trajectories", 200, {"tid": "traj_example"}),
            (
                "/v1/chat/completions",
                model_status,
                {
                    "id": "chat_example",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "policy",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": answer},
                        }
                    ],
                }
                if model_status == 200
                else {"detail": "model failed"},
            ),
        ]
    )
    if expected_reward is not None:
        pending.append(
            (
                "/api/v1/trajectories/traj_example/rewards",
                200,
                {"ok": True, "trajectory_id": "traj_example", "status": "RUNNING"},
            )
        )
    pending.append(
        (
            "/api/v1/trajectories/traj_example/complete",
            completion_status,
            {"trajectory_id": "traj_example", "status": "COMPLETED"}
            if completion_status == 200
            else {"detail": "completion failed"},
        )
    )
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        path, status, body = pending.popleft()
        assert request.method == "POST" and request.url.path == path
        requests.append(request)
        return httpx.Response(status, json=body)

    with trajectory.Client(
        api_key="test-only",
        base_url="https://api.example.com",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        monkeypatch.setattr(trajectory, "Client", lambda: client)
        if model_status == 500:
            with pytest.raises(trajectory.InternalServerError) as caught:
                runpy.run_path(str(ROOT / "runtime/harness.py"))
            if completion_status == 403:
                assert "PermissionDeniedError" in caught.value.__notes__[0]
        else:
            runpy.run_path(str(ROOT / "runtime/harness.py"))

    assert not pending
    model_body = json.loads(requests[1].content)
    assert model_body["messages"][-1]["content"] == task.env_vars["QUESTION"]
    assert task.env_vars["EXPECTED_ANSWER"] not in json.dumps(model_body)
    assert requests[1].headers["X-Trajectory-Id"] == "traj_example"
    if expected_reward is not None:
        assert json.loads(requests[-2].content)["value"] == expected_reward
        assert json.loads(requests[-1].content) == {}
    else:
        assert json.loads(requests[-1].content) == {"termination_reason": "ERROR"}


def test_status_reads_saved_run_without_starting_another(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        requests.append(request.url.path)
        if request.url.path.endswith("/progress"):
            return httpx.Response(
                200,
                json={
                    "eval_run_id": "eval_example",
                    "status": "completed",
                    "failure": None,
                    "terminal_rollouts": 1,
                    "total_rollouts": 1,
                },
            )
        return httpx.Response(
            200,
            json={
                "trajectory_rewards": [
                    {
                        "trajectory_id": "traj_example",
                        "task_id": "task_example",
                        "reward": 1.0,
                    }
                ]
            },
        )

    with trajectory.Client(
        api_key="test-only",
        base_url="https://api.example.com",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        monkeypatch.setattr(trajectory, "Client", lambda: client)
        monkeypatch.setattr(
            sys, "argv", ["evaluate.py", "status", "--eval-run-id", "eval_example"]
        )
        runpy.run_path(str(ROOT / "evaluate.py"), run_name="__main__")
    assert len(requests) == 2 and all("eval_example" in path for path in requests)
    assert '"reward": 1.0' in capsys.readouterr().out
