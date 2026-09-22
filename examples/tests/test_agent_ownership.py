import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import httpx
import pytest
from trajectory import Client

_EXAMPLES = Path(__file__).resolve().parents[1]
_HARNESSES = {
    "gsm8k": "gsm8k_harness.py",
    "constraint_challenge": "constraint_harness.py",
}


def _load_example(example: str, filename: str) -> ModuleType:
    path = _EXAMPLES / example / filename
    spec = importlib.util.spec_from_file_location(f"{example}_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=_HARNESSES)
def example(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    for name in (
        "TRAJECTORY_API_KEY",
        "TRAJECTORY_TOKEN",
        "TRAJECTORY_TID",
        "MODEL_ENDPOINT_ACCESS_TOKEN",
        "MODEL_ENDPOINT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAJECTORY_BASE_URL", "https://api.example.com")
    return request.param


def test_ingestion_requires_agent_before_preparing_benchmark(
    example: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_example(example, "ingest.py")
    work = Mock(side_effect=AssertionError("Ingestion started without an agent"))
    for name in ("Client", "build_benchmark", "push"):
        monkeypatch.setattr(module, name, work)
    if example == "gsm8k":
        monkeypatch.setattr(module, "_load_rows", work)
    monkeypatch.setattr(sys, "argv", ["ingest.py", "--skip-build"])

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2
    assert "--agent-id" in capsys.readouterr().err
    work.assert_not_called()


def test_ingestion_sends_selected_agent_to_registration(
    example: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_example(example, "ingest.py")
    if example == "gsm8k":
        monkeypatch.setattr(
            module,
            "_load_rows",
            lambda split, limit: [
                {"question": "What is 6 times 7?", "answer": "#### 42"}
            ],
        )
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/benchmark-ingestion/sessions":
            return httpx.Response(
                201,
                json={
                    "session_id": "iop_test",
                    "bench_id": "bm_test",
                    "accepted": True,
                },
            )
        assert request.url.path == "/api/v1/benchmark-ingestion/operations/iop_test"
        return httpx.Response(
            200,
            json={
                "operation_id": "iop_test",
                "bench_id": "bm_test",
                "status": "succeeded",
                "stage": "registering",
                "attempt": 1,
                "max_attempts": 1,
                "build_images": False,
                "registered_tasks": 2,
            },
        )

    with Client(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        max_retries=0,
    ) as client:
        monkeypatch.setattr(module, "Client", lambda: client)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "ingest.py",
                "--name",
                "selected-agent-benchmark",
                "--agent-id",
                "agt_selected",
                "--skip-build",
            ],
        )
        assert module.main() == 0

    assert len(requests) == 2
    assert requests[0].method == "POST"
    body = json.loads(requests[0].content)
    assert body["agent_id"] == "agt_selected"
    assert body["metadata"]["name"] == "selected-agent-benchmark"
    assert body["build_images"] is False
    assert "bench_id=bm_test" in capsys.readouterr().out


@pytest.mark.parametrize("positive_reward", [True, False])
def test_runtime_uses_resolved_trajectory_for_inference_and_grading(
    example: str, positive_reward: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_example(example, f"runtime/{_HARNESSES[example]}")
    monkeypatch.setenv("TRAJECTORY_TOKEN", "test-trajectory-token")
    monkeypatch.setenv("MODEL_ENDPOINT_ACCESS_TOKEN", "test-endpoint-token")
    monkeypatch.setenv("MODEL_ENDPOINT_ID", "test-model-endpoint")
    monkeypatch.setenv("GSM8K_QUESTION", "What is 6 times 7?")
    monkeypatch.setenv("GSM8K_ANSWER", "#### 42")
    monkeypatch.setenv("USER_PROMPT", "Share your view on arithmetic.")
    message = {"role": "assistant", "content": "Tt a" if positive_reward else "a"}
    if example == "gsm8k":
        message["tool_calls"] = [
            {
                "id": "call_submit",
                "type": "function",
                "function": {
                    "name": "submit_answer",
                    "arguments": json.dumps(
                        {"answer": "42" if positive_reward else "41"}
                    ),
                },
            }
        ]
    trajectory_id = "tid_resolved_for_selected_agent"
    requests = []
    responses = [
        {"tid": trajectory_id},
        {
            "id": "chat_test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        },
        {"ok": True, "trajectory_id": trajectory_id, "status": "running"},
        {"trajectory_id": trajectory_id, "status": "completed"},
    ]

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=responses[len(requests) - 1])

    with Client(
        http_client=httpx.Client(transport=httpx.MockTransport(handle)), max_retries=0
    ) as client:
        monkeypatch.setattr(module, "Client", lambda: client)
        module.main()

    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/api/v1/trajectories"),
        ("POST", "/v1/chat/completions"),
        ("POST", f"/api/v1/trajectories/{trajectory_id}/rewards"),
        ("POST", f"/api/v1/trajectories/{trajectory_id}/complete"),
    ]
    for request in requests:
        assert request.headers["Authorization"] == "Bearer test-trajectory-token"
    for request in requests[:2]:
        assert request.headers["x-model-endpoint-access-token"] == "test-endpoint-token"
    assert requests[1].headers["X-Trajectory-Id"] == trajectory_id
    assert requests[1].headers["X-Model-Endpoint-Id"] == "test-model-endpoint"
    reward = json.loads(requests[2].content)
    expected_positive = 1.0 if example == "gsm8k" else 0.5
    assert reward["value"] == (expected_positive if positive_reward else 0.0)
    assert reward["name"] == (
        "reward_accuracy" if example == "gsm8k" else "reward_t_density"
    )
