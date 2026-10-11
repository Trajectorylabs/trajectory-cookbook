"""Verify the packaged harness uses the published SDK reward and completion API."""

import importlib.util
import json
import sys
from pathlib import Path

import httpx2
import pytest
from trajectory import Client

ROOT = Path(__file__).parent


@pytest.mark.parametrize("answer, expected_reward", [("42", 1.0), ("41", 0.0)])
def test_task_records_reward_and_completes_with_published_sdk(
    tmp_path, monkeypatch, answer, expected_reward
):
    spec = importlib.util.spec_from_file_location(
        "gsm8k_harness", ROOT / "runtime/gsm8k_harness.py"
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    requests = []

    def respond(request):
        body = json.loads(request.content or b"{}")
        requests.append((request.url.path, request.headers, body))
        if request.url.path == "/api/v1/trajectories":
            return httpx2.Response(200, json={"tid": "tid_test"})
        if request.url.path == "/v1/chat/completions":
            return httpx2.Response(
                200,
                json={
                    "id": "chat_test",
                    "model": "policy",
                    "object": "chat.completion",
                    "created": 0,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_test",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_answer",
                                            "arguments": json.dumps({"answer": answer}),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )
        if request.url.path.endswith("/rewards"):
            return httpx2.Response(
                200, json={"ok": True, "trajectory_id": "tid_test", "status": "active"}
            )
        if request.url.path.endswith("/complete"):
            return httpx2.Response(
                200, json={"trajectory_id": "tid_test", "status": "completed"}
            )
        raise AssertionError(request.url.path)

    task = tmp_path / "task.json"
    task.write_text(
        json.dumps({"question": "What is 6 times 7?", "expected_answer": "#### 42"})
    )
    monkeypatch.setattr(sys, "argv", ["gsm8k_harness", "--task-file", str(task)])
    with Client(
        api_key="test",
        base_url="https://example.com",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:
        monkeypatch.setattr(harness, "Client", lambda **kwargs: client)
        harness.main()
    assert len(requests) == 4
    assert requests[1][1]["X-Trajectory-Id"] == "tid_test"
    assert requests[2][2]["name"] == "reward_accuracy"
    assert requests[2][2]["value"] == expected_reward
    assert requests[3][2] == {"termination_reason": "ENV_DONE"}
