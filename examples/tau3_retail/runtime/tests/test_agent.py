"""Run one real retail task offline with a scripted user and a mocked Trajectory API."""

import json
import os

import httpx2
import pytest
import tau2.runner.batch
from tau2.data_model.message import AssistantMessage
from tau3_retail import agent
from trajectory import Client

pytestmark = pytest.mark.skipif(
    os.environ.get("TAU2_DATA_DIR") is None,
    reason="Set TAU2_DATA_DIR to a tau2-bench checkout's data directory",
)

# Task 5 has no natural-language assertions, so grading needs no judge model.
TASK_ID = "5"


def _chat_completion(content: str) -> dict:
    return {
        "id": "chat_test",
        "model": "policy",
        "object": "chat.completion",
        "created": 0,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


@pytest.mark.parametrize(
    ("reply", "leaks", "expected"),
    [
        ("Could you share your order ID?", 0, 1.0),
        ("I should ask for the order.</think>\n\nCould you share your order ID?", 1, 0.0),
    ],
)
def test_task_publishes_gated_reward_and_completes(monkeypatch, reply, leaks, expected):
    requests = []

    def respond(request):
        body = json.loads(request.content or b"{}")
        requests.append((request.url.path, request.headers, body))
        path = request.url.path
        if path == "/api/v1/trajectories":
            return httpx2.Response(200, json={"tid": "tid_test"})
        if path == "/v1/chat/completions":
            return httpx2.Response(200, json=_chat_completion(reply))
        if path.endswith(("/rewards", "/events")):
            return httpx2.Response(
                200, json={"ok": True, "trajectory_id": "tid_test", "status": "active"}
            )
        if path.endswith("/complete"):
            return httpx2.Response(200, json={"trajectory_id": "tid_test", "status": "completed"})
        raise AssertionError(path)

    user_turns = iter(
        [
            AssistantMessage(role="assistant", content="I need help with my order."),
            AssistantMessage(role="assistant", content="Never mind. ###STOP###"),
        ]
    )
    monkeypatch.setattr("tau2.user.user_simulator.generate", lambda **_: next(user_turns))

    run_single_task = tau2.runner.batch.run_single_task

    def solved(*args, **kwargs):
        result = run_single_task(*args, **kwargs)
        result.reward_info.reward = 1.0
        return result

    monkeypatch.setattr("tau2.runner.batch.run_single_task", solved)

    with Client(
        api_key="test",
        base_url="https://example.com",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:
        assert agent.run_task(client, TASK_ID) == "tid_test"

    paths = [path for path, _, _ in requests]
    assert paths[0] == "/api/v1/trajectories"
    assert paths[-1] == "/api/v1/trajectories/tid_test/complete"
    chat = [entry for entry in requests if entry[0] == "/v1/chat/completions"]
    assert len(chat) == 1
    assert chat[0][1]["X-Trajectory-Id"] == "tid_test"
    assert chat[0][2]["max_tokens"] == 2048
    assert any(m["role"] == "system" and "retail" in m["content"] for m in chat[0][2]["messages"])

    reward = next(body for path, _, body in requests if path.endswith("/rewards"))
    assert reward["name"] == "reward_accuracy"
    assert reward["value"] == expected
    event = next(body for path, _, body in requests if path.endswith("/events"))
    assert event["payload"]["task_id"] == TASK_ID
    assert event["payload"]["upstream_termination_reason"] == "user_stop"
    assert event["payload"]["reasoning_leaks"] == leaks
    assert event["payload"]["upstream_reward"] == 1.0
    assert requests[-1][2] == {"termination_reason": "ENV_DONE"}


def test_runner_exception_completes_trajectory_as_error(monkeypatch):
    requests = []

    def respond(request):
        requests.append((request.url.path, json.loads(request.content or b"{}")))
        if request.url.path == "/api/v1/trajectories":
            return httpx2.Response(200, json={"tid": "tid_test"})
        if request.url.path.endswith("/complete"):
            return httpx2.Response(200, json={"trajectory_id": "tid_test", "status": "completed"})
        raise AssertionError(request.url.path)

    def fail(*_args, **_kwargs):
        raise ConnectionError("user simulator unavailable")

    monkeypatch.setattr("tau2.runner.batch.run_single_task", fail)
    with Client(
        api_key="test",
        base_url="https://example.com",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:
        with pytest.raises(ConnectionError):
            agent.run_task(client, TASK_ID)
    assert requests[-1] == (
        "/api/v1/trajectories/tid_test/complete",
        {"termination_reason": "ERROR"},
    )
