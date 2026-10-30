import json
from types import SimpleNamespace

import httpx2
import pytest
from tau2.data_model.simulation import RewardInfo, TerminationReason
from tau3_banking import agent
from tau3_banking.policy_provider import TrajectoryPolicy
from trajectory import Client


def sdk_client(requests, finish_reason="stop"):
    def respond(request):
        body = json.loads(request.content or b"{}")
        requests.append((request.url.path.rsplit("/", 1)[-1], request.headers, body))
        if request.url.path == "/api/v1/trajectories":
            return httpx2.Response(200, json={"tid": "tid_test"})
        if request.url.path == "/v1/chat/completions":
            message = {"role": "assistant", "content": "Hello"}
            return httpx2.Response(
                200,
                json={
                    "id": "chat_test",
                    "model": "policy",
                    "object": "chat.completion",
                    "created": 0,
                    "choices": [
                        {"index": 0, "finish_reason": finish_reason, "message": message}
                    ],
                },
            )
        status = "finalizing" if request.url.path.endswith("/complete") else "active"
        return httpx2.Response(
            200, json={"ok": True, "trajectory_id": "tid_test", "status": status}
        )

    return Client(
        api_key="test",
        base_url="https://example.com",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    )


@pytest.mark.parametrize(
    "upstream, expected",
    [
        (TerminationReason.USER_STOP, "ENV_DONE"),
        (TerminationReason.MAX_STEPS, "MAX_STEPS"),
        (TerminationReason.AGENT_ERROR, "ERROR"),
        (TerminationReason.TOO_MANY_ERRORS, "ERROR"),
    ],
)
def test_error_endings_complete_without_reward(monkeypatch, upstream, expected):
    reward_info = RewardInfo(reward=0.0, info={"note": "test"})
    result = SimpleNamespace(termination_reason=upstream, reward_info=reward_info)
    monkeypatch.setattr(agent, "verify_source", lambda: None)
    monkeypatch.setattr(agent, "configure_nl_assertion_judge", lambda: None)
    monkeypatch.setattr(agent, "load_task", lambda task_id: object())
    monkeypatch.setattr(agent, "register_user_simulator", lambda: None)
    monkeypatch.setattr("tau2.runner.batch.run_single_task", lambda *a, **k: result)
    requests = []
    with sdk_client(requests) as client:
        agent.run_task(client, "task_001")

    endpoints = [endpoint for endpoint, _, _ in requests]
    assert endpoints[-2:] == ["events", "complete"]
    assert ("rewards" in endpoints) == (expected != "ERROR")
    event = requests[-2][2]["payload"]
    assert event["reward_info"] == reward_info.model_dump(mode="json")
    assert requests[-1][2] == {"termination_reason": expected}


@pytest.mark.parametrize(
    "finish_reason, truncated",
    [("stop", False), ("length", True), ("max_tokens", True)],
)
def test_policy_forwards_params_and_detects_cut_off(finish_reason, truncated):
    requests = []
    with sdk_client(requests, finish_reason) as client:
        policy = TrajectoryPolicy(client, "tid_test")
        policy.completion(
            model="trajectory_sdk/policy",
            messages=[{"role": "user", "content": "Hi"}],
            optional_params={"max_tokens": 16, "max_retries": 30},
        )

    _, headers, body = requests[0]
    assert headers["X-Trajectory-Id"] == "tid_test"
    assert "X-Model-Request-Id" not in headers
    assert body["max_tokens"] == 16
    assert "max_retries" not in body
    assert policy.truncated is truncated
