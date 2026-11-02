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
        (TerminationReason.TOO_MANY_ERRORS, "ENV_DONE"),
    ],
)
def test_source_reward_and_endings(monkeypatch, upstream, expected):
    reward_info = RewardInfo(reward=0.0, info={"note": "test"})
    result = SimpleNamespace(termination_reason=upstream, reward_info=reward_info)
    monkeypatch.setattr(agent, "verify_source", lambda: None)
    monkeypatch.setattr(agent, "configure_nl_assertion_judge", lambda model: None)
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


def test_source_grade_survives_earlier_output_limit(monkeypatch):
    reward_info = RewardInfo(reward=1.0, info={"note": "test"})
    result = SimpleNamespace(
        termination_reason=TerminationReason.USER_STOP, reward_info=reward_info
    )
    monkeypatch.setattr(agent, "verify_source", lambda: None)
    monkeypatch.setattr(agent, "configure_nl_assertion_judge", lambda model: None)
    monkeypatch.setattr(agent, "load_task", lambda task_id: object())
    monkeypatch.setattr(agent, "register_user_simulator", lambda: None)
    monkeypatch.setattr(
        agent,
        "register_policy",
        lambda *a: SimpleNamespace(output_limited_replies=1, calls=1),
    )
    monkeypatch.setattr("tau2.runner.batch.run_single_task", lambda *a, **k: result)
    requests = []
    with sdk_client(requests) as client:
        agent.run_task(client, "task_001")

    assert "rewards" in [endpoint for endpoint, _, _ in requests]
    assert requests[-1][2] == {"termination_reason": "ENV_DONE"}


@pytest.mark.parametrize(
    "finish_reason, output_limited_replies",
    [("stop", 0), ("length", 1), ("max_tokens", 1)],
)
def test_policy_forwards_params_and_counts_output_limits(
    finish_reason, output_limited_replies
):
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
    assert policy.output_limited_replies == output_limited_replies
