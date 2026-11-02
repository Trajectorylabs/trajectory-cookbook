import json
from types import SimpleNamespace

import httpx2
import pytest
from tau2.data_model.simulation import RewardInfo, TerminationReason
from tau3_retail import agent
from trajectory import Client


@pytest.mark.parametrize(
    "reason, output_limited_replies, leaked, expected_reward, expected_ending",
    [
        (TerminationReason.USER_STOP, False, 1, 1.0, "ENV_DONE"),
        (TerminationReason.USER_STOP, 1, 0, 1.0, "ENV_DONE"),
        (TerminationReason.AGENT_ERROR, False, 0, None, "ERROR"),
    ],
)
def test_upstream_reward_and_endings(
    monkeypatch,
    reason,
    output_limited_replies,
    leaked,
    expected_reward,
    expected_ending,
):
    requests = []

    def respond(request):
        requests.append((request.url.path, json.loads(request.content or b"{}")))
        if request.url.path == "/api/v1/trajectories":
            return httpx2.Response(200, json={"tid": "tid_test"})
        return httpx2.Response(
            200,
            json={
                "ok": True,
                "trajectory_id": "tid_test",
                "status": "finalizing"
                if request.url.path.endswith("/complete")
                else "active",
            },
        )

    source_result = SimpleNamespace(
        reward_info=RewardInfo(reward=1.0, info={}),
        termination_reason=reason,
    )
    models = []
    monkeypatch.setattr(agent, "configure_nl_assertion_judge", models.append)
    monkeypatch.setattr(agent, "load_task", lambda task_id: object())
    monkeypatch.setattr(
        agent,
        "register_policy",
        lambda *args: SimpleNamespace(
            calls=1,
            reasoning_leaks=leaked,
            malformed_tool_calls=0,
            output_limited_replies=output_limited_replies,
        ),
    )
    monkeypatch.setattr(
        "tau2.runner.batch.run_single_task", lambda *args, **kwargs: source_result
    )

    with Client(
        api_key="test",
        base_url="https://example.com",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:
        agent.run_task(client, "0")

    rewards = [body for path, body in requests if path.endswith("/rewards")]
    event = next(body for path, body in requests if path.endswith("/events"))
    assert (rewards[0]["value"] if rewards else None) == expected_reward
    assert event["payload"]["reasoning_leaks"] == leaked
    assert event["payload"]["output_limited_replies"] == output_limited_replies
    assert requests[-1][1] == {"termination_reason": expected_ending}
    assert models == ["gpt-4.1-2025-04-14"]


@pytest.mark.parametrize("output_limited_replies", [0, 1])
def test_malformed_tool_arguments_get_zero_grade(monkeypatch, output_limited_replies):
    requests = []

    def respond(request):
        requests.append((request.url.path, json.loads(request.content or b"{}")))
        if request.url.path == "/api/v1/trajectories":
            return httpx2.Response(200, json={"tid": "tid_test"})
        return httpx2.Response(
            200, json={"ok": True, "trajectory_id": "tid_test", "status": "completed"}
        )

    monkeypatch.setattr(agent, "configure_nl_assertion_judge", lambda model: None)
    monkeypatch.setattr(agent, "load_task", lambda task_id: object())
    monkeypatch.setattr(
        agent,
        "register_policy",
        lambda *args: SimpleNamespace(
            calls=1,
            reasoning_leaks=0,
            malformed_tool_calls=1,
            output_limited_replies=output_limited_replies,
        ),
    )

    def bad_call(*args, **kwargs):
        raise json.JSONDecodeError("bad tool arguments", "{", 1)

    monkeypatch.setattr("tau2.runner.batch.run_single_task", bad_call)
    with Client(
        api_key="test",
        base_url="https://example.com",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:
        agent.run_task(client, "0")

    reward = next(body for path, body in requests if path.endswith("/rewards"))
    assert reward["value"] == 0.0
    assert requests[-1][1] == {"termination_reason": "ENV_DONE"}
