from unittest.mock import Mock

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.user.user_simulator_base import UserState
from tau3_banking.user_simulator import BankingUserSimulator, EmptyUserResponseError


def simulator(monkeypatch, responses):
    generate = Mock(side_effect=responses)
    monkeypatch.setattr("tau2.user.user_simulator.generate", generate)
    user = BankingUserSimulator(llm="test-model", instructions="Test scenario")
    monkeypatch.setattr(user.generate_next_message.retry, "wait", lambda _: 0)
    return user, generate


def test_empty_response_retries_same_history_without_mutating_state(monkeypatch):
    user, generate = simulator(
        monkeypatch,
        [
            AssistantMessage(role="assistant", content=None),
            AssistantMessage(role="assistant", content="  "),
            AssistantMessage(role="assistant", content="Please check my balance."),
        ],
    )
    state = UserState(system_messages=[], messages=[])
    incoming = AssistantMessage(role="assistant", content="How can I help?")

    response, next_state = user.generate_next_message(incoming, state)

    assert response.content == "Please check my balance."
    assert state.messages == []
    assert len(next_state.messages) == 2
    assert generate.call_count == 3
    assert all(
        call.kwargs["messages"] == generate.call_args_list[0].kwargs["messages"]
        for call in generate.call_args_list
    )


def test_tool_only_user_response_is_valid_without_retry(monkeypatch):
    tool = ToolCall(id="call_test", name="test_tool", arguments={})
    user, generate = simulator(
        monkeypatch,
        [
            AssistantMessage(role="assistant", content=None, tool_calls=[tool]),
        ],
    )
    response, _ = user.generate_next_message(
        AssistantMessage(role="assistant", content="Continue"),
        UserState(system_messages=[], messages=[]),
    )
    response.validate()
    assert response.tool_calls[0].requestor == "user"
    assert generate.call_count == 1


def test_repeated_empty_responses_fail_after_three_attempts(monkeypatch):
    user, generate = simulator(
        monkeypatch,
        [
            AssistantMessage(role="assistant", content=None),
        ]
        * 3,
    )
    state = UserState(system_messages=[], messages=[])
    with pytest.raises(EmptyUserResponseError):
        user.generate_next_message(
            AssistantMessage(role="assistant", content="Continue"), state
        )
    assert generate.call_count == 3
    assert state.messages == []


def test_other_simulator_errors_propagate_without_retry(monkeypatch):
    user, generate = simulator(monkeypatch, [ValueError("Invalid model configuration")])
    with pytest.raises(ValueError, match="Invalid model configuration"):
        user.generate_next_message(
            AssistantMessage(role="assistant", content="Continue"),
            UserState(system_messages=[], messages=[]),
        )
    assert generate.call_count == 1
