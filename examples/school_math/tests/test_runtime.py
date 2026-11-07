import copy
import json
from unittest.mock import create_autospec

import pytest
from trajectory import Client
from trajectory.types.create_trajectory_response import CreateTrajectoryResponse
from trajectory.types.inference.chat_completion_response import ChatCompletionResponse

from examples.school_math.runtime.math_harness import (
    actor_messages,
    calculate,
    parse_number,
    run_task,
)

TASK = {
    "question": "Solve 2*x = 3 for x.",
    "answer": "3/2",
    "source_module": "algebra__linear_1d",
}


def response(
    calls: list[tuple[str, str]], finish: str = "tool_calls"
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chat_test",
        object="chat.completion",
        created=0,
        model="policy",
        choices=[
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                        for index, (name, arguments) in enumerate(calls)
                    ],
                },
            }
        ],
    )


def mock_client(responses: list[ChatCompletionResponse]) -> Client:
    client = Client(api_key="test")
    client.chat.completions = create_autospec(client.chat.completions, instance=True)
    client.trajectories = create_autospec(client.trajectories, instance=True)
    client.trajectories.create.return_value = CreateTrajectoryResponse(tid="traj_test")
    client.chat.completions.create.side_effect = responses
    return client


@pytest.mark.parametrize("answer", ["1.5", "3/2", "+1.50", " 6/4 "])
def test_numeric_grading_accepts_exact_equivalent_values(answer: str) -> None:
    client = mock_client(
        [response([("submit_answer", json.dumps({"answer": answer}))])]
    )
    run_task(client, TASK)
    assert client.trajectories.log_reward.call_args.kwargs["value"] == 1
    client.trajectories.complete.assert_called_once_with(
        "traj_test", termination_reason="ENV_DONE"
    )


def test_wrong_answer_is_a_valid_zero_grade() -> None:
    client = mock_client([response([("submit_answer", '{"answer":"2"}')])])
    run_task(client, TASK)
    assert client.trajectories.log_reward.call_args.kwargs["value"] == 0


@pytest.mark.parametrize(
    "value",
    [
        "The answer is 3/2",
        "1 then 1.5",
        "NaN",
        "Infinity",
        "1/0",
        "__import__('os')",
        "1,500",
        "1.5 units",
    ],
)
def test_numeric_grader_rejects_prose_and_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_number(value)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').getcwd()", "2**100000", "[1][0]", "(1).__class__", "x+1", "1/0"],
)
def test_calculator_rejects_non_arithmetic_or_unbounded_work(expression: str) -> None:
    with pytest.raises(ValueError):
        calculate(expression)


def test_calculator_is_exact_and_tool_responses_are_returned() -> None:
    client = mock_client(
        [
            response([("calculate", '{"expression":"(1/3 + 1/6)*3"}')]),
            response([("submit_answer", '{"answer":"3/2"}')]),
        ]
    )
    captured = []
    replies = list(client.chat.completions.create.side_effect)

    def complete(**kwargs):
        captured.append(copy.deepcopy(kwargs["messages"]))
        return replies[len(captured) - 1]

    client.chat.completions.create.side_effect = complete
    run_task(client, TASK)
    assert json.loads(captured[1][2]["content"]) == {"result": "3/2"}
    assert captured[1][2]["tool_call_id"] == "call_0"


def test_actor_has_only_question_and_narrow_tools() -> None:
    task = TASK | {"answer": "private_expected", "private_data": "secret_fixture"}
    payload = json.dumps(actor_messages(task))
    assert "private_expected" not in payload
    assert "secret_fixture" not in payload
    client = mock_client([response([("submit_answer", '{"answer":"3/2"}')])])
    run_task(client, TASK)
    definitions = client.chat.completions.create.call_args.kwargs["extra_body"]["tools"]
    assert {tool["function"]["name"] for tool in definitions} == {
        "calculate",
        "submit_answer",
    }


def test_cutoff_does_not_dispatch_or_grade_submission() -> None:
    client = mock_client([response([("submit_answer", '{"answer":"3/2"}')], "length")])
    run_task(client, TASK)
    client.trajectories.log_reward.assert_not_called()
    client.trajectories.complete.assert_called_once_with(
        "traj_test", termination_reason="TRUNCATION"
    )


def test_malformed_tool_json_records_error_without_grade() -> None:
    client = mock_client([response([("submit_answer", '{"answer":')])])
    with pytest.raises(json.JSONDecodeError):
        run_task(client, TASK)
    client.trajectories.log_reward.assert_not_called()
    client.trajectories.complete.assert_called_once_with(
        "traj_test", termination_reason="ERROR"
    )


def test_model_service_failure_records_error_without_grade() -> None:
    client = mock_client([])
    client.chat.completions.create.side_effect = RuntimeError("model unavailable")
    with pytest.raises(RuntimeError):
        run_task(client, TASK)
    client.trajectories.log_reward.assert_not_called()
    client.trajectories.complete.assert_called_once_with(
        "traj_test", termination_reason="ERROR"
    )


def test_unexpected_finish_reason_does_not_dispatch_tools() -> None:
    client = mock_client(
        [response([("submit_answer", '{"answer":"3/2"}')], "content_filter")]
    )
    with pytest.raises(ValueError, match="finish reason"):
        run_task(client, TASK)
    client.trajectories.log_reward.assert_not_called()
    client.trajectories.complete.assert_called_once_with(
        "traj_test", termination_reason="ERROR"
    )


def test_turn_limit_is_truncation_without_grade() -> None:
    client = mock_client([response([], "stop")])
    run_task(client, TASK, max_turns=1)
    client.trajectories.log_reward.assert_not_called()
    client.trajectories.complete.assert_called_once_with(
        "traj_test", termination_reason="TRUNCATION"
    )
