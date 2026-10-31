from unittest.mock import MagicMock, patch

import pytest
from lab_core.harness.adapters import trajectory
from lab_core.harness.run import create_adapter


def _adapter():
    adapter = object.__new__(trajectory.TrajectoryAdapter)
    adapter.trajectory = MagicMock()
    adapter.tid = "tid_test"
    adapter.judge_model = "gpt-5.4-mini"
    adapter._open = True
    return adapter


def test_session_sends_reasoning_effort_without_summary():
    client = MagicMock()
    responses = trajectory._Responses(client, "tid_test")

    responses.create(
        model="trajectory-session",
        input=[],
        reasoning={"effort": "low", "summary": "auto"},
    )

    assert client.responses.create.call_args.kwargs["extra_body"]["reasoning"] == {
        "effort": "low"
    }


def test_grader_failure_completes_error_without_reward():
    adapter = _adapter()

    with (
        patch.object(trajectory, "Judge"),
        patch.object(
            trajectory, "evaluate_run", side_effect=RuntimeError("judge down")
        ),
        pytest.raises(RuntimeError, match="judge down"),
    ):
        adapter.finalize("run", "task", {"finish_reason": "finish_tool"})

    adapter.trajectory.trajectories.log_reward.assert_not_called()
    adapter.trajectory.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="ERROR"
    )
    adapter.fail()
    adapter.trajectory.trajectories.complete.assert_called_once()


def test_reward_logging_failure_completes_error():
    adapter = _adapter()
    adapter.trajectory.trajectories.log_reward.side_effect = RuntimeError("api down")

    with (
        patch.object(trajectory, "Judge"),
        patch.object(trajectory, "evaluate_run", return_value={"all_pass": 1}),
        pytest.raises(RuntimeError, match="api down"),
    ):
        adapter.finalize("run", "task", {"finish_reason": "finish_tool"})

    adapter.trajectory.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="ERROR"
    )


def test_exit_hook_completes_open_session_once():
    adapter = _adapter()

    adapter.fail()
    adapter.fail()

    adapter.trajectory.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="ERROR"
    )


def test_create_adapter_passes_judge_model():
    with patch.object(trajectory, "TrajectoryAdapter") as adapter_class:
        create_adapter("trajectory/session", judge_model="gpt-5.4-mini")

    adapter_class.assert_called_once_with(0.0, None, "gpt-5.4-mini")


def test_create_adapter_requires_judge_model():
    with pytest.raises(ValueError, match="--judge-model is required"):
        create_adapter("trajectory/session")
