from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from lab_core.harness.adapters import trajectory
from lab_core.harness.run import create_adapter

COMPLETED = SimpleNamespace(status="completed")
SCORES = {"all_pass": False, "n_passed": 3, "n_criteria": 4}
DUAL_SCORES = {"dual_criterion_pass": 0.625, "dual_all_pass_rate": 0.5}


def _adapter(judges=("claude-sonnet-4-6", "gpt-5.5")):
    adapter = object.__new__(trajectory.TrajectoryAdapter)
    adapter.trajectory = MagicMock()
    adapter.trajectory.trajectories.complete.return_value = COMPLETED
    adapter.tid = "tid_test"
    adapter.judges = judges
    adapter._open = True
    return adapter


def _rewards(adapter):
    return {
        c.kwargs["name"]: (c.kwargs["value"], c.kwargs.get("weight"))
        for c in adapter.trajectory.trajectories.log_reward.call_args_list
    }


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


def test_dual_judge_reward_is_the_dual_criterion_pass():
    adapter = _adapter()

    with patch.object(
        trajectory, "evaluate_run_dual", return_value=DUAL_SCORES
    ) as evaluate:
        adapter.finalize(
            "run", "task", {"finish_reason": "finish_tool", "incomplete_details": None}
        )

    evaluate.assert_called_once_with(
        "run", "task", judge_models=("claude-sonnet-4-6", "gpt-5.5")
    )
    assert _rewards(adapter) == {
        "criteria_fraction": (0.625, None),
        "all_pass": (0.5, 0.0),
    }
    adapter.trajectory.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="ENV_DONE"
    )


def test_single_judge_reward_is_the_criteria_fraction():
    adapter = _adapter(("openai/gpt-6-luna",))

    with (
        patch.object(trajectory, "Judge") as judge,
        patch.object(trajectory, "evaluate_run", return_value=SCORES),
    ):
        adapter.finalize(
            "run", "task", {"finish_reason": "finish_tool", "incomplete_details": None}
        )

    judge.assert_called_once_with(model="openai/gpt-6-luna")
    assert _rewards(adapter) == {
        "criteria_fraction": (0.75, None),
        "all_pass": (0.0, 0.0),
    }


def test_grader_failure_leaves_exit_hook_an_error_completion_without_reward():
    adapter = _adapter()

    with (
        patch.object(
            trajectory, "evaluate_run_dual", side_effect=ValueError("judge down")
        ),
        pytest.raises(ValueError, match="judge down"),
    ):
        adapter.finalize("run", "task", {"finish_reason": "finish_tool"})

    adapter.trajectory.trajectories.complete.assert_not_called()
    adapter.fail()
    adapter.trajectory.trajectories.log_reward.assert_not_called()
    adapter.trajectory.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="ERROR"
    )


def test_reward_logging_failure_leaves_exit_hook_an_error_completion():
    adapter = _adapter()
    adapter.trajectory.trajectories.log_reward.side_effect = RuntimeError("api down")

    with (
        patch.object(trajectory, "evaluate_run_dual", return_value=DUAL_SCORES),
        pytest.raises(RuntimeError, match="api down"),
    ):
        adapter.finalize("run", "task", {"finish_reason": "finish_tool"})

    adapter.fail()
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


def test_failed_final_completion_leaves_exit_hook_an_error_completion():
    adapter = _adapter()
    adapter.trajectory.trajectories.complete.side_effect = [
        RuntimeError("api down"),
        COMPLETED,
    ]

    with (
        patch.object(trajectory, "evaluate_run_dual", return_value=DUAL_SCORES),
        pytest.raises(RuntimeError, match="api down"),
    ):
        adapter.finalize(
            "run", "task", {"finish_reason": "finish_tool", "incomplete_details": None}
        )

    adapter.fail()
    adapter.fail()

    assert adapter.trajectory.trajectories.complete.call_args_list[-1] == call(
        "tid_test", termination_reason="ERROR"
    )
    assert adapter.trajectory.trajectories.complete.call_count == 2


def test_exit_hook_retries_a_failed_error_completion():
    adapter = _adapter()
    adapter.trajectory.trajectories.complete.side_effect = [
        RuntimeError("api down"),
        COMPLETED,
    ]

    with pytest.raises(RuntimeError, match="api down"):
        adapter.fail()
    adapter.fail()
    adapter.fail()

    assert adapter.trajectory.trajectories.complete.call_count == 2
    adapter.trajectory.close.assert_called_once()


def test_unexpected_completion_status_raises_and_stays_open():
    adapter = _adapter()
    adapter.trajectory.trajectories.complete.side_effect = [
        SimpleNamespace(status="running"),
        COMPLETED,
    ]

    with (
        patch.object(trajectory, "evaluate_run_dual", return_value=DUAL_SCORES),
        pytest.raises(RuntimeError, match="returned running"),
    ):
        adapter.finalize(
            "run", "task", {"finish_reason": "finish_tool", "incomplete_details": None}
        )

    adapter.trajectory.close.assert_not_called()
    adapter.fail()
    adapter.fail()

    assert adapter.trajectory.trajectories.complete.call_args_list[-1] == call(
        "tid_test", termination_reason="ERROR"
    )
    assert adapter.trajectory.trajectories.complete.call_count == 2
    adapter.trajectory.close.assert_called_once()


def test_create_adapter_passes_judges():
    with patch.object(trajectory, "TrajectoryAdapter") as adapter_class:
        create_adapter("trajectory/session", judges=["openai/gpt-6-luna"])

    adapter_class.assert_called_once_with(0.0, None, ["openai/gpt-6-luna"])


def test_non_openai_judge_needs_no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sdk = SimpleNamespace(Client=MagicMock())
    with (
        patch.object(trajectory.atexit, "register"),
        patch("importlib.import_module", return_value=sdk),
    ):
        adapter = trajectory.TrajectoryAdapter(0.0, None, ["claude-sonnet-4-6"])

    assert adapter.judges == ("claude-sonnet-4-6",)
    assert adapter.client.responses.tid == adapter.tid
