"""Check the Trajectory overlay inside the runtime image built from the pinned upstream."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from trajectory.types.inference.chat_completion_response import ChatCompletionResponse

trajectory_adapter = pytest.importorskip("big_finance_harness.models.trajectory")
python_exec = pytest.importorskip("big_finance_harness.tools.python_exec")

HARNESS_ROOT = Path("/opt/big_finance")


def load_orchestrator():
    path = HARNESS_ROOT / "scripts/run_eval_set.py"
    spec = importlib.util.spec_from_file_location("run_eval_set", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_exec_hides_harness_environment(monkeypatch):
    monkeypatch.setenv("TRAJECTORY_API_KEY", "secret-value")
    code = "import os; print(os.environ.get('TRAJECTORY_API_KEY'))"
    out = asyncio.run(python_exec.PythonExecTool().run({"code": code}))
    assert "secret-value" not in out
    assert "None" in out


@pytest.mark.parametrize(
    "finish_reason, stop_reason, truncated",
    [
        ("tool_calls", "tool_use", False),
        ("stop", "end_turn", False),
        ("length", "max_tokens", True),
    ],
)
def test_adapter_forwards_through_the_session(
    monkeypatch, finish_reason, stop_reason, truncated
):
    client = MagicMock()
    client.trajectories.create.return_value.tid = "traj_test"
    client.chat.completions.create.return_value = ChatCompletionResponse.model_validate(
        {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 0,
            "model": "trajectory-session",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": "Looking it up.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "x"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    monkeypatch.setattr(trajectory_adapter, "Client", lambda: client)

    adapter = trajectory_adapter.TrajectoryClient()
    response = asyncio.run(adapter.chat("system", [], []))

    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "trajectory-session"
    assert request["x_trajectory_id"] == "traj_test"
    assert response.stop_reason == stop_reason
    assert response.tool_calls[0].input == {"query": "x"}
    assert adapter.truncated is truncated


@pytest.mark.parametrize(
    "stop_reason, truncated, ending",
    [
        ("final_answer", False, "ENV_DONE"),
        ("max_steps", False, "MAX_STEPS"),
        ("context_exceeded", False, "TRUNCATION"),
        ("token_budget", False, "TRUNCATION"),
        ("final_answer", True, "TRUNCATION"),
    ],
)
def test_trajectory_mode_grades_and_ends(
    tmp_path, monkeypatch, stop_reason, truncated, ending
):
    result, client, dataset = run_trajectory_mode(
        tmp_path, monkeypatch, stop_reason, truncated
    )

    assert result.exit_code == 0, result.output
    assert not dataset.exists()
    client.trajectories.log_reward.assert_called_once_with(
        "traj_test", name="reward_accuracy", value=0.75, explanation="3/4 rubric points"
    )
    client.trajectories.complete.assert_called_once_with(
        "traj_test", termination_reason=ending
    )


def test_trajectory_mode_harness_error_ends_without_reward(tmp_path, monkeypatch):
    result, client, _ = run_trajectory_mode(tmp_path, monkeypatch, "error", False)

    assert result.exit_code != 0
    client.trajectories.log_reward.assert_not_called()
    client.trajectories.complete.assert_called_once_with(
        "traj_test", termination_reason="ERROR"
    )


def run_trajectory_mode(tmp_path, monkeypatch, stop_reason, truncated):
    orchestrator = load_orchestrator()
    rows = (HARNESS_ROOT / "data/big_finance_subset.jsonl").read_text().splitlines()
    dataset = tmp_path / "subset.jsonl"
    dataset.write_text(rows[0] + "\n")
    task_id = json.loads(rows[0])["id"]

    client = MagicMock()
    adapter = SimpleNamespace(client=client, tid="traj_test", truncated=truncated)
    monkeypatch.setattr(trajectory_adapter, "TrajectoryClient", lambda: adapter)

    async def run_question(**_):
        return SimpleNamespace(stop_reason=stop_reason, error="boom")

    async def grade(**_):
        return SimpleNamespace(rubric_points_earned=3, rubric_points_possible=4)

    monkeypatch.setattr(orchestrator, "run_question", run_question)
    monkeypatch.setattr(orchestrator, "grade", grade)
    monkeypatch.setattr(orchestrator, "default_tools", list)
    monkeypatch.chdir(tmp_path)

    args = [
        "--trajectory",
        "--task-id",
        task_id,
        "--dataset",
        str(dataset),
        "--run-id",
        "test",
    ]
    return CliRunner().invoke(orchestrator.main, args), client, dataset
