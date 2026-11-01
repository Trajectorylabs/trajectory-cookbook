"""Check the Trajectory overlay inside the runtime image built from the pinned upstream."""

import asyncio
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from trajectory.types.inference.chat_completion_response import ChatCompletionResponse

trajectory_adapter = pytest.importorskip("big_finance_harness.models.trajectory")
python_exec = pytest.importorskip("big_finance_harness.tools.python_exec")
CliRunner = pytest.importorskip("click.testing").CliRunner
run_question = pytest.importorskip("big_finance_harness.agent").run_question
Tool = pytest.importorskip("big_finance_harness.tools.base").Tool

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


def test_python_exec_keeps_numpy_calculations_available():
    out = asyncio.run(
        python_exec.PythonExecTool().run(
            {"code": "import numpy as np; print(np.sum([2, 3, 5]))"}
        )
    )
    assert "exit_code: 0" in out
    assert "stdout:\n10" in out


def test_python_exec_limits_memory_and_files_outside_working_directory():
    tool = python_exec.PythonExecTool()
    memory = asyncio.run(tool.run({"code": "bytearray(600 << 20)"}))
    assert "exit_code: 1" in memory
    assert "MemoryError" in memory

    file_growth = asyncio.run(
        tool.run(
            {"code": "with open('/tmp/python-exec-limit-check', 'wb') as f: "
             "f.write(b'x' * (2 << 20))"}
        )
    )
    assert "exit_code: 1" in file_growth
    assert "File too large" in file_growth


@pytest.mark.skipif(os.geteuid() != 0, reason="the image runs the harness as root")
def test_python_exec_child_cannot_read_root_secrets(tmp_path):
    secret = tmp_path / "root-only"
    secret.write_text("secret-value")
    secret.chmod(0o400)
    assert Path(f"/proc/{os.getpid()}/environ").read_bytes()
    code = (
        "import os\n"
        "print('uid', os.getuid(), os.getgid(), os.getgroups(), os.getcwd())\n"
        f"for path in [f'/proc/{{os.getppid()}}/environ', {str(secret)!r}]:\n"
        "    try:\n"
        "        print(open(path).read())\n"
        "    except PermissionError:\n"
        "        print('denied', path)\n"
        "print(6 * 7)\n"
    )
    out = asyncio.run(python_exec.PythonExecTool().run({"code": code}))
    assert "exit_code: 0" in out
    assert "uid 65534 65534 []" in out
    assert out.count("denied") == 2
    assert "secret-value" not in out
    assert "42" in out


@pytest.mark.skipif(os.geteuid() != 0, reason="the image runs the harness as root")
def test_python_exec_cannot_start_detached_descendants():
    shared = Path(tempfile.mkdtemp(dir="/tmp"))
    shared.chmod(0o777)
    pid_file = shared / "pid"
    code = (
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
    )
    out = asyncio.run(python_exec.PythonExecTool().run({"code": code}))
    assert "exit_code: 1" in out
    assert "Resource temporarily unavailable" in out
    assert not pid_file.exists()


def test_python_exec_kills_timed_out_snippet():
    with pytest.raises(python_exec.ToolError, match="timed out"):
        asyncio.run(
            python_exec.PythonExecTool(timeout_s=0.1).run(
                {"code": "import time; time.sleep(30)"}
            )
        )


def test_python_exec_caps_output_while_reading():
    code = (
        "import sys\nfor _ in range(200): sys.stdout.write('x' * 10**6)\nprint('end')"
    )
    tool = python_exec.PythonExecTool(timeout_s=20)
    out = asyncio.run(tool.run({"code": code}))
    head, stdout = out.split("stdout:\n")
    assert head == "exit_code: 0\n\n"
    assert stdout == "x" * python_exec.MAX_OUTPUT_CHARS
    assert "end" not in out


def test_python_exec_holds_at_most_the_cap_per_stream():
    async def read():
        stream = asyncio.StreamReader()
        stream.feed_data(b"y" * (python_exec.MAX_OUTPUT_BYTES + 10**6))
        stream.feed_eof()
        return await python_exec._read_capped(stream)

    assert len(asyncio.run(read())) == python_exec.MAX_OUTPUT_BYTES


def session_response(finish_reason, tool_name="web_search"):
    return ChatCompletionResponse.model_validate(
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
                                    "name": tool_name,
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


def session_adapter(monkeypatch, *responses):
    client = MagicMock()
    client.trajectories.create.return_value.tid = "traj_test"
    client.chat.completions.create.side_effect = list(responses)
    monkeypatch.setattr(trajectory_adapter, "Client", lambda: client)
    return trajectory_adapter.TrajectoryClient(), client


@pytest.mark.parametrize(
    "finish_reason, stop_reason, tool_calls, truncated",
    [
        ("tool_calls", "tool_use", 1, False),
        ("stop", "end_turn", 1, False),
        ("length", "max_tokens", 0, True),
        ("max_tokens", "other", 0, True),
    ],
)
def test_adapter_forwards_through_the_session(
    monkeypatch, finish_reason, stop_reason, tool_calls, truncated
):
    adapter, client = session_adapter(monkeypatch, session_response(finish_reason))
    response = asyncio.run(adapter.chat("system", [], []))

    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "trajectory-session"
    assert request["x_trajectory_id"] == "traj_test"
    assert response.stop_reason == stop_reason
    assert [call.input for call in response.tool_calls] == [{"query": "x"}] * tool_calls
    assert adapter.truncated is truncated


class RecordingTool(Tool):
    name = "web_search"
    description = "Search."

    def __init__(self):
        self.input_schema = {"type": "object"}
        self.calls = []

    async def run(self, args):
        self.calls.append(args)
        return "result"


def test_cut_off_reply_stops_the_loop_without_running_its_tools(monkeypatch):
    adapter, _ = session_adapter(
        monkeypatch, session_response("tool_calls"), session_response("length")
    )
    tool = RecordingTool()

    run = asyncio.run(
        run_question(
            question_id="q",
            question="What?",
            reference_answer=None,
            client=adapter,
            tools=[tool],
            system_prompt="system",
            max_steps=5,
        )
    )

    assert tool.calls == [{"query": "x"}]
    assert len(run.steps) == 2
    assert run.stop_reason == "no_tool_call"
    assert run.final_answer == "Looking it up."
    assert adapter.truncated


@pytest.mark.parametrize(
    "stop_reason, truncated, ending",
    [
        ("final_answer", False, "ENV_DONE"),
        ("max_steps", False, "MAX_STEPS"),
        ("context_exceeded", False, "TRUNCATION"),
        ("token_budget", False, "TRUNCATION"),
        ("no_tool_call", False, "ENV_DONE"),
        ("no_tool_call", True, "TRUNCATION"),
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


def test_trajectory_mode_refuses_to_run_without_root(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    result, client, dataset = run_trajectory_mode(
        tmp_path, monkeypatch, "final_answer", False
    )

    assert result.exit_code != 0
    assert "must run as root" in result.output
    assert dataset.exists()
    client.trajectories.complete.assert_not_called()


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
