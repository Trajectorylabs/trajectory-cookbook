"""Check the harness endings, grade parsing, and split without Docker or the platform."""

import importlib.util
import json
from pathlib import Path

import httpx2
import pytest
from trajectory import Client

ROOT = Path(__file__).parent
PROMPTS = {
    "system_template": "You are on {{system}}.",
    "instance_template": "Please solve this issue: {{task}}",
    "observation_template": "{{output.returncode}}: {{output.output}}",
    "format_error_template": "Tool call error: {{error}}",
    "env": {},
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = load("swe_bench_pro_harness", "runtime/swe_bench_pro_harness.py")
ingest = load("swe_bench_pro_ingest", "ingest.py")


def bash(command, call_id="call_1"):
    arguments = json.dumps({"command": command})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": arguments},
    }


def reply(*tool_calls, finish_reason="tool_calls", content=None, metadata=None):
    body = {
        "id": "chat_test",
        "model": "policy",
        "object": "chat.completion",
        "created": 0,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": list(tool_calls),
                },
            }
        ],
    }
    if metadata:
        body["metadata"] = metadata
    return body


def sdk_client(replies, requests):
    def respond(request):
        body = json.loads(request.content or b"{}")
        requests.append((request.url.path, body))
        if request.url.path == "/api/v1/trajectories":
            return httpx2.Response(200, json={"tid": "tid_test"})
        if request.url.path == "/v1/chat/completions":
            return httpx2.Response(200, json=replies.pop(0))
        status = "finalizing" if request.url.path.endswith("/complete") else "active"
        return httpx2.Response(
            200, json={"ok": True, "trajectory_id": "tid_test", "status": status}
        )

    return Client(
        api_key="test",
        base_url="https://example.com",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    )


def run_agent(replies, outputs=(), time_limit=3000):
    requests, executed = [], []
    outputs = list(outputs)

    def execute(command, timeout):
        executed.append(command)
        return outputs.pop(0)

    with sdk_client(list(replies), requests) as client:
        ending = harness.run_agent(
            client, "tid_test", PROMPTS, "Fix it.", execute, time_limit
        )
    calls = [body for path, body in requests if path == "/v1/chat/completions"]
    return ending, executed, calls


def output(text, returncode=0):
    return {"output": text, "returncode": returncode, "exception_info": ""}


def test_submit_command_ends_normally():
    ending, executed, calls = run_agent(
        [reply(bash("ls")), reply(bash(f"echo {harness.SUBMIT_MARKER}"))],
        [output("src\n"), output(f"{harness.SUBMIT_MARKER}\n")],
    )
    assert ending == ("ENV_DONE", 2)
    assert executed == ["ls", f"echo {harness.SUBMIT_MARKER}"]
    assert calls[0]["messages"][1]["content"] == "Please solve this issue: Fix it."
    assert calls[0]["tools"] == [harness.BASH_TOOL]
    assert calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "0: src\n",
    }


def test_failed_submit_command_does_not_end():
    ending, _, _ = run_agent(
        [
            reply(bash(f"echo {harness.SUBMIT_MARKER}; false")),
            reply(finish_reason="length"),
        ],
        [output(f"{harness.SUBMIT_MARKER}\n", returncode=1)],
    )
    assert ending == ("TRUNCATION", 2)


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
def test_cut_off_reply_runs_no_tools(finish_reason):
    ending, executed, _ = run_agent(
        [reply(bash("rm -rf /app"), finish_reason=finish_reason)]
    )
    assert ending == ("TRUNCATION", 1)
    assert executed == []


def test_reply_without_tool_call_gets_format_error():
    ending, _, calls = run_agent(
        [
            reply(finish_reason="stop", content="Done."),
            reply(bash(f"echo {harness.SUBMIT_MARKER}")),
        ],
        [output(f"{harness.SUBMIT_MARKER}\n")],
    )
    assert ending == ("ENV_DONE", 2)
    assert calls[1]["messages"][-2] == {"role": "assistant", "content": "Done."}
    assert calls[1]["messages"][-1]["content"].startswith(
        "Tool call error: No tool calls found"
    )


def test_malformed_tool_call_is_answered_without_running():
    malformed = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": "{"},
    }
    ending, executed, calls = run_agent(
        [reply(malformed), reply(bash(f"echo {harness.SUBMIT_MARKER}"))],
        [output(f"{harness.SUBMIT_MARKER}\n")],
    )
    assert ending == ("ENV_DONE", 2)
    assert executed == [f"echo {harness.SUBMIT_MARKER}"]
    assert calls[1]["messages"][-1]["role"] == "tool"
    assert calls[1]["messages"][-1]["content"].startswith("Tool call error:")


def test_turn_limit_ends_with_max_steps(monkeypatch):
    monkeypatch.setattr(harness, "MAX_TURNS", 2)
    ending, executed, _ = run_agent(
        [reply(bash("ls")), reply(bash("ls"))], [output(""), output("")]
    )
    assert ending == ("MAX_STEPS", 2)
    assert len(executed) == 2


def test_platform_limit_stops_the_loop():
    cancelled = reply(
        finish_reason="stop",
        content="",
        metadata={"trajectory_termination_reason": "trajectory_cancelled"},
    )
    ending, executed, _ = run_agent([cancelled])
    assert ending == ("MAX_STEPS", 1)
    assert executed == []


def test_time_limit_ends_with_timeout():
    ending, executed, _ = run_agent([reply(bash("ls"))], time_limit=0)
    assert ending == ("TIMEOUT", 0)
    assert executed == []


@pytest.mark.parametrize("as_python_literal", [False, True])
def test_read_grade_counts_required_tests(tmp_path, as_python_literal):
    (tmp_path / "reward.txt").write_text("1\n")
    tests = [{"name": "a", "status": "PASSED"}, {"name": "b", "status": "FAILED"}]
    (tmp_path / "output.json").write_text(json.dumps({"tests": tests}))
    fail_to_pass = "['a', 'b']" if as_python_literal else json.dumps(["a", "b"])
    reward, counts = harness.read_grade(
        tmp_path, {"fail_to_pass": fail_to_pass, "pass_to_pass": "[]"}
    )
    assert reward == 1.0
    assert counts == {
        "fail_to_pass": {"passed": 1, "total": 2},
        "pass_to_pass": {"passed": 0, "total": 0},
    }


def test_read_grade_rejects_missing_or_invalid_reward(tmp_path):
    with pytest.raises(FileNotFoundError):
        harness.read_grade(tmp_path, {})
    (tmp_path / "reward.txt").write_text("0.5")
    with pytest.raises(ValueError):
        harness.read_grade(tmp_path, {})


@pytest.mark.parametrize("ending", ["ENV_DONE", "MAX_STEPS"])
def test_main_completes_with_the_agent_ending(monkeypatch, tmp_path, ending):
    (tmp_path / "tasks/instance_x").mkdir(parents=True)
    (tmp_path / "tasks/instance_x/task.toml").write_text("")
    monkeypatch.setattr(harness, "UPSTREAM", tmp_path)
    monkeypatch.setattr(harness, "run_task", lambda client, tid, instance_id: ending)
    monkeypatch.setattr("sys.argv", ["harness", "instance_x"])
    requests = []
    with sdk_client([], requests) as client:
        monkeypatch.setattr(harness, "Client", lambda **kwargs: client)
        harness.main()
    assert requests[-1] == (
        "/api/v1/trajectories/tid_test/complete",
        {"termination_reason": ending},
    )


def test_harness_failure_completes_error_without_reward(monkeypatch, tmp_path):
    (tmp_path / "tasks/instance_x").mkdir(parents=True)
    (tmp_path / "tasks/instance_x/task.toml").write_text("")
    monkeypatch.setattr(harness, "UPSTREAM", tmp_path)

    def fail(client, tid, instance_id):
        raise RuntimeError("docker pull failed")

    monkeypatch.setattr(harness, "run_task", fail)
    monkeypatch.setattr("sys.argv", ["harness", "instance_x"])
    requests = []
    with sdk_client([], requests) as client:
        monkeypatch.setattr(harness, "Client", lambda **kwargs: client)
        with pytest.raises(RuntimeError, match="docker pull failed"):
            harness.main()
    assert not any(path.endswith("/rewards") for path, _ in requests)
    assert requests[-1] == (
        "/api/v1/trajectories/tid_test/complete",
        {"termination_reason": "ERROR"},
    )


def test_split_holds_out_each_repository_deterministically():
    ids = [
        f"instance_{owner}__{name}-{index:040x}"
        for owner, name, count in [
            ("ansible", "ansible", 40),
            ("flipt-io", "flipt", 20),
        ]
        for index in range(count)
    ]
    splits = ingest.split_tasks(ids)
    assert splits == ingest.split_tasks(list(reversed(ids)))
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert sorted(splits["train"] + splits["test"]) == sorted(ids)
    held_out = [ingest.repository(instance_id) for instance_id in splits["test"]]
    assert held_out.count("ansible/ansible") == 6
    assert held_out.count("flipt-io/flipt") == 3


def test_observation_json_keeps_quotes_and_brackets():
    rendered = harness.render(
        "{{ output.output | tojson }}", output={"output": "it's <a>"}
    )
    assert rendered == '"it\'s <a>"'
