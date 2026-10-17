"""Verify the packaged harness uses the published SDK reward and completion API."""

import importlib.util
import json
import sys
from pathlib import Path

import httpx2
import pytest
from trajectory import Client

ROOT = Path(__file__).parent


@pytest.mark.parametrize(
    "answer, expected_reward, finish_reason, expected_reason",
    [
        ("42", 1.0, "tool_calls", "ENV_DONE"),
        ("41", 0.0, "tool_calls", "ENV_DONE"),
        ("42", 1.0, "length", "TRUNCATION"),
        ("42", 1.0, "max_tokens", "TRUNCATION"),
    ],
)
def test_task_records_reward_and_completes_with_published_sdk(
    tmp_path, monkeypatch, answer, expected_reward, finish_reason, expected_reason
):
    spec = importlib.util.spec_from_file_location(
        "gsm8k_harness", ROOT / "runtime/gsm8k_harness.py"
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    requests = []

    def respond(request):
        body = json.loads(request.content or b"{}")
        requests.append((request.url.path, request.headers, body))
        if request.url.path == "/api/v1/trajectories":
            return httpx2.Response(200, json={"tid": "tid_test"})
        if request.url.path == "/v1/chat/completions":
            return httpx2.Response(
                200,
                json={
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
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_test",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_answer",
                                            "arguments": json.dumps({"answer": answer}),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )
        if request.url.path.endswith("/rewards"):
            return httpx2.Response(
                200, json={"ok": True, "trajectory_id": "tid_test", "status": "active"}
            )
        if request.url.path.endswith("/complete"):
            return httpx2.Response(
                200, json={"trajectory_id": "tid_test", "status": "completed"}
            )
        raise AssertionError(request.url.path)

    task = tmp_path / "task.json"
    task.write_text(
        json.dumps({"question": "What is 6 times 7?", "expected_answer": "#### 42"})
    )
    monkeypatch.setattr(sys, "argv", ["gsm8k_harness", "--task-file", str(task)])
    with Client(
        api_key="test",
        base_url="https://example.com",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:
        monkeypatch.setattr(harness, "Client", lambda **kwargs: client)
        harness.main()
    assert len(requests) == 4
    assert requests[1][1]["X-Trajectory-Id"] == "tid_test"
    assert requests[2][2]["name"] == "reward_accuracy"
    assert requests[2][2]["value"] == expected_reward
    assert requests[3][2] == {"termination_reason": expected_reason}


@pytest.mark.parametrize(
    "rewards,grades,total,mean,reason,rollout_status,expected",
    [
        ([None, None], [False, False], 2, 0.0, "ENV_DONE", "COMPLETED", None),
        ([0.0], [False], 1, 0.0, "ENV_DONE", "COMPLETED", None),
        ([0.0], [True], 1, 0.0, "ENV_DONE", "COMPLETED", 0.0),
        ([0.0, None], [True, False], 2, 0.0, "ENV_DONE", "COMPLETED", None),
        ([1.0], [True], 2, 1.0, "ENV_DONE", "COMPLETED", None),
        ([1.0], [True], 1, 1.0, "TRUNCATION", "COMPLETED", None),
        ([1.0], [True], 1, 1.0, "ERROR", "FAILED", None),
    ],
)
def test_completed_eval_requires_grades_before_training(
    rewards, grades, total, mean, reason, rollout_status, expected
):
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("gsm8k_train_test", ROOT / "train.py")
    train = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = train
    spec.loader.exec_module(train)
    client = SimpleNamespace(
        evals=SimpleNamespace(
            runs=SimpleNamespace(
                progress=lambda _: SimpleNamespace(
                    status="completed", terminal_rollouts=total, total_rollouts=total
                ),
                retrieve=lambda _: SimpleNamespace(reward_mean=mean),
            )
        ),
        trajectories=SimpleNamespace(
            list=lambda **_: iter(
                SimpleNamespace(
                    trajectory_id=str(i),
                    reward=r,
                    termination_reason=reason,
                    rollout=SimpleNamespace(status=rollout_status),
                )
                for i, r in enumerate(rewards)
            ),
            retrieve=lambda tid, **_: SimpleNamespace(
                reward_components=[SimpleNamespace(name="reward_accuracy")]
                if grades[int(tid)]
                else []
            ),
        ),
    )
    if expected is None:
        with pytest.raises(RuntimeError, match="fully graded normal completions"):
            train._wait_for_evaluation(client, "eval_fixture", "fixture", 0)
    else:
        assert (
            train._wait_for_evaluation(client, "eval_fixture", "fixture", 0) == expected
        )


def test_evaluations_keep_baseline_version_when_latest_changes(monkeypatch):
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "gsm8k_comparison_test", ROOT / "train.py"
    )
    train = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = train
    spec.loader.exec_module(train)
    eval_calls, training_calls = [], []

    def create_eval(**kwargs):
        eval_calls.append(kwargs)
        # A name lookup after the baseline would select a different upload.
        selected = kwargs.get(
            "bench_id", "bm_old" if len(eval_calls) == 1 else "bm_new"
        )
        return SimpleNamespace(bench_id=selected, eval_run_id=f"eval_{len(eval_calls)}")

    def create_training(**kwargs):
        training_calls.append(kwargs)
        return SimpleNamespace(training_run_id="train_1")

    client = SimpleNamespace(
        evals=SimpleNamespace(create=create_eval),
        training=SimpleNamespace(
            create=create_training,
            checkpoints=SimpleNamespace(
                retrieve=lambda *_: SimpleNamespace(checkpoint_id="checkpoint_1")
            ),
        ),
    )
    monkeypatch.setattr(train, "_wait_for_evaluation", lambda *args: 0.5)
    monkeypatch.setattr(
        train, "_wait_for_training", lambda *args: SimpleNamespace(status="succeeded")
    )
    train.train_and_evaluate(client, "agent", "benchmark", "model", 3, 0)
    assert training_calls[0]["bench_id"] == eval_calls[1]["bench_id"] == "bm_old"
    assert "benchmark_name" not in training_calls[0]
    assert "benchmark_name" not in eval_calls[1]
    assert eval_calls[0]["options"] == eval_calls[1]["options"]
    assert eval_calls[1]["parent_checkpoint_id"] == "checkpoint_1"
    assert train.eval_smoketest(client, "bm_old") == 0.5
    assert eval_calls[2]["bench_id"] == "bm_old"
    assert "benchmark_name" not in eval_calls[2]


@pytest.mark.parametrize("outcome", ["ready", "build_failed", "timeout"])
def test_upload_keeps_operation_id_and_reports_build_failure(
    monkeypatch, capsys, outcome
):
    from types import SimpleNamespace

    from trajectory.lib.benchmark_operations import BenchmarkOperationError

    spec = importlib.util.spec_from_file_location(
        "gsm8k_ingest_test", ROOT / "ingest.py"
    )
    ingest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest)
    failure = SimpleNamespace(
        to_json=lambda: '{"message":"Python version unsupported"}'
    )
    result = SimpleNamespace(
        status=SimpleNamespace(
            operation_id="op_fixture", status="partial_failure", bench_id="bm_fixture"
        ),
        failures=lambda: iter([failure]),
    )

    def wait(**kwargs):
        if outcome == "build_failed":
            raise BenchmarkOperationError(result)
        if outcome == "timeout":
            raise TimeoutError("server processing continues")
        return result

    def start(client, manifest, **kwargs):
        assert kwargs["build_images"] is True
        assert (kwargs["root"] / "Dockerfile").is_file()
        return SimpleNamespace(id="op_fixture", result=wait)

    monkeypatch.setattr(ingest, "Client", lambda: object())
    monkeypatch.setattr(ingest, "start_push", start)
    rows = {"train": [{"question": "6 * 7?", "answer": "#### 42"}]}
    if outcome == "ready":
        assert ingest._push_benchmark(rows, "fixture", "agent") == "bm_fixture"
    else:
        error = BenchmarkOperationError if outcome == "build_failed" else TimeoutError
        with pytest.raises(error):
            ingest._push_benchmark(rows, "fixture", "agent")
    output = capsys.readouterr().out
    assert "operation_id=op_fixture" in output
    if outcome == "build_failed":
        assert "Python version unsupported" in output
