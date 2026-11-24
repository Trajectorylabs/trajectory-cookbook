"""Check native grading, actor isolation, bounded tools and trajectory endings."""

import contextlib
import json
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest
from trajectory import Client
from trajectory.lib.benchmarks import _BenchmarkPackage
from trajectory.types.create_trajectory_response import CreateTrajectoryResponse
from trajectory.types.inference.chat_completion_response import ChatCompletionResponse

from examples.finqa.ingest import build_benchmark, package_tasks
from examples.finqa.prepare import ROOT, SOURCES, checked_bytes, prepare, select_rows
from examples.finqa.runtime.finqa_harness import (
    MAX_TURNS,
    TOOLS,
    compile_steps,
    dispatch_tool,
    execute_program,
    grade_submission,
    load_evaluator,
    public_input,
    run_task,
)


@pytest.fixture(scope="module")
def evaluator():
    return load_evaluator(ROOT / ".source/code/evaluate/evaluate.py")


@pytest.fixture
def row():
    return {
        "id": "report_test_a/page_1.pdf-0",
        "pre_text": ["Revenue increased after a launch."],
        "post_text": ["Amounts are in millions."],
        "table": [["year", "2020", "2021"], ["revenue", "$100", "$120"]],
        "qa": {
            "question": "How much did revenue grow?",
            "program": "subtract(120, 100)",
            "exe_ans": 20.0,
            "gold_inds": {"text_0": "PRIVATE_SENTINEL"},
        },
        "golden_doc": "PRIVATE_SENTINEL",
    }


def reply(content=None, calls=None, ending="stop"):
    return ChatCompletionResponse(
        id="reply_test_a",
        object="chat.completion",
        created=0,
        model="model_test_a",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": calls,
                },
                "finish_reason": ending,
            }
        ],
    )


def tool(name, arguments):
    return {
        "id": "call_test_a",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def structured_steps(program):
    return [
        {
            "op": program[index][:-1],
            "left": program[index + 1],
            "right": program[index + 2],
        }
        for index in range(0, len(program) - 1, 4)
    ]


@contextlib.contextmanager
def mocked_client(responses):
    with Client(api_key="test_key") as client, contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(
                client.trajectories,
                "create",
                autospec=True,
                return_value=CreateTrajectoryResponse(tid="trajectory_test_a"),
            )
        )
        create = stack.enter_context(
            patch.object(
                client.chat.completions, "create", autospec=True, side_effect=responses
            )
        )
        reward = stack.enter_context(
            patch.object(client.trajectories, "log_reward", autospec=True)
        )
        complete = stack.enter_context(
            patch.object(client.trajectories, "complete", autospec=True)
        )
        yield client, create, reward, complete


@pytest.mark.parametrize(
    "program, expected",
    [
        (["subtract(", "120", "100", ")", "EOF"], 20),
        (["divide(", "1", "3", ")", "EOF"], 0.33333),
        (["multiply(", "25%", "const_100", ")", "EOF"], 25),
        (["table_sum(", "revenue", "none", ")", "EOF"], 220),
        (["greater(", "120", "100", ")", "EOF"], "yes"),
        (["subtract(", "120", "100", ")", "divide(", "#0", "100", ")", "EOF"], 0.2),
    ],
)
def test_calculator_preserves_upstream_execution(program, expected, row, evaluator):
    assert compile_steps({"steps": structured_steps(program)}) == program
    assert (
        execute_program(program, row["table"], evaluator)
        == evaluator.eval_program(program, row["table"])
        == (0, expected)
    )


@pytest.mark.parametrize(
    "program",
    [
        ["exp(", "2", "99999", ")", "EOF"],
        ["add(", "#0", "1", ")", "EOF"],
        ["table_sum(", "../../private_tasks.jsonl", "none", ")", "EOF"],
        ["add(", "nan", "1", ")", "EOF"],
        ["add(", "__import__('os')", "1", ")", "EOF"],
        ["add(", "1", "1", ")"] * 17 + ["EOF"],
        ["multiply(", "20", "20", ")", "exp(", "2", "#0", ")", "EOF"],
    ],
)
def test_calculator_rejects_unbounded_or_non_numeric_programs(program, row, evaluator):
    with pytest.raises(ValueError):
        execute_program(program, row["table"], evaluator)


@pytest.mark.parametrize(
    "program",
    [
        ["divide(", "1", "0", ")", "EOF"],
        ["exp(", "-1", "0.5", ")", "EOF"],
        ["greater(", "2", "1", ")", "add(", "#0", "1", ")", "EOF"],
        ["exp(", "1e15", "20", ")", "exp(", "#0", "20", ")", "EOF"],
    ],
)
def test_invalid_arithmetic_returns_a_serializable_invalid_result(
    program, row, evaluator
):
    result = dispatch_tool(
        "calculate", {"steps": structured_steps(program)}, public_input(row), evaluator
    )
    assert result == {"invalid": True, "result": "n/a"}
    assert evaluator.eval_program(program, row["table"])[0] == 1
    json.dumps(result)


@pytest.mark.parametrize(
    "content, score",
    [
        ('{"program":["subtract(","120","100",")","EOF"]}', 1),
        ('{"program":["add(","120","100",")","EOF"]}', 0),
        ('{"program":["divide(","1","0",")","EOF"]}', 0),
        ('{"answer":"20"}', 0),
        ("the answer is 20", 0),
        ('{"program":null}', 0),
    ],
)
def test_completed_submissions_are_correct_wrong_or_invalid(
    content, score, row, evaluator
):
    assert grade_submission(content, row, evaluator)["execution_accuracy"] == score


def test_actor_tools_cannot_access_gold_or_other_files(row, evaluator):
    actor = public_input(row)
    results = [
        dispatch_tool("read_report", {"section": section}, actor, evaluator)
        for section in ("pre_text", "post_text", "table")
    ]
    results += [
        dispatch_tool("search_report", {"query": "PRIVATE_SENTINEL"}, actor, evaluator)
    ]
    assert "PRIVATE_SENTINEL" not in json.dumps([actor, results])
    assert {definition["function"]["name"] for definition in TOOLS} == {
        "search_report",
        "read_report",
        "calculate",
        "submit_program",
    }
    with pytest.raises(ValueError):
        dispatch_tool("read_report", {"section": "qa"}, actor, evaluator)
    with pytest.raises(ValueError):
        dispatch_tool(
            "read_file", {"path": "/opt/finqa/private_tasks.jsonl"}, actor, evaluator
        )


def test_agent_calls_tools_and_publishes_the_native_grade(row, evaluator):
    program = ["subtract(", "120", "100", ")", "EOF"]
    responses = [
        reply(calls=[tool("read_report", {"section": "table"})], ending="tool_calls"),
        reply(
            calls=[tool("calculate", {"steps": structured_steps(program)})],
            ending="tool_calls",
        ),
        reply(
            calls=[tool("submit_program", {"steps": structured_steps(program)})],
            ending="tool_calls",
        ),
    ]
    with mocked_client(responses) as (client, create, reward, complete):
        run_task(row, client, evaluator)
        assert create.call_count == 3
        assert all(
            call.kwargs["x_trajectory_id"] == "trajectory_test_a"
            for call in create.call_args_list
        )
        assert "PRIVATE_SENTINEL" not in json.dumps(create.call_args.kwargs["messages"])
        assert any(
            message["role"] == "tool" for message in create.call_args.kwargs["messages"]
        )
        assert reward.call_args.kwargs["value"] == 1
        assert complete.call_args.kwargs["termination_reason"] == "ENV_DONE"


def test_invalid_tool_action_can_be_corrected(row, evaluator):
    responses = [
        reply(calls=[tool("read_report", {"section": "qa"})], ending="tool_calls"),
        reply(
            calls=[
                tool(
                    "submit_program",
                    {"steps": [{"op": "subtract", "left": "120", "right": "100"}]},
                )
            ],
            ending="tool_calls",
        ),
    ]
    with mocked_client(responses) as (client, create, reward, complete):
        run_task(row, client, evaluator)
        assert any(
            "tool_error" in message.get("content", "")
            for message in create.call_args.kwargs["messages"]
        )
        assert reward.call_args.kwargs["value"] == 1
        assert complete.call_args.kwargs["termination_reason"] == "ENV_DONE"


@pytest.mark.parametrize("ending", ["length", "max_tokens"])
def test_cutoff_tool_calls_are_not_executed(ending, row, evaluator):
    response = reply(
        calls=[
            tool(
                "submit_program",
                {"steps": [{"op": "subtract", "left": "120", "right": "100"}]},
            )
        ],
        ending=ending,
    )
    with (
        mocked_client([response]) as (client, _, reward, complete),
        patch(
            "examples.finqa.runtime.finqa_harness.dispatch_tool", autospec=True
        ) as dispatch,
    ):
        run_task(row, client, evaluator)
        dispatch.assert_not_called()
        reward.assert_not_called()
        assert complete.call_args.kwargs["termination_reason"] == "TRUNCATION"


def test_service_failure_is_not_a_wrong_answer(row, evaluator):
    with mocked_client([RuntimeError("service unavailable")]) as (
        client,
        _,
        reward,
        complete,
    ):
        with pytest.raises(RuntimeError, match="service unavailable"):
            run_task(row, client, evaluator)
        reward.assert_not_called()
        assert complete.call_args.kwargs["termination_reason"] == "ERROR"


def test_malformed_tool_json_is_a_protocol_error(row, evaluator):
    call = tool("read_report", {"section": "table"})
    call["function"]["arguments"] = '{"section":'
    with mocked_client([reply(calls=[call], ending="tool_calls")]) as (
        client,
        _,
        reward,
        complete,
    ):
        with pytest.raises(json.JSONDecodeError):
            run_task(row, client, evaluator)
        reward.assert_not_called()
        assert complete.call_args.kwargs["termination_reason"] == "ERROR"


def test_turn_limit_has_no_fabricated_grade(row, evaluator):
    responses = [
        reply(calls=[tool("read_report", {"section": "table"})], ending="tool_calls")
        for _ in range(MAX_TURNS)
    ]
    with mocked_client(responses) as (client, _, reward, complete):
        run_task(row, client, evaluator)
        reward.assert_not_called()
        assert complete.call_args.kwargs["termination_reason"] == "TRUNCATION"


def test_prepared_gold_and_wrong_predictions_match_official_reader(tmp_path, evaluator):
    manifest = prepare(tmp_path / "data", ROOT / ".source")
    records, loaded = package_tasks(tmp_path / "data", tmp_path / "package")
    assert loaded == manifest
    assert len(records) == 80
    assert not (tmp_path / "package/dev.json").exists()
    benchmark = build_benchmark("finqa-test", records, manifest)
    assert sum(task.split == "train" for task in benchmark.tasks) == 64
    assert sum(task.split == "test" for task in benchmark.tasks) == 16
    assert {task.name for task in benchmark.tasks} == {
        f"finqa/{record['source_id']}" for record in records
    }
    predictions, gold = [], []
    expected = []
    for index, record in enumerate(records):
        row = record["source"]
        program = (
            evaluator.program_tokenization(row["qa"]["program"])
            if index % 2 == 0
            else ["add(", "const_100", "const_100", ")", "EOF"]
        )
        compiled = compile_steps({"steps": structured_steps(program)})
        assert compiled == program
        expected.append(
            grade_submission(json.dumps({"program": compiled}), row, evaluator)[
                "execution_accuracy"
            ]
        )
        predictions.append({"id": row["id"], "predicted": program})
        gold.append(row)
    (tmp_path / "predictions.json").write_text(json.dumps(predictions))
    (tmp_path / "gold.json").write_text(json.dumps(gold))
    native = evaluator.evaluate_result(
        str(tmp_path / "predictions.json"), str(tmp_path / "gold.json")
    )
    assert native[0] == sum(expected) / len(expected)
    assert sum(expected) == 40
    completed = subprocess.run(
        [sys.executable, "-c", "import finqa_harness; print('packaged import ok')"],
        cwd=tmp_path / "package",
        text=True,
        capture_output=True,
        check=True,
    )
    assert "packaged import ok" in completed.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        {"program": ["add(", "1", "1", ")", "EOF"]},
        {"steps": ["table_sum(", "revenue", ",EOF"]},
        {"steps": [{"op": "add(", "left": "1", "right": "1"}]},
        {"steps": [{"op": "add", "left": 1, "right": "1"}]},
        {"steps": [{"op": "add", "left": "1", "right": "1", "EOF": True}]},
        {"steps": []},
    ],
)
def test_structured_interface_rejects_tokens_and_wrong_step_shapes(arguments):
    with pytest.raises(ValueError):
        compile_steps(arguments)


@pytest.mark.parametrize(
    "steps, score, invalid",
    [
        ([{"op": "subtract", "left": "120", "right": "100"}], 1, False),
        ([{"op": "add", "left": "120", "right": "100"}], 0, False),
        ([{"op": "divide", "left": "1", "right": "0"}], 0, True),
        (["subtract(", "120", "100", ")", "EOF"], 0, True),
    ],
)
def test_submit_program_ends_with_native_correct_wrong_or_invalid_grade(
    steps, score, invalid, row, evaluator
):
    with mocked_client(
        [reply(calls=[tool("submit_program", {"steps": steps})], ending="tool_calls")]
    ) as (client, create, reward, complete):
        run_task(row, client, evaluator)
        assert create.call_count == 1
        assert reward.call_args.kwargs["value"] == score
        assert (
            json.loads(reward.call_args.kwargs["explanation"])["invalid_program"]
            == invalid
        )
        assert complete.call_args.kwargs["termination_reason"] == "ENV_DONE"


def test_text_without_submit_tool_is_not_accepted(row, evaluator):
    with mocked_client([reply('{"program":["subtract(","120","100",")","EOF"]}')]) as (
        client,
        _,
        reward,
        complete,
    ):
        run_task(row, client, evaluator)
        reward.assert_not_called()
        assert complete.call_args.kwargs["termination_reason"] == "TRUNCATION"


def test_selection_is_stable_and_rejects_source_corruption(tmp_path, row):
    assert select_rows([row], 1, "train")[0]["source_id"] == row["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        select_rows([row, row], 1, "train")
    path = tmp_path / "evaluator.py"
    shutil.copyfile(ROOT / ".source/code/evaluate/evaluate.py", path)
    path.write_text(path.read_text() + "\n# altered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        checked_bytes(path, SOURCES["code/evaluate/evaluate.py"])


def test_full_ingestion_upload_preserves_source_provenance(tmp_path):
    data = tmp_path / "data"
    package = tmp_path / "package"
    prepare(data, ROOT / ".source", train_count=1, test_count=1)
    records, manifest = package_tasks(data, package)
    upload_files = {
        file.path
        for file in _BenchmarkPackage(
            build_benchmark("finqa-package-test", records, manifest), package
        ).files()
    }
    assert "source_manifest.json" in upload_files
    assert "manifest.json" not in upload_files
    assert json.loads((package / "source_manifest.json").read_text()) == manifest
    copy_line = next(
        line
        for line in (package / "Dockerfile").read_text().splitlines()
        if line.startswith("COPY ")
    )
    assert set(copy_line.split()[1:-1]) <= upload_files
