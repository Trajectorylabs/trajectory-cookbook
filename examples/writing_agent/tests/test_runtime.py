import json
from unittest.mock import create_autospec

import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion
from trajectory import Client
from trajectory.types.complete_trajectory_response import CompleteTrajectoryResponse
from trajectory.types.create_trajectory_response import CreateTrajectoryResponse
from trajectory.types.inference.chat_completion_response import ChatCompletionResponse

from examples.writing_agent.runtime.writing_harness import (
    grade_draft,
    run_task,
    score_reward,
    writer_messages,
)


def writing_task() -> dict:
    return {
        "task_id": "test_001",
        "split": "test",
        "input": {
            "notes": "We replaced seven steps with one guided task. Still measuring the effect.",
            "instructions": "Make the draft concise without claiming a measured improvement.",
            "audience": "Founders",
            "facts": ["We replaced seven steps with one guided task."],
            "constraints": {
                "max_characters": 280,
                "lowercase": False,
                "avoid": ["hashtags"],
            },
        },
        "private_grading": {
            "source_post_ids": ["post_test"],
            "brief_origin": "synthetic_author_reviewed",
            "reference_post": "Private reference that must not reach the writer.",
            "must_preserve": ["The effect is not measured yet."],
            "author_reviewed": True,
        },
    }


def clients(
    monkeypatch: pytest.MonkeyPatch, finish_reason: str = "stop"
) -> tuple[Client, OpenAI]:
    actor = Client(api_key="fixture")
    judge = OpenAI(api_key="fixture")
    actor_reply = ChatCompletionResponse.model_validate(
        {
            "id": "response_test",
            "object": "chat.completion",
            "created": 0,
            "model": "actor",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": "One guided task. Still measuring.",
                    },
                }
            ],
        }
    )
    judge_reply = ChatCompletion.model_validate(
        {
            "id": "judge_test",
            "object": "chat.completion",
            "created": 0,
            "model": "judge",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "faithful": 1,
                                "instruction_free": 1,
                                "unsupported_claims": [],
                                "meaning_changes": [],
                                "postability": "minor_edit",
                                "reason": "Faithful draft in the author voice.",
                                "voice": 0.8,
                                "brief_adherence": 1,
                                "quality": 0.9,
                            }
                        ),
                    },
                }
            ],
        }
    )
    calls = [
        (actor.trajectories, "create", CreateTrajectoryResponse(tid="tid_test")),
        (actor.chat.completions, "create", actor_reply),
        (actor.trajectories, "log_reward", None),
        (
            actor.trajectories,
            "complete",
            CompleteTrajectoryResponse(trajectory_id="tid_test", status="completed"),
        ),
        (judge.chat.completions, "create", judge_reply),
    ]
    for resource, name, result in calls:
        monkeypatch.setattr(
            resource,
            name,
            create_autospec(getattr(resource, name), return_value=result),
        )
    meaning_reply = judge_reply.model_copy(deep=True)
    meaning_reply.choices[0].message.content = json.dumps(
        {
            "status": "preserved",
            "literal_reading": "The author replaced seven steps with one guided task and is still measuring.",
            "issues": [],
        }
    )
    judge.chat.completions.create.side_effect = [meaning_reply, judge_reply]
    return actor, judge


def test_private_reference_is_visible_only_to_separate_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, judge = clients(monkeypatch)
    task = writing_task()
    voice = [{"id": "voice_001", "text": "Small steps. Clear outcomes."}]
    assert run_task(task, voice, actor, judge, "fixed-judge") == "tid_test"
    writer_call = actor.chat.completions.create.call_args.kwargs
    judge_call = judge.chat.completions.create.call_args.kwargs
    reference = task["private_grading"]["reference_post"]
    assert reference not in json.dumps(writer_call)
    assert reference in json.dumps(judge_call)
    assert judge_call["model"] == "fixed-judge"
    assert "tools" not in writer_call and "extra_body" not in writer_call
    reward = actor.trajectories.log_reward.call_args.kwargs
    assert reward["name"] == "writing_quality"
    assert reward["value"] == pytest.approx(0.89)
    assert (
        json.loads(reward["explanation"])["judge_scores"]["postability"] == "minor_edit"
    )
    actor.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="ENV_DONE"
    )


def test_truncated_output_never_gets_a_grade(monkeypatch: pytest.MonkeyPatch) -> None:
    actor, judge = clients(monkeypatch, "length")
    run_task(writing_task(), [], actor, judge, "fixed-judge")
    judge.chat.completions.create.assert_not_called()
    actor.trajectories.log_reward.assert_not_called()
    actor.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="TRUNCATION"
    )


def test_broken_judging_is_an_error_without_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, judge = clients(monkeypatch)
    judge.chat.completions.create.side_effect = ValueError("invalid judge response")
    with pytest.raises(ValueError, match="invalid judge"):
        run_task(writing_task(), [], actor, judge, "fixed-judge")
    actor.trajectories.log_reward.assert_not_called()
    actor.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="ERROR"
    )


@pytest.mark.parametrize("draft", ["", "  ", "x" * 281])
def test_invalid_format_has_zero_reward_without_judge_call(
    monkeypatch: pytest.MonkeyPatch, draft: str
) -> None:
    _, judge = clients(monkeypatch)
    assert grade_draft(writing_task(), [], draft, judge, "fixed-judge")["reward"] == 0
    judge.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_malformed_scores_are_not_silently_turned_into_rewards(bad: object) -> None:
    with pytest.raises(ValueError):
        score_reward(
            {
                "faithful": 1,
                "instruction_free": 1,
                "unsupported_claims": [],
                "meaning_changes": [],
                "postability": "as_is",
                "reason": "Good",
                "voice": bad,
                "brief_adherence": 1,
                "quality": 1,
            }
        )


def test_unfaithful_writing_cannot_earn_style_reward() -> None:
    assert (
        score_reward(
            {
                "faithful": 0,
                "instruction_free": 1,
                "unsupported_claims": [],
                "meaning_changes": [],
                "postability": "as_is",
                "reason": "Changed meaning",
                "voice": 1,
                "brief_adherence": 1,
                "quality": 1,
            }
        )
        == 0
    )


def test_writer_does_not_receive_task_metadata_or_private_reference() -> None:
    payload = json.loads(writer_messages(writing_task(), [])[1]["content"])
    assert set(payload) == {"writing_instructions", "source_material", "voice_examples"}
    assert set(payload["source_material"]) == {"notes", "facts"}
    assert set(payload["writing_instructions"]) == {
        "request",
        "audience",
        "constraints",
    }


@pytest.mark.parametrize(
    "draft, expected",
    [
        ("an idea. keep it grounded and casual.", "instruction_echo"),
        ("A new idea", "uppercase"),
    ],
)
def test_lowercase_and_instruction_echo_fail_before_quality_judging(
    monkeypatch: pytest.MonkeyPatch, draft: str, expected: str
) -> None:
    _, judge = clients(monkeypatch)
    task = writing_task()
    task["input"]["instructions"] = "keep it grounded and casual"
    task["input"]["constraints"]["lowercase"] = True
    review = grade_draft(task, [], draft, judge, "fixed-judge")
    assert review["reward"] == 0
    assert expected in review["deterministic_failures"]
    judge.chat.completions.create.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"faithful": 0},
        {"instruction_free": 0},
        {"unsupported_claims": ["an invented feeling"]},
        {"meaning_changes": ["a reversed intention"]},
        {"postability": "major_edit"},
        {"postability": "reject"},
    ],
)
def test_quality_failures_cannot_earn_a_perfect_format_pass(overrides: dict) -> None:
    scores = {
        "faithful": 1,
        "instruction_free": 1,
        "unsupported_claims": [],
        "meaning_changes": [],
        "postability": "as_is",
        "reason": "Reviewed",
        "voice": 1,
        "brief_adherence": 1,
        "quality": 1,
        **overrides,
    }
    assert score_reward(scores) == 0


def test_fresh_notes_can_be_judged_without_a_finished_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, judge = clients(monkeypatch)
    task = writing_task()
    del task["private_grading"]
    run_task(task, [], actor, judge, "fixed-judge")
    payload = json.loads(
        judge.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    )
    assert payload["private_grading"] is None
    assert actor.trajectories.log_reward.call_args.kwargs["name"] == "writing_quality"


@pytest.mark.parametrize(
    "status, issues",
    [
        ("changed", ["reversed intent"]),
        ("ambiguous", ["unclear negation"]),
        ("preserved", ["unsupported personal claim"]),
    ],
)
def test_meaning_failures_stop_before_style_can_override_them(
    monkeypatch: pytest.MonkeyPatch, status: str, issues: list[str]
) -> None:
    _, judge = clients(monkeypatch)
    reply = next(iter(judge.chat.completions.create.side_effect))
    reply.choices[0].message.content = json.dumps(
        {
            "status": status,
            "literal_reading": "A different or ambiguous intent.",
            "issues": issues,
        }
    )
    judge.chat.completions.create.side_effect = [reply]
    review = grade_draft(
        writing_task(), [], "an apparently fluent draft", judge, "fixed-judge"
    )
    assert review["reward"] == 0
    assert review["judge_scores"] is None
    assert judge.chat.completions.create.call_count == 1
