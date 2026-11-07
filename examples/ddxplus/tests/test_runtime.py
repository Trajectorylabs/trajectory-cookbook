import json
from unittest.mock import create_autospec

import pytest
from trajectory import Client
from trajectory.types.inference.chat_completion_response import ChatCompletionResponse

from examples.ddxplus.runtime.ddxplus_harness import Interview, run


def interview():
    metadata = {}
    for key, kind, default, parent in [
        ("fever", "B", 0, "fever"),
        ("rash", "B", 0, "rash"),
        ("color", "C", "none", "rash"),
        ("sites", "M", "none", "rash"),
        ("absent", "B", 0, "absent"),
        ("missing_root", "B", 0, "omitted"),
    ]:
        metadata[key] = {
            "data_type": kind,
            "default_value": default,
            "code_question": parent,
            "question_en": f"question {key}",
            "value_meaning": {
                "red": {"en": "red"},
                "arm": {"en": "arm"},
                "leg": {"en": "leg"},
                "none": {"en": "none"},
            },
        }
    patient = {
        "age": 23,
        "sex": "F",
        "evidences": ["fever", "rash", "color_@_red", "sites_@_arm", "sites_@_leg"],
        "initial_evidence": "fever",
        "pathology": "condition_a",
        "differential": [["condition_a", 0.7], ["condition_b", 0.3]],
    }
    return Interview(patient, metadata, ["condition_a", "condition_b"])


def response(name="submit_differential", arguments=None, finish="tool_calls"):
    return ChatCompletionResponse.model_validate(
        {
            "id": "reply_test",
            "object": "chat.completion",
            "created": 0,
            "model": "actor",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_test",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(
                                        arguments
                                        if arguments is not None
                                        else {"conditions": ["condition_a"]}
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )


def client_with_responses(replies):
    client = Client(api_key="test")
    client.chat.completions.create = create_autospec(
        client.chat.completions.create, side_effect=replies
    )
    client.trajectories.complete = create_autospec(client.trajectories.complete)
    client.trajectories.log_reward = create_autospec(client.trajectories.log_reward)
    return client


def test_static_values_dependencies_defaults_and_multichoice():
    state = interview()
    assert state.ask({"evidence_id": "color"}) == {
        "error": "Ask parent evidence first",
        "parent_evidence_id": "rash",
        "remaining_questions": 15,
    }
    assert state.ask({"evidence_id": "rash"})["values"] == [True]
    assert state.ask({"evidence_id": "color"})["values"] == ["red"]
    assert state.ask({"evidence_id": "sites"})["values"] == ["arm", "leg"]
    assert state.ask({"evidence_id": "absent"})["values"] == [False]
    assert state.ask({"evidence_id": "missing_root"})["values"] == [False]
    assert state.ask({"evidence_id": "sites"})["values"] == ["arm", "leg"]
    assert len(state.questions) == 5
    assert state.ask({"evidence_id": "../../tasks.jsonl"}) == {
        "error": "Unknown evidence_id",
        "remaining_questions": 10,
    }


def test_inactive_parent_returns_metadata_default_without_inventing_symptoms():
    state = interview()
    state.profile.pop("rash")
    state.profile.pop("color")
    state.ask({"evidence_id": "rash"})
    assert state.ask({"evidence_id": "color"}) == {
        "evidence_id": "color",
        "question": "question color",
        "data_type": "C",
        "values": ["none"],
        "meanings": ["none"],
        "not_applicable": True,
        "remaining_questions": 13,
    }


def test_source_contradiction_remains_an_error():
    state = interview()
    state.profile.pop("rash")
    state.ask({"evidence_id": "rash"})
    with pytest.raises(ValueError, match="inactive parent"):
        state.ask({"evidence_id": "color"})


def test_prompt_exposes_only_initial_observation_and_public_catalog():
    state = interview()
    prompt = state.prompt()
    assert "initial_evidence" in prompt
    assert "pathology" not in prompt and 'differential":' not in prompt
    assert "0.7" not in prompt and '"values": ["red"]' not in prompt


@pytest.mark.parametrize(
    ("conditions", "score"),
    [
        (["condition_a"], 1),
        (["condition_b", "condition_a"], 0),
        ([], 0),
        (["condition_a", "condition_a"], 1),
        (["unknown"], 0),
        ([{}], 0),
        (["condition_a", "unknown"], 1),
        (
            [
                "condition_a",
                "condition_b",
                "condition_a",
                "condition_b",
                "condition_a",
                "condition_b",
            ],
            1,
        ),
    ],
)
def test_only_valid_top_one_earns_reward(conditions, score):
    client = client_with_responses([response(arguments={"conditions": conditions})])
    run(interview(), client, "tid_test")
    assert client.trajectories.log_reward.call_count == 1
    assert client.trajectories.log_reward.call_args.kwargs["value"] == score
    client.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="ENV_DONE"
    )


@pytest.mark.parametrize("finish", ["length", "max_tokens", "content_filter"])
def test_cutoff_or_rejected_output_never_dispatches_or_grades(finish):
    client = client_with_responses([response(finish=finish)])
    run(interview(), client, "tid_test")
    client.trajectories.log_reward.assert_not_called()
    client.trajectories.complete.assert_called_once_with(
        "tid_test",
        termination_reason="ERROR" if finish == "content_filter" else "TRUNCATION",
    )


def test_questions_reach_next_actor_turn_but_private_profile_does_not():
    client = client_with_responses(
        [response("ask_evidence", {"evidence_id": "rash"}), response()]
    )
    run(interview(), client, "tid_test")
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert json.loads(
        next(message for message in messages if message["role"] == "tool")["content"]
    )["values"] == [True]
    assert (
        next(message for message in messages if message["role"] == "tool")[
            "tool_call_id"
        ]
        == "call_test"
    )
    assert client.trajectories.log_reward.call_count == 1


def test_malformed_tool_json_is_protocol_error_not_wrong_answer():
    reply = response()
    reply.choices[0].message.tool_calls[0].function.arguments = "{broken"
    client = client_with_responses([reply])
    with pytest.raises(json.JSONDecodeError):
        run(interview(), client, "tid_test")
    client.trajectories.log_reward.assert_not_called()
    client.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="ERROR"
    )


def test_model_service_failure_is_not_a_zero_grade():
    client = client_with_responses([RuntimeError("service unavailable")])
    with pytest.raises(RuntimeError, match="service unavailable"):
        run(interview(), client, "tid_test")
    client.trajectories.log_reward.assert_not_called()
    client.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="ERROR"
    )


def test_stop_with_tool_payload_is_protocol_error():
    client = client_with_responses([response(finish="stop")])
    run(interview(), client, "tid_test")
    client.trajectories.log_reward.assert_not_called()
    client.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="ERROR"
    )


def test_no_final_submission_is_truncation_not_wrong_answer():
    reply = response(finish="stop")
    reply.choices[0].message.tool_calls = []
    client = client_with_responses([reply])
    run(interview(), client, "tid_test")
    client.trajectories.log_reward.assert_not_called()
    client.trajectories.complete.assert_called_once_with(
        "tid_test", termination_reason="TRUNCATION"
    )


def test_budget_never_reveals_extra_evidence(monkeypatch):
    monkeypatch.setattr("examples.ddxplus.runtime.ddxplus_harness.QUESTION_LIMIT", 1)
    state = interview()
    state.ask({"evidence_id": "rash"})
    assert state.ask({"evidence_id": "color"}) == {
        "error": "Question budget exhausted; submit your differential",
        "remaining_questions": 0,
    }
    assert "color" not in state.revealed
    assert state.ask({"evidence_id": "rash"})["values"] == [True]


def test_schema_enums_match_public_catalogs_and_are_not_shared_mutably():
    state = interview()
    tools = state.tools()
    assert tools[0]["function"]["parameters"]["properties"]["evidence_id"][
        "enum"
    ] == list(state.evidences)
    assert (
        tools[1]["function"]["parameters"]["properties"]["conditions"]["items"]["enum"]
        == state.conditions
    )
    tools[0]["function"]["parameters"]["properties"]["evidence_id"]["enum"].append(
        "private_key"
    )
    assert (
        "private_key"
        not in interview().tools()[0]["function"]["parameters"]["properties"][
            "evidence_id"
        ]["enum"]
    )


def test_primary_accuracy_is_independent_of_invalid_secondary_differential():
    metrics = interview().grade({"conditions": ["condition_a", "outside_catalog"]})
    assert metrics["accuracy"] == 1.0
    assert metrics["valid_primary"] is True
    assert metrics["differential_valid"] is False
    assert metrics["differential_recall"] == 0.0
