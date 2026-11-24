import json
from unittest.mock import create_autospec

import pytest
from trajectory import Client
from trajectory.lib.platform_links import PlatformLinks
from trajectory.types.benchmarks.benchmark_list_item import BenchmarkListItem
from trajectory.types.benchmarks.specs.benchmark_details import BenchmarkDetails
from trajectory.types.benchmarks.tasks.benchmark_spec_task import BenchmarkSpecTask
from trajectory.types.eval_progress_response import EvalProgressResponse
from trajectory.types.start_eval_response import StartEvalResponse

from examples.writing_agent import train


def test_failed_baseline_never_starts_paid_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Client(api_key="fixture")
    benchmark = BenchmarkDetails(
        bench_id="bm_test",
        task_count=2,
        tasks=[
            BenchmarkSpecTask(task_id="task_train", split="train"),
            BenchmarkSpecTask(task_id="task_test", split="test"),
        ],
    )
    monkeypatch.setattr(
        client.benchmarks.specs,
        "retrieve",
        create_autospec(
            client.benchmarks.specs.retrieve,
            return_value=benchmark,
        ),
    )
    monkeypatch.setattr(
        client.benchmarks,
        "retrieve",
        create_autospec(
            client.benchmarks.retrieve,
            return_value=BenchmarkListItem(
                bench_id="bm_test",
                organization_id="org_test",
                name="writing",
                description="{}",
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            ),
        ),
    )
    monkeypatch.setattr(
        PlatformLinks,
        "from_client",
        create_autospec(
            PlatformLinks.from_client,
            return_value=PlatformLinks("org_test"),
        ),
    )
    monkeypatch.setattr(
        client.evals,
        "create",
        create_autospec(
            client.evals.create,
            return_value=StartEvalResponse(
                bench_id="bm_test", eval_run_id="eval_test", display_name=None
            ),
        ),
    )
    monkeypatch.setattr(
        client.training, "create", create_autospec(client.training.create)
    )
    monkeypatch.setattr(
        train,
        "validate_calibration",
        create_autospec(train.validate_calibration),
    )
    monkeypatch.setattr(
        train,
        "wait_evaluation",
        create_autospec(
            train.wait_evaluation,
            side_effect=RuntimeError("baseline failed"),
        ),
    )
    with pytest.raises(RuntimeError, match="baseline failed"):
        train.train_and_evaluate(
            client, "bm_test", "model_fixture", 3, 60, calibration={}
        )
    client.training.create.assert_not_called()


def test_invalid_run_budget_never_contacts_platform() -> None:
    client = Client(api_key="fixture")
    with pytest.raises(ValueError, match="positive"):
        train.train_and_evaluate(client, "bm_test", "model_fixture", 0, 60)


def test_training_requires_author_calibration_before_contacting_platform() -> None:
    with pytest.raises(ValueError, match="calibration"):
        train.train_and_evaluate(
            Client(api_key="fixture"), "bm_test", "model_fixture", 3, 60
        )


@pytest.mark.parametrize(
    "mistake",
    [
        "unconfirmed",
        "wrong_scorer",
        "wrong_voice",
        "too_few",
        "no_negatives",
        "disagreement",
    ],
)
def test_unusable_judge_calibration_blocks_training(mistake: str) -> None:
    contract = {
        "scorer_sha256": "hash_fixture",
        "voice_sha256": "voice_fixture",
        "judge_model": "judge_fixture",
        "judge_provider": "openai",
    }
    pairs = [
        {
            "id": f"pair_{index}",
            "author_label": "as_is" if index < 10 else "reject",
            "judge_label": "as_is" if index < 10 else "reject",
        }
        for index in range(20)
    ]
    report = {**contract, "author_confirmed": True, "pairs": pairs}
    if mistake == "unconfirmed":
        report["author_confirmed"] = False
    elif mistake == "wrong_scorer":
        report["scorer_sha256"] = "old_scorer"
    elif mistake == "wrong_voice":
        report["voice_sha256"] = "old_voice"
    elif mistake == "too_few":
        report["pairs"] = pairs[:5]
    elif mistake == "no_negatives":
        for pair in pairs:
            pair["author_label"] = pair["judge_label"] = "as_is"
    elif mistake == "disagreement":
        for pair in pairs[:5]:
            pair["judge_label"] = "reject"
    with pytest.raises(ValueError):
        train.validate_calibration(report, json.dumps(contract))


def test_author_calibration_with_varied_agreed_labels_is_accepted() -> None:
    contract = {
        "scorer_sha256": "hash_fixture",
        "voice_sha256": "voice_fixture",
        "judge_model": "judge_fixture",
        "judge_provider": "openai",
    }
    report = {
        **contract,
        "author_confirmed": True,
        "pairs": [
            {
                "id": f"pair_{index}",
                "author_label": "as_is" if index < 10 else "reject",
                "judge_label": "as_is" if index < 10 else "reject",
            }
            for index in range(20)
        ],
    }
    train.validate_calibration(report, json.dumps(contract))


def test_partial_evaluation_cannot_be_used_to_start_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Client(api_key="fixture")
    progress = EvalProgressResponse.model_validate(
        {
            "eval_run_id": "eval_test",
            "status": "completed",
            "terminal_rollouts": 2,
            "total_rollouts": 2,
            "live": {
                "scored_rollouts": 1,
                "waiting_rollouts": 0,
                "running_rollouts": 0,
                "awaiting_result_rollouts": 0,
                "provisional_reward_mean": 0.5,
                "final_reward_mean": 0.5,
                "failed_rollouts": 1,
                "cancelled_rollouts": 0,
            },
        }
    )
    monkeypatch.setattr(
        client.evals.runs,
        "progress",
        create_autospec(client.evals.runs.progress, return_value=progress),
    )
    with pytest.raises(RuntimeError, match="failed, cancelled or ungraded"):
        train.wait_evaluation(client, "eval_test", 60)
