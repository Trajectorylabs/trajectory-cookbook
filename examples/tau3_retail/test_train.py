from types import SimpleNamespace

import pytest
import train
from train import _wait_for_evaluation


def test_completed_evaluation_rejects_ungraded_attempt():
    page = SimpleNamespace(
        items=[
            SimpleNamespace(reward=1.0, termination_reason="ENV_DONE"),
            SimpleNamespace(reward=None, termination_reason="ERROR"),
        ],
        has_more=False,
        next_cursor=None,
    )

    def list_attempts(**query):
        assert "cursor" not in query
        return page

    client = SimpleNamespace(
        evals=SimpleNamespace(
            runs=SimpleNamespace(
                progress=lambda _: SimpleNamespace(
                    status="completed", terminal_rollouts=2, total_rollouts=2
                ),
                retrieve=lambda _: SimpleNamespace(reward_mean=0.5),
            )
        ),
        trajectories=SimpleNamespace(list=list_attempts),
    )

    with pytest.raises(RuntimeError, match="missing or failed grades"):
        _wait_for_evaluation(client, "partial-run", "baseline", 0)


def test_workflow_reuses_baseline_benchmark_id(monkeypatch):
    calls = []

    def evaluate(*args, bench_id=None, **kwargs):
        calls.append(("evaluate", bench_id))
        return (0.5, "bm_pinned")

    def start_training(client, bench_id, model, num_steps):
        calls.append(("train", bench_id))
        return "trn_1"

    monkeypatch.setattr(train, "evaluate", evaluate)
    monkeypatch.setattr(train, "start_training", start_training)
    monkeypatch.setattr(
        train,
        "wait_for_training",
        lambda *args: SimpleNamespace(status="succeeded"),
    )
    client = SimpleNamespace(
        training=SimpleNamespace(
            checkpoints=SimpleNamespace(
                list=lambda _: [SimpleNamespace(step_index=2, checkpoint_id="cpt_2")]
            )
        )
    )

    train.train_and_evaluate(client, "agent", "name", "model", 2, 0)

    assert calls == [
        ("evaluate", None),
        ("train", "bm_pinned"),
        ("evaluate", "bm_pinned"),
    ]
