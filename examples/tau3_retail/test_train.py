from types import SimpleNamespace

import pytest
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
