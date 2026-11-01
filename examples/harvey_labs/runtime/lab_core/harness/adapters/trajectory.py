import atexit
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from lab_core.evaluation.judge import Judge
from lab_core.evaluation.run_eval import (
    evaluate_run,
    evaluate_run_dual,
    resolve_judge_models,
)
from lab_core.harness.adapters.base import ModelAdapter
from lab_core.harness.adapters.openai import OpenAIAdapter


class _Responses:
    def __init__(self, client: Any, tid: str):
        self.client = client
        self.tid = tid

    def create(self, **kwargs):
        if "reasoning" in kwargs:
            kwargs["reasoning"] = {"effort": kwargs["reasoning"]["effort"]}
        return self.client.responses.create(
            model="trajectory-session",
            input=kwargs.pop("input"),
            x_trajectory_id=self.tid,
            extra_body={key: value for key, value in kwargs.items() if key != "model"},
            timeout=None,
        )


class TrajectoryAdapter(OpenAIAdapter):
    """OpenAI Responses adapter backed by one managed Trajectory session."""

    def __init__(
        self,
        temperature: float,
        reasoning_effort: str | None,
        judges: Sequence[str] | None = None,
    ):
        from importlib import import_module

        Client = getattr(import_module("trajectory"), "Client")

        self.judges = resolve_judge_models(judges, None)
        # OpenAIAdapter.__init__ builds an OpenAI client, which needs a key that
        # a non-OpenAI judge doesn't bring; set its state with upstream's defaults.
        ModelAdapter.__init__(self, "trajectory-session", temperature, reasoning_effort)
        self.max_tokens = 128000
        self._context = []
        self._system_instructions = None
        # The agent loop doesn't retry; the SDK retries each call under one idempotency key.
        self.trajectory = Client()
        self.tid = self.trajectory.trajectories.create().tid
        self._open = True
        # Process exit doesn't complete a session; end any failed run as ERROR.
        atexit.register(self.fail)
        self.client = SimpleNamespace(responses=_Responses(self.trajectory, self.tid))

    def finalize(self, run_id: str, task: str, metrics: dict) -> None:
        if len(self.judges) == 2:
            scores = evaluate_run_dual(run_id, task, judge_models=self.judges)
            fraction = scores["dual_criterion_pass"]
            all_pass = scores["dual_all_pass_rate"]
        else:
            scores = evaluate_run(run_id, task, Judge(model=self.judges[0]))
            fraction = scores["n_passed"] / scores["n_criteria"]
            all_pass = float(scores["all_pass"])
        judges = " + ".join(self.judges)
        # All-pass rarely differs within a group, so training uses the passed fraction.
        self.trajectory.trajectories.log_reward(
            self.tid,
            name="criteria_fraction",
            value=fraction,
            explanation=f"Harvey LAB {judges} criteria passed",
        )
        self.trajectory.trajectories.log_reward(
            self.tid,
            name="all_pass",
            value=all_pass,
            weight=0.0,
            explanation=f"Harvey LAB {judges} all-pass score",
        )
        reason = "ENV_DONE"
        if (
            metrics["finish_reason"] == "context_overflow"
            or metrics["incomplete_details"]
        ):
            reason = "TRUNCATION"
        elif metrics["finish_reason"] == "max_turns_exceeded":
            reason = "MAX_STEPS"
        self._complete(reason)

    def fail(self) -> None:
        if self._open:
            self._complete("ERROR")

    def _complete(self, reason: str) -> None:
        # Stay open until a completion succeeds, so the exit hook can retry it.
        completed = self.trajectory.trajectories.complete(
            self.tid, termination_reason=reason
        )
        if completed.status not in ("completed", "finalizing"):
            raise RuntimeError(f"Trajectory completion returned {completed.status}")
        self._open = False
        self.trajectory.close()
