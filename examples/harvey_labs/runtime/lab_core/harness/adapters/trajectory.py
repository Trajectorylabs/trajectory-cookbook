import atexit
from types import SimpleNamespace
from typing import Any

from lab_core.evaluation.judge import Judge
from lab_core.evaluation.run_eval import evaluate_run
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
        judge_model: str,
    ):
        from importlib import import_module

        Client = getattr(import_module("trajectory"), "Client")

        super().__init__(
            model="trajectory-session",
            temperature=temperature,
            # Managed runs replace this with the run's max_output_tokens_per_step.
            max_tokens=28_000,
            reasoning_effort=reasoning_effort,
        )
        self.client.close()
        # The agent loop doesn't retry; the SDK retries each call under one idempotency key.
        self.trajectory = Client()
        self.tid = self.trajectory.trajectories.create().tid
        self._open = True
        # Process exit doesn't complete a session; end any failed run as ERROR.
        atexit.register(self.fail)
        self.client = SimpleNamespace(responses=_Responses(self.trajectory, self.tid))
        self.judge_model = judge_model

    def finalize(self, run_id: str, task: str, metrics: dict) -> None:
        try:
            scores = evaluate_run(run_id, task, Judge(model=self.judge_model))
            self.trajectory.trajectories.log_reward(
                self.tid,
                name="reward_accuracy",
                value=float(scores["all_pass"]),
                explanation=f"Harvey LAB {self.judge_model} all-pass score",
            )
        except Exception:
            self.fail()
            raise
        reason = "ENV_DONE"
        if metrics["finish_reason"] == "max_turns_exceeded":
            reason = "MAX_STEPS"
        elif (
            metrics["finish_reason"] == "context_overflow"
            or metrics["incomplete_details"]
        ):
            reason = "TRUNCATION"
        self.trajectory.trajectories.complete(self.tid, termination_reason=reason)
        self._open = False
        self.trajectory.close()

    def fail(self) -> None:
        # Stay open until a completion succeeds, so the exit hook can retry it.
        if not self._open:
            return
        self.trajectory.trajectories.complete(self.tid, termination_reason="ERROR")
        self._open = False
        self.trajectory.close()
