"""Run one BigFinanceBench question with the upstream agent, tools, and rubric grader."""

import argparse
import asyncio
import json
import os
from pathlib import Path

import litellm
from big_finance_harness.agent import run_question
from big_finance_harness.grader import grade
from big_finance_harness.models.base import ModelClient, _to_oai_messages, _to_oai_tools
from big_finance_harness.prompts import SYSTEM_PROMPT
from big_finance_harness.tools import default_tools
from big_finance_harness.types import (
    DatasetItem,
    ModelResponse,
    ThinkingBlock,
    ToolUseBlock,
)
from trajectory import BadRequestError, Client

DATASET = Path("/opt/big_finance_bench/data/big_finance_subset.jsonl")
# The judge in upstream's README quickstart.
JUDGE = "openai:gpt-5.5"
# Upstream LiteLLMClient's per-call request_timeout.
REQUEST_TIMEOUT_SECONDS = 1800
STOP_REASONS = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


class TrajectoryModelClient(ModelClient):
    """Upstream's model client contract, served by the Trajectory model endpoint.

    Sampling and reasoning settings come from the run options.
    """

    snapshot = "big-finance-bench"

    def __init__(self, client: Client, tid: str):
        self.client = client
        self.tid = tid
        self.cancelled = False

    async def chat(
        self,
        system,
        messages,
        tools,
        temperature=None,
        thinking="off",
        max_output_tokens=65536,
    ) -> ModelResponse:
        request = _to_oai_messages(system, messages)
        for message in request:
            # Upstream returns reasoning to the model; the endpoint names it `reasoning`.
            if "reasoning_content" in message:
                message["reasoning"] = message.pop("reasoning_content")
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.snapshot,
                messages=request,
                x_trajectory_id=self.tid,
                max_tokens=max_output_tokens,
                extra_body={"tools": _to_oai_tools(tools), "tool_choice": "auto"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except BadRequestError as error:
            if error.code != "context_length_exceeded":
                raise
            raise litellm.ContextWindowExceededError(
                str(error), self.snapshot, "trajectory"
            ) from error
        metadata = (response.model_extra or {}).get("metadata") or {}
        # A run limit ended the trajectory; the empty reply ends upstream's loop.
        if metadata.get("trajectory_termination_reason") == "trajectory_cancelled":
            self.cancelled = True
        choice = response.choices[0]
        tool_calls = []
        for call in choice.message.tool_calls or []:
            # As upstream LiteLLMClient parses tool arguments.
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_unparsed_arguments": call.function.arguments}
            tool_calls.append(
                ToolUseBlock(id=call.id, name=call.function.name, input=arguments)
            )
        reasoning = (choice.message.model_extra or {}).get("reasoning")
        return ModelResponse(
            text=choice.message.content or "",
            tool_calls=tool_calls,
            stop_reason=STOP_REASONS.get(choice.finish_reason, "other"),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            resolved_model=response.model,
            request_id=response.id,
            thinking=ThinkingBlock(reasoning_content=reasoning) if reasoning else None,
        )


async def solve_and_grade(client: Client, tid: str, item: DatasetItem):
    policy = TrajectoryModelClient(client, tid)
    run = await run_question(
        question_id=item.id,
        question=item.query,
        reference_answer=item.reference_answer,
        client=policy,
        tools=default_tools(),
        system_prompt=SYSTEM_PROMPT,
    )
    if run.stop_reason == "error":
        raise RuntimeError(f"BigFinanceBench agent failed: {run.error}")
    graded = await grade(run=run, item=item, judge_model_id=JUDGE)
    truncated = policy.cancelled or run.stop_reason in ("max_steps", "context_exceeded")
    return run, graded, "TRUNCATION" if truncated else "ENV_DONE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    args = parser.parse_args()
    items = {
        item.id: item
        for item in map(
            DatasetItem.model_validate_json, DATASET.read_text().splitlines()
        )
    }
    if args.task not in items:
        parser.error("unknown task")
    # ingest.py --judge-model serves the same judge call through OpenRouter.
    judge = os.environ.get("BIG_FINANCE_BENCH_JUDGE_MODEL")
    if judge:
        litellm.model_alias_map[JUDGE.removeprefix("openai:")] = judge

    with Client() as client:
        tid = client.trajectories.create().tid
        try:
            run, graded, reason = asyncio.run(
                solve_and_grade(client, tid, items[args.task])
            )
            client.trajectories.log_event(
                tid,
                event_id="big-finance-bench-grade",
                name="big_finance_bench_grade",
                payload={
                    "stop_reason": run.stop_reason,
                    **graded.model_dump(mode="json"),
                },
            )
            earned = graded.rubric_points_earned
            possible = graded.rubric_points_possible
            answer = "correct" if graded.final_answer_correct else "incorrect"
            client.trajectories.log_reward(
                tid,
                name="rubric_points",
                value=earned,
                max_value=possible,
                explanation=f"{earned}/{possible} rubric points; final answer {answer} ({judge or JUDGE})",
            )
        except Exception:
            client.trajectories.complete(tid, termination_reason="ERROR")
            raise
        completed = client.trajectories.complete(tid, termination_reason=reason)
        if completed.status not in ("completed", "finalizing"):
            raise RuntimeError(f"Trajectory completion returned {completed.status}")
        print(
            f"task={args.task} ending={reason} points={earned}/{possible} "
            f"final_answer_correct={graded.final_answer_correct}",
            flush=True,
        )


if __name__ == "__main__":
    main()
