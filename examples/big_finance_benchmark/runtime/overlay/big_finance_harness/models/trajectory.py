from __future__ import annotations

import asyncio
import json

from big_finance_harness.models.base import (
    ModelClient,
    ThinkingLevel,
    _to_oai_messages,
    _to_oai_tools,
)
from big_finance_harness.types import Message, ModelResponse, ToolSpec, ToolUseBlock
from trajectory import Client


class TrajectoryClient(ModelClient):
    """Model client backed by one Trajectory session."""

    snapshot = "trajectory-session"

    def __init__(self) -> None:
        self.client = Client()
        self.tid = self.client.trajectories.create().tid
        self.truncated = False

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float | None = None,
        thinking: ThinkingLevel = "off",
        max_output_tokens: int = 65536,
    ) -> ModelResponse:
        # The managed run's options set max_tokens and sampling, so they are not sent here.
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.snapshot,
            messages=_to_oai_messages(system, messages),
            x_trajectory_id=self.tid,
            extra_body={"tools": _to_oai_tools(tools), "tool_choice": "auto"},
        )
        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolUseBlock] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {"_unparsed_arguments": tc.function.arguments}
            tool_calls.append(ToolUseBlock(id=tc.id, name=tc.function.name, input=args))

        # The run's per-step output limit cut this reply off. Returning no tool calls
        # ends the harness loop without running them, and the partial text is graded.
        self.truncated = choice.finish_reason in ("length", "max_tokens")
        if self.truncated:
            tool_calls = []

        stop_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }
        usage = response.usage
        return ModelResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            stop_reason=stop_map.get(choice.finish_reason or "", "other"),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            resolved_model=response.model,
            request_id=response.id,
            raw_response={"response": response.model_dump(mode="json")},
        )
