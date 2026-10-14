"""LiteLLM provider that routes the Tau agent through a Trajectory SDK session."""

import json
from typing import Any
from uuid import uuid4

import litellm
from litellm import CustomLLM, ModelResponse
from trajectory import Client


class TrajectoryPolicy(CustomLLM):
    def __init__(self, client: Client, trajectory_id: str) -> None:
        super().__init__()
        self.client = client
        self.trajectory_id = trajectory_id
        self.calls = 0
        self.reasoning_leaks = 0
        self.malformed_tool_calls = 0
        self.truncated = False

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        optional_params: dict[str, Any],
        **kwargs: Any,
    ) -> ModelResponse:
        del model, kwargs
        params = {
            key: value for key, value in optional_params.items() if key != "max_retries"
        }
        response = self.client.chat.completions.create(
            model="trajectory-session",
            x_trajectory_id=self.trajectory_id,
            messages=messages,
            extra_headers={"X-Model-Request-Id": str(uuid4())},
            extra_body=params,
        )
        self.calls += 1
        message = response.choices[0].message
        self.reasoning_leaks += "</think>" in (message.content or "")
        for tool_call in message.tool_calls or []:
            try:
                json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                self.malformed_tool_calls += 1
        self.truncated |= response.choices[0].finish_reason == "length"
        return ModelResponse(**response.model_dump(mode="json"))


def register_policy(client: Client, trajectory_id: str) -> TrajectoryPolicy:
    provider = TrajectoryPolicy(client, trajectory_id)
    litellm.custom_provider_map.append(
        {"provider": "trajectory_sdk", "custom_handler": provider}
    )
    return provider
