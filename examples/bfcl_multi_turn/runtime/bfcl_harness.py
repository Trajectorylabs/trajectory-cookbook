"""Run one BFCL multi-turn entry with BFCL's own inference loop and state/response checker."""

import argparse
import time
from typing import Any

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.eval_checker.eval_runner import _evaluate_single_multi_turn_entry
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry
from protocol import BFCL_COMMIT, BFCL_PACKAGE, POLICY_NAME, split_of
from trajectory import APIError, Client


class TrajectoryHandler(OpenAICompletionsHandler):
    """BFCL's OpenAI function-calling handler with inference routed through a Trajectory session."""

    def __init__(self, client: Client, trajectory_id: str) -> None:
        BaseHandler.__init__(
            self,
            model_name=POLICY_NAME,
            temperature=None,
            registry_name=f"{POLICY_NAME}-FC",
            is_fc_model=True,
        )
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.trajectory = client
        self.trajectory_id = trajectory_id
        self.calls = 0
        self.output_limited_replies = 0

    def generate_with_backoff(self, **kwargs):
        start = time.time()
        response = self.trajectory.chat.completions.create(
            model="bfcl",  # Any model name.
            messages=kwargs["messages"],
            extra_body={"tools": kwargs["tools"]} if kwargs.get("tools") else None,
            # Managed runs apply the run's output limit and sampling settings instead.
            max_tokens=8_192,
            x_trajectory_id=self.trajectory_id,
        )
        self.calls += 1
        self.output_limited_replies += response.choices[0].finish_reason in ("length", "max_tokens")
        return response, time.time() - start

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        message = api_response.choices[0].message
        calls = message.tool_calls or []
        usage = getattr(api_response, "usage", None)
        return {
            # Same shapes as the upstream handler: tool calls as {name: json_args}, else text.
            "model_responses": (
                [{call.function.name: call.function.arguments} for call in calls]
                if calls
                else message.content
            ),
            # A plain dict, so the history serializes on the next call.
            "model_responses_message_for_chat_history": {
                "role": "assistant",
                "content": message.content or "",
                **(
                    {
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in calls
                        ]
                    }
                    if calls
                    else {}
                ),
            },
            "tool_call_ids": [call.id for call in calls],
            "input_token": getattr(usage, "prompt_tokens", 0) or 0,
            "output_token": getattr(usage, "completion_tokens", 0) or 0,
        }


def load_entry(entry_id: str) -> tuple[str, int, dict, list]:
    category, scenario = entry_id.rsplit("_", 1)
    entry = next(e for e in load_dataset_entry(category) if e["id"] == entry_id)
    answer = next(a for a in load_ground_truth_entry(category) if a["id"] == entry_id)
    return category, int(scenario), entry, answer["ground_truth"]


def main(entry_id: str) -> None:
    category, scenario, entry, ground_truth = load_entry(entry_id)
    client = Client(max_retries=20, timeout=1200)
    trajectory_id = client.trajectories.create().tid
    handler = TrajectoryHandler(client, trajectory_id)
    try:
        model_result, metadata = handler.inference(
            entry, include_input_log=False, exclude_state_log=True
        )
    except APIError:
        client.trajectories.complete(trajectory_id, termination_reason="ERROR")
        raise

    # BFCL counts an entry correct only if every turn's end state and responses match.
    result = _evaluate_single_multi_turn_entry(
        handler,
        entry_id,
        model_result,
        ground_truth,
        dict(entry),
        handler.model_name_underline_replaced,
        category,
    )
    valid = bool(result["valid"])
    error = result.get("error", {})
    forced_quit = error.get("error_type") == "multi_turn:force_terminated"

    client.trajectories.log_reward(trajectory_id, name="reward_accuracy", value=float(valid))
    client.trajectories.log_event(
        trajectory_id,
        event_id="bfcl-protocol",
        name="bfcl_protocol",
        payload={
            "entry_id": entry_id,
            "category": category,
            "scenario": scenario,
            "split": split_of(scenario),
            "bfcl_package": BFCL_PACKAGE,
            "bfcl_commit": BFCL_COMMIT,
            "turns": len(ground_truth),
            "policy_calls": handler.calls,
            "output_limited_replies": handler.output_limited_replies,
            "error_type": error.get("error_type"),
            "error_message": error.get("error_message"),
        },
    )
    reason = "TRUNCATION" if forced_quit else "ENV_DONE"
    completed = client.trajectories.complete(trajectory_id, termination_reason=reason)
    if completed.status not in ("completed", "finalizing"):
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True, help="e.g. multi_turn_base_12")
    main(parser.parse_args().entry)
