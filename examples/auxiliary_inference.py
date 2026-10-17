"""Run an auxiliary text request and return its trace ID with the response."""

from trajectory import Client


def call_auxiliary(
    prompt: str, model: str, api_key: str, base_url: str, timeout: float
) -> tuple[dict, str]:
    # The harness owns retries; do not reuse the actor's session key or TID.
    with Client(
        api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout
    ) as client:
        tid = client.trajectories.create().tid
        reason = "ERROR"
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                x_trajectory_id=tid,
            )
            choice = response.choices[0]
            if choice.finish_reason in ("length", "max_tokens"):
                reason = "TRUNCATION"
                raise ValueError("Auxiliary model response was truncated")
            if (
                choice.finish_reason != "stop"
                or not (choice.message.content or "").strip()
            ):
                raise ValueError("Auxiliary model returned no completed text answer")
            reason = "ENV_DONE"
        except Exception as error:
            try:
                client.trajectories.complete(tid, termination_reason=reason)
            except Exception as cleanup_error:
                error.add_note(
                    f"Trace completion also failed: {type(cleanup_error).__name__}"
                )
            raise
        client.trajectories.complete(tid, termination_reason=reason)
        return response.model_dump(), tid
