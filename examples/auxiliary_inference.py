"""Run an auxiliary text request on its own organization-owned trajectory."""

from trajectory import Client


def call_auxiliary(
    prompt: str, model: str, api_key: str, base_url: str, timeout: float
) -> dict:
    origin = base_url.rstrip("/")
    if origin.endswith("/v1"):
        origin = origin[:-3]
    # The calling harness owns retries. Never inherit the actor's session credential or TID.
    with Client(
        api_key=api_key, base_url=origin, max_retries=0, timeout=timeout
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
            if choice.finish_reason != "stop" or not choice.message.content:
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
        return response.model_dump()


def server_allows_retry(error: Exception) -> bool:
    headers = getattr(getattr(error, "response", None), "headers", {})
    return str(headers.get("x-should-retry", "")).strip().lower() != "false"
