"""Write a Breakout policy in one reply, then score it on private episode seeds."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import trajectory

from scorer import evaluate, parse_weights

_ROOT = Path(__file__).resolve().parent


def run(client, prompt, seeds):
    trajectory_id = client.trajectories.create().tid
    try:
        response = client.chat.completions.create(
            model="policy",  # Managed runs select the model.
            messages=[{"role": "user", "content": prompt}],
            # Managed runs apply their output limit and sampling settings.
            max_tokens=32_768,
            x_trajectory_id=trajectory_id,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        termination = (
            "TRUNCATION"
            if choice.finish_reason in {"length", "max_tokens"}
            else "ENV_DONE"
        )
        result = {
            "trajectory_id": trajectory_id,
            "finish_reason": choice.finish_reason,
            "termination_reason": termination,
            "reply_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "reward": 0.0,
        }
        if termination == "TRUNCATION":
            result.update(valid=False, error="Editor output token cap")
        else:
            try:
                weights, audit = parse_weights(content)
            except ValueError as exc:
                result.update(valid=False, error=str(exc))
            else:
                metrics = evaluate(weights, seeds)
                result.update(
                    valid=True,
                    audit=audit,
                    metrics=metrics,
                    reward=metrics["mean_return"],
                )

        # The raw component is in game points; the platform aggregate is /10000.
        client.trajectories.log_reward(
            trajectory_id,
            name="breakout_score",
            value=result["reward"],
            min_value=0.0,
            max_value=10_000.0,
        )
        state = client.trajectories.complete(
            trajectory_id, termination_reason=termination
        )
        deadline = time.monotonic() + 300
        while state.status == "finalizing" and time.monotonic() < deadline:
            time.sleep(2)
            state = client.trajectories.retrieve(trajectory_id, include_steps=False)
        if state.status != "completed":
            raise RuntimeError("Capture did not finish finalizing")
        return result
    except Exception:
        # Infrastructure failures must not become zero-scoring game submissions.
        try:
            client.trajectories.complete(trajectory_id, termination_reason="ERROR")
        except Exception:
            pass
        raise


def main(task_id):
    rows = (_ROOT / "tasks.jsonl").read_text().splitlines()
    if not 0 <= task_id < len(rows):
        raise ValueError("task_id is outside the packaged task roster")
    task = json.loads(rows[task_id])
    prompt = (_ROOT / "prompt.txt").read_text()
    # Allow slow reasoning responses without changing the SDK's retry policy.
    with trajectory.Client(timeout=5_400) as client:
        print(json.dumps(run(client, prompt, task["seeds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_id", type=int, required=True)
    main(parser.parse_args().task_id)
