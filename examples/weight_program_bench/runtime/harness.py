"""One tool-free weight-writing reply, scored with the frozen expanded benchmark."""

import argparse
import json
from pathlib import Path

from trajectory import Client

from verify import apply_edits, evaluate, score_cap

ROOT = Path(__file__).resolve().parent


def run(client, task):
    trajectory_id = client.trajectories.create().tid
    try:
        response = client.chat.completions.create(
            model="policy",
            messages=[
                {"role": "system", "content": task["system_prompt"]},
                {"role": "user", "content": task["prompt"]},
            ],
            max_tokens=32_768,
            x_trajectory_id=trajectory_id,
        )
        choice = response.choices[0]
        result = {
            "game": task["game"],
            "source": task["source"],
            "episodes": task["episodes"],
        }
        try:
            weights, edit_count = apply_edits(choice.message.content or "")
        except (IndexError, KeyError, SyntaxError, TypeError, ValueError) as error:
            reward = 0.0
            result.update(parse_success=0.0, error=str(error))
        else:
            # Keep simulator/import failures distinct from invalid model submissions.
            raw_score, mean_length = evaluate(
                task["game"], task["source"], weights, task["episodes"]
            )
            cap = score_cap(task["game"], task["source"])
            reward = min(max(raw_score, 0.0), cap) / cap
            result.update(
                parse_success=1.0,
                edit_count=edit_count,
                raw_score=raw_score,
                score_cap=cap,
                mean_episode_length=mean_length,
            )
        termination = (
            "TRUNCATION"
            if choice.finish_reason in ("length", "max_tokens")
            else "ENV_DONE"
        )
        # The source grades any readable edits even when the response was truncated.
        client.trajectories.log_reward(
            trajectory_id, name="reward_normalized_minatar_score", value=reward
        )
    except Exception:
        try:
            client.trajectories.complete(trajectory_id, termination_reason="ERROR")
        except Exception:
            pass
        raise

    completed = client.trajectories.complete(
        trajectory_id, termination_reason=termination
    )
    if completed.status not in ("completed", "finalizing"):
        raise RuntimeError(f"Trajectory completion returned {completed.status!r}")
    return {
        **result,
        "trajectory_id": trajectory_id,
        "reward": reward,
        "finish_reason": choice.finish_reason,
        "termination_reason": termination,
        "capture_status": completed.status,
    }


def main(task_id):
    rows = (ROOT / "tasks.jsonl").read_text().splitlines()
    if not 0 <= task_id < len(rows):
        raise ValueError("task_id is outside the packaged roster")
    task = json.loads(rows[task_id])
    # Preserve the source harness's retry policy and SDK timeout default.
    with Client(max_retries=20) as client:
        print(json.dumps(run(client, task)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_id", type=int, required=True)
    main(parser.parse_args().task_id)
