# /// script
# dependencies = ["trajectory-sdk==0.7.1"]
# ///
"""Start an evaluation or inspect an existing run without starting another."""

import argparse

from trajectory import Client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List models available for evaluation.")
    start = commands.add_parser("start", help="Evaluate the single test task.")
    start.add_argument("--bench-id", required=True)
    start.add_argument("--model", required=True)
    status = commands.add_parser("status", help="Inspect a saved evaluation ID.")
    status.add_argument("--eval-run-id", required=True)
    args = parser.parse_args()
    client = Client()
    if args.command == "models":
        for model in client.evals.list_models().items:
            print(model.model_slug)
    elif args.command == "start":
        run = client.evals.start(
            args.bench_id,
            model_slug=args.model,
            display_name="Minimal arithmetic evaluation",
            eval_options={"max_samples": 1, "pass_at_k": 1},
        )
        print(f"eval_run_id={run.eval_run_id}", flush=True)
    else:
        progress = client.evals.runs.retrieve_progress(args.eval_run_id)
        print(progress.model_dump_json(indent=2))
        rewards = client.evals.runs.list_trajectory_rewards(args.eval_run_id)
        for reward in rewards.trajectory_rewards:
            print(reward.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
