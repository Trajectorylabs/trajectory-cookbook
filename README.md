# Trajectory Cookbook

Examples for evaluating and training agents with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/). Each example packages a task harness
and grader, uploads a benchmark, and shows how to launch runs. Use **trajectory-sdk 0.10.13
or newer** and run commands from this repository’s root.

**Start with [GSM8K](examples/gsm8k/README.md):** create an agent, diagnose one math problem,
upload the benchmark, evaluate a model, and launch training. The walkthrough includes SDK
snippets for fetching evaluation scores and the training reward history shown in the product.

For a smaller introduction to responses, rewards, and trajectory completion, try
[T Factory](examples/t_factory/README.md#run-one-task).

## Use with a coding agent

The SDK bundles two skills: `trajectory` for integration and run inspection, and
`agent-feedback` for customer-authorized product feedback. Install their entrypoints
using the same Python environment you will use for the example:

```bash
uv run --with 'trajectory-sdk>=0.10.13' trajectory skills install --directory .agents/skills
# For Claude Code, use --directory .claude/skills instead.
```

Ask your coding agent to use the `trajectory` skill. Each entrypoint reads the full
instructions from the project's installed SDK, so an SDK upgrade updates the guidance
it loads. Reload the skill in an already-running agent after upgrading. Existing custom
skills are preserved; [the setup guide](https://docs.trajectory.ai/coding-agents) explains
how to migrate old copied skills with a backup.

The agent should share a run link when work starts, then link the relevant evidence when
reporting results or failures. Installing `agent-feedback` does not authorize sending feedback.

Examples use the production API and platform by default. For another environment, set
`TRAJECTORY_BASE_URL` and `TRAJECTORY_PLATFORM_URL` to the matching API and platform origins.
The scripts pass the latter to [PlatformLinks](https://docs.trajectory.ai/platform); notebook
or Python-session snippets can pass `platform_url=` explicitly. Never use a production link
for a run in another environment.

## Examples

| Benchmark | What it shows |
| --- | --- |
| [GSM8K](examples/gsm8k/README.md) | Diagnose one task, upload 64 training and 16 held-out problems, evaluate, and train. |
| [Harvey LAB](examples/harvey_labs/README.md) | Legal research tasks with the source harness, native judge, and nested Podman. |
| [Tau 3 Banking](examples/tau3_banking/README.md) | Multi-turn banking tasks with the upstream simulator and evaluator. |
| [SWE-Bench Pro](examples/swe_bench_pro/README.md) | A bash-tool coding agent in an offline container, graded on a fresh one. |
| [Terminal-Bench 4.0](examples/terminal_bench_4/README.md) | Evaluation-only tasks using Harbor’s environment and verifier. |
| [T Factory](examples/t_factory/README.md) | Start with one task, then evaluate and train against a toy reward. |

For SDK adapter details, see the guides for [source graders](https://docs.trajectory.ai/guides/source-graders),
[Inspect](https://docs.trajectory.ai/guides/inspect), [diagnostics](https://docs.trajectory.ai/guides/spec#validate-one-task-before-uploading),
[diagnostic artifacts](https://docs.trajectory.ai/guides/diagnostic-artifacts), and
[termination reasons](https://docs.trajectory.ai/guides/termination-reasons).
For the full SDK reference and guides, see [Trajectory docs](https://docs.trajectory.ai). For application traffic,
see [production traces](https://docs.trajectory.ai/guides/production-traces) (private beta).

When adapting another benchmark, keep its solver, grading rule, and actor/private-test
boundary. Validate the packaged path on one task and compare its native grade with the
source evaluator before scaling up.

## Beta testing and support

- [Book a beta testing call](https://calendly.com/d/dvxq-4dj-6pt/trajectory-beta-testing)
- [Join the Trajectory Discord](https://discord.gg/s5t2tpbNEE)

## License

Cookbook-authored code is Apache 2.0; see [LICENSE](LICENSE). External benchmarks and
datasets retain their source licenses, linked from their example READMEs.
