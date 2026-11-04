# Trajectory Cookbook

Examples for evaluating and training agents with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/). Each example packages a task harness
and grader, uploads a benchmark, and shows how to launch runs. Use **trajectory-sdk 0.10.1
or newer** and run commands from this repository’s root.

**Start with [GSM8K](examples/gsm8k/README.md):** create an agent, diagnose one math problem,
upload the benchmark, evaluate a model, and launch training. The walkthrough includes SDK
snippets for fetching evaluation scores and the training reward history shown in the product.

For a smaller introduction to responses, rewards, and trajectory completion, try
[T Factory](examples/t_factory/README.md#run-one-task).

The SDK also installs a Trajectory skill for coding agents. Before adapting a benchmark, copy it
into your project, and copy it again after upgrading the SDK:

```bash
dest=.claude/skills/trajectory  # Codex: .agents/skills/trajectory
mkdir -p "$dest"
uv run --no-project --with 'trajectory-sdk>=0.10.1' python -c 'import importlib.metadata as m, shutil, sys; shutil.copy(next(f.locate() for f in m.files("trajectory-sdk") if f.as_posix().endswith("skills/trajectory/SKILL.md")), sys.argv[1])' "$dest"
```

To share feedback on Trajectory's API, SDK, documentation, or workflow, use the
[agent-feedback skill](https://github.com/Trajectorylabs/trajectory-platform/blob/main/skills/agent-feedback/SKILL.md).
A free-text message is enough; a category, custom label, or extra details are optional.

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
