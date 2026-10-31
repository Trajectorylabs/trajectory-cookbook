# Trajectory Cookbook

Worked integrations for bringing a harness and grader to the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/). The
[Trajectory docs](https://docs.trajectory.ai) cover setup, benchmark registration, evaluation,
training, deployment, and trace capture.

The SDK also installs a Trajectory skill for coding agents. Before adapting a benchmark, copy it
into your project, and copy it again after upgrading the SDK:

```bash
dest=.claude/skills/trajectory  # Codex: .agents/skills/trajectory
mkdir -p "$dest"
uv run --no-project --with trajectory-sdk python -c 'import importlib.metadata as m, shutil, sys; shutil.copy(next(f.locate() for f in m.files("trajectory-sdk") if f.name == "SKILL.md"), sys.argv[1])' "$dest"
```

**Start here:** [Run one T Factory task](examples/t_factory/README.md) to see a model response,
recorded reward, and completed trajectory.

## Examples

| Example | What it shows |
| --- | --- |
| [T Factory: first task](examples/t_factory/README.md) | One response, reward, and completion. |
| [GSM8K: full workflow](examples/gsm8k/README.md) | Validate one task, evaluate and train, compare a checkpoint, and optionally deploy. |
| [T Factory: evaluation and training](examples/t_factory/training.md) | A small training walkthrough with a toy reward. |
| [Number guessing](examples/number_guessing/README.md) | Multi-turn execution, efficiency rewards, and context compaction. |
| [Inspect](examples/inspect.md) | Actor routing, native scoring, errors, and a separate judge. |
| [Harvey LAB](examples/harvey_labs.md) | An existing harness, native judge, and nested Podman. |
| [Big Finance Benchmark](examples/big_finance_benchmark.md) | An existing research agent, web tools, and rubric grading. |
| [Tau 3 Banking](examples/tau3_banking/banking.md) | An upstream evaluator and multi-turn customer-agent task. |
| [SWE-Bench Pro](examples/swe_bench_pro/README.md) | A bash-tool coding agent in an offline container, graded on a fresh one. |

[Reward integration](examples/reward_integration.md) works through an adapter that preserves
the source scorer. [Cut-off reply handling](examples/termination_reasons.md) shows when to
stop the actor loop. [Task diagnostics](examples/task_diagnostics.md) and
[diagnostic artifacts](examples/diagnostic_artifacts.md) show how to inspect a packaged run.
For application traffic, use the [production traces guide](https://docs.trajectory.ai/guides/production-traces)
(private beta).

When adapting another benchmark, keep its solver, grading rule, and actor/private-test
boundary. Validate the packaged path on one task and compare its native grade with the
source evaluator before scaling up.

## Beta testing and support

- [Book a beta testing call](https://calendly.com/d/dvxq-4dj-6pt/trajectory-beta-testing)
- [Join the Trajectory Discord](https://discord.gg/s5t2tpbNEE)

## License

Apache 2.0. See [LICENSE](LICENSE).
