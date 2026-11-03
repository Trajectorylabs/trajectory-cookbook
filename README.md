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

## Benchmarks

| Benchmark | What it shows |
| --- | --- |
| [GSM8K](examples/gsm8k/README.md) | Validate one task, evaluate and train, compare a checkpoint, and optionally deploy. |
| [Chess reasoning](examples/chess/README.md) | Stockfish-labeled best moves across four categories, with reasoning and Python agents, model evaluation, and checkpoint comparison. |
| [Harvey LAB](examples/harvey_labs/README.md) | An existing harness, native judge, and nested Podman. |
| [Tau 3 Banking](examples/tau3_banking/README.md) | An upstream evaluator and multi-turn customer-agent task. |
| [SWE-Bench Pro](examples/swe_bench_pro/README.md) | A bash-tool coding agent in an offline container, graded on a fresh one. |
| [T Factory](examples/t_factory/README.md) | Start with one task, then evaluate and train against a toy reward. |

## Guides

| Guide | What it shows |
| --- | --- |
| [Reward integration](reward_integration.md) | Preserve a benchmark's source scorer behind an adapter. |
| [Cut-off reply handling](termination_reasons.md) | Stop the actor loop and record termination correctly. |
| [Task diagnostics](task_diagnostics.md) | Validate a packaged task before evaluation or training. |
| [Diagnostic artifacts](diagnostic_artifacts.md) | Retain and inspect task output from a packaged run. |
| [Inspect integration](inspect.md) | Route actors, preserve native scoring, and handle errors and judges. |
| [T Factory training](examples/t_factory/training.md) | Evaluate and train against a small reward-hacking example. |

For application traffic, use the [production traces guide](https://docs.trajectory.ai/guides/production-traces)
(private beta).

When adapting another benchmark, keep its solver, grading rule, and actor/private-test
boundary. Validate the packaged path on one task and compare its native grade with the
source evaluator before scaling up.

## Beta testing and support

- [Book a beta testing call](https://calendly.com/d/dvxq-4dj-6pt/trajectory-beta-testing)
- [Join the Trajectory Discord](https://discord.gg/s5t2tpbNEE)

## License

Cookbook-authored code is Apache 2.0; see [LICENSE](LICENSE). External benchmarks and
datasets retain their source licenses, linked from their example READMEs.
