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
| [Harvey LAB](examples/harvey_labs/README.md) | An existing harness, native judge, and nested Podman. |
| [Tau 3 Banking](examples/tau3_banking/README.md) | An upstream evaluator and multi-turn customer-agent task. |
| [SWE-Bench Pro](examples/swe_bench_pro/README.md) | A bash-tool coding agent in an offline container, graded on a fresh one. |
| [Terminal-Bench 4.0](examples/terminal_bench_4/README.md) | Evaluate a published task with Harbor's environment and verifier. |
| [T Factory](examples/t_factory/README.md) | Start with one task, then evaluate and train against a toy reward. |

For SDK adapter details, see the guides for [source graders](https://docs.trajectory.ai/guides/source-graders),
[Inspect](https://docs.trajectory.ai/guides/inspect), [diagnostics](https://docs.trajectory.ai/guides/spec#validate-one-task-before-uploading),
[diagnostic artifacts](https://docs.trajectory.ai/guides/diagnostic-artifacts), and
[termination reasons](https://docs.trajectory.ai/guides/termination-reasons).
For training, start with [T Factory](examples/t_factory/training.md). For application traffic,
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
