# Trajectory Cookbook

Worked integrations for bringing a harness and grader to the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/). The
[Trajectory docs](https://docs.trajectory.ai) cover setup, benchmark registration, evaluation,
training, deployment, and trace capture. The SDK installs the same guides and a Trajectory skill
for coding agents; `python -m pip show -f trajectory-sdk` lists them under
`share/doc/trajectory-sdk/`. When adapting a benchmark, copy `skills/trajectory/SKILL.md` to
`.claude/skills/trajectory/` for Claude Code or `.agents/skills/trajectory/` for Codex.

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
