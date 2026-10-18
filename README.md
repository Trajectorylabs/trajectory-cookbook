# Trajectory Cookbook

Worked integrations for bringing a harness and grader to the
[Trajectory SDK](https://github.com/Trajectorylabs/trajectory-platform). The SDK's
[guides](https://github.com/Trajectorylabs/trajectory-platform/tree/main/docs/guides)
cover setup, benchmark registration, evaluation, training, deployment, and trace capture.
Give your coding agent the SDK's
[Trajectory skill](https://github.com/Trajectorylabs/trajectory-platform/blob/main/skills/trajectory/SKILL.md)
when adapting a benchmark.

The runnable examples pin `trajectory-sdk==0.9.24`.

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
For application traffic, use the SDK's [production traces guide](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/guides/production-traces.mdx)
(private beta).

When adapting another benchmark, keep its solver, grading rule, and actor/private-test
boundary. Validate the packaged path on one task and compare its native grade with the
source evaluator before scaling up.

## Complete native integrations

- [Harvey LAB SDK integration](https://github.com/Trajectorylabs/harvey-labs/pull/11)
- [Big Finance Benchmark SDK integration](https://github.com/Trajectorylabs/big-finance-benchmark-public/pull/1)

## Beta testing and support

- [Book a beta testing call](https://calendly.com/d/dvxq-4dj-6pt/trajectory-beta-testing)
- [Join the Trajectory Discord](https://discord.gg/s5t2tpbNEE)

## License

Apache 2.0. See [LICENSE](LICENSE).
