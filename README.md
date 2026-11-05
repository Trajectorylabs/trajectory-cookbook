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
`agent-feedback` for customer-authorized product feedback. Install both complete skills:

```bash
npx skills add Trajectorylabs/trajectory-cookbook --skill trajectory --skill agent-feedback
```

Choose your coding agent when prompted. For a specific target, add `--agent codex` or
`--agent claude-code`. Installation is project-local by default and needs no Trajectory API
key or access to a private repository. Review existing skill files before replacing them.

For a Python-only project using uv, add `trajectory-sdk>=0.10.13` to that project,
then run its SDK installer from the project environment:

```bash
uv run trajectory skills install --directory .agents/skills
# For Claude Code, use --directory .claude/skills instead.
```

Choose one installer and ask your agent to read the `trajectory` skill. Use the Python
installer for skills matching your project's SDK; public copies come from the SDK's
canonical skills.

SDK upgrades do not refresh skills. Review local edits, then run `npx skills update` or
repeat the Python installation command. The Python installer preserves differing files
unless you pass `--replace`; see [updates and backups](https://docs.trajectory.ai/coding-agents#update).

Expect a platform link when work starts and an outcome/link update when it finishes,
with evidence for results or failures. Installing `agent-feedback` does not authorize reports.

The GSM8K link snippets assume the production API and platform. With another API
(`TRAJECTORY_BASE_URL`), pass its matching platform origin as `platform_url=` to
`PlatformLinks.from_client(client)`; never use a production link for another environment.

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

## Maintaining the skills

`skills/` contains complete copies of the SDK's canonical skills. After publishing an SDK
release that changes them, update the public copies:

```bash
uv run --no-project --with 'trajectory-sdk>=0.10.13' python scripts/sync_skills.py --write
```

CI checks these copies against the released SDK on relevant changes and weekly, and tests
installation of both skills for Codex and Claude Code. Publish the SDK release before
updating copies that depend on it. SDK and skill upgrades are separate customer actions.
