# Public Tau 3 Banking

This example packages Sierra's public 97-task Banking benchmark from Tau 2 commit
`fc0055dc4e0a316c3f83133267fbd6faaa770992` (v1.0.1). The original task bytes, retrieval,
user simulator, evaluator, and task digest checks are preserved. Task IDs are sorted into
80 training and 17 held-out test tasks.

The runtime locks `trajectory-sdk==0.9.16` and calls the SDK for policy inference, reward
publication, protocol events, and completion. It invokes the virtual environment's Python
by absolute path and declares `TAU2_DATA_DIR` in the task environment. Neither depends on
the sandbox launcher restoring Docker's PATH or working directory.

Choose an existing agent and organization API key using the
[Cookbook guidance](../../README.md#choose-or-create-an-agent). Register the organization
secret `OPENAI_API_KEY` for the upstream user simulator and judge; its value is never
included in the image or upload.

```sh
uv run examples/tau3_banking/ingest.py --agent-name YOUR_AGENT_NAME
```

Ingestion waits for a new runtime build. Follow the
[benchmark diagnostic guidance](../task_diagnostics.md) to execute a subset before training.
Image build readiness alone does not verify policy inference, grading, or reward publication.
