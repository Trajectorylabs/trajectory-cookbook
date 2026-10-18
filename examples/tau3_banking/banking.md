# Public Tau 3 Banking

This example packages Sierra's public 97-task Banking benchmark from Tau 2 commit
`fc0055dc4e0a316c3f83133267fbd6faaa770992` (v1.0.1). The original task bytes, retrieval,
user simulator prompts/model, evaluator, and task digest checks are preserved. Task IDs are sorted into
80 training and 17 held-out test tasks.

The runtime locks `trajectory-sdk==0.9.24` and calls the SDK for policy inference, reward
publication, protocol events, and completion. It invokes the virtual environment's Python
by absolute path and declares `TAU2_DATA_DIR` in the task environment. Neither depends on
the sandbox launcher restoring Docker's PATH or working directory.

The runtime retries user responses containing neither text nor tool calls, with at most
three attempts and a one-second wait. Each attempt receives the same conversation state;
empty responses do not enter the conversation. Exhaustion still fails the task. This runtime
recovery policy is recorded in task specifications and trajectory protocol events. It does
not replace the simulator with a solo agent or fabricate customer messages.

Choose an existing agent and organization API key using the
[SDK agent guide](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/agent.mdx). Register the organization
secret `OPENAI_API_KEY` for the upstream user simulator and judge; its value is never
included in the image or upload.

```sh
uv run examples/tau3_banking/ingest.py --agent-name YOUR_AGENT_NAME
```

Ingestion waits for a new runtime build. Follow the
[benchmark diagnostic guidance](../task_diagnostics.md) to execute a subset before training.
Image build readiness alone does not verify policy inference, grading, or reward publication.
