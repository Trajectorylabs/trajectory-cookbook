# Ingest a benchmark and get its runtimes ready

Choose how to supply the runtime, submit the original task inventory, then use the ingestion
records to fix failures. You are done with ingestion when every submitted task is accounted
for and its required runtime is ready, or its failure is recorded with a cause and next action.
Runtime readiness does not prove that the task's solver or grader works.

> Release note: automatic context archives, private GAR pull credentials and the expanded
> ingestion-history examples below target the upcoming matching API/SDK release. Their fresh-client
> walkthrough and full-population qualification are still pending. Do not infer availability
> from installing an older published SDK.

## 1. Choose the runtime path

Both paths produce the same task/runtime records and use the same inspection and repair loop.
Keep the benchmark's native files, commands, task IDs, splits and actor/private-grader boundary.

| You have | Use | Trajectory does |
| --- | --- | --- |
| An ordinary Dockerfile and local files | `DockerfileBuild("runtime/Dockerfile")` | Uploads the selected context privately and builds the runtime. |
| An image you already build and publish | `ImageRef("registry.example.com/runtime@sha256:...")` | Pulls the pinned image and adds the execution bootstrap. |
| A ready runtime in your organization | `RuntimeRef("rt_YOUR_RUNTIME")` | Reuses that runtime. This is not a fresh build or registry-pull test. |

### Managed build: keep your Dockerfile and files

```python
from trajectory.lib import DockerfileBuild

runtime = DockerfileBuild("runtime/Dockerfile")
```

With `root=Path("my-benchmark")`, the context is `my-benchmark/runtime/`. The SDK selects
files beneath that Dockerfile's directory, filtered by `.dockerignore` at the benchmark root.
It always includes the selected Dockerfile; it does not infer the context from `COPY` statements.
A root-level Dockerfile therefore selects the checkout unless you exclude files.

The SDK packages each selected context automatically and uses the existing upload session.
You do not create an archive, change `COPY` commands or upload each file yourself. The worker
verifies and restores the context before building. File modes and safe relative symlinks within
the selected context are preserved; escaping links, links to excluded files and unsupported
file types fail validation. Put shared code in each context or provide it through your base
image. Packaging does not deduplicate different contexts into shared cloud layers.

Keep build inputs stable until upload finishes. Older SDKs can continue sending loose files;
existing ready runtimes remain usable. The packed representation has a different identity, so
its first submission can require a new build even when equivalent loose files were built before.

### Prebuilt image: build and publish once

```python
from trajectory.lib import ImageRef

runtime = ImageRef("registry.example.com/runtime@sha256:YOUR_DIGEST")
```

Use the image digest returned by your registry. On the initial Modal path, build for
`linux/amd64`. Include the harness, task files, verifier dependencies and the harness's SDK
installation in the image; the execution bootstrap does not install them. Public image
references require no pull-secret field and keep their existing behavior.

Your local or CI credentials authorize building and pushing the image. They do not automatically
authorize Trajectory to pull it. For a private image, use the pull-secret setup below. Private
base images in a managed Dockerfile and credentials needed by a Dockerfile `RUN` command are
separate capabilities; this pull-secret field does not supply them. Build and publish the image
with your own tooling when those credentials are needed.

### Authenticate each operation

Set `TRAJECTORY_API_KEY` for the organization receiving the benchmark. `Client()` uses that key
for Trajectory API calls. Trajectory supplies signed upload authorization and its worker/provider
credentials; you do not need cloud-storage credentials for managed context delivery.

The candidate private-image adapter supports **Google Artifact Registry on Modal**. Save a
pull credential as an existing organization secret, then reference its **name**, not its value
or secret ID:

```python
import json
import os

from trajectory import Client, SecretRef
from trajectory.lib import ImageRef

client = Client()
created = client.secrets.create(
    name="gar-pull",
    value=json.dumps({
        "username": "oauth2accesstoken",
        "password": os.environ["GAR_ACCESS_TOKEN"],
    }),
)
runtime = ImageRef(
    "us-central1-docker.pkg.dev/PROJECT/REPOSITORY/runtime@sha256:YOUR_DIGEST",
    registry_secret=SecretRef(secret_ref="gar-pull"),
)
```

Obtain a fresh token from an identity authorized to read that repository; see Google's
[Artifact Registry authentication instructions](https://docs.cloud.google.com/artifact-registry/docs/docker/authentication#token).
The token normally lasts 60 minutes, including time spent queued before the provider pulls it.
Trajectory does not refresh it. The exact username is `oauth2accesstoken`.

If your organization already provides a Google service-account JSON key, the secret value can
instead be the contents of that key file:

```python
from pathlib import Path

created = client.secrets.create(
    name="gar-pull-key",
    value=Path(os.environ["GAR_SERVICE_ACCOUNT_JSON_FILE"]).read_text(),
)
# Use SecretRef(secret_ref="gar-pull-key") in ImageRef.
```

Use a repository-scoped reader for pulling. Keep push credentials in your own build environment
and keep all credential values out of Dockerfiles, benchmark manifests and logs. The API key's
organization controls access to the saved secret. The pull secret is not injected into task `env_vars`; reference separate runtime credentials
there when your harness needs them. Other private registries/providers are not qualified by
this adapter.

To replace a credential, let its active build settle, revoke it with
`client.secrets.revoke(created.secret.secret_id)`, then create the replacement using the same
name. There is no in-place secret update or automatic token refresh. A ready imported image can
still be reused after revocation; that reuse does not prove the replacement credential can pull.
A fresh pull is required to verify new registry access. Where key creation is disabled, use an authorized access token. Rotate any long-lived key
through your organization's credential process; Trajectory does not rotate it for you.

## 2. Submit and keep the operation ID

Use the chosen `runtime` with your original tasks. Preserve dataset identifiers in `name` and
set each original `split`; omit platform task IDs for new tasks. Here is a one-task example:

```python
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import start_push

client = Client()
benchmark = BenchmarkSpec(
    name="my-benchmark",
    runtime=runtime,
    tasks=[TaskSpec(
        name="source-task-001",
        split="test",
        run_command="python /app/harness.py",
    )],
)
operation = start_push(
    client, benchmark,
    agent_id="agt_YOUR_AGENT",
    root=Path("my-benchmark"),
    build_images=True,
    idempotency_key="YOUR_SAVED_SUBMISSION_KEY",
)
print(operation.id)
status = operation.refresh()
print(status.status, status.stage, status.registered_tasks, status.total_tasks,
      status.ready_runtimes, status.total_runtimes, status.last_error)
```

Save a stable idempotency key **before** starting a large submission. Repeating unchanged input
with that key lets the SDK reuse completed objects with matching checksums; an interrupted
single-object upload may restart. Changed files, tasks or settings require a new key. Do not
rename the benchmark merely to retry: keep its agent and name so versions stay grouped.

`start_push()` returns after transfer and acceptance; registration/builds continue on the server.
`operation.refresh()` reconnects to that work. A client timeout does not cancel it. After restarting
your client, use `get_operation(client, operation_id)` from `trajectory.lib` before submitting again.
`push()` is the convenience that also waits for requested image builds.

The upload callback reports completed or reused objects/bytes, not instantaneous network traffic.
Operation `upload_progress` distinguishes declared and verified totals and includes task parts.
Verified upload bytes do not establish task registration or runtime readiness.

## 3. Inspect every input and failed runtime

Find an operation in history, including one that failed before registration:

```python
for operation in client.benchmarks.ingestion.list():
    print(operation.operation_id, operation.bench_id, operation.status)
```

Use `bench_id=...` to restrict that history to one benchmark version. For a selected operation,
account for original inputs even when they have no registered task ID:

```python
operation_id = "iop_YOUR_OPERATION"
cursor = None
while True:
    page = client.benchmarks.ingestion.list_inputs(operation_id, cursor=cursor, limit=100)
    for item in page.items:
        print(item.part_path, item.task_index, item.name, item.status,
              item.task_id, item.runtime, item.failure)
    if page.unavailable_parts:
        print("Unresolved input parts:", page.unavailable_parts)
    cursor = page.next_cursor
    if cursor is None:
        break
```

`pending` means registration has not finished; `rejected` has a recorded task failure;
`not_registered` means the operation ended without a registration receipt. `registered` does
not establish runtime readiness. Use `(operation_id, part_path, task_index)` to identify an input
before it receives a task ID. These reads use retained upload parts and registration receipts;
they do not register or rebuild anything.

Malformed or unverified parts appear in `unavailable_parts`. Follow the cursor even when a page
contains no task rows, and refresh from the first page to observe progress. Keep your original
source count: the platform cannot infer identities from an unreadable part or a task never submitted.

Inspect task results, failures and runtime records through the same paginated interface:

```python
for kind in ("result", "failure", "runtime"):
    after = ""
    while True:
        page = client.benchmarks.ingestion.list_items(
            operation_id, kind=kind, after=after, limit=100,
        )
        for item in page.items:
            print(kind, item.model_dump())
        if page.next_cursor is None:
            break
        after = page.next_cursor
```

A shared-runtime failure can affect many tasks. Use `kind="runtime_task"` and
`runtime_id="rt_YOUR_RUNTIME"` with that pagination loop to identify them. Preserve unaffected
work and the original task denominator while repairing the shared cause.

Read fresh build output from the failed operation/runtime pair before rebuilding:

```python
logs = client.benchmarks.ingestion.build_logs(
    operation_id, runtime_id="rt_YOUR_RUNTIME",
)
print(logs.observed_at, logs.provider_ref, logs.available)
if logs.available:
    print(logs.excerpt)
```

This is a bounded read of the original build, with registered secret values redacted. It returns
up to 6,000 characters. `available=True` means text arrived, not that the provider's final error
has arrived. A later read can reveal the final lines; rebuilding is unnecessary to obtain them.
A timeout or missing historical reference may return `available=False`. Keep that diagnostic
limitation alongside the original failure. Credentials printed by build code outside the known
redaction set may remain visible to your organization's authorized users.

## 4. Fix the cause and retry the affected work

| Observed failure | Next action |
| --- | --- |
| Transfer interrupted; inputs unchanged | Repeat the original submission with the saved key. Completed verified objects can be reused. |
| Invalid context, changed Dockerfile/dependencies or task settings | Correct the owning source or integration, then submit the intended task set with a new key. |
| Private pull denied or credential expired | Check repository read access and replace the saved credential. For a registered runtime, use the runtime retry below; for a rejected input, resubmit with a new key. Verify a fresh pull. |
| Transient runtime build failure; stored source unchanged | After ingestion settles, call `client.benchmarks.images.build(bench_id)`, then inspect `images.list(bench_id)`. |

`images.build` retries eligible pending/failed runtime work for the benchmark, not a selected
single task. Ready runtimes and identified in-flight builds are skipped. Do not overlap it with
an active ingestion operation to bypass scheduling. Earlier operation failures remain historical;
a successful retry does not erase them.

For corrected source, keep the same agent and benchmark name and save the new operation and
benchmark IDs. A new version contains only the tasks you submit; a subset does not inherit the
rest of the old version. Keep first-pass and recovered outcomes separate. A deterministic shared
failure should be diagnosed once rather than repeated across the queue; task-local failures need
not block independent healthy tasks.

Managed upload/context bounds protect worker resources and are distinct from provider image
constraints. If a bound rejects input, retain its dimension, observed value and allowed value.
The new archive path's supported capacity is still being measured; this draft does not publish
its provisional member or memory guards as qualified customer limits. Image layers do not become
managed source contexts merely because they belong to the same benchmark.

## 5. Confirm readiness, then test the task

```python
bench_id = status.bench_id
images = client.benchmarks.images.list(bench_id)
for image in images.images:
    print(image.task_id, image.build_status, image.provider_ref, image.failure_message)
```

Reconcile every input with its registered task and runtime result. Registration counts, operation
completion and image readiness answer different questions. Keep unresolved inputs and failed
runtimes visible; a subset of ready images is not full-population success.

Use the [task-diagnostic walkthrough](task_diagnostics.md) to check native startup,
imports, services and credentials on representative ready runtimes. That runs task code and is a
separate gate from ingestion. Once you need solver/grader evidence, continue with
[execution, native reports and repair](diagnostic_artifacts.md#1-check-native-prerequisites-cheaply).
An unresolved grader configuration alone is not a reason to exclude a task from ingestion;
record the stage it prevents and preserve its source identity.
