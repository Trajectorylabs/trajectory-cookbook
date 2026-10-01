# BigCodeBench through the public SDK

Reuses the full-source adapter from [BigCodeBench PR #5](https://github.com/Trajectorylabs/bigcodebench/pull/5)
at `9fb55cf671385b2f21de0e62e095d108f3478d46`. The original author source is pinned to
`09dd993f46c3fbf3a799465bb96d524edcb0b199`, and the v0.1.4 dataset to
`b74c0d0bf70d2c0bc459be537895cca163007f1a` with upstream hash verification.

The small `mainline-sdk.patch` upgrades the adapter's SDK pin from 0.6.12 to 0.6.20
and uses the existing `client.chat.completions.create` hook. It preserves the
trajectory header, event/reward/completion acknowledgment checks, original grader,
network access, executable temporary filesystem, and isolated grader children.

From the cookbook root, prepare a fresh export and ingest it:

```bash
bash examples/bigcodebench/prepare.sh
uv run --env-file .env examples/bigcodebench/ingest.py
```

Preparation requires Git access to `Trajectorylabs/bigcodebench`, `uv`, and internet
access to the pinned public dataset. It creates `.work/source` and `.work/prepared`;
it deliberately refuses to overwrite an existing checkout or export. The source
adapter's 26 tests run before preparation. All 1,140 tasks are TEST, and all nine
original fields are retained. The output token budget is 32,768.

Ingestion uses the public `trajectory.lib.submit(..., build_images=True)` workflow.
It verifies exported file hashes, saves an operation receipt before waiting, and
checks every task acknowledgment. Rerun the ingestion command to reconnect with
`get_operation`; it does not create another benchmark. Results and runtime IDs are
saved to `.work/prepared/ingestion-results.json`.

Verified ingestion: `bm_06ab3469bf6a7b4d80009f8804847780`, with all 1,140 task
acknowledgments, zero failures, and one ready runtime. [verification.json](verification.json)
records the source/SDK revisions, hashes, operation ID, and remaining validation gates.

To use a freshly pulled SDK checkout instead of the equivalent published release:

```bash
uv run --env-file .env --no-project --python 3.12 \
  --with /absolute/path/to/trajectory-platform \
  python examples/bigcodebench/ingest.py
```

Registration and a ready runtime do not prove hosted grading. The adapter's
`sdk.check_runtime --full-reference` separately checks the exported runtime,
network/shell behavior, isolation, and original canonical solutions. Its official
grader dependency image has about 9.27 GB of compressed layers; tasks request
8 GiB memory and a Docker engine. Those native checks have not been rerun here.
Historical native results in TRA-1444 are evidence for their original revision only.

The original grader's assertion/loader-mutation defect remains open under
[TRA-1270](https://linear.app/trajectoryai/issue/TRA-1270). No TRAIN split is invented,
and training/overlap and hosted-admission gates from
[TRA-1262](https://linear.app/trajectoryai/issue/TRA-1262) remain in force.
