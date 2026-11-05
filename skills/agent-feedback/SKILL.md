---
name: agent-feedback
description: Send customer-authorized Trajectory feedback, suggestions, or positive observations as prose or partial structured details about the API, SDK, documentation, or workflow.
---

Load the full `agent-feedback` instructions from the Python environment you will use for
this project's Trajectory SDK calls. Use the project's interpreter (for example,
`uv run python`); a globally installed CLI can have a different SDK version.

Run this with that interpreter, then read the printed file before acting:

```python
import importlib.metadata as metadata

files = metadata.files("trajectory-sdk")
print(next(f.locate().resolve() for f in files
           if f.as_posix().endswith("skills/agent-feedback/SKILL.md")))
```

If the SDK is absent, install it using the project's dependency workflow first.
Read the bundled instructions again when the project's SDK version changes.
Installing this entrypoint does not authorize reporting feedback or launching runs.
