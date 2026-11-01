# Modified by Trajectory: isolates agent code (nobody, no secrets), reaps it, and caps output.
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import tempfile
from typing import Any

from big_finance_harness.tools.base import Tool, ToolError

DEFAULT_TIMEOUT_S = 5.0
MAX_OUTPUT_CHARS = 8000
NOBODY = 65534
# A UTF-8 character is at most 4 bytes, so this many bytes always cover MAX_OUTPUT_CHARS.
MAX_OUTPUT_BYTES = 4 * MAX_OUTPUT_CHARS
NO_CHILDREN = (
    "import resource, sys; "
    "resource.setrlimit(resource.RLIMIT_NPROC, (1, 1)); "
    "exec(compile(sys.argv[1], '<python_exec>', 'exec'))"
)


async def _read_capped(stream: asyncio.StreamReader) -> bytes:
    kept = bytearray()
    while chunk := await stream.read(65536):
        kept += chunk[: MAX_OUTPUT_BYTES - len(kept)]
    return bytes(kept)


class PythonExecTool(Tool):
    name = "python_exec"
    description = (
        "Execute a short Python snippet and return its captured stdout/stderr. Use it "
        "for arithmetic, unit conversions, and small calculations against values you have "
        "already retrieved. The interpreter runs with a 5-second timeout in an isolated "
        "child process; it is not a security sandbox. There is no persistent state "
        "between calls — each call runs a fresh interpreter. Subprocesses are unavailable."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s

    async def run(self, args: dict[str, Any]) -> str:
        code = args.get("code", "")
        if not code.strip():
            raise ToolError("code is required and must be non-empty")

        with tempfile.TemporaryDirectory() as cwd:
            # As root, a separate uid keeps /proc/<parent>/environ and its API keys unreadable.
            drop = {}
            if os.geteuid() == 0:
                os.chown(cwd, NOBODY, NOBODY)
                drop = {"user": NOBODY, "group": NOBODY, "extra_groups": []}
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",  # isolated mode: ignore PYTHON* env vars and user site-packages
                "-c",
                NO_CHILDREN,
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={"OPENBLAS_NUM_THREADS": "1"},
                cwd=cwd,
                start_new_session=True,
                **drop,
            )
            try:
                stdout_b, stderr_b, _ = await asyncio.wait_for(
                    asyncio.gather(
                        _read_capped(proc.stdout), _read_capped(proc.stderr), proc.wait()
                    ),
                    timeout=self.timeout_s,
                )
            except asyncio.TimeoutError:
                raise ToolError(f"python_exec timed out after {self.timeout_s}s") from None
            finally:
                # Processes the snippet started must not outlive the call.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()

        stdout = stdout_b.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]
        stderr = stderr_b.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]
        rc = proc.returncode

        parts: list[str] = [f"exit_code: {rc}"]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        return "\n\n".join(parts)
