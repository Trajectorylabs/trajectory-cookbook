"""Podman flags for nested runtimes that disable cgroups (no podman needed)."""

from __future__ import annotations

import subprocess

import pytest
from lab_core.sandbox import sandbox as sandbox_module
from lab_core.sandbox.sandbox import Sandbox


@pytest.mark.parametrize(
    ("disable", "runtime", "expect_limits"),
    [("1", "crun", False), ("1", "runc", True), (None, "crun", True)],
)
def test_resource_limits_follow_cgroups(
    tmp_path, monkeypatch, disable, runtime, expect_limits
):
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        stdout = runtime if cmd[:2] == ["podman", "info"] else "container"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sandbox_module, "_cgroup_controller_available", lambda name: True
    )
    if disable is None:
        monkeypatch.delenv("HARVEY_PODMAN_DISABLE_CGROUPS", raising=False)
    else:
        monkeypatch.setenv("HARVEY_PODMAN_DISABLE_CGROUPS", disable)

    sandbox = Sandbox(
        documents_dir=tmp_path / "documents",
        output_dir=tmp_path / "output",
        workspace_dir=tmp_path / "workspace",
    )
    sandbox._start_container()

    run_cmd = next(cmd for cmd in commands if cmd[:2] == ["podman", "run"])
    limits = [
        flag
        for flag in run_cmd
        if flag.startswith(("--cpus", "--memory", "--pids-limit"))
    ]
    assert bool(limits) is expect_limits
    assert ("--cgroups=disabled" in run_cmd) is (disable == "1" and runtime == "crun")
